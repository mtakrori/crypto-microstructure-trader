#!/usr/bin/env python3
"""
Data Quality Validator for Crypto Microstructure Trading System
UPDATED: Enhanced validation with live data checks and microstructure-specific requirements
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from config import (
    DATABASE_FILE, SYMBOLS, TIMEFRAMES, 
    DATA_VALIDATION_CONFIG, get_table_name
)

@dataclass
class DataQualityReport:
    """Data quality assessment report with microstructure enhancements"""
    symbol: str
    timeframe: str
    is_valid: bool
    total_candles: int
    recent_candles: int
    missing_candles: int
    invalid_candles: int
    latest_timestamp: Optional[datetime]
    data_age_minutes: float
    issues: List[str]
    quality_score: float  # 0-1 score
    has_live_data: bool = False  # New field
    live_candle_age_seconds: float = 0.0  # New field
    is_microstructure_ready: bool = False  # New field

@dataclass
class MicrostructureReadiness:
    """Microstructure trading readiness assessment"""
    symbol: str
    is_ready: bool
    has_live_1m: bool
    has_live_5m: bool
    live_data_age_seconds: float
    complete_candle_count: int
    issues: List[str]
    readiness_score: float

class DataValidator:
    """Comprehensive data quality validator with microstructure support"""
    
    def __init__(self):
        self.logger = logging.getLogger('DataValidator')
        self.config = DATA_VALIDATION_CONFIG
        
    def validate_symbol_timeframe(self, symbol: str, timeframe: str) -> DataQualityReport:
        """Validate data quality for a specific symbol/timeframe with live data checks"""
        table_name = get_table_name(symbol, timeframe)
        issues = []
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                ''', (table_name,))
                
                if not cursor.fetchone():
                    return DataQualityReport(
                        symbol=symbol,
                        timeframe=timeframe,
                        is_valid=False,
                        total_candles=0,
                        recent_candles=0,
                        missing_candles=0,
                        invalid_candles=0,
                        latest_timestamp=None,
                        data_age_minutes=float('inf'),
                        issues=[f"Table {table_name} does not exist"],
                        quality_score=0.0,
                        has_live_data=False,
                        live_candle_age_seconds=float('inf'),
                        is_microstructure_ready=False
                    )
                
                # Get statistics INCLUDING incomplete candles
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as total_candles,
                        MAX(open_time) as latest_timestamp,
                        MIN(open_time) as earliest_timestamp,
                        SUM(CASE WHEN is_complete = 0 THEN 1 ELSE 0 END) as incomplete_count,
                        SUM(CASE WHEN is_complete = 1 THEN 1 ELSE 0 END) as complete_count
                    FROM {table_name}
                ''')
                
                stats = cursor.fetchone()
                total_candles = stats[0]
                latest_timestamp_ms = stats[1]
                earliest_timestamp_ms = stats[2]
                incomplete_count = stats[3] or 0
                complete_count = stats[4] or 0
                
                if total_candles == 0:
                    return DataQualityReport(
                        symbol=symbol,
                        timeframe=timeframe,
                        is_valid=False,
                        total_candles=0,
                        recent_candles=0,
                        missing_candles=0,
                        invalid_candles=0,
                        latest_timestamp=None,
                        data_age_minutes=float('inf'),
                        issues=["No data available"],
                        quality_score=0.0,
                        has_live_data=False,
                        live_candle_age_seconds=float('inf'),
                        is_microstructure_ready=False
                    )
                
                # Convert timestamps
                latest_timestamp = datetime.fromtimestamp(latest_timestamp_ms / 1000, tz=timezone.utc)
                earliest_timestamp = datetime.fromtimestamp(earliest_timestamp_ms / 1000, tz=timezone.utc)
                
                # Calculate data age
                current_time = datetime.now(timezone.utc)
                data_age_minutes = (current_time - latest_timestamp).total_seconds() / 60
                
                # Check for LIVE data (incomplete candle) - CRITICAL FOR MICROSTRUCTURE
                cursor.execute(f'''
                    SELECT open_time, close, volume 
                    FROM {table_name}
                    WHERE is_complete = 0
                    ORDER BY open_time DESC
                    LIMIT 1
                ''')
                
                live_candle = cursor.fetchone()
                has_live_data = live_candle is not None
                live_candle_age_seconds = float('inf')
                
                if not has_live_data:
                    issues.append("❌ No live/incomplete candle available")
                else:
                    # Check how fresh the live candle is
                    live_candle_age_ms = current_time.timestamp() * 1000 - live_candle[0]
                    live_candle_age_seconds = live_candle_age_ms / 1000
                    
                    # Different freshness requirements based on timeframe
                    max_age_seconds = {
                        '1m': 65,    # 1 minute + 5 second buffer
                        '5m': 305,   # 5 minutes + 5 second buffer
                        '15m': 905,  # 15 minutes + 5 second buffer
                        '1h': 3605,  # 1 hour + 5 second buffer
                        '4h': 14405, # 4 hours + 5 second buffer
                        '1d': 86405  # 1 day + 5 second buffer
                    }.get(timeframe, 65)
                    
                    if live_candle_age_seconds > max_age_seconds:
                        issues.append(f"⚠️ Live candle stale: {live_candle_age_seconds:.1f}s old")
                
                # Check recent COMPLETE candles for analysis
                one_hour_ago_ms = int((current_time - timedelta(hours=1)).timestamp() * 1000)
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE open_time > ? AND is_complete = 1
                ''', (one_hour_ago_ms,))
                recent_complete_candles = cursor.fetchone()[0]
                
                # Check all recent candles including incomplete
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE open_time > ?
                ''', (one_hour_ago_ms,))
                recent_candles = cursor.fetchone()[0]
                
                # Validate minimum complete candles for indicators
                if complete_count < self.config['min_candles_required']:
                    issues.append(f"Insufficient complete data: {complete_count} < {self.config['min_candles_required']} required")
                
                # MICROSTRUCTURE-SPECIFIC CHECKS
                is_microstructure_ready = False
                if timeframe in ['1m', '5m']:
                    # For microstructure, we need VERY fresh data
                    max_age_for_microstructure = {
                        '1m': 1.0,  # 1 minute max
                        '5m': 2.0   # 2 minutes max
                    }.get(timeframe, 1.0)
                    
                    if data_age_minutes > max_age_for_microstructure:
                        issues.append(f"🚫 Data too stale for microstructure: {data_age_minutes:.1f} min old")
                    
                    # Must have live data for microstructure
                    if not has_live_data:
                        issues.append("🚫 No live data for microstructure trading")
                    elif live_candle_age_seconds > 10:  # Live candle must be very fresh
                        issues.append(f"🚫 Live candle too old for microstructure: {live_candle_age_seconds:.1f}s")
                    
                    # Check if microstructure ready
                    is_microstructure_ready = (
                        has_live_data and
                        live_candle_age_seconds <= 10 and
                        data_age_minutes <= max_age_for_microstructure and
                        complete_count >= 50  # Need at least 50 complete candles for indicators
                    )
                
                # Standard freshness check
                if data_age_minutes > self.config['max_data_age_minutes']:
                    issues.append(f"Stale data: {data_age_minutes:.1f} minutes old")
                
                # Check for data gaps and anomalies
                missing_candles = self._check_data_gaps(conn, table_name, timeframe, earliest_timestamp, latest_timestamp)
                invalid_candles = self._check_data_anomalies(conn, table_name)
                
                if missing_candles > 0:
                    missing_percentage = (missing_candles / total_candles) * 100
                    if missing_percentage > self.config['missing_data_tolerance'] * 100:
                        issues.append(f"Too many gaps: {missing_candles} missing ({missing_percentage:.1f}%)")
                
                if invalid_candles > 0:
                    invalid_percentage = (invalid_candles / total_candles) * 100
                    if invalid_percentage > 0.01:  # More than 1% invalid
                        issues.append(f"Invalid candles: {invalid_candles} ({invalid_percentage:.1f}%)")
                
                # Calculate quality score with microstructure emphasis
                quality_score = self._calculate_quality_score_microstructure(
                    total_candles, recent_candles, missing_candles, 
                    invalid_candles, data_age_minutes, has_live_data,
                    live_candle_age_seconds, timeframe
                )
                
                # Determine overall validity
                is_valid = (
                    len([i for i in issues if '🚫' not in i and '❌' not in i]) == 0 and
                    complete_count >= self.config['min_candles_required'] and
                    data_age_minutes <= self.config['max_data_age_minutes'] and
                    quality_score >= 0.7
                )
                
                # Add summary status to issues if needed
                if is_microstructure_ready:
                    issues.insert(0, "✅ READY FOR MICROSTRUCTURE TRADING")
                elif timeframe in ['1m', '5m'] and not is_microstructure_ready:
                    issues.insert(0, "❌ NOT READY FOR MICROSTRUCTURE TRADING")
                
                return DataQualityReport(
                    symbol=symbol,
                    timeframe=timeframe,
                    is_valid=is_valid,
                    total_candles=total_candles,
                    recent_candles=recent_candles,
                    missing_candles=missing_candles,
                    invalid_candles=invalid_candles,
                    latest_timestamp=latest_timestamp,
                    data_age_minutes=data_age_minutes,
                    issues=issues,
                    quality_score=quality_score,
                    has_live_data=has_live_data,
                    live_candle_age_seconds=live_candle_age_seconds,
                    is_microstructure_ready=is_microstructure_ready
                )
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error validating {symbol} {timeframe}: {e}")
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                is_valid=False,
                total_candles=0,
                recent_candles=0,
                missing_candles=0,
                invalid_candles=0,
                latest_timestamp=None,
                data_age_minutes=float('inf'),
                issues=[f"Database error: {str(e)}"],
                quality_score=0.0,
                has_live_data=False,
                live_candle_age_seconds=float('inf'),
                is_microstructure_ready=False
            )
    
    def _check_data_gaps(self, conn: sqlite3.Connection, table_name: str, 
                        timeframe: str, start_time: datetime, end_time: datetime) -> int:
        """Check for missing candles in the expected sequence"""
        try:
            # Calculate expected interval in minutes
            interval_minutes = {
                '1m': 1, '5m': 5, '15m': 15, 
                '1h': 60, '4h': 240, '1d': 1440
            }.get(timeframe, 1)
            
            # Get all timestamps (including incomplete)
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT open_time FROM {table_name} 
                ORDER BY open_time
            ''')
            
            timestamps = [row[0] for row in cursor.fetchall()]
            
            if len(timestamps) < 2:
                return 0
            
            # Check for gaps
            missing_count = 0
            expected_interval_ms = interval_minutes * 60 * 1000
            
            for i in range(1, len(timestamps)):
                time_diff = timestamps[i] - timestamps[i-1]
                expected_candles = time_diff // expected_interval_ms
                
                if expected_candles > 1:
                    missing_count += expected_candles - 1
            
            return int(missing_count)
            
        except Exception as e:
            self.logger.warning(f"Error checking data gaps for {table_name}: {e}")
            return 0
    
    def _check_data_anomalies(self, conn: sqlite3.Connection, table_name: str) -> int:
        """Check for data anomalies and invalid candles"""
        try:
            cursor = conn.cursor()
            
            # Check for invalid OHLC relationships
            cursor.execute(f'''
                SELECT COUNT(*) FROM {table_name}
                WHERE high < open OR high < close OR high < low
                   OR low > open OR low > close OR low > high
                   OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                   OR volume < 0
            ''')
            invalid_ohlc = cursor.fetchone()[0]
            
            # Check for extreme price movements (>10% in single candle)
            cursor.execute(f'''
                SELECT COUNT(*) FROM {table_name}
                WHERE ABS(close - open) / open > {self.config['price_change_threshold']}
                   AND is_complete = 1
            ''')
            extreme_moves = cursor.fetchone()[0]
            
            # Check for volume spikes (>10x average) - only on complete candles
            cursor.execute(f'''
                SELECT AVG(volume) FROM {table_name}
                WHERE volume > 0 AND is_complete = 1
            ''')
            avg_volume = cursor.fetchone()[0] or 0
            
            if avg_volume > 0:
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table_name}
                    WHERE volume > {avg_volume * self.config['volume_spike_threshold']}
                       AND is_complete = 1
                ''')
                volume_spikes = cursor.fetchone()[0]
            else:
                volume_spikes = 0
            
            return invalid_ohlc + extreme_moves + volume_spikes
            
        except Exception as e:
            self.logger.warning(f"Error checking anomalies for {table_name}: {e}")
            return 0
    
    def _calculate_quality_score_microstructure(self, total_candles: int, recent_candles: int,
                                missing_candles: int, invalid_candles: int,
                                data_age_minutes: float, has_live_data: bool,
                                live_candle_age_seconds: float, timeframe: str) -> float:
        """Calculate quality score with emphasis on microstructure requirements"""
        try:
            score = 1.0
            
            # For microstructure timeframes (1m, 5m), live data is CRITICAL
            if timeframe in ['1m', '5m']:
                # Live data availability (40% weight for microstructure)
                if not has_live_data:
                    score *= 0.6  # Heavy penalty for no live data
                elif live_candle_age_seconds > 10:
                    # Penalty based on staleness of live candle
                    staleness_penalty = min(0.5, live_candle_age_seconds / 60)
                    score *= (1 - staleness_penalty * 0.4)
                
                # Data freshness (30% weight for microstructure)
                if data_age_minutes > 1.0:
                    freshness_penalty = min(0.9, data_age_minutes / 5.0)
                    score *= (1 - freshness_penalty * 0.3)
            else:
                # For non-microstructure timeframes, standard scoring
                # Live data (10% weight)
                if not has_live_data:
                    score *= 0.9
                
                # Data freshness (20% weight)
                if data_age_minutes > self.config['max_data_age_minutes']:
                    age_penalty = min(0.5, data_age_minutes / (self.config['max_data_age_minutes'] * 2))
                    score *= (1 - age_penalty * 0.2)
            
            # Sufficient historical data (20% weight)
            if total_candles < self.config['min_candles_required']:
                score *= (total_candles / self.config['min_candles_required']) * 0.8 + 0.2
            
            # Missing data penalty (5% weight)
            if total_candles > 0:
                missing_ratio = missing_candles / total_candles
                if missing_ratio > self.config['missing_data_tolerance']:
                    score *= (1 - min(0.05, missing_ratio * 0.05))
            
            # Invalid data penalty (5% weight)
            if total_candles > 0:
                invalid_ratio = invalid_candles / total_candles
                if invalid_ratio > 0.01:  # Only penalize if >1% invalid
                    score *= (1 - min(0.05, invalid_ratio * 0.05))
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    def check_microstructure_readiness(self, symbol: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive check for microstructure trading readiness
        Returns (is_ready, list_of_issues)
        """
        issues = []
        
        try:
            # Check 1m data (primary for microstructure)
            report_1m = self.validate_symbol_timeframe(symbol, '1m')
            
            if not report_1m.is_microstructure_ready:
                issues.extend([f"1m: {issue}" for issue in report_1m.issues if '❌' in issue or '🚫' in issue])
            
            # Check 5m data (secondary for context)
            report_5m = self.validate_symbol_timeframe(symbol, '5m')
            
            # 5m is optional but helpful
            if not report_5m.has_live_data:
                issues.append("5m: No live data (optional but recommended)")
            
            # Additional microstructure-specific checks
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                
                # Check 1m live candle freshness
                table_1m = get_table_name(symbol, '1m')
                cursor.execute(f'''
                    SELECT open_time, close, volume
                    FROM {table_1m}
                    WHERE is_complete = 0
                    ORDER BY open_time DESC
                    LIMIT 1
                ''')
                
                live_candle = cursor.fetchone()
                
                if not live_candle:
                    issues.append("No live 1m candle available")
                    return False, issues
                
                # Check live candle age
                current_time_ms = datetime.now(timezone.utc).timestamp() * 1000
                candle_age_seconds = (current_time_ms - live_candle[0]) / 1000
                
                if candle_age_seconds > 10:  # Maximum 10 seconds for microstructure
                    issues.append(f"Live 1m candle too stale: {candle_age_seconds:.1f}s old")
                    return False, issues
                
                # Check volume is reasonable (not zero)
                if live_candle[2] == 0:
                    issues.append("Live candle has zero volume")
                
                # Check sufficient complete candles for indicators
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table_1m}
                    WHERE is_complete = 1
                ''')
                
                complete_count = cursor.fetchone()[0]
                
                if complete_count < 50:
                    issues.append(f"Insufficient 1m history: {complete_count} complete candles (need 50+)")
                    return False, issues
                
                # Check for recent data consistency
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table_1m}
                    WHERE open_time > {current_time_ms - 600000}  -- Last 10 minutes
                ''')
                
                recent_count = cursor.fetchone()[0]
                
                if recent_count < 5:  # Should have at least 5 candles in last 10 minutes
                    issues.append(f"Sparse recent data: only {recent_count} candles in last 10 minutes")
            
            # If we get here with no critical issues, we're ready
            is_ready = len(issues) == 0 or all('optional' in issue.lower() for issue in issues)
            
            return is_ready, issues
            
        except Exception as e:
            issues.append(f"Error checking readiness: {str(e)}")
            return False, issues
    
    def get_microstructure_readiness_report(self, symbol: str) -> MicrostructureReadiness:
        """Get detailed microstructure readiness report"""
        try:
            # Get basic readiness
            is_ready, issues = self.check_microstructure_readiness(symbol)
            
            # Get detailed metrics
            report_1m = self.validate_symbol_timeframe(symbol, '1m')
            report_5m = self.validate_symbol_timeframe(symbol, '5m')
            
            # Calculate readiness score
            readiness_score = 0.0
            
            # 1m data (60% weight)
            if report_1m.has_live_data:
                readiness_score += 0.3
            if report_1m.live_candle_age_seconds <= 10:
                readiness_score += 0.3
            
            # 5m data (20% weight)
            if report_5m.has_live_data:
                readiness_score += 0.1
            if report_5m.live_candle_age_seconds <= 30:
                readiness_score += 0.1
            
            # Complete data (20% weight)
            if report_1m.total_candles >= 100:
                readiness_score += 0.2
            
            return MicrostructureReadiness(
                symbol=symbol,
                is_ready=is_ready,
                has_live_1m=report_1m.has_live_data,
                has_live_5m=report_5m.has_live_data,
                live_data_age_seconds=report_1m.live_candle_age_seconds,
                complete_candle_count=report_1m.total_candles - (1 if report_1m.has_live_data else 0),
                issues=issues,
                readiness_score=readiness_score
            )
            
        except Exception as e:
            self.logger.error(f"Error generating readiness report for {symbol}: {e}")
            return MicrostructureReadiness(
                symbol=symbol,
                is_ready=False,
                has_live_1m=False,
                has_live_5m=False,
                live_data_age_seconds=float('inf'),
                complete_candle_count=0,
                issues=[f"Error: {str(e)}"],
                readiness_score=0.0
            )
    
    def validate_all_data(self) -> Dict[str, Dict[str, DataQualityReport]]:
        """Validate data quality for all symbols and timeframes"""
        results = {}
        
        for symbol in SYMBOLS:
            results[symbol] = {}
            for timeframe in TIMEFRAMES:
                results[symbol][timeframe] = self.validate_symbol_timeframe(symbol, timeframe)
        
        return results
    
    def get_trading_ready_pairs(self) -> List[Tuple[str, str]]:
        """Get list of symbol/timeframe pairs ready for trading"""
        ready_pairs = []
        
        # For microstructure, we need 1m and 5m with live data
        priority_timeframes = ['1m', '5m']
        
        for symbol in SYMBOLS:
            # Check microstructure readiness
            is_ready, issues = self.check_microstructure_readiness(symbol)
            
            if is_ready:
                for timeframe in priority_timeframes:
                    ready_pairs.append((symbol, timeframe))
                self.logger.info(f"✅ {symbol} ready for microstructure trading")
            else:
                self.logger.warning(f"❌ {symbol} not ready: {issues[:2]}")  # Show first 2 issues
        
        return ready_pairs
    
    def generate_health_report(self) -> Dict:
        """Generate comprehensive data health report with microstructure focus"""
        all_results = self.validate_all_data()
        
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'healthy',
            'symbols_ready': 0,
            'symbols_microstructure_ready': 0,
            'symbols_total': len(SYMBOLS),
            'issues_found': [],
            'critical_issues': [],
            'quality_scores': {},
            'microstructure_readiness': {},
            'recommendations': []
        }
        
        total_score = 0
        valid_pairs = 0
        total_pairs = 0
        microstructure_ready_count = 0
        
        for symbol, timeframes_data in all_results.items():
            symbol_scores = []
            symbol_issues = []
            symbol_critical_issues = []
            
            # Check microstructure readiness for this symbol
            readiness = self.get_microstructure_readiness_report(symbol)
            summary['microstructure_readiness'][symbol] = {
                'ready': readiness.is_ready,
                'score': readiness.readiness_score,
                'live_1m': readiness.has_live_1m,
                'live_5m': readiness.has_live_5m,
                'issues': readiness.issues[:2]  # Top 2 issues
            }
            
            if readiness.is_ready:
                microstructure_ready_count += 1
            
            for timeframe, report in timeframes_data.items():
                total_pairs += 1
                
                if report.is_valid:
                    valid_pairs += 1
                
                symbol_scores.append(report.quality_score)
                total_score += report.quality_score
                
                if report.issues:
                    for issue in report.issues:
                        if '❌' in issue or '🚫' in issue:
                            symbol_critical_issues.append(f"{symbol} {timeframe}: {issue}")
                        else:
                            symbol_issues.append(f"{symbol} {timeframe}: {issue}")
            
            avg_symbol_score = np.mean(symbol_scores) if symbol_scores else 0
            summary['quality_scores'][symbol] = avg_symbol_score
            
            if avg_symbol_score >= 0.8:
                summary['symbols_ready'] += 1
            
            summary['issues_found'].extend(symbol_issues)
            summary['critical_issues'].extend(symbol_critical_issues)
        
        summary['symbols_microstructure_ready'] = microstructure_ready_count
        
        # Overall status determination
        overall_score = total_score / total_pairs if total_pairs > 0 else 0
        ready_ratio = valid_pairs / total_pairs if total_pairs > 0 else 0
        microstructure_ratio = microstructure_ready_count / len(SYMBOLS)
        
        if microstructure_ratio < 0.3:
            summary['overall_status'] = 'critical'
            summary['recommendations'].append("🚨 Less than 30% of symbols ready for microstructure trading")
        elif ready_ratio < 0.5:
            summary['overall_status'] = 'degraded'
            summary['recommendations'].append("⚠️ Many symbols have data quality issues")
        elif ready_ratio < 0.8:
            summary['overall_status'] = 'warning'
        elif overall_score < 0.7:
            summary['overall_status'] = 'degraded'
        
        # Recommendations
        if len(summary['critical_issues']) > 0:
            summary['recommendations'].append(f"Fix {len(summary['critical_issues'])} critical issues")
        
        if microstructure_ready_count < len(SYMBOLS):
            summary['recommendations'].append(f"Only {microstructure_ready_count}/{len(SYMBOLS)} symbols have live data")
            summary['recommendations'].append("Ensure fetcher.py is running and updating every 15 seconds")
        
        if len(summary['issues_found']) > 10:
            summary['recommendations'].append("Consider data repair/refetch for problematic symbols")
        
        return summary
    
    def wait_for_data_ready(self, symbol: str, timeframe: str = '1m',
                           timeout_minutes: int = 10, 
                           microstructure: bool = True) -> bool:
        """Wait for data to become ready for trading"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout_minutes * 60:
            if microstructure:
                # Check microstructure readiness
                is_ready, issues = self.check_microstructure_readiness(symbol)
                
                if is_ready:
                    self.logger.info(f"✅ {symbol} ready for microstructure trading")
                    return True
                
                self.logger.debug(f"Waiting for {symbol} microstructure readiness... Issues: {issues[:2]}")
            else:
                # Standard readiness check
                report = self.validate_symbol_timeframe(symbol, timeframe)
                
                if report.is_valid:
                    self.logger.info(f"✅ Data ready for {symbol} {timeframe}")
                    return True
                
                self.logger.debug(f"Waiting for {symbol} {timeframe} data... "
                                f"Issues: {', '.join(report.issues[:2])}")
            
            # Wait 15 seconds before next check (aligned with fetcher update cycle)
            import time
            time.sleep(15)
        
        self.logger.error(f"Timeout waiting for {symbol} data readiness")
        return False

def main():
    """Test data validation with microstructure focus"""
    validator = DataValidator()
    
    print("\n" + "=" * 60)
    print("MICROSTRUCTURE DATA VALIDATION")
    print("=" * 60)
    
    # Check microstructure readiness for each symbol
    print("\n📊 Microstructure Readiness Check:")
    print("-" * 40)
    
    ready_count = 0
    for symbol in SYMBOLS[:3]:  # Test first 3 symbols
        readiness = validator.get_microstructure_readiness_report(symbol)
        
        status = "✅ READY" if readiness.is_ready else "❌ NOT READY"
        print(f"\n{symbol}: {status}")
        print(f"  Readiness Score: {readiness.readiness_score:.1%}")
        print(f"  Live 1m: {'Yes' if readiness.has_live_1m else 'No'}")
        print(f"  Live 5m: {'Yes' if readiness.has_live_5m else 'No'}")
        print(f"  Live Data Age: {readiness.live_data_age_seconds:.1f}s")
        print(f"  Complete Candles: {readiness.complete_candle_count}")
        
        if readiness.issues:
            print(f"  Issues:")
            for issue in readiness.issues[:3]:
                print(f"    - {issue}")
        
        if readiness.is_ready:
            ready_count += 1
    
    # Generate health report
    print("\n" + "=" * 60)
    print("OVERALL HEALTH REPORT")
    print("=" * 60)
    
    health_report = validator.generate_health_report()
    
    print(f"Status: {health_report['overall_status'].upper()}")
    print(f"Symbols Ready: {health_report['symbols_ready']}/{health_report['symbols_total']}")
    print(f"Microstructure Ready: {health_report['symbols_microstructure_ready']}/{health_report['symbols_total']}")
    
    if health_report['critical_issues']:
        print(f"\n🚨 Critical Issues ({len(health_report['critical_issues'])}):")
        for issue in health_report['critical_issues'][:5]:
            print(f"  - {issue}")
    
    if health_report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in health_report['recommendations']:
            print(f"  - {rec}")
    
    # Get trading ready pairs
    ready_pairs = validator.get_trading_ready_pairs()
    print(f"\n✅ Trading Ready Pairs: {len(ready_pairs)}")
    for symbol, timeframe in ready_pairs[:5]:
        print(f"  - {symbol} {timeframe}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()