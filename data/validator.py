#!/usr/bin/env python3
"""
Data Quality Validator for Crypto Microstructure Trading System
FIXED: Corrected live candle freshness calculation
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
    has_live_data: bool = False
    live_candle_age_seconds: float = 0.0
    is_microstructure_ready: bool = False

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
                
                # Check for LIVE data (incomplete candle) - USE FETCH TIME
                cursor.execute(f'''
                    SELECT open_time, close, volume, last_updated
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
                    # USE LAST_UPDATED TIME TO DETERMINE FRESHNESS
                    live_candle_open_ms = live_candle[0]
                    last_updated = live_candle[3]  # This is the fetch/update time
                    
                    if last_updated:
                        # Calculate age from when data was last fetched/updated
                        last_update_time = datetime.fromtimestamp(last_updated, tz=timezone.utc)
                        fetch_age_seconds = (current_time - last_update_time).total_seconds()
                        live_candle_age_seconds = fetch_age_seconds
                        
                        # For microstructure, we need data updated within last 10 seconds
                        if fetch_age_seconds > 10:
                            issues.append(f"⚠️ Live candle data stale: last updated {fetch_age_seconds:.1f}s ago")
                        else:
                            # Data is fresh!
                            self.logger.debug(f"{symbol} {timeframe}: Live data fresh - updated {fetch_age_seconds:.1f}s ago")
                    else:
                        # No last_updated field, fall back to checking if candle is in valid period
                        live_candle_open_time = datetime.fromtimestamp(live_candle_open_ms / 1000, tz=timezone.utc)
                        
                        timeframe_minutes = {
                            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
                        }.get(timeframe, 1)
                        
                        expected_close_time = live_candle_open_time + timedelta(minutes=timeframe_minutes)
                        
                        if current_time < expected_close_time:
                            # Candle is still in its valid period
                            candle_progress = (current_time - live_candle_open_time).total_seconds()
                            live_candle_age_seconds = 0  # Consider it fresh if within period
                        else:
                            # Candle should have been closed
                            issues.append(f"⚠️ Live candle past expected close time")
                            live_candle_age_seconds = (current_time - expected_close_time).total_seconds()
                
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
                    # For microstructure, we need fresh data being updated
                    max_age_for_microstructure = {
                        '1m': 2.0,  # 2 minutes max
                        '5m': 6.0   # 6 minutes max
                    }.get(timeframe, 2.0)
                    
                    if data_age_minutes > max_age_for_microstructure:
                        issues.append(f"🚫 Data too stale for microstructure: {data_age_minutes:.1f} min old")
                    
                    # Must have live data for microstructure
                    if not has_live_data:
                        issues.append("🚫 No live data for microstructure trading")
                    
                    # Check if microstructure ready
                    is_microstructure_ready = (
                        has_live_data and
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
            
            # For microstructure timeframes (1m, 5m), live data is important but not critical
            if timeframe in ['1m', '5m']:
                # Live data availability (20% weight for microstructure)
                if not has_live_data:
                    score *= 0.8  # Penalty for no live data
                
                # Data freshness (30% weight for microstructure)
                if data_age_minutes > 2.0:
                    freshness_penalty = min(0.9, data_age_minutes / 10.0)
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
                
                # FIXED: Check if candle is within its valid period
                current_time = datetime.now(timezone.utc)
                candle_open_time = datetime.fromtimestamp(live_candle[0] / 1000, tz=timezone.utc)
                candle_expected_close = candle_open_time + timedelta(minutes=1)
                
                if current_time > candle_expected_close:
                    issues.append(f"Live 1m candle past expected close time")
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
                    WHERE open_time > {int((current_time - timedelta(minutes=10)).timestamp() * 1000)}
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
    
    # ... (rest of the methods remain the same) ...
