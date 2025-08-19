#!/usr/bin/env python3
"""
Stop Hunt Detection Strategy for Crypto Microstructure Trading
Detects liquidity grabs and stop hunting patterns using 1m data
UPDATED: Now uses live/incomplete candle data for real-time detection
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from config import (
    DATABASE_FILE, STOP_HUNT_CONFIG, get_table_name
)

class StopHuntType(Enum):
    LONG_HUNT = "long_hunt"      # Hunt long stops (price spikes up then down)
    SHORT_HUNT = "short_hunt"    # Hunt short stops (price spikes down then up)

@dataclass
class StopHuntSignal:
    """Stop hunt trading signal"""
    symbol: str
    timestamp: datetime
    hunt_type: StopHuntType
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    volume_spike_ratio: float
    price_spike_percentage: float
    wick_rejection_percentage: float
    hunt_high: Optional[float] = None
    hunt_low: Optional[float] = None
    expected_hold_minutes: int = 5
    is_live_signal: bool = False  # New field to track if signal is from live data

@dataclass
class StopHuntMetrics:
    """Metrics for stop hunt analysis"""
    recent_high: float
    recent_low: float
    avg_volume: float
    price_volatility: float
    volume_volatility: float
    support_levels: List[float]
    resistance_levels: List[float]
    has_live_data: bool  # New field

class StopHuntDetector:
    """Advanced stop hunt pattern detection with live data support"""
    
    def __init__(self):
        self.logger = logging.getLogger('StopHuntDetector')
        self.config = STOP_HUNT_CONFIG
        
    def get_recent_data(self, symbol: str, timeframe: str = '1m', 
                       limit: int = None, include_incomplete: bool = True) -> pd.DataFrame:
        """Get recent candle data for analysis INCLUDING LIVE CANDLES"""
        if limit is None:
            limit = self.config['lookback_candles']
            
        table_name = get_table_name(symbol, timeframe)
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                # CRITICAL CHANGE: Include incomplete candles for real-time detection
                if include_incomplete:
                    query = f"""
                    SELECT open_time, open, high, low, close, volume, is_complete
                    FROM {table_name}
                    ORDER BY open_time DESC
                    LIMIT {limit}
                    """
                else:
                    # Fallback for historical analysis only
                    query = f"""
                    SELECT open_time, open, high, low, close, volume, is_complete
                    FROM {table_name}
                    WHERE is_complete = 1
                    ORDER BY open_time DESC
                    LIMIT {limit}
                    """
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return df
                
                # Convert timestamp and sort chronologically
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                df = df.sort_values('open_time').reset_index(drop=True)
                
                # Mark which candles are live
                df['is_live'] = df['is_complete'] == 0
                
                # Add symbol for tracking
                df['symbol'] = symbol
                
                # Add technical indicators
                df['price_change_pct'] = df['close'].pct_change()
                df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
                df['wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']
                df['wick_upper_pct'] = df['wick_upper'] / df['close']
                df['wick_lower_pct'] = df['wick_lower'] / df['close']
                
                # Log if we have live data
                if df['is_live'].any():
                    live_age = (datetime.now(timezone.utc) - df[df['is_live']].iloc[0]['timestamp']).total_seconds()
                    self.logger.debug(f"{symbol}: Using live candle, age: {live_age:.1f}s")
                
                return df
                
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_metrics(self, df: pd.DataFrame) -> StopHuntMetrics:
        """Calculate metrics needed for stop hunt detection"""
        if df.empty or len(df) < 10:
            return StopHuntMetrics(0, 0, 0, 0, 0, [], [], False)
        
        # Check for live data
        has_live_data = df['is_live'].any() if 'is_live' in df.columns else False
        
        # For reference levels, exclude the current live candle to avoid bias
        if has_live_data:
            reference_data = df[~df['is_live']]
        else:
            reference_data = df.iloc[:-1]
        
        if len(reference_data) < 5:
            reference_data = df.iloc[:-1]  # Fallback
        
        recent_high = reference_data['high'].max()
        recent_low = reference_data['low'].min()
        avg_volume = reference_data['volume'].mean()
        
        # Calculate volatility
        price_returns = reference_data['close'].pct_change().dropna()
        price_volatility = price_returns.std() if len(price_returns) > 1 else 0
        
        volume_volatility = (reference_data['volume'].std() / avg_volume) if avg_volume > 0 else 0
        
        # Identify support and resistance levels using volume-weighted approach
        support_levels = []
        resistance_levels = []
        
        try:
            # Simple pivot point analysis for support/resistance
            highs = reference_data['high'].rolling(window=3, center=True).max()
            lows = reference_data['low'].rolling(window=3, center=True).min()
            
            for i in range(1, len(reference_data) - 1):
                if reference_data.iloc[i]['high'] == highs.iloc[i]:
                    resistance_levels.append(reference_data.iloc[i]['high'])
                if reference_data.iloc[i]['low'] == lows.iloc[i]:
                    support_levels.append(reference_data.iloc[i]['low'])
            
            # Keep only strongest levels
            resistance_levels = sorted(resistance_levels, reverse=True)[:3]
            support_levels = sorted(support_levels)[:3]
            
        except Exception as e:
            self.logger.debug(f"Error calculating support/resistance: {e}")
        
        return StopHuntMetrics(
            recent_high=recent_high,
            recent_low=recent_low,
            avg_volume=avg_volume,
            price_volatility=price_volatility,
            volume_volatility=volume_volatility,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            has_live_data=has_live_data
        )
    
    def detect_long_stop_hunt(self, df: pd.DataFrame, metrics: StopHuntMetrics) -> Optional[StopHuntSignal]:
        """
        Detect long stop hunt pattern (REAL-TIME):
        1. Price spikes above recent high
        2. Volume surge during spike
        3. Immediate rejection with large upper wick
        4. Entry on rejection for short position
        """
        if df.empty or len(df) < 2:
            return None
            
        latest_candle = df.iloc[-1]
        is_live = latest_candle.get('is_live', False)
        
        # Check for price spike above recent high
        spike_threshold = metrics.recent_high * (1 + self.config['min_spike_threshold'])
        if latest_candle['high'] < spike_threshold:
            return None
        
        # Check for volume spike
        volume_spike_ratio = latest_candle['volume_ratio']
        if volume_spike_ratio < self.config['min_volume_spike']:
            return None
        
        # Check for wick rejection
        wick_rejection_pct = latest_candle['wick_upper_pct']
        if wick_rejection_pct < self.config['wick_rejection_min']:
            return None
        
        # Calculate price spike percentage
        price_spike_pct = (latest_candle['high'] - metrics.recent_high) / metrics.recent_high
        
        # Check minimum spike threshold
        if price_spike_pct < self.config['min_spike_threshold']:
            return None
        
        # Ensure strong rejection (close well below high)
        rejection_strength = (latest_candle['high'] - latest_candle['close']) / latest_candle['high']
        if rejection_strength < self.config['wick_rejection_min']:
            return None
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            volume_spike_ratio=volume_spike_ratio,
            price_spike_pct=price_spike_pct,
            wick_rejection_pct=wick_rejection_pct,
            rejection_strength=rejection_strength,
            volatility=metrics.price_volatility,
            is_live=is_live
        )
        
        if confidence < self.config['confidence_threshold']:
            return None
        
        # Calculate entry, target, and stop loss
        entry_price = latest_candle['close']
        target_price = entry_price * (1 - self.config['target_profit'])
        stop_loss = latest_candle['high'] * (1 + self.config['stop_loss_buffer'])
        
        return StopHuntSignal(
            symbol=df.iloc[0].get('symbol', 'UNKNOWN'),
            timestamp=latest_candle['timestamp'],
            hunt_type=StopHuntType.LONG_HUNT,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            volume_spike_ratio=volume_spike_ratio,
            price_spike_percentage=price_spike_pct,
            wick_rejection_percentage=wick_rejection_pct,
            hunt_high=latest_candle['high'],
            expected_hold_minutes=self.config['max_hold_time'],
            is_live_signal=is_live
        )
    
    def detect_short_stop_hunt(self, df: pd.DataFrame, metrics: StopHuntMetrics) -> Optional[StopHuntSignal]:
        """
        Detect short stop hunt pattern (REAL-TIME):
        1. Price spikes below recent low
        2. Volume surge during spike
        3. Immediate rejection with large lower wick
        4. Entry on rejection for long position
        """
        if df.empty or len(df) < 2:
            return None
            
        latest_candle = df.iloc[-1]
        is_live = latest_candle.get('is_live', False)
        
        # Check for price spike below recent low
        spike_threshold = metrics.recent_low * (1 - self.config['min_spike_threshold'])
        if latest_candle['low'] > spike_threshold:
            return None
        
        # Check for volume spike
        volume_spike_ratio = latest_candle['volume_ratio']
        if volume_spike_ratio < self.config['min_volume_spike']:
            return None
        
        # Check for wick rejection
        wick_rejection_pct = latest_candle['wick_lower_pct']
        if wick_rejection_pct < self.config['wick_rejection_min']:
            return None
        
        # Calculate price spike percentage
        price_spike_pct = (metrics.recent_low - latest_candle['low']) / metrics.recent_low
        
        # Check minimum spike threshold
        if price_spike_pct < self.config['min_spike_threshold']:
            return None
        
        # Ensure strong rejection (close well above low)
        rejection_strength = (latest_candle['close'] - latest_candle['low']) / latest_candle['close']
        if rejection_strength < self.config['wick_rejection_min']:
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            volume_spike_ratio=volume_spike_ratio,
            price_spike_pct=price_spike_pct,
            wick_rejection_pct=wick_rejection_pct,
            rejection_strength=rejection_strength,
            volatility=metrics.price_volatility,
            is_live=is_live
        )
        
        if confidence < self.config['confidence_threshold']:
            return None
        
        # Calculate entry, target, and stop loss
        entry_price = latest_candle['close']
        target_price = entry_price * (1 + self.config['target_profit'])
        stop_loss = latest_candle['low'] * (1 - self.config['stop_loss_buffer'])
        
        return StopHuntSignal(
            symbol=df.iloc[0].get('symbol', 'UNKNOWN'),
            timestamp=latest_candle['timestamp'],
            hunt_type=StopHuntType.SHORT_HUNT,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            volume_spike_ratio=volume_spike_ratio,
            price_spike_percentage=price_spike_pct,
            wick_rejection_percentage=wick_rejection_pct,
            hunt_low=latest_candle['low'],
            expected_hold_minutes=self.config['max_hold_time'],
            is_live_signal=is_live
        )
    
    def _calculate_confidence(self, volume_spike_ratio: float, price_spike_pct: float,
                             wick_rejection_pct: float, rejection_strength: float,
                             volatility: float, is_live: bool = False) -> float:
        """Calculate confidence score for stop hunt signal"""
        
        # Base confidence from volume spike (0.3 weight)
        volume_confidence = min(1.0, (volume_spike_ratio - 1) / 2) * 0.3
        
        # Price spike confidence (0.25 weight)
        price_confidence = min(1.0, price_spike_pct / 0.01) * 0.25  # Normalize to 1% spike
        
        # Wick rejection confidence (0.25 weight)
        wick_confidence = min(1.0, wick_rejection_pct / 0.005) * 0.25  # Normalize to 0.5% wick
        
        # Rejection strength confidence (0.2 weight)
        rejection_confidence = min(1.0, rejection_strength / 0.005) * 0.2
        
        base_confidence = volume_confidence + price_confidence + wick_confidence + rejection_confidence
        
        # Adjust for market volatility (more volatile = lower confidence)
        volatility_adjustment = max(0.7, 1 - (volatility * 10))  # Reduce confidence in high vol
        
        # Boost confidence slightly if using live data (fresher signal)
        live_data_boost = 1.05 if is_live else 1.0
        
        final_confidence = base_confidence * volatility_adjustment * live_data_boost
        
        return min(0.95, max(0.1, final_confidence))
    
    def scan_symbol(self, symbol: str) -> Optional[StopHuntSignal]:
        """Scan a symbol for stop hunt opportunities using LIVE DATA"""
        try:
            # Get recent 1m data INCLUDING LIVE CANDLE
            df = self.get_recent_data(symbol, '1m', self.config['lookback_candles'], include_incomplete=True)
            
            if df.empty or len(df) < 10:
                self.logger.debug(f"Insufficient data for {symbol}")
                return None
            
            # Calculate metrics
            metrics = self.calculate_metrics(df)
            
            # Log live data status
            if metrics.has_live_data:
                self.logger.debug(f"{symbol}: Scanning with live data")
            
            # Check for long stop hunt (hunt longs, go short)
            long_hunt = self.detect_long_stop_hunt(df, metrics)
            if long_hunt:
                self.logger.info(f"Long stop hunt detected: {symbol} at {long_hunt.entry_price:.6f} "
                               f"(confidence: {long_hunt.confidence:.2f}, live: {long_hunt.is_live_signal})")
                return long_hunt
            
            # Check for short stop hunt (hunt shorts, go long)
            short_hunt = self.detect_short_stop_hunt(df, metrics)
            if short_hunt:
                self.logger.info(f"Short stop hunt detected: {symbol} at {short_hunt.entry_price:.6f} "
                                f"(confidence: {short_hunt.confidence:.2f}, live: {short_hunt.is_live_signal})")
                return short_hunt
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning {symbol} for stop hunts: {e}")
            return None
    
    def scan_all_symbols(self, symbols: List[str]) -> List[StopHuntSignal]:
        """Scan all symbols for stop hunt opportunities"""
        signals = []
        
        for symbol in symbols:
            signal = self.scan_symbol(symbol)
            if signal:
                signals.append(signal)
        
        # Sort by confidence and prioritize live signals
        signals.sort(key=lambda x: (x.is_live_signal, x.confidence), reverse=True)
        
        return signals
    
    def validate_signal(self, signal: StopHuntSignal) -> bool:
        """Validate signal with STRICT freshness requirements for microstructure"""
        try:
            # CRITICAL: For microstructure, signals must be VERY fresh
            signal_age = (datetime.now(timezone.utc) - signal.timestamp).total_seconds()
            
            # Maximum 10 seconds for microstructure trading
            if signal_age > 10:
                self.logger.debug(f"Signal too old for microstructure: {signal_age:.1f}s")
                return False
            
            # Get CURRENT price including incomplete candle
            current_df = self.get_recent_data(signal.symbol, '1m', 1, include_incomplete=True)
            if current_df.empty:
                return False
            
            current_candle = current_df.iloc[-1]
            current_price = current_candle['close']
            is_live = current_candle.get('is_live', False)
            
            # Log validation status
            self.logger.debug(f"Validating {signal.symbol}: age={signal_age:.1f}s, live_data={is_live}")
            
            # Check price drift - VERY tight for microstructure
            price_drift = abs(current_price - signal.entry_price) / signal.entry_price
            
            # Maximum 0.02% drift for stop hunt entries
            if price_drift > 0.0002:
                self.logger.debug(f"Price drifted too much: {price_drift:.4%}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def get_current_price_live(self, symbol: str) -> Optional[float]:
        """Get the absolute latest price including incomplete candle"""
        try:
            table_name = get_table_name(symbol, '1m')
            
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT close, is_complete, open_time
                    FROM {table_name}
                    ORDER BY open_time DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                if result:
                    price, is_complete, open_time = result
                    
                    if is_complete == 0:
                        candle_age = (datetime.now(timezone.utc).timestamp() * 1000 - open_time) / 1000
                        self.logger.debug(f"Using live price for {symbol}, candle age: {candle_age:.1f}s")
                    
                    return float(price)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting live price for {symbol}: {e}")
            return None
    
    def get_signal_summary(self, signal: StopHuntSignal) -> Dict:
        """Get human-readable signal summary"""
        direction = "SHORT" if signal.hunt_type == StopHuntType.LONG_HUNT else "LONG"
        hunt_price = signal.hunt_high if signal.hunt_high else signal.hunt_low
        
        risk_reward = abs(signal.target_price - signal.entry_price) / abs(signal.stop_loss - signal.entry_price)
        
        return {
            'symbol': signal.symbol,
            'direction': direction,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'confidence': f"{signal.confidence:.1%}",
            'hunt_type': signal.hunt_type.value,
            'hunt_price': hunt_price,
            'volume_spike': f"{signal.volume_spike_ratio:.1f}x",
            'price_spike': f"{signal.price_spike_percentage:.2%}",
            'wick_rejection': f"{signal.wick_rejection_percentage:.2%}",
            'risk_reward': f"1:{risk_reward:.1f}",
            'max_hold_time': f"{signal.expected_hold_minutes}min",
            'is_live': "Yes" if signal.is_live_signal else "No"
        }

def main():
    """Test stop hunt detection with live data"""
    detector = StopHuntDetector()
    
    # Test with a specific symbol
    test_symbol = 'BTCUSDT'
    
    print(f"\nChecking for stop hunts in {test_symbol}...")
    signal = detector.scan_symbol(test_symbol)
    
    if signal:
        summary = detector.get_signal_summary(signal)
        print(f"\n🎯 Stop Hunt Signal Found:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Validate the signal
        if detector.validate_signal(signal):
            print("\n✅ Signal is VALID for trading")
        else:
            print("\n❌ Signal is STALE or invalid")
    else:
        print(f"No stop hunt signals found for {test_symbol}")
    
    # Test scanning all symbols
    from config import SYMBOLS
    print(f"\n\nScanning all symbols...")
    all_signals = detector.scan_all_symbols(SYMBOLS)
    
    print(f"\nTotal signals found: {len(all_signals)}")
    for i, signal in enumerate(all_signals[:3], 1):  # Show top 3
        summary = detector.get_signal_summary(signal)
        print(f"\nSignal {i}: {summary['symbol']} {summary['direction']}")
        print(f"  Confidence: {summary['confidence']}")
        print(f"  Entry: {summary['entry_price']}")
        print(f"  Live Data: {summary['is_live']}")

if __name__ == "__main__":
    main()