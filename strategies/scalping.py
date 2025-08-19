#!/usr/bin/env python3
"""
High-Frequency Scalping Strategy for Crypto Microstructure Trading
Focuses on mean reversion and momentum scalps for 0.2-0.3% moves
UPDATED: Now uses live/incomplete candle data for real-time scalping
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
    DATABASE_FILE, SCALPING_CONFIG, get_table_name
)

class ScalpType(Enum):
    MEAN_REVERSION_LONG = "mean_reversion_long"
    MEAN_REVERSION_SHORT = "mean_reversion_short"
    MOMENTUM_LONG = "momentum_long"
    MOMENTUM_SHORT = "momentum_short"
    RANGE_BOUNCE_LONG = "range_bounce_long"
    RANGE_BOUNCE_SHORT = "range_bounce_short"

@dataclass
class ScalpSignal:
    """Scalping trading signal"""
    symbol: str
    timestamp: datetime
    scalp_type: ScalpType
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    momentum_strength: float
    volume_confirmation: float
    mean_reversion_score: float
    volatility_adjusted: bool
    expected_hold_minutes: int = 3
    risk_reward_ratio: float = 1.5
    is_live_signal: bool = False  # New field for live data tracking

@dataclass
class ScalpingMetrics:
    """Metrics for scalping analysis"""
    sma_5: float
    sma_10: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_mid: float
    rsi: float
    volume_ma: float
    price_velocity: float
    volatility: float
    support_level: float
    resistance_level: float
    has_live_data: bool = False  # New field

class ScalpingStrategy:
    """High-frequency scalping strategy engine with live data support"""
    
    def __init__(self):
        self.logger = logging.getLogger('ScalpingStrategy')
        self.config = SCALPING_CONFIG
        
    def get_recent_data(self, symbol: str, timeframe: str = '1m', 
                       limit: int = 50, include_incomplete: bool = True) -> pd.DataFrame:
        """Get recent candle data with technical indicators INCLUDING LIVE DATA"""
        table_name = get_table_name(symbol, timeframe)
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                # CRITICAL CHANGE: Include incomplete candles for real-time scalping
                if include_incomplete:
                    query = f"""
                    SELECT open_time, open, high, low, close, volume, is_complete
                    FROM {table_name}
                    ORDER BY open_time DESC
                    LIMIT {limit}
                    """
                else:
                    # Fallback for historical analysis
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
                
                # Mark live candles
                df['is_live'] = df['is_complete'] == 0
                
                # Add symbol for tracking
                df['symbol'] = symbol
                
                # Log live data status
                if df['is_live'].any():
                    live_idx = df[df['is_live']].index[0]
                    live_age = (datetime.now(timezone.utc) - df.loc[live_idx, 'timestamp']).total_seconds()
                    self.logger.debug(f"{symbol}: Using live candle, age: {live_age:.1f}s")
                
                # Add technical indicators (handle incomplete data carefully)
                df = self._add_technical_indicators(df)
                
                return df
                
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with proper handling of live/incomplete data"""
        if df.empty or len(df) < 20:
            return df
        
        try:
            # Identify complete vs incomplete data
            complete_mask = df['is_complete'] == 1
            has_live = df['is_live'].any() if 'is_live' in df.columns else False
            
            # Calculate indicators on complete data first
            if complete_mask.sum() >= 5:
                # Moving averages (use complete data)
                df.loc[complete_mask, 'sma_5'] = df.loc[complete_mask, 'close'].rolling(window=5, min_periods=1).mean()
                df.loc[complete_mask, 'sma_10'] = df.loc[complete_mask, 'close'].rolling(window=10, min_periods=1).mean()
                df.loc[complete_mask, 'sma_20'] = df.loc[complete_mask, 'close'].rolling(window=20, min_periods=1).mean()
                
                # For live candle, extrapolate from last complete values
                if has_live:
                    live_idx = df[df['is_live']].index
                    if len(live_idx) > 0 and complete_mask.sum() > 0:
                        last_complete_idx = df[complete_mask].index[-1]
                        
                        # Use last complete candle's indicators as baseline
                        for col in ['sma_5', 'sma_10', 'sma_20']:
                            if col in df.columns and not pd.isna(df.loc[last_complete_idx, col]):
                                df.loc[live_idx[0], col] = df.loc[last_complete_idx, col]
            
            # Bollinger Bands (calculate on all data including live)
            bb_window = 20
            bb_std = 2
            df['bb_mid'] = df['close'].rolling(window=bb_window, min_periods=1).mean()
            bb_std_dev = df['close'].rolling(window=bb_window, min_periods=1).std()
            df['bb_upper'] = df['bb_mid'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_mid'] - (bb_std_dev * bb_std)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_position'] = df['bb_position'].clip(0, 1)  # Clip to [0, 1]
            
            # RSI (use complete data, extrapolate for live)
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price momentum and velocity
            df['price_change'] = df['close'].pct_change()
            df['price_velocity'] = df['price_change'].rolling(window=3, min_periods=1).mean()
            df['price_acceleration'] = df['price_velocity'].diff()
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=10, min_periods=1).std() / df['close'].rolling(window=10, min_periods=1).mean()
            
            # Support and resistance levels (exclude live candle from calculation)
            if complete_mask.sum() >= 10:
                complete_data = df[complete_mask]
                df['resistance'] = complete_data['high'].rolling(window=10, min_periods=1).max().iloc[-1]
                df['support'] = complete_data['low'].rolling(window=10, min_periods=1).min().iloc[-1]
            else:
                df['resistance'] = df['high'].rolling(window=10, min_periods=1).max()
                df['support'] = df['low'].rolling(window=10, min_periods=1).min()
            
            # Candle patterns
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_metrics(self, df: pd.DataFrame) -> ScalpingMetrics:
        """Calculate metrics for scalping analysis"""
        if df.empty or len(df) < 10:
            return ScalpingMetrics(0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, False)
        
        latest = df.iloc[-1]
        has_live_data = latest.get('is_live', False)
        
        return ScalpingMetrics(
            sma_5=latest.get('sma_5', latest['close']),
            sma_10=latest.get('sma_10', latest['close']),
            bollinger_upper=latest.get('bb_upper', latest['close'] * 1.01),
            bollinger_lower=latest.get('bb_lower', latest['close'] * 0.99),
            bollinger_mid=latest.get('bb_mid', latest['close']),
            rsi=latest.get('rsi', 50),
            volume_ma=latest.get('volume_ma', latest['volume']),
            price_velocity=latest.get('price_velocity', 0),
            volatility=latest.get('volatility', 0.01),
            support_level=latest.get('support', latest['low']),
            resistance_level=latest.get('resistance', latest['high']),
            has_live_data=has_live_data
        )
    
    def detect_mean_reversion_scalp(self, df: pd.DataFrame, metrics: ScalpingMetrics) -> Optional[ScalpSignal]:
        """
        Detect mean reversion scalping opportunities (REAL-TIME):
        1. Price moves significantly away from mean
        2. Volume confirmation
        3. Oversold/overbought conditions
        """
        if df.empty or len(df) < 10:
            return None
        
        latest = df.iloc[-1]
        is_live = latest.get('is_live', False)
        prev_candles = df.iloc[-self.config['momentum_lookback']:-1] if len(df) > self.config['momentum_lookback'] else df.iloc[:-1]
        
        # Calculate cumulative move from SMA
        distance_from_sma = (latest['close'] - metrics.sma_10) / metrics.sma_10
        
        # Check for significant move threshold
        if abs(distance_from_sma) < self.config['min_move_threshold']:
            return None
        
        # Volume confirmation
        volume_confirmation = latest['volume_ratio']
        if volume_confirmation < self.config['volume_confirmation']:
            return None
        
        # Determine direction and conditions
        if distance_from_sma > 0:  # Price above SMA, look for short scalp
            # Check RSI overbought
            if metrics.rsi < 70:
                return None
            
            # Check Bollinger position
            bb_position = latest.get('bb_position', 0.5)
            if bb_position < 0.8:
                return None
            
            # Mean reversion strength
            mr_score = min(1.0, (metrics.rsi - 70) / 20 + bb_position - 0.5)
            
            if mr_score < self.config['mean_reversion_strength']:
                return None
            
            # Calculate trade parameters
            entry_price = latest['close']
            target_price = entry_price * (1 - self.config['target_profit'])
            stop_loss = entry_price * (1 + self.config['target_profit'] * self.config['stop_loss_ratio'])
            
            # Confidence calculation (boost for live data)
            confidence = self._calculate_scalp_confidence(
                mr_score, volume_confirmation, metrics.volatility, 'short', is_live
            )
            
            return ScalpSignal(
                symbol=latest.get('symbol', 'UNKNOWN'),
                timestamp=latest['timestamp'],
                scalp_type=ScalpType.MEAN_REVERSION_SHORT,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                momentum_strength=abs(distance_from_sma),
                volume_confirmation=volume_confirmation,
                mean_reversion_score=mr_score,
                volatility_adjusted=True,
                expected_hold_minutes=self.config['max_hold_time'],
                is_live_signal=is_live
            )
        
        else:  # Price below SMA, look for long scalp
            # Check RSI oversold
            if metrics.rsi > 30:
                return None
            
            # Check Bollinger position
            bb_position = latest.get('bb_position', 0.5)
            if bb_position > 0.2:
                return None
            
            # Mean reversion strength
            mr_score = min(1.0, (30 - metrics.rsi) / 20 + 0.5 - bb_position)
            
            if mr_score < self.config['mean_reversion_strength']:
                return None
            
            # Calculate trade parameters
            entry_price = latest['close']
            target_price = entry_price * (1 + self.config['target_profit'])
            stop_loss = entry_price * (1 - self.config['target_profit'] * self.config['stop_loss_ratio'])
            
            # Confidence calculation (boost for live data)
            confidence = self._calculate_scalp_confidence(
                mr_score, volume_confirmation, metrics.volatility, 'long', is_live
            )
            
            return ScalpSignal(
                symbol=latest.get('symbol', 'UNKNOWN'),
                timestamp=latest['timestamp'],
                scalp_type=ScalpType.MEAN_REVERSION_LONG,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                momentum_strength=abs(distance_from_sma),
                volume_confirmation=volume_confirmation,
                mean_reversion_score=mr_score,
                volatility_adjusted=True,
                expected_hold_minutes=self.config['max_hold_time'],
                is_live_signal=is_live
            )
    
    def detect_momentum_scalp(self, df: pd.DataFrame, metrics: ScalpingMetrics) -> Optional[ScalpSignal]:
        """
        Detect momentum scalping opportunities (REAL-TIME):
        1. Strong price velocity in one direction
        2. Volume confirmation
        3. Breakout conditions
        """
        if df.empty or len(df) < 10:
            return None
        
        latest = df.iloc[-1]
        is_live = latest.get('is_live', False)
        
        # Check price velocity
        velocity = metrics.price_velocity
        if abs(velocity) < self.config['min_move_threshold'] / 3:  # Lower threshold for momentum
            return None
        
        # Volume confirmation
        volume_confirmation = latest['volume_ratio']
        if volume_confirmation < self.config['volume_confirmation']:
            return None
        
        # Momentum direction
        if velocity > 0:  # Upward momentum, long scalp
            # Check if breaking resistance (allow for live price action)
            if latest['close'] <= metrics.resistance_level * 1.0005:  # Need 0.05% break
                return None
            
            # Check RSI not too overbought
            if metrics.rsi > 80:
                return None
            
            entry_price = latest['close']
            target_price = entry_price * (1 + self.config['target_profit'])
            stop_loss = entry_price * (1 - self.config['target_profit'] * self.config['stop_loss_ratio'])
            
            confidence = self._calculate_scalp_confidence(
                abs(velocity) * 100, volume_confirmation, metrics.volatility, 'long', is_live
            )
            
            return ScalpSignal(
                symbol=latest.get('symbol', 'UNKNOWN'),
                timestamp=latest['timestamp'],
                scalp_type=ScalpType.MOMENTUM_LONG,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                momentum_strength=abs(velocity),
                volume_confirmation=volume_confirmation,
                mean_reversion_score=0,
                volatility_adjusted=True,
                expected_hold_minutes=self.config['max_hold_time'],
                is_live_signal=is_live
            )
        
        else:  # Downward momentum, short scalp
            # Check if breaking support (allow for live price action)
            if latest['close'] >= metrics.support_level * 0.9995:  # Need 0.05% break
                return None
            
            # Check RSI not too oversold
            if metrics.rsi < 20:
                return None
            
            entry_price = latest['close']
            target_price = entry_price * (1 - self.config['target_profit'])
            stop_loss = entry_price * (1 + self.config['target_profit'] * self.config['stop_loss_ratio'])
            
            confidence = self._calculate_scalp_confidence(
                abs(velocity) * 100, volume_confirmation, metrics.volatility, 'short', is_live
            )
            
            return ScalpSignal(
                symbol=latest.get('symbol', 'UNKNOWN'),
                timestamp=latest['timestamp'],
                scalp_type=ScalpType.MOMENTUM_SHORT,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                momentum_strength=abs(velocity),
                volume_confirmation=volume_confirmation,
                mean_reversion_score=0,
                volatility_adjusted=True,
                expected_hold_minutes=self.config['max_hold_time'],
                is_live_signal=is_live
            )
    
    def detect_range_bounce_scalp(self, df: pd.DataFrame, metrics: ScalpingMetrics) -> Optional[ScalpSignal]:
        """
        Detect range bounce scalping opportunities (REAL-TIME):
        1. Price near support/resistance levels
        2. Rejection signals
        3. Volume confirmation
        """
        if df.empty or len(df) < 10:
            return None
        
        latest = df.iloc[-1]
        is_live = latest.get('is_live', False)
        
        # Calculate distance to support/resistance
        distance_to_support = (latest['close'] - metrics.support_level) / latest['close']
        distance_to_resistance = (metrics.resistance_level - latest['close']) / latest['close']
        
        # Volume confirmation
        volume_confirmation = latest['volume_ratio']
        if volume_confirmation < self.config['volume_confirmation']:
            return None
        
        # Check for bounce at support (long)
        if distance_to_support <= 0.002:  # Within 0.2% of support
            # Look for rejection signals
            lower_wick_strength = latest.get('lower_wick', 0)
            if lower_wick_strength < 0.001:  # Need significant wick
                return None
            
            # Check RSI not too low
            if metrics.rsi < 25:
                return None
            
            entry_price = latest['close']
            target_price = entry_price * (1 + self.config['target_profit'])
            stop_loss = metrics.support_level * 0.998  # Below support
            
            confidence = self._calculate_scalp_confidence(
                lower_wick_strength * 1000, volume_confirmation, metrics.volatility, 'long', is_live
            )
            
            return ScalpSignal(
                symbol=latest.get('symbol', 'UNKNOWN'),
                timestamp=latest['timestamp'],
                scalp_type=ScalpType.RANGE_BOUNCE_LONG,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                momentum_strength=lower_wick_strength,
                volume_confirmation=volume_confirmation,
                mean_reversion_score=1 - distance_to_support * 500,
                volatility_adjusted=True,
                expected_hold_minutes=self.config['max_hold_time'],
                is_live_signal=is_live
            )
        
        # Check for bounce at resistance (short)
        elif distance_to_resistance <= 0.002:  # Within 0.2% of resistance
            # Look for rejection signals
            upper_wick_strength = latest.get('upper_wick', 0)
            if upper_wick_strength < 0.001:  # Need significant wick
                return None
            
            # Check RSI not too high
            if metrics.rsi > 75:
                return None
            
            entry_price = latest['close']
            target_price = entry_price * (1 - self.config['target_profit'])
            stop_loss = metrics.resistance_level * 1.002  # Above resistance
            
            confidence = self._calculate_scalp_confidence(
                upper_wick_strength * 1000, volume_confirmation, metrics.volatility, 'short', is_live
            )
            
            return ScalpSignal(
                symbol=latest.get('symbol', 'UNKNOWN'),
                timestamp=latest['timestamp'],
                scalp_type=ScalpType.RANGE_BOUNCE_SHORT,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                momentum_strength=upper_wick_strength,
                volume_confirmation=volume_confirmation,
                mean_reversion_score=1 - distance_to_resistance * 500,
                volatility_adjusted=True,
                expected_hold_minutes=self.config['max_hold_time'],
                is_live_signal=is_live
            )
        
        return None
    
    def _calculate_scalp_confidence(self, signal_strength: float, volume_confirmation: float,
                                   volatility: float, direction: str, is_live: bool = False) -> float:
        """Calculate confidence score for scalping signal"""
        
        # Base confidence from signal strength (0.4 weight)
        signal_confidence = min(1.0, signal_strength) * 0.4
        
        # Volume confirmation (0.3 weight)
        volume_confidence = min(1.0, (volume_confirmation - 1) / 1) * 0.3
        
        # Volatility adjustment (0.3 weight) - prefer moderate volatility
        optimal_volatility = 0.02  # 2% is optimal
        volatility_score = 1 - abs(volatility - optimal_volatility) / optimal_volatility
        volatility_confidence = max(0.3, volatility_score) * 0.3
        
        base_confidence = signal_confidence + volume_confidence + volatility_confidence
        
        # Market condition adjustment
        market_adjustment = 1.0  # Could add market trend analysis here
        
        # BONUS for live data (fresher signals are better for scalping)
        live_data_boost = 1.1 if is_live else 1.0
        
        final_confidence = base_confidence * market_adjustment * live_data_boost
        
        return min(0.95, max(0.1, final_confidence))
    
    def scan_symbol(self, symbol: str) -> Optional[ScalpSignal]:
        """Scan a symbol for scalping opportunities using LIVE DATA"""
        try:
            # Get recent 1m data INCLUDING LIVE CANDLE
            df = self.get_recent_data(symbol, '1m', 50, include_incomplete=True)
            
            if df.empty or len(df) < 20:
                self.logger.debug(f"Insufficient data for {symbol}")
                return None
            
            # Calculate metrics
            metrics = self.calculate_metrics(df)
            
            # Log live data status
            if metrics.has_live_data:
                self.logger.debug(f"{symbol}: Scanning with live data for scalping")
            
            # Try different scalping strategies in order of preference
            strategies = [
                self.detect_range_bounce_scalp,   # Highest win rate
                self.detect_mean_reversion_scalp, # Good risk/reward
                self.detect_momentum_scalp        # Trend following
            ]
            
            for strategy in strategies:
                signal = strategy(df, metrics)
                if signal and signal.confidence >= 0.6:  # Minimum confidence threshold
                    self.logger.info(f"Scalp signal: {symbol} {signal.scalp_type.value} "
                                   f"at {signal.entry_price:.6f} (confidence: {signal.confidence:.2f}, "
                                   f"live: {signal.is_live_signal})")
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning {symbol} for scalps: {e}")
            return None
    
    def scan_all_symbols(self, symbols: List[str]) -> List[ScalpSignal]:
        """Scan all symbols for scalping opportunities"""
        signals = []
        
        for symbol in symbols:
            signal = self.scan_symbol(symbol)
            if signal:
                signals.append(signal)
        
        # Sort by confidence and prioritize live signals
        signals.sort(key=lambda x: (x.is_live_signal, x.confidence, x.risk_reward_ratio), reverse=True)
        
        return signals
    
    def validate_signal(self, signal: ScalpSignal) -> bool:
        """Validate signal with ULTRA-STRICT freshness for scalping"""
        try:
            # CRITICAL: Scalping signals must be EXTREMELY fresh
            signal_age = (datetime.now(timezone.utc) - signal.timestamp).total_seconds()
            
            # Maximum 5 seconds for scalping (even stricter than stop hunts)
            if signal_age > 5:
                self.logger.debug(f"Signal too old for scalping: {signal_age:.1f}s")
                return False
            
            # Get CURRENT price including incomplete candle
            current_df = self.get_recent_data(signal.symbol, '1m', 1, include_incomplete=True)
            if current_df.empty:
                return False
            
            current_candle = current_df.iloc[-1]
            current_price = current_candle['close']
            is_live = current_candle.get('is_live', False)
            
            # Warn if no live data available
            if not is_live:
                self.logger.warning(f"No live candle for {signal.symbol} scalp validation")
            
            # Check price drift - EXTREMELY tight for scalping
            price_drift = abs(current_price - signal.entry_price) / signal.entry_price
            
            # Maximum 0.01% drift for scalping entries
            if price_drift > 0.0001:  # 0.01% - tighter than stop hunts
                self.logger.debug(f"Price drifted too much for scalping: {price_drift:.5%}")
                return False
            
            self.logger.debug(f"Scalp validated: age={signal_age:.1f}s, drift={price_drift:.5%}, live={is_live}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating scalp signal: {e}")
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
                        self.logger.debug(f"Live price for {symbol}: {price:.2f} (age: {candle_age:.1f}s)")
                    
                    return float(price)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting live price for {symbol}: {e}")
            return None
    
    def get_signal_summary(self, signal: ScalpSignal) -> Dict:
        """Get human-readable signal summary"""
        direction = "LONG" if "LONG" in signal.scalp_type.value.upper() else "SHORT"
        
        risk_amount = abs(signal.stop_loss - signal.entry_price)
        reward_amount = abs(signal.target_price - signal.entry_price)
        risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'symbol': signal.symbol,
            'direction': direction,
            'scalp_type': signal.scalp_type.value.replace('_', ' ').title(),
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'confidence': f"{signal.confidence:.1%}",
            'momentum_strength': f"{signal.momentum_strength:.3f}",
            'volume_confirmation': f"{signal.volume_confirmation:.1f}x",
            'mean_reversion_score': f"{signal.mean_reversion_score:.2f}",
            'risk_reward': f"1:{risk_reward:.1f}",
            'max_hold_time': f"{signal.expected_hold_minutes}min",
            'profit_target': f"{abs(reward_amount/signal.entry_price):.2%}",
            'is_live': "Yes" if signal.is_live_signal else "No"
        }

def main():
    """Test scalping strategy with live data"""
    strategy = ScalpingStrategy()
    
    # Test with a specific symbol
    test_symbol = 'BTCUSDT'
    
    print(f"\nChecking for scalping opportunities in {test_symbol}...")
    signal = strategy.scan_symbol(test_symbol)
    
    if signal:
        summary = strategy.get_signal_summary(signal)
        print(f"\n💰 Scalping Signal Found:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Validate the signal
        if strategy.validate_signal(signal):
            print("\n✅ Signal is VALID for trading (fresh)")
        else:
            print("\n❌ Signal is STALE (too old for scalping)")
    else:
        print(f"No scalping signals found for {test_symbol}")
    
    # Test scanning all symbols
    from config import SYMBOLS
    print(f"\n\nScanning all symbols for scalps...")
    all_signals = strategy.scan_all_symbols(SYMBOLS[:3])  # Test first 3 symbols
    
    print(f"\nTotal scalping signals found: {len(all_signals)}")
    for i, signal in enumerate(all_signals, 1):
        summary = strategy.get_signal_summary(signal)
        print(f"\nSignal {i}: {summary['symbol']} {summary['direction']}")
        print(f"  Type: {summary['scalp_type']}")
        print(f"  Confidence: {summary['confidence']}")
        print(f"  Live Data: {summary['is_live']}")

if __name__ == "__main__":
    main()