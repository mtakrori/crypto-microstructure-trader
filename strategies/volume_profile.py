#!/usr/bin/env python3
"""
Volume Profile Analysis Strategy for Crypto Microstructure Trading
Uses volume-at-price analysis to identify high-probability support/resistance levels
UPDATED: Now includes live/incomplete candle data for real-time level validation
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
    DATABASE_FILE, VOLUME_PROFILE_CONFIG, get_table_name
)

class VolumeProfileSignalType(Enum):
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    VOLUME_BREAKOUT_LONG = "volume_breakout_long"
    VOLUME_BREAKOUT_SHORT = "volume_breakout_short"
    POC_REVERSION = "poc_reversion"  # Point of Control reversion

@dataclass
class VolumeLevel:
    """Volume profile level data"""
    price: float
    volume: float
    percentage_of_total: float
    is_high_volume_node: bool
    is_low_volume_node: bool
    strength_score: float

@dataclass
class VolumeProfileData:
    """Complete volume profile analysis"""
    levels: List[VolumeLevel]
    poc_price: float  # Point of Control (highest volume)
    vah_price: float  # Value Area High
    val_price: float  # Value Area Low
    total_volume: float
    price_range: Tuple[float, float]
    analysis_timeframe: str
    has_live_data: bool = False  # New field

@dataclass
class VolumeProfileSignal:
    """Volume profile trading signal"""
    symbol: str
    timestamp: datetime
    signal_type: VolumeProfileSignalType
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    volume_level: VolumeLevel
    distance_to_level: float
    volume_confirmation: bool
    poc_distance: float
    value_area_position: str  # "above", "below", "inside"
    expected_hold_minutes: int = 10
    is_live_signal: bool = False  # New field

class VolumeProfileStrategy:
    """Volume profile analysis and trading strategy with live data support"""
    
    def __init__(self):
        self.logger = logging.getLogger('VolumeProfileStrategy')
        self.config = VOLUME_PROFILE_CONFIG
        
    def get_volume_data(self, symbol: str, timeframe: str = '1m', 
                       hours_back: int = None, include_incomplete: bool = True) -> pd.DataFrame:
        """Get volume data for profile analysis INCLUDING LIVE DATA"""
        if hours_back is None:
            hours_back = self.config['lookback_hours']
            
        table_name = get_table_name(symbol, timeframe)
        
        try:
            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            start_time_ms = int(start_time.timestamp() * 1000)
            
            with sqlite3.connect(DATABASE_FILE) as conn:
                # CRITICAL CHANGE: Include incomplete candles for current market state
                if include_incomplete:
                    query = f"""
                    SELECT open_time, open, high, low, close, volume, is_complete
                    FROM {table_name}
                    WHERE open_time >= {start_time_ms}
                    ORDER BY open_time ASC
                    """
                else:
                    # Fallback for pure historical profile
                    query = f"""
                    SELECT open_time, open, high, low, close, volume, is_complete
                    FROM {table_name}
                    WHERE is_complete = 1 
                    AND open_time >= {start_time_ms}
                    ORDER BY open_time ASC
                    """
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return df
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                df['is_live'] = df['is_complete'] == 0
                df['symbol'] = symbol
                
                # Log live data status
                if df['is_live'].any():
                    live_count = df['is_live'].sum()
                    self.logger.debug(f"{symbol}: Volume profile includes {live_count} live candle(s)")
                
                return df
                
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching volume data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_recent_price_data(self, symbol: str, limit: int = 10, include_incomplete: bool = True) -> pd.DataFrame:
        """Get recent price data for signal detection"""
        table_name = get_table_name(symbol, '1m')
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                if include_incomplete:
                    query = f"""
                    SELECT open_time, open, high, low, close, volume, is_complete
                    FROM {table_name}
                    ORDER BY open_time DESC
                    LIMIT {limit}
                    """
                else:
                    query = f"""
                    SELECT open_time, open, high, low, close, volume, is_complete
                    FROM {table_name}
                    WHERE is_complete = 1
                    ORDER BY open_time DESC
                    LIMIT {limit}
                    """
                
                df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                    df['is_live'] = df['is_complete'] == 0
                    df['symbol'] = symbol
                    df = df.sort_values('open_time').reset_index(drop=True)
                    
                    # Add volume ratio
                    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=min(10, len(df)), min_periods=1).mean()
                
                return df
                
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching recent data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> VolumeProfileData:
        """Calculate volume profile from price/volume data"""
        if df.empty:
            return VolumeProfileData([], 0, 0, 0, 0, (0, 0), "", False)
        
        try:
            has_live_data = df['is_live'].any() if 'is_live' in df.columns else False
            
            # For profile calculation, we can use all data including incomplete
            # but weight the incomplete candle less since it's partial
            profile_df = df.copy()
            if has_live_data:
                # Reduce weight of incomplete candle volume (it's partial)
                live_mask = profile_df['is_live'] == True
                if live_mask.any():
                    # Estimate completion ratio based on time
                    current_ms = datetime.now(timezone.utc).timestamp() * 1000
                    for idx in profile_df[live_mask].index:
                        candle_start = profile_df.loc[idx, 'open_time']
                        elapsed_ms = current_ms - candle_start
                        completion_ratio = min(1.0, elapsed_ms / 60000)  # 60000ms = 1 minute
                        profile_df.loc[idx, 'volume'] *= completion_ratio
            
            # Get price range
            min_price = profile_df['low'].min()
            max_price = profile_df['high'].max()
            price_range = max_price - min_price
            
            if price_range == 0:
                return VolumeProfileData([], 0, 0, 0, 0, (min_price, max_price), "1m", has_live_data)
            
            # Create price bins
            num_bins = min(self.config['num_bins'], len(profile_df) * 2)
            bin_size = price_range / num_bins
            
            # Initialize volume profile
            volume_profile = {}
            total_volume = 0
            
            # Distribute volume across price levels for each candle
            for _, candle in profile_df.iterrows():
                candle_volume = candle['volume']
                candle_range = candle['high'] - candle['low']
                
                if candle_range == 0:
                    # Single price level
                    bin_price = round(candle['close'] / bin_size) * bin_size
                    volume_profile[bin_price] = volume_profile.get(bin_price, 0) + candle_volume
                else:
                    # Distribute volume across price range
                    # Use typical price weighting (more volume at close)
                    typical_price = (candle['high'] + candle['low'] + candle['close']) / 3
                    
                    # Create sub-bins within candle range
                    candle_bins = max(1, int(candle_range / bin_size))
                    for i in range(candle_bins):
                        sub_price = candle['low'] + (i * candle_range / candle_bins)
                        bin_price = round(sub_price / bin_size) * bin_size
                        
                        # Weight volume distribution (more at typical price)
                        distance_to_typical = abs(sub_price - typical_price)
                        weight = max(0.1, 1 - (distance_to_typical / (candle_range / 2)))
                        
                        distributed_volume = (candle_volume / candle_bins) * weight
                        volume_profile[bin_price] = volume_profile.get(bin_price, 0) + distributed_volume
                
                total_volume += candle_volume
            
            # Convert to VolumeLevel objects
            levels = []
            for price, volume in volume_profile.items():
                percentage = (volume / total_volume * 100) if total_volume > 0 else 0
                
                # Determine if high/low volume node
                is_hvn = percentage >= (100 / num_bins) * 2  # 2x average
                is_lvn = percentage <= (100 / num_bins) * 0.5  # 0.5x average
                
                # Calculate strength score
                strength_score = min(1.0, percentage / 10)  # Normalize to 10% max
                
                levels.append(VolumeLevel(
                    price=price,
                    volume=volume,
                    percentage_of_total=percentage,
                    is_high_volume_node=is_hvn,
                    is_low_volume_node=is_lvn,
                    strength_score=strength_score
                ))
            
            # Sort by volume (highest first)
            levels.sort(key=lambda x: x.volume, reverse=True)
            
            # Find Point of Control (POC) - highest volume level
            poc_price = levels[0].price if levels else 0
            
            # Calculate Value Area (70% of volume)
            value_area_volume = total_volume * 0.7
            accumulated_volume = 0
            value_area_levels = []
            
            for level in levels:
                accumulated_volume += level.volume
                value_area_levels.append(level)
                
                if accumulated_volume >= value_area_volume:
                    break
            
            # Value Area High and Low
            value_area_prices = [level.price for level in value_area_levels]
            vah_price = max(value_area_prices) if value_area_prices else poc_price
            val_price = min(value_area_prices) if value_area_prices else poc_price
            
            return VolumeProfileData(
                levels=levels,
                poc_price=poc_price,
                vah_price=vah_price,
                val_price=val_price,
                total_volume=total_volume,
                price_range=(min_price, max_price),
                analysis_timeframe="1m",
                has_live_data=has_live_data
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return VolumeProfileData([], 0, 0, 0, 0, (0, 0), "1m", False)
    
    def find_nearest_significant_level(self, price: float, 
                                     volume_profile: VolumeProfileData) -> Optional[VolumeLevel]:
        """Find the nearest significant volume level to current price"""
        if not volume_profile.levels:
            return None
        
        # Filter for significant levels only
        significant_levels = [
            level for level in volume_profile.levels 
            if level.strength_score >= self.config['min_level_strength']
        ]
        
        if not significant_levels:
            return None
        
        # Find nearest level
        nearest_level = min(
            significant_levels,
            key=lambda level: abs(level.price - price)
        )
        
        # Check if within maximum distance
        distance = abs(nearest_level.price - price) / price
        if distance <= self.config['max_distance_to_level']:
            return nearest_level
        
        return None
    
    def detect_support_bounce(self, current_price: float, volume_profile: VolumeProfileData,
                             recent_df: pd.DataFrame, symbol: str) -> Optional[VolumeProfileSignal]:
        """Detect bounce off volume-based support level (REAL-TIME)"""
        if recent_df.empty:
            return None
        
        latest_candle = recent_df.iloc[-1]
        is_live = latest_candle.get('is_live', False)
        
        # Find support levels (below current price)
        support_levels = [
            level for level in volume_profile.levels
            if level.price < current_price and level.is_high_volume_node
        ]
        
        if not support_levels:
            return None
        
        # Find nearest support
        nearest_support = max(support_levels, key=lambda x: x.price)
        distance_to_support = (current_price - nearest_support.price) / current_price
        
        # Check if price is near support
        if distance_to_support > self.config['max_distance_to_level']:
            return None
        
        # Look for bounce signals
        # 1. Price touched or went below support
        # 2. Current candle shows rejection (lower wick)
        # 3. Volume confirmation
        
        touched_support = latest_candle['low'] <= nearest_support.price * 1.001
        
        if not touched_support:
            return None
        
        # Check for rejection (lower wick)
        lower_wick = latest_candle['close'] - latest_candle['low']
        wick_strength = lower_wick / latest_candle['close']
        
        if wick_strength < 0.001:  # Need at least 0.1% wick
            return None
        
        # Volume confirmation
        volume_confirmation = latest_candle.get('volume_ratio', 1) > 1.2
        
        # Calculate confidence (boost for live data)
        confidence = self._calculate_volume_confidence(
            nearest_support, distance_to_support, wick_strength, volume_confirmation, is_live
        )
        
        if confidence < 0.6:
            return None
        
        # Calculate trade parameters
        entry_price = current_price
        target_price = entry_price * (1 + self.config['target_profit'])
        stop_loss = nearest_support.price * (1 - self.config['stop_buffer'])
        
        # Determine value area position
        value_area_position = self._get_value_area_position(current_price, volume_profile)
        
        return VolumeProfileSignal(
            symbol=symbol,
            timestamp=latest_candle['timestamp'],
            signal_type=VolumeProfileSignalType.SUPPORT_BOUNCE,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            volume_level=nearest_support,
            distance_to_level=distance_to_support,
            volume_confirmation=volume_confirmation,
            poc_distance=abs(current_price - volume_profile.poc_price) / current_price,
            value_area_position=value_area_position,
            expected_hold_minutes=self.config['max_hold_time'],
            is_live_signal=is_live
        )
    
    def detect_resistance_rejection(self, current_price: float, volume_profile: VolumeProfileData,
                                  recent_df: pd.DataFrame, symbol: str) -> Optional[VolumeProfileSignal]:
        """Detect rejection at volume-based resistance level (REAL-TIME)"""
        if recent_df.empty:
            return None
        
        latest_candle = recent_df.iloc[-1]
        is_live = latest_candle.get('is_live', False)
        
        # Find resistance levels (above current price)
        resistance_levels = [
            level for level in volume_profile.levels
            if level.price > current_price and level.is_high_volume_node
        ]
        
        if not resistance_levels:
            return None
        
        # Find nearest resistance
        nearest_resistance = min(resistance_levels, key=lambda x: x.price)
        distance_to_resistance = (nearest_resistance.price - current_price) / current_price
        
        # Check if price is near resistance
        if distance_to_resistance > self.config['max_distance_to_level']:
            return None
        
        # Look for rejection signals
        touched_resistance = latest_candle['high'] >= nearest_resistance.price * 0.999
        
        if not touched_resistance:
            return None
        
        # Check for rejection (upper wick)
        upper_wick = latest_candle['high'] - latest_candle['close']
        wick_strength = upper_wick / latest_candle['close']
        
        if wick_strength < 0.001:  # Need at least 0.1% wick
            return None
        
        # Volume confirmation
        volume_confirmation = latest_candle.get('volume_ratio', 1) > 1.2
        
        # Calculate confidence (boost for live data)
        confidence = self._calculate_volume_confidence(
            nearest_resistance, distance_to_resistance, wick_strength, volume_confirmation, is_live
        )
        
        if confidence < 0.6:
            return None
        
        # Calculate trade parameters
        entry_price = current_price
        target_price = entry_price * (1 - self.config['target_profit'])
        stop_loss = nearest_resistance.price * (1 + self.config['stop_buffer'])
        
        # Determine value area position
        value_area_position = self._get_value_area_position(current_price, volume_profile)
        
        return VolumeProfileSignal(
            symbol=symbol,
            timestamp=latest_candle['timestamp'],
            signal_type=VolumeProfileSignalType.RESISTANCE_REJECTION,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            volume_level=nearest_resistance,
            distance_to_level=distance_to_resistance,
            volume_confirmation=volume_confirmation,
            poc_distance=abs(current_price - volume_profile.poc_price) / current_price,
            value_area_position=value_area_position,
            expected_hold_minutes=self.config['max_hold_time'],
            is_live_signal=is_live
        )
    
    def detect_poc_reversion(self, current_price: float, volume_profile: VolumeProfileData,
                           recent_df: pd.DataFrame, symbol: str) -> Optional[VolumeProfileSignal]:
        """Detect reversion to Point of Control (POC) with LIVE DATA - FIXED TO BE LESS SENSITIVE"""
        if recent_df.empty or volume_profile.poc_price == 0:
            return None
        
        latest_candle = recent_df.iloc[-1]
        is_live = latest_candle.get('is_live', False)
        
        # Calculate distance from POC
        distance_from_poc = abs(current_price - volume_profile.poc_price) / current_price
        
        # FIX: Use configured minimum distance (default 0.8%)
        min_distance = self.config.get('min_poc_distance', 0.008)
        
        # Only consider if significantly away from POC
        if distance_from_poc < min_distance:
            return None
        
        # FIX: Also add maximum distance check (don't trade if too far from POC)
        if distance_from_poc > 0.03:  # More than 3% away is too far
            return None
        
        # Check if price is moving toward POC
        price_direction_to_poc = 1 if current_price < volume_profile.poc_price else -1
        
        # Check recent momentum toward POC (need stronger momentum)
        recent_candles = recent_df.tail(5)  # Increased from 3 to 5 for better confirmation
        if len(recent_candles) < 5:
            return None
        
        momentum_toward_poc = 0
        for _, candle in recent_candles.iterrows():
            if price_direction_to_poc > 0:  # POC above, check for upward momentum
                if candle['close'] > candle['open']:
                    momentum_toward_poc += 1
            else:  # POC below, check for downward momentum
                if candle['close'] < candle['open']:
                    momentum_toward_poc += 1
        
        # FIX: Need at least 3 out of 5 candles moving toward POC (increased from 2/3)
        if momentum_toward_poc < 3:
            return None
        
        # Volume confirmation (require stronger volume)
        volume_confirmation = latest_candle.get('volume_ratio', 1) > 1.2  # Increased from 1.0
        
        # Calculate confidence based on distance and momentum (more conservative)
        base_confidence = min(0.9, 0.2 + (distance_from_poc * 15) + (momentum_toward_poc * 0.15))
        confidence = base_confidence * (1.05 if is_live else 1.0)
        
        # FIX: Use configured confidence threshold
        min_confidence = self.config.get('confidence_threshold', 0.75)
        if confidence < min_confidence:
            return None
        
        # Calculate trade parameters
        entry_price = current_price
        
        if price_direction_to_poc > 0:  # Long toward POC
            target_price = volume_profile.poc_price * 0.998  # Don't aim for exact POC
            stop_loss = entry_price * (1 - self.config['target_profit'] * 1.5)
            signal_type = VolumeProfileSignalType.POC_REVERSION
        else:  # Short toward POC
            target_price = volume_profile.poc_price * 1.002  # Don't aim for exact POC
            stop_loss = entry_price * (1 + self.config['target_profit'] * 1.5)
            signal_type = VolumeProfileSignalType.POC_REVERSION
        
        # Create dummy volume level for POC
        poc_level = VolumeLevel(
            price=volume_profile.poc_price,
            volume=volume_profile.levels[0].volume if volume_profile.levels else 0,
            percentage_of_total=volume_profile.levels[0].percentage_of_total if volume_profile.levels else 0,
            is_high_volume_node=True,
            is_low_volume_node=False,
            strength_score=1.0
        )
        
        value_area_position = self._get_value_area_position(current_price, volume_profile)
        
        return VolumeProfileSignal(
            symbol=symbol,
            timestamp=latest_candle['timestamp'],
            signal_type=signal_type,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            volume_level=poc_level,
            distance_to_level=distance_from_poc,
            volume_confirmation=volume_confirmation,
            poc_distance=distance_from_poc,
            value_area_position=value_area_position,
            expected_hold_minutes=self.config['max_hold_time'],
            is_live_signal=is_live
        )
    
    def _calculate_volume_confidence(self, volume_level: VolumeLevel, distance: float,
                                   wick_strength: float, volume_confirmation: bool,
                                   is_live: bool = False) -> float:
        """Calculate confidence for volume-based signals"""
        
        # Level strength (0.4 weight)
        level_confidence = volume_level.strength_score * 0.4
        
        # Distance to level (0.3 weight) - closer is better
        distance_confidence = (1 - (distance / self.config['max_distance_to_level'])) * 0.3
        
        # Wick strength (0.2 weight)
        wick_confidence = min(1.0, wick_strength * 200) * 0.2  # Normalize 0.5% wick to 1.0
        
        # Volume confirmation (0.1 weight)
        volume_conf = 0.1 if volume_confirmation else 0.05
        
        base_confidence = level_confidence + distance_confidence + wick_confidence + volume_conf
        
        # BONUS for live data (fresher signals)
        live_data_boost = 1.05 if is_live else 1.0
        
        final_confidence = base_confidence * live_data_boost
        
        return min(0.95, max(0.1, final_confidence))
    
    def _get_value_area_position(self, price: float, volume_profile: VolumeProfileData) -> str:
        """Determine if price is above, below, or inside value area"""
        if price > volume_profile.vah_price:
            return "above"
        elif price < volume_profile.val_price:
            return "below"
        else:
            return "inside"
    
    def scan_symbol(self, symbol: str) -> Optional[VolumeProfileSignal]:
        """Scan symbol for volume profile opportunities using LIVE DATA"""
        try:
            # Get volume data for profile calculation (include incomplete)
            volume_df = self.get_volume_data(symbol, '1m', self.config['lookback_hours'], include_incomplete=True)
            
            if volume_df.empty or len(volume_df) < 50:
                self.logger.debug(f"Insufficient volume data for {symbol}")
                return None
            
            # Calculate volume profile
            volume_profile = self.calculate_volume_profile(volume_df)
            
            if not volume_profile.levels:
                return None
            
            # Get recent price data INCLUDING LIVE CANDLE
            recent_df = self.get_recent_price_data(symbol, 10, include_incomplete=True)
            if recent_df.empty:
                return None
            
            current_price = recent_df.iloc[-1]['close']
            
            # Log profile status
            if volume_profile.has_live_data:
                self.logger.debug(f"{symbol}: Volume profile includes live data")
            
            # Try different volume profile strategies
            strategies = [
                lambda: self.detect_support_bounce(current_price, volume_profile, recent_df, symbol),
                lambda: self.detect_resistance_rejection(current_price, volume_profile, recent_df, symbol),
                lambda: self.detect_poc_reversion(current_price, volume_profile, recent_df, symbol)
            ]
            
            for strategy in strategies:
                signal = strategy()
                if signal and signal.confidence >= 0.6:
                    self.logger.info(f"Volume profile signal: {symbol} {signal.signal_type.value} "
                                   f"at {signal.entry_price:.6f} (confidence: {signal.confidence:.2f}, "
                                   f"live: {signal.is_live_signal})")
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning {symbol} for volume profile signals: {e}")
            return None
    
    def scan_all_symbols(self, symbols: List[str]) -> List[VolumeProfileSignal]:
        """Scan all symbols for volume profile opportunities"""
        signals = []
        
        for symbol in symbols:
            signal = self.scan_symbol(symbol)
            if signal:
                signals.append(signal)
        
        # Sort by confidence, level strength, and prioritize live signals
        signals.sort(key=lambda x: (x.is_live_signal, x.confidence, x.volume_level.strength_score), reverse=True)
        
        return signals
    
    def validate_signal(self, signal: VolumeProfileSignal) -> bool:
        """Validate signal with appropriate freshness for volume profile trading"""
        try:
            # Volume profile signals can be slightly less time-sensitive than scalps
            # but still need to be fresh for microstructure trading
            signal_age = (datetime.now(timezone.utc) - signal.timestamp).total_seconds()
            
            # Maximum 15 seconds for volume profile (more lenient than scalping)
            if signal_age > 15:
                self.logger.debug(f"Signal too old: {signal_age:.1f}s")
                return False
            
            # Get CURRENT price including incomplete candle
            current_df = self.get_recent_price_data(signal.symbol, 1, include_incomplete=True)
            if current_df.empty:
                return False
            
            current_candle = current_df.iloc[-1]
            current_price = current_candle['close']
            is_live = current_candle.get('is_live', False)
            
            # Log validation details
            self.logger.debug(f"Validating {signal.symbol} volume signal: age={signal_age:.1f}s, live={is_live}")
            
            # Check price drift - moderate tolerance for volume profile
            price_drift = abs(current_price - signal.entry_price) / signal.entry_price
            
            # Maximum 0.05% drift for volume profile entries
            if price_drift > 0.0005:  # 0.05% - between scalping and stop hunt tolerance
                self.logger.debug(f"Price drifted too much: {price_drift:.4%}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating volume signal: {e}")
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
    
    def get_signal_summary(self, signal: VolumeProfileSignal) -> Dict:
        """Get human-readable signal summary"""
        direction = "LONG" if "LONG" in signal.signal_type.value.upper() or signal.target_price > signal.entry_price else "SHORT"
        
        risk_amount = abs(signal.stop_loss - signal.entry_price)
        reward_amount = abs(signal.target_price - signal.entry_price)
        risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'symbol': signal.symbol,
            'direction': direction,
            'signal_type': signal.signal_type.value.replace('_', ' ').title(),
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'confidence': f"{signal.confidence:.1%}",
            'volume_level_price': signal.volume_level.price,
            'level_strength': f"{signal.volume_level.strength_score:.2f}",
            'distance_to_level': f"{signal.distance_to_level:.3%}",
            'poc_distance': f"{signal.poc_distance:.3%}",
            'value_area_position': signal.value_area_position,
            'volume_confirmation': "Yes" if signal.volume_confirmation else "No",
            'risk_reward': f"1:{risk_reward:.1f}",
            'max_hold_time': f"{signal.expected_hold_minutes}min",
            'is_live': "Yes" if signal.is_live_signal else "No"
        }

def main():
    """Test volume profile strategy with live data"""
    strategy = VolumeProfileStrategy()
    
    # Test with a specific symbol
    test_symbol = 'BTCUSDT'
    
    print(f"\nAnalyzing volume profile for {test_symbol}...")
    signal = strategy.scan_symbol(test_symbol)
    
    if signal:
        summary = strategy.get_signal_summary(signal)
        print(f"\n📊 Volume Profile Signal Found:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Validate the signal
        if strategy.validate_signal(signal):
            print("\n✅ Signal is VALID for trading")
        else:
            print("\n❌ Signal is STALE or invalid")
    else:
        print(f"No volume profile signals found for {test_symbol}")
    
    # Test scanning all symbols
    from config import SYMBOLS
    print(f"\n\nScanning all symbols for volume profile signals...")
    all_signals = strategy.scan_all_symbols(SYMBOLS[:2])  # Test first 2 symbols
    
    print(f"\nTotal volume profile signals found: {len(all_signals)}")
    for i, signal in enumerate(all_signals, 1):
        summary = strategy.get_signal_summary(signal)
        print(f"\nSignal {i}: {summary['symbol']} {summary['direction']}")
        print(f"  Type: {summary['signal_type']}")
        print(f"  Level Strength: {summary['level_strength']}")
        print(f"  Live Data: {summary['is_live']}")

if __name__ == "__main__":
    main()
