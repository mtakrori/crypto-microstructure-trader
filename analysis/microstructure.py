#!/usr/bin/env python3
"""
Core Microstructure Analysis Tools for Crypto Trading
Provides advanced analysis capabilities for 1m/5m data
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

from config import DATABASE_FILE, get_table_name

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class MarketMicrostructure:
    """Market microstructure analysis result"""
    symbol: str
    timestamp: datetime
    regime: MarketRegime
    volatility: float
    volume_profile_strength: float
    order_flow_imbalance: float
    bid_ask_pressure: float
    liquidity_score: float
    mean_reversion_strength: float
    momentum_strength: float
    
@dataclass
class OrderFlowAnalysis:
    """Order flow analysis result"""
    symbol: str
    timestamp: datetime
    buyer_initiated_volume: float
    seller_initiated_volume: float
    volume_imbalance: float
    aggressive_buying: float
    aggressive_selling: float
    passive_activity: float
    
@dataclass
class LiquidityAnalysis:
    """Liquidity analysis result"""
    symbol: str
    timestamp: datetime
    effective_spread: float
    market_impact: float
    liquidity_pools: List[float]
    thin_areas: List[float]
    liquidity_score: float

class MicrostructureAnalyzer:
    """Advanced microstructure analysis engine"""
    
    def __init__(self):
        self.logger = logging.getLogger('MicrostructureAnalyzer')
    
    def get_tick_data(self, symbol: str, timeframe: str = '1m', 
                     hours_back: int = 1) -> pd.DataFrame:
        """Get high-frequency tick data"""
        table_name = get_table_name(symbol, timeframe)
        
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            start_time_ms = int(start_time.timestamp() * 1000)
            
            with sqlite3.connect(DATABASE_FILE) as conn:
                query = f"""
                SELECT open_time, open, high, low, close, volume, 
                       taker_buy_volume, taker_buy_quote_volume, count
                FROM {table_name}
                WHERE is_complete = 1 AND open_time >= {start_time_ms}
                ORDER BY open_time ASC
                """
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return df
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                
                # Calculate additional metrics
                df = self._add_microstructure_indicators(df)
                
                return df
                
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching tick data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure indicators to dataframe"""
        if df.empty:
            return df
        
        try:
            # Price-based indicators
            df['mid_price'] = (df['high'] + df['low']) / 2
            df['price_range'] = df['high'] - df['low']
            df['price_volatility'] = df['close'].pct_change().rolling(window=10).std()
            
            # Volume-based indicators
            df['volume_ma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_volatility'] = df['volume'].rolling(window=10).std() / df['volume_ma']
            
            # Order flow indicators
            df['buy_volume'] = df['taker_buy_volume']
            df['sell_volume'] = df['volume'] - df['taker_buy_volume']
            df['buy_ratio'] = df['buy_volume'] / df['volume']
            df['sell_ratio'] = df['sell_volume'] / df['volume']
            df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / df['volume']
            
            # VWAP (Volume Weighted Average Price)
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
            
            # Liquidity indicators
            df['effective_spread'] = df['price_range'] / df['mid_price']
            df['market_impact'] = df['price_range'] / df['volume']
            
            # Momentum indicators
            df['price_momentum'] = df['close'].pct_change(periods=3)
            df['volume_momentum'] = df['volume'].pct_change(periods=3)
            
            # Mean reversion indicators
            df['bollinger_mid'] = df['close'].rolling(window=20).mean()
            df['bollinger_std'] = df['close'].rolling(window=20).std()
            df['bollinger_upper'] = df['bollinger_mid'] + (2 * df['bollinger_std'])
            df['bollinger_lower'] = df['bollinger_mid'] - (2 * df['bollinger_std'])
            df['bollinger_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
            
            # RSI for mean reversion
            df['rsi'] = self._calculate_rsi(df['close'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding microstructure indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def analyze_market_regime(self, symbol: str, timeframe: str = '1m') -> MarketMicrostructure:
        """Analyze current market microstructure regime"""
        try:
            df = self.get_tick_data(symbol, timeframe, hours_back=2)
            
            if df.empty or len(df) < 20:
                return self._default_microstructure(symbol)
            
            latest = df.iloc[-1]
            
            # Determine market regime
            regime = self._classify_market_regime(df)
            
            # Calculate volatility
            volatility = df['price_volatility'].iloc[-1] if not pd.isna(df['price_volatility'].iloc[-1]) else 0.01
            
            # Volume profile strength
            volume_profile_strength = self._calculate_volume_profile_strength(df)
            
            # Order flow imbalance
            order_flow_imbalance = latest.get('volume_imbalance', 0)
            
            # Bid-ask pressure (approximated from order flow)
            bid_ask_pressure = self._calculate_bid_ask_pressure(df)
            
            # Liquidity score
            liquidity_score = self._calculate_liquidity_score(df)
            
            # Mean reversion strength
            mean_reversion_strength = self._calculate_mean_reversion_strength(df)
            
            # Momentum strength
            momentum_strength = self._calculate_momentum_strength(df)
            
            return MarketMicrostructure(
                symbol=symbol,
                timestamp=latest['timestamp'],
                regime=regime,
                volatility=volatility,
                volume_profile_strength=volume_profile_strength,
                order_flow_imbalance=order_flow_imbalance,
                bid_ask_pressure=bid_ask_pressure,
                liquidity_score=liquidity_score,
                mean_reversion_strength=mean_reversion_strength,
                momentum_strength=momentum_strength
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime for {symbol}: {e}")
            return self._default_microstructure(symbol)
    
    def _default_microstructure(self, symbol: str) -> MarketMicrostructure:
        """Return default microstructure when analysis fails"""
        return MarketMicrostructure(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            regime=MarketRegime.RANGING,
            volatility=0.02,
            volume_profile_strength=0.5,
            order_flow_imbalance=0.0,
            bid_ask_pressure=0.0,
            liquidity_score=0.5,
            mean_reversion_strength=0.5,
            momentum_strength=0.0
        )
    
    def _classify_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify current market regime"""
        try:
            if len(df) < 20:
                return MarketRegime.RANGING
            
            # Price trend analysis
            recent_prices = df['close'].tail(20)
            price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Volatility analysis
            vol = df['price_volatility'].tail(10).mean()
            
            # Volume trend
            vol_trend = df['volume'].tail(10).mean() / df['volume'].tail(20).mean()
            
            # Regime classification
            if vol > 0.03:  # High volatility threshold
                return MarketRegime.HIGH_VOLATILITY
            elif vol < 0.005:  # Low volatility threshold
                return MarketRegime.LOW_VOLATILITY
            elif price_trend > 0.01:  # Strong uptrend
                return MarketRegime.TRENDING_UP
            elif price_trend < -0.01:  # Strong downtrend
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            self.logger.error(f"Error classifying market regime: {e}")
            return MarketRegime.RANGING
    
    def _calculate_volume_profile_strength(self, df: pd.DataFrame) -> float:
        """Calculate strength of volume profile levels"""
        try:
            if len(df) < 10:
                return 0.5
            
            # Simple volume concentration measure
            volume_std = df['volume'].std()
            volume_mean = df['volume'].mean()
            
            # Higher std relative to mean = more concentrated volume = stronger levels
            concentration = volume_std / volume_mean if volume_mean > 0 else 0.5
            
            return min(1.0, concentration)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile strength: {e}")
            return 0.5
    
    def _calculate_bid_ask_pressure(self, df: pd.DataFrame) -> float:
        """Calculate bid-ask pressure from order flow"""
        try:
            if len(df) < 5:
                return 0.0
            
            # Use recent buy/sell volume ratio as proxy
            recent_df = df.tail(10)
            
            total_buy_volume = recent_df['buy_volume'].sum()
            total_sell_volume = recent_df['sell_volume'].sum()
            total_volume = total_buy_volume + total_sell_volume
            
            if total_volume == 0:
                return 0.0
            
            # Pressure ranges from -1 (all selling) to +1 (all buying)
            pressure = (total_buy_volume - total_sell_volume) / total_volume
            
            return pressure
            
        except Exception as e:
            self.logger.error(f"Error calculating bid-ask pressure: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate overall liquidity score"""
        try:
            if len(df) < 10:
                return 0.5
            
            # Factors affecting liquidity score:
            # 1. Volume consistency
            # 2. Spread tightness (price_range relative to price)
            # 3. Market impact (price range relative to volume)
            
            recent_df = df.tail(20)
            
            # Volume consistency (lower CV = higher liquidity)
            volume_cv = recent_df['volume'].std() / recent_df['volume'].mean() if recent_df['volume'].mean() > 0 else 1
            volume_score = max(0, 1 - volume_cv)
            
            # Spread tightness
            avg_spread = recent_df['effective_spread'].mean()
            spread_score = max(0, 1 - (avg_spread * 1000))  # Scale to reasonable range
            
            # Market impact (lower impact = higher liquidity)
            market_impact = recent_df['market_impact'].mean()
            impact_score = max(0, 1 - min(1, market_impact * 10000))  # Scale appropriately
            
            # Combined score
            liquidity_score = (volume_score * 0.4 + spread_score * 0.3 + impact_score * 0.3)
            
            return min(1.0, max(0.0, liquidity_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    def _calculate_mean_reversion_strength(self, df: pd.DataFrame) -> float:
        """Calculate mean reversion tendency"""
        try:
            if len(df) < 20:
                return 0.5
            
            # Use Bollinger Band position and RSI
            latest = df.iloc[-1]
            
            bb_position = latest.get('bollinger_position', 0.5)
            rsi = latest.get('rsi', 50)
            
            # Distance from mean (0.5 = at center, 0/1 = at extremes)
            bb_extremity = abs(bb_position - 0.5) * 2
            
            # RSI extremity (distance from 50)
            rsi_extremity = abs(rsi - 50) / 50
            
            # Mean reversion strength increases at extremes
            mean_reversion_strength = (bb_extremity + rsi_extremity) / 2
            
            return min(1.0, mean_reversion_strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion strength: {e}")
            return 0.5
    
    def _calculate_momentum_strength(self, df: pd.DataFrame) -> float:
        """Calculate momentum strength"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Price momentum
            price_momentum = df['price_momentum'].tail(5).mean()
            
            # Volume momentum
            volume_momentum = df['volume_momentum'].tail(5).mean()
            
            # Combined momentum (normalized)
            momentum_strength = abs(price_momentum) * (1 + volume_momentum / 2)
            
            return min(1.0, momentum_strength * 10)  # Scale appropriately
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {e}")
            return 0.0
    
    def analyze_order_flow(self, symbol: str, timeframe: str = '1m', 
                          minutes_back: int = 30) -> OrderFlowAnalysis:
        """Analyze order flow patterns"""
        try:
            df = self.get_tick_data(symbol, timeframe, hours_back=minutes_back/60)
            
            if df.empty:
                return OrderFlowAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    buyer_initiated_volume=0,
                    seller_initiated_volume=0,
                    volume_imbalance=0,
                    aggressive_buying=0,
                    aggressive_selling=0,
                    passive_activity=0
                )
            
            # Calculate order flow metrics
            total_buy_volume = df['buy_volume'].sum()
            total_sell_volume = df['sell_volume'].sum()
            total_volume = total_buy_volume + total_sell_volume
            
            volume_imbalance = (total_buy_volume - total_sell_volume) / total_volume if total_volume > 0 else 0
            
            # Aggressive vs passive activity (based on volume spikes)
            volume_threshold = df['volume'].quantile(0.8)  # 80th percentile
            aggressive_periods = df[df['volume'] > volume_threshold]
            
            if len(aggressive_periods) > 0:
                aggressive_buying = aggressive_periods['buy_volume'].sum() / total_buy_volume if total_buy_volume > 0 else 0
                aggressive_selling = aggressive_periods['sell_volume'].sum() / total_sell_volume if total_sell_volume > 0 else 0
            else:
                aggressive_buying = 0
                aggressive_selling = 0
            
            passive_activity = 1 - max(aggressive_buying, aggressive_selling)
            
            return OrderFlowAnalysis(
                symbol=symbol,
                timestamp=df.iloc[-1]['timestamp'],
                buyer_initiated_volume=total_buy_volume,
                seller_initiated_volume=total_sell_volume,
                volume_imbalance=volume_imbalance,
                aggressive_buying=aggressive_buying,
                aggressive_selling=aggressive_selling,
                passive_activity=passive_activity
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow for {symbol}: {e}")
            return OrderFlowAnalysis(symbol, datetime.now(timezone.utc), 0, 0, 0, 0, 0, 0)
    
    def analyze_liquidity(self, symbol: str, timeframe: str = '1m',
                         hours_back: int = 1) -> LiquidityAnalysis:
        """Analyze market liquidity characteristics"""
        try:
            df = self.get_tick_data(symbol, timeframe, hours_back)
            
            if df.empty or len(df) < 10:
                return LiquidityAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    effective_spread=0.001,
                    market_impact=0.001,
                    liquidity_pools=[],
                    thin_areas=[],
                    liquidity_score=0.5
                )
            
            # Calculate liquidity metrics
            avg_effective_spread = df['effective_spread'].mean()
            avg_market_impact = df['market_impact'].mean()
            
            # Identify liquidity pools and thin areas
            volume_threshold_high = df['volume'].quantile(0.8)
            volume_threshold_low = df['volume'].quantile(0.2)
            
            liquidity_pools = df[df['volume'] > volume_threshold_high]['close'].tolist()
            thin_areas = df[df['volume'] < volume_threshold_low]['close'].tolist()
            
            # Overall liquidity score
            liquidity_score = self._calculate_liquidity_score(df)
            
            return LiquidityAnalysis(
                symbol=symbol,
                timestamp=df.iloc[-1]['timestamp'],
                effective_spread=avg_effective_spread,
                market_impact=avg_market_impact,
                liquidity_pools=liquidity_pools[-10:],  # Last 10 pools
                thin_areas=thin_areas[-10:],  # Last 10 thin areas
                liquidity_score=liquidity_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity for {symbol}: {e}")
            return LiquidityAnalysis(symbol, datetime.now(timezone.utc), 0.001, 0.001, [], [], 0.5)
    
    def get_microstructure_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive microstructure summary"""
        try:
            # Get all analyses
            market_structure = self.analyze_market_regime(symbol)
            order_flow = self.analyze_order_flow(symbol)
            liquidity = self.analyze_liquidity(symbol)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'market_regime': {
                    'regime': market_structure.regime.value,
                    'volatility': market_structure.volatility,
                    'volume_profile_strength': market_structure.volume_profile_strength,
                    'liquidity_score': market_structure.liquidity_score,
                    'mean_reversion_strength': market_structure.mean_reversion_strength,
                    'momentum_strength': market_structure.momentum_strength
                },
                'order_flow': {
                    'volume_imbalance': order_flow.volume_imbalance,
                    'buyer_volume': order_flow.buyer_initiated_volume,
                    'seller_volume': order_flow.seller_initiated_volume,
                    'aggressive_buying': order_flow.aggressive_buying,
                    'aggressive_selling': order_flow.aggressive_selling,
                    'passive_activity': order_flow.passive_activity
                },
                'liquidity': {
                    'effective_spread': liquidity.effective_spread,
                    'market_impact': liquidity.market_impact,
                    'liquidity_score': liquidity.liquidity_score,
                    'liquidity_pools_count': len(liquidity.liquidity_pools),
                    'thin_areas_count': len(liquidity.thin_areas)
                },
                'trading_conditions': {
                    'favorable_for_scalping': self._assess_scalping_conditions(market_structure, liquidity),
                    'favorable_for_mean_reversion': market_structure.mean_reversion_strength > 0.6,
                    'favorable_for_momentum': market_structure.momentum_strength > 0.6,
                    'favorable_for_stop_hunting': market_structure.volatility > 0.02 and liquidity.liquidity_score < 0.4
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating microstructure summary for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _assess_scalping_conditions(self, market_structure: MarketMicrostructure, 
                                  liquidity: LiquidityAnalysis) -> bool:
        """Assess if conditions are favorable for scalping"""
        try:
            # Good scalping conditions:
            # 1. Moderate volatility (not too high, not too low)
            # 2. Good liquidity
            # 3. Tight spreads
            # 4. Not in strong trending regime
            
            volatility_ok = 0.005 < market_structure.volatility < 0.025
            liquidity_ok = liquidity.liquidity_score > 0.6
            spread_ok = liquidity.effective_spread < 0.002
            regime_ok = market_structure.regime in [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY]
            
            return volatility_ok and liquidity_ok and spread_ok and regime_ok
            
        except Exception as e:
            self.logger.error(f"Error assessing scalping conditions: {e}")
            return False

def main():
    """Test microstructure analyzer"""
    analyzer = MicrostructureAnalyzer()
    
    # Test with a symbol
    test_symbol = 'BTCUSDT'
    
    # Get microstructure summary
    summary = analyzer.get_microstructure_summary(test_symbol)
    
    print(f"Microstructure Analysis for {test_symbol}:")
    print(f"  Market Regime: {summary['market_regime']['regime']}")
    print(f"  Volatility: {summary['market_regime']['volatility']:.4f}")
    print(f"  Liquidity Score: {summary['liquidity']['liquidity_score']:.2f}")
    print(f"  Volume Imbalance: {summary['order_flow']['volume_imbalance']:.3f}")
    print(f"  Favorable for Scalping: {summary['trading_conditions']['favorable_for_scalping']}")

if __name__ == "__main__":
    main()