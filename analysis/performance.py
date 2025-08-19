#!/usr/bin/env python3
"""
Performance Tracking and Analytics for Crypto Microstructure Trading
Comprehensive performance analysis, comparison, and reporting
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from config import PATHS, PERFORMANCE_CONFIG

class PerformancePeriod(Enum):
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "7d"
    MONTHLY = "30d"

@dataclass
class TradePerformance:
    """Individual trade performance metrics"""
    trade_id: str
    symbol: str
    strategy: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percentage: float
    commission: float
    slippage: float
    hold_time_seconds: int
    max_adverse_excursion: float
    max_favorable_excursion: float
    confidence_score: float
    market_conditions: Dict

@dataclass
class StrategyPerformance:
    """Strategy-specific performance metrics"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_hold_time_minutes: float
    trades_per_day: float

@dataclass
class PerformanceComparison:
    """Performance comparison between systems"""
    old_system_metrics: Dict
    new_system_metrics: Dict
    improvement_metrics: Dict
    statistical_significance: Dict

class PerformanceTracker:
    """Comprehensive performance tracking and analytics"""
    
    def __init__(self):
        self.logger = logging.getLogger('PerformanceTracker')
        self.config = PERFORMANCE_CONFIG
        
        # Performance storage
        self.trades: List[TradePerformance] = []
        self.daily_metrics: List[Dict] = []
        self.strategy_metrics: Dict[str, StrategyPerformance] = {}
        
        # Load existing data
        self.load_performance_history()
        
        # Initialize tracking
        self.start_time = datetime.now(timezone.utc)
        self.last_save_time = self.start_time
        
    def load_performance_history(self):
        """Load performance history from files"""
        try:
            # Load trades history
            trades_file = PATHS['trades_file']
            if Path(trades_file).exists():
                with open(trades_file, 'r') as f:
                    trades_data = json.load(f)
                    
                    for trade_data in trades_data:
                        trade = TradePerformance(
                            trade_id=trade_data['trade_id'],
                            symbol=trade_data['symbol'],
                            strategy=trade_data['strategy'],
                            side=trade_data['side'],
                            entry_time=datetime.fromisoformat(trade_data['entry_time']),
                            exit_time=datetime.fromisoformat(trade_data['exit_time']),
                            entry_price=trade_data['entry_price'],
                            exit_price=trade_data['exit_price'],
                            quantity=trade_data['quantity'],
                            pnl=trade_data['pnl'],
                            pnl_percentage=trade_data['pnl_percentage'],
                            commission=trade_data.get('commission', 0),
                            slippage=trade_data.get('slippage', 0),
                            hold_time_seconds=trade_data['hold_time_seconds'],
                            max_adverse_excursion=trade_data.get('max_adverse_excursion', 0),
                            max_favorable_excursion=trade_data.get('max_favorable_excursion', 0),
                            confidence_score=trade_data.get('confidence_score', 0.5),
                            market_conditions=trade_data.get('market_conditions', {})
                        )
                        self.trades.append(trade)
            
            # Load daily metrics
            performance_file = PATHS['performance_file']
            if Path(performance_file).exists():
                with open(performance_file, 'r') as f:
                    self.daily_metrics = json.load(f)
                    
            self.logger.info(f"Loaded {len(self.trades)} trades and {len(self.daily_metrics)} daily metrics")
            
        except Exception as e:
            self.logger.error(f"Error loading performance history: {e}")
    
    def save_performance_data(self):
        """Save performance data to files"""
        try:
            # Save trades
            trades_data = []
            for trade in self.trades:
                trade_dict = asdict(trade)
                trade_dict['entry_time'] = trade.entry_time.isoformat()
                trade_dict['exit_time'] = trade.exit_time.isoformat()
                trades_data.append(trade_dict)
            
            with open(PATHS['trades_file'], 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)
            
            # Save daily metrics
            with open(PATHS['performance_file'], 'w') as f:
                json.dump(self.daily_metrics, f, indent=2, default=str)
                
            self.last_save_time = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    def record_trade(self, trade_id: str, symbol: str, strategy: str, side: str,
                    entry_time: datetime, exit_time: datetime, entry_price: float,
                    exit_price: float, quantity: float, commission: float = 0,
                    slippage: float = 0, confidence_score: float = 0.5,
                    market_conditions: Dict = None):
        """Record a completed trade"""
        try:
            # Calculate metrics
            if side.lower() in ['buy', 'long']:
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            pnl_percentage = pnl / (entry_price * quantity) if entry_price > 0 else 0
            hold_time_seconds = int((exit_time - entry_time).total_seconds())
            
            # Create trade record
            trade = TradePerformance(
                trade_id=trade_id,
                symbol=symbol,
                strategy=strategy,
                side=side,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl=pnl - commission,  # Net PnL after commission
                pnl_percentage=pnl_percentage,
                commission=commission,
                slippage=slippage,
                hold_time_seconds=hold_time_seconds,
                max_adverse_excursion=0,  # Would need real-time tracking
                max_favorable_excursion=0,  # Would need real-time tracking
                confidence_score=confidence_score,
                market_conditions=market_conditions or {}
            )
            
            self.trades.append(trade)
            
            # Update strategy metrics
            self._update_strategy_metrics(strategy)
            
            self.logger.info(f"Trade recorded: {symbol} {strategy} {side} "
                           f"PnL: ${pnl:.4f} ({pnl_percentage:.2%})")
            
            # Auto-save if configured
            if self.config['track_all_signals']:
                self._check_auto_save()
                
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def _update_strategy_metrics(self, strategy_name: str):
        """Update metrics for a specific strategy"""
        try:
            strategy_trades = [t for t in self.trades if t.strategy == strategy_name]
            
            if not strategy_trades:
                return
            
            # Calculate metrics
            total_trades = len(strategy_trades)
            winning_trades = sum(1 for t in strategy_trades if t.pnl > 0)
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t.pnl for t in strategy_trades)
            
            wins = [t.pnl for t in strategy_trades if t.pnl > 0]
            losses = [t.pnl for t in strategy_trades if t.pnl <= 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio
            returns = [t.pnl_percentage for t in strategy_trades]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(strategy_trades)
            
            # Average hold time
            avg_hold_time_minutes = np.mean([t.hold_time_seconds / 60 for t in strategy_trades])
            
            # Trades per day
            if strategy_trades:
                days_active = (strategy_trades[-1].exit_time - strategy_trades[0].entry_time).days + 1
                trades_per_day = total_trades / max(1, days_active)
            else:
                trades_per_day = 0
            
            # Update strategy performance
            self.strategy_metrics[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_hold_time_minutes=avg_hold_time_minutes,
                trades_per_day=trades_per_day
            )
            
        except Exception as e:
            self.logger.error(f"Error updating strategy metrics for {strategy_name}: {e}")
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for returns"""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming returns are trade-level)
            excess_return = mean_return - (risk_free_rate / 252)  # Daily risk-free rate
            sharpe = excess_return / std_return * np.sqrt(252)  # Annualize
            
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, trades: List[TradePerformance]) -> float:
        """Calculate maximum drawdown for trades"""
        try:
            if not trades:
                return 0.0
            
            # Calculate cumulative PnL
            cumulative_pnl = np.cumsum([t.pnl for t in trades])
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_pnl)
            
            # Calculate drawdown at each point
            drawdowns = running_max - cumulative_pnl
            
            # Return maximum drawdown
            max_drawdown = np.max(drawdowns)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _check_auto_save(self):
        """Check if auto-save is needed"""
        try:
            time_since_save = (datetime.now(timezone.utc) - self.last_save_time).total_seconds()
            save_interval_seconds = self.config['save_frequency_minutes'] * 60
            
            if time_since_save >= save_interval_seconds:
                self.save_performance_data()
                
        except Exception as e:
            self.logger.error(f"Error checking auto-save: {e}")
    
    def get_performance_summary(self, period: PerformancePeriod = PerformancePeriod.DAILY) -> Dict:
        """Get comprehensive performance summary"""
        try:
            # Filter trades by period
            cutoff_time = self._get_period_cutoff(period)
            recent_trades = [t for t in self.trades if t.exit_time >= cutoff_time]
            
            if not recent_trades:
                return self._empty_performance_summary()
            
            # Overall metrics
            total_trades = len(recent_trades)
            winning_trades = sum(1 for t in recent_trades if t.pnl > 0)
            win_rate = winning_trades / total_trades
            
            total_pnl = sum(t.pnl for t in recent_trades)
            total_commission = sum(t.commission for t in recent_trades)
            total_slippage = sum(abs(t.slippage) for t in recent_trades)
            
            # PnL statistics
            wins = [t.pnl for t in recent_trades if t.pnl > 0]
            losses = [t.pnl for t in recent_trades if t.pnl <= 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Time-based metrics
            avg_hold_time = np.mean([t.hold_time_seconds for t in recent_trades]) / 60  # minutes
            
            # Strategy breakdown
            strategy_breakdown = {}
            for strategy_name, metrics in self.strategy_metrics.items():
                strategy_trades_recent = [t for t in recent_trades if t.strategy == strategy_name]
                if strategy_trades_recent:
                    strategy_breakdown[strategy_name] = {
                        'trades': len(strategy_trades_recent),
                        'win_rate': sum(1 for t in strategy_trades_recent if t.pnl > 0) / len(strategy_trades_recent),
                        'total_pnl': sum(t.pnl for t in strategy_trades_recent),
                        'avg_hold_time': np.mean([t.hold_time_seconds for t in strategy_trades_recent]) / 60
                    }
            
            # Symbol breakdown
            symbol_breakdown = {}
            symbols = set(t.symbol for t in recent_trades)
            for symbol in symbols:
                symbol_trades = [t for t in recent_trades if t.symbol == symbol]
                symbol_breakdown[symbol] = {
                    'trades': len(symbol_trades),
                    'win_rate': sum(1 for t in symbol_trades if t.pnl > 0) / len(symbol_trades),
                    'total_pnl': sum(t.pnl for t in symbol_trades)
                }
            
            return {
                'period': period.value,
                'summary': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'net_pnl': total_pnl - total_commission,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'avg_hold_time_minutes': avg_hold_time,
                    'total_commission': total_commission,
                    'total_slippage': total_slippage
                },
                'strategy_breakdown': strategy_breakdown,
                'symbol_breakdown': symbol_breakdown,
                'best_trade': max(recent_trades, key=lambda x: x.pnl).pnl if recent_trades else 0,
                'worst_trade': min(recent_trades, key=lambda x: x.pnl).pnl if recent_trades else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return self._empty_performance_summary()
    
    def _get_period_cutoff(self, period: PerformancePeriod) -> datetime:
        """Get cutoff time for performance period"""
        now = datetime.now(timezone.utc)
        
        if period == PerformancePeriod.HOURLY:
            return now - timedelta(hours=1)
        elif period == PerformancePeriod.DAILY:
            return now - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            return now - timedelta(days=7)
        elif period == PerformancePeriod.MONTHLY:
            return now - timedelta(days=30)
        else:
            return now - timedelta(days=1)
    
    def _empty_performance_summary(self) -> Dict:
        """Return empty performance summary"""
        return {
            'period': 'none',
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'net_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_hold_time_minutes': 0,
                'total_commission': 0,
                'total_slippage': 0
            },
            'strategy_breakdown': {},
            'symbol_breakdown': {},
            'best_trade': 0,
            'worst_trade': 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def compare_with_baseline(self, baseline_metrics: Dict) -> PerformanceComparison:
        """Compare current performance with baseline (old system)"""
        try:
            current_metrics = self.get_performance_summary(PerformancePeriod.WEEKLY)
            
            # Calculate improvements
            improvements = {}
            
            current_summary = current_metrics['summary']
            baseline_summary = baseline_metrics.get('summary', {})
            
            for metric in ['win_rate', 'total_pnl', 'profit_factor', 'avg_hold_time_minutes']:
                current_val = current_summary.get(metric, 0)
                baseline_val = baseline_summary.get(metric, 0)
                
                if baseline_val != 0:
                    improvement_pct = ((current_val - baseline_val) / baseline_val) * 100
                    improvements[metric] = improvement_pct
                else:
                    improvements[metric] = 0
            
            # Statistical significance (simplified)
            significance = {}
            if current_summary['total_trades'] >= 30 and baseline_summary.get('total_trades', 0) >= 30:
                significance['sample_size_adequate'] = True
                significance['win_rate_improvement_significant'] = abs(improvements.get('win_rate', 0)) > 10
            else:
                significance['sample_size_adequate'] = False
                significance['win_rate_improvement_significant'] = False
            
            return PerformanceComparison(
                old_system_metrics=baseline_metrics,
                new_system_metrics=current_metrics,
                improvement_metrics=improvements,
                statistical_significance=significance
            )
            
        except Exception as e:
            self.logger.error(f"Error comparing with baseline: {e}")
            return PerformanceComparison({}, {}, {}, {})
    
    def generate_daily_report(self) -> Dict:
        """Generate daily performance report"""
        try:
            daily_summary = self.get_performance_summary(PerformancePeriod.DAILY)
            weekly_summary = self.get_performance_summary(PerformancePeriod.WEEKLY)
            
            # Key highlights
            highlights = []
            
            daily_win_rate = daily_summary['summary']['win_rate']
            if daily_win_rate > 0.7:
                highlights.append(f"Excellent win rate: {daily_win_rate:.1%}")
            elif daily_win_rate < 0.4:
                highlights.append(f"Low win rate alert: {daily_win_rate:.1%}")
            
            daily_trades = daily_summary['summary']['total_trades']
            if daily_trades > 20:
                highlights.append(f"High activity: {daily_trades} trades")
            elif daily_trades == 0:
                highlights.append("No trades executed today")
            
            daily_pnl = daily_summary['summary']['total_pnl']
            if daily_pnl > 100:
                highlights.append(f"Strong profits: ${daily_pnl:.2f}")
            elif daily_pnl < -50:
                highlights.append(f"Loss alert: ${daily_pnl:.2f}")
            
            # Strategy performance ranking
            strategy_ranking = []
            for strategy, data in daily_summary['strategy_breakdown'].items():
                strategy_ranking.append({
                    'strategy': strategy,
                    'pnl': data['total_pnl'],
                    'win_rate': data['win_rate'],
                    'trades': data['trades']
                })
            
            strategy_ranking.sort(key=lambda x: x['pnl'], reverse=True)
            
            report = {
                'date': datetime.now(timezone.utc).date().isoformat(),
                'daily_summary': daily_summary,
                'weekly_context': weekly_summary,
                'highlights': highlights,
                'top_strategies': strategy_ranking[:3],
                'recommendations': self._generate_recommendations(daily_summary, weekly_summary)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
            return {}
    
    def _generate_recommendations(self, daily_summary: Dict, weekly_summary: Dict) -> List[str]:
        """Generate performance-based recommendations"""
        recommendations = []
        
        try:
            daily_win_rate = daily_summary['summary']['win_rate']
            weekly_win_rate = weekly_summary['summary']['win_rate']
            
            # Win rate recommendations
            if daily_win_rate < 0.4:
                recommendations.append("Consider reducing position sizes due to low win rate")
            
            if weekly_win_rate > 0.6:
                recommendations.append("Strong weekly performance - consider increasing allocation")
            
            # Strategy recommendations
            best_strategies = []
            for strategy, data in weekly_summary['strategy_breakdown'].items():
                if data['win_rate'] > 0.6 and data['total_pnl'] > 0:
                    best_strategies.append(strategy)
            
            if best_strategies:
                recommendations.append(f"Focus on top performing strategies: {', '.join(best_strategies)}")
            
            # Hold time recommendations
            daily_hold_time = daily_summary['summary']['avg_hold_time_minutes']
            if daily_hold_time > 15:
                recommendations.append("Average hold times above target - review exit criteria")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

def main():
    """Test performance tracker"""
    tracker = PerformanceTracker()
    
    # Test recording a trade
    tracker.record_trade(
        trade_id="TEST_001",
        symbol="BTCUSDT",
        strategy="stop_hunt_long",
        side="long",
        entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
        exit_time=datetime.now(timezone.utc),
        entry_price=45000.0,
        exit_price=45150.0,
        quantity=0.01,
        commission=0.50,
        confidence_score=0.8
    )
    
    # Get performance summary
    summary = tracker.get_performance_summary()
    print(f"Performance Summary:")
    print(f"  Total Trades: {summary['summary']['total_trades']}")
    print(f"  Win Rate: {summary['summary']['win_rate']:.1%}")
    print(f"  Total PnL: ${summary['summary']['total_pnl']:.2f}")
    
    # Generate daily report
    report = tracker.generate_daily_report()
    print(f"\nDaily Report Highlights:")
    for highlight in report.get('highlights', []):
        print(f"  - {highlight}")

if __name__ == "__main__":
    main()