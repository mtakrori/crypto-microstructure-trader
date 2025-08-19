#!/usr/bin/env python3
"""
Advanced Risk Management for Crypto Microstructure Trading System
Handles position sizing, risk limits, and portfolio protection
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from config import (
    RISK_CONFIG, ACCOUNT_BALANCE, LEVERAGE, MAX_CONCURRENT_TRADES,
    MAX_DAILY_TRADES, PATHS
)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Current risk metrics for the account"""
    account_balance: float
    current_exposure: float
    daily_pnl: float
    open_positions: int
    daily_trades: int
    max_drawdown: float
    win_rate_24h: float
    risk_level: RiskLevel
    available_buying_power: float

@dataclass
class PositionSizeCalculation:
    """Position size calculation result"""
    symbol: str
    signal_type: str
    recommended_size: float
    max_allowed_size: float
    risk_amount: float
    confidence_adjustment: float
    reason: str
    approved: bool

@dataclass
class RiskEvent:
    """Risk event logging"""
    timestamp: datetime
    event_type: str
    severity: RiskLevel
    description: str
    action_taken: str
    metrics_snapshot: Dict

class AdvancedRiskManager:
    """Advanced risk management system"""
    
    def __init__(self, initial_balance: float = ACCOUNT_BALANCE):
        self.logger = logging.getLogger('RiskManager')
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.config = RISK_CONFIG
        
        # Risk tracking
        self.open_positions: Dict[str, Dict] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.risk_events: List[RiskEvent] = []
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        # Load existing state
        self.load_state()
        
    def load_state(self):
        """Load risk manager state from file"""
        try:
            with open(PATHS['state_file'], 'r') as f:
                state = json.load(f)
                risk_state = state.get('risk_manager', {})
                
                self.current_balance = risk_state.get('current_balance', self.initial_balance)
                self.daily_trades = risk_state.get('daily_trades', 0)
                self.daily_pnl = risk_state.get('daily_pnl', 0.0)
                self.max_drawdown = risk_state.get('max_drawdown', 0.0)
                self.open_positions = risk_state.get('open_positions', {})
                
                # Check if new day (reset daily counters)
                last_date = risk_state.get('last_reset_date')
                if last_date != str(self.last_reset_date):
                    self.reset_daily_counters()
                    
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.info("No existing risk state found, starting fresh")
    
    def save_state(self):
        """Save risk manager state to file"""
        try:
            # Load existing state file
            state = {}
            try:
                with open(PATHS['state_file'], 'r') as f:
                    state = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            # Update risk manager state
            state['risk_manager'] = {
                'current_balance': self.current_balance,
                'daily_trades': self.daily_trades,
                'daily_pnl': self.daily_pnl,
                'max_drawdown': self.max_drawdown,
                'open_positions': self.open_positions,
                'last_reset_date': str(self.last_reset_date)
            }
            
            # Save back to file
            with open(PATHS['state_file'], 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving risk state: {e}")
    
    def reset_daily_counters(self):
        """Reset daily counters for new trading day"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.logger.info("Daily risk counters reset")
    
    def calculate_current_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        try:
            # Calculate current exposure
            current_exposure = sum(
                pos.get('exposure', 0) for pos in self.open_positions.values()
            )
            
            # Calculate 24h win rate
            win_rate_24h = self._calculate_recent_win_rate()
            
            # Determine risk level
            risk_level = self._assess_risk_level(current_exposure, win_rate_24h)
            
            # Available buying power
            used_margin = current_exposure / LEVERAGE
            available_buying_power = max(0, self.current_balance - used_margin)
            
            return RiskMetrics(
                account_balance=self.current_balance,
                current_exposure=current_exposure,
                daily_pnl=self.daily_pnl,
                open_positions=len(self.open_positions),
                daily_trades=self.daily_trades,
                max_drawdown=self.max_drawdown,
                win_rate_24h=win_rate_24h,
                risk_level=risk_level,
                available_buying_power=available_buying_power
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, RiskLevel.CRITICAL, 0)
    
    def _calculate_recent_win_rate(self, hours: int = 24) -> float:
        """Calculate win rate for recent trades"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_trades = [
                trade for trade in self.trade_history 
                if datetime.fromisoformat(trade['timestamp']) > cutoff_time
                and trade.get('status') == 'closed'
            ]
            
            if not recent_trades:
                return 0.5  # Default to 50% if no recent trades
            
            wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            return wins / len(recent_trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.5
    
    def _assess_risk_level(self, current_exposure: float, win_rate: float) -> RiskLevel:
        """Assess current risk level"""
        try:
            # Exposure risk
            exposure_ratio = current_exposure / self.current_balance
            
            # Drawdown risk
            drawdown_ratio = self.max_drawdown / self.initial_balance
            
            # Performance risk
            performance_risk = 0.5 - win_rate  # Higher when win rate is low
            
            # Daily loss risk
            daily_loss_ratio = max(0, -self.daily_pnl / self.current_balance)
            
            # Combine risk factors
            risk_score = (
                exposure_ratio * 0.3 +
                drawdown_ratio * 0.3 +
                performance_risk * 0.2 +
                daily_loss_ratio * 0.2
            )
            
            if risk_score > 0.8:
                return RiskLevel.CRITICAL
            elif risk_score > 0.6:
                return RiskLevel.HIGH
            elif risk_score > 0.4:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Error assessing risk level: {e}")
            return RiskLevel.CRITICAL
    
    def calculate_position_size(self, signal_type: str, entry_price: float,
                              stop_loss: float, confidence: float,
                              symbol: str) -> PositionSizeCalculation:
        """Calculate appropriate position size for a trade"""
        try:
            metrics = self.calculate_current_metrics()
            
            # Check if we can take new positions
            if not self._can_take_new_position(metrics):
                return PositionSizeCalculation(
                    symbol=symbol,
                    signal_type=signal_type,
                    recommended_size=0,
                    max_allowed_size=0,
                    risk_amount=0,
                    confidence_adjustment=0,
                    reason="Risk limits exceeded",
                    approved=False
                )
            
            # Calculate base risk amount
            base_risk_amount = self.current_balance * self.config['max_risk_per_trade']
            
            # Adjust for confidence
            confidence_adjustment = confidence
            adjusted_risk_amount = base_risk_amount * confidence_adjustment
            
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss) / entry_price
            
            if stop_distance == 0:
                return PositionSizeCalculation(
                    symbol=symbol,
                    signal_type=signal_type,
                    recommended_size=0,
                    max_allowed_size=0,
                    risk_amount=0,
                    confidence_adjustment=confidence_adjustment,
                    reason="Invalid stop loss",
                    approved=False
                )
            
            # Calculate position size
            position_value = adjusted_risk_amount / stop_distance
            position_size = (position_value / entry_price) * LEVERAGE
            
            # Apply risk level adjustments
            position_size = self._apply_risk_level_adjustment(position_size, metrics.risk_level)
            
            # Apply performance-based adjustments
            position_size = self._apply_performance_adjustment(position_size, metrics.win_rate_24h)
            
            # Calculate maximum allowed size
            max_position_value = self.current_balance * self.config['max_position_size']
            max_allowed_size = (max_position_value / entry_price) * LEVERAGE
            
            # Ensure we don't exceed maximum
            final_size = min(position_size, max_allowed_size)
            
            # Final checks
            approved = self._final_position_checks(final_size, entry_price, metrics)
            
            reason = "Approved" if approved else "Failed final checks"
            
            return PositionSizeCalculation(
                symbol=symbol,
                signal_type=signal_type,
                recommended_size=final_size,
                max_allowed_size=max_allowed_size,
                risk_amount=adjusted_risk_amount,
                confidence_adjustment=confidence_adjustment,
                reason=reason,
                approved=approved
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return PositionSizeCalculation(
                symbol=symbol,
                signal_type=signal_type,
                recommended_size=0,
                max_allowed_size=0,
                risk_amount=0,
                confidence_adjustment=0,
                reason=f"Calculation error: {str(e)}",
                approved=False
            )
    
    def _can_take_new_position(self, metrics: RiskMetrics) -> bool:
        """Check if we can take a new position"""
        # Check concurrent position limit
        if metrics.open_positions >= MAX_CONCURRENT_TRADES:
            return False
        
        # Check daily trade limit
        if metrics.daily_trades >= MAX_DAILY_TRADES:
            return False
        
        # Check daily loss limit
        daily_loss_limit = self.current_balance * self.config['daily_loss_limit']
        if -metrics.daily_pnl >= daily_loss_limit:
            return False
        
        # Check risk level
        if metrics.risk_level == RiskLevel.CRITICAL:
            return False
        
        # Check available buying power
        if metrics.available_buying_power < self.current_balance * 0.1:  # Need 10% available
            return False
        
        return True
    
    def _apply_risk_level_adjustment(self, position_size: float, risk_level: RiskLevel) -> float:
        """Apply position size adjustment based on current risk level"""
        adjustments = {
            RiskLevel.LOW: 1.0,      # No adjustment
            RiskLevel.MEDIUM: 0.8,   # Reduce by 20%
            RiskLevel.HIGH: 0.5,     # Reduce by 50%
            RiskLevel.CRITICAL: 0.0  # No new positions
        }
        
        return position_size * adjustments.get(risk_level, 0.5)
    
    def _apply_performance_adjustment(self, position_size: float, win_rate: float) -> float:
        """Apply position size adjustment based on recent performance"""
        if win_rate < self.config['win_rate_threshold']:
            # Reduce size when performance is poor
            reduction_factor = max(0.3, win_rate / self.config['win_rate_threshold'])
            return position_size * reduction_factor
        elif win_rate > 0.7:
            # Slightly increase size when performance is good
            return position_size * 1.1
        else:
            return position_size
    
    def _final_position_checks(self, position_size: float, entry_price: float,
                              metrics: RiskMetrics) -> bool:
        """Final validation checks for position"""
        # Check minimum position size
        min_order_value = 10  # $10 minimum
        position_value = position_size * entry_price / LEVERAGE
        
        if position_value < min_order_value:
            return False
        
        # Check if this would exceed total exposure limits
        new_exposure = position_size * entry_price
        total_exposure = metrics.current_exposure + new_exposure
        max_total_exposure = self.current_balance * 3  # 3x leverage limit
        
        if total_exposure > max_total_exposure:
            return False
        
        return True
    
    def add_position(self, symbol: str, signal_type: str, size: float,
                    entry_price: float, stop_loss: float, target_price: float):
        """Add a new position to risk tracking"""
        try:
            position_id = f"{symbol}_{int(datetime.now().timestamp())}"
            
            exposure = size * entry_price
            
            self.open_positions[position_id] = {
                'symbol': symbol,
                'signal_type': signal_type,
                'size': size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'exposure': exposure,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'unrealized_pnl': 0.0
            }
            
            self.daily_trades += 1
            self.save_state()
            
            self.logger.info(f"Position added: {symbol} size={size:.6f} exposure=${exposure:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
    
    def close_position(self, position_id: str, exit_price: float, exit_reason: str):
        """Close a position and update metrics"""
        try:
            if position_id not in self.open_positions:
                self.logger.warning(f"Position {position_id} not found")
                return
            
            position = self.open_positions[position_id]
            
            # Calculate PnL
            size = position['size']
            entry_price = position['entry_price']
            
            # Determine if long or short based on signal type
            is_long = any(keyword in position['signal_type'].lower() 
                         for keyword in ['long', 'bounce', 'support'])
            
            if is_long:
                pnl_points = exit_price - entry_price
            else:
                pnl_points = entry_price - exit_price
            
            pnl_percentage = pnl_points / entry_price
            pnl_usd = (size * entry_price / LEVERAGE) * pnl_percentage * LEVERAGE
            
            # Update balance and daily PnL
            self.current_balance += pnl_usd
            self.daily_pnl += pnl_usd
            
            # Update max drawdown
            if self.current_balance < self.initial_balance:
                drawdown = self.initial_balance - self.current_balance
                self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': position['symbol'],
                'signal_type': position['signal_type'],
                'size': size,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl_usd,
                'pnl_percentage': pnl_percentage,
                'exit_reason': exit_reason,
                'status': 'closed'
            }
            
            self.trade_history.append(trade_record)
            
            # Remove from open positions
            del self.open_positions[position_id]
            
            self.save_state()
            
            self.logger.info(f"Position closed: {position['symbol']} PnL=${pnl_usd:.2f} "
                           f"({pnl_percentage:.2%}) Reason: {exit_reason}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def update_position_pnl(self, position_id: str, current_price: float):
        """Update unrealized PnL for open position"""
        try:
            if position_id not in self.open_positions:
                return
            
            position = self.open_positions[position_id]
            entry_price = position['entry_price']
            size = position['size']
            
            # Determine direction
            is_long = any(keyword in position['signal_type'].lower() 
                         for keyword in ['long', 'bounce', 'support'])
            
            if is_long:
                pnl_points = current_price - entry_price
            else:
                pnl_points = entry_price - current_price
            
            pnl_percentage = pnl_points / entry_price
            unrealized_pnl = (size * entry_price / LEVERAGE) * pnl_percentage * LEVERAGE
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
        except Exception as e:
            self.logger.error(f"Error updating position PnL: {e}")
    
    def check_stop_loss_triggers(self) -> List[str]:
        """Check if any positions should be stopped out"""
        triggered_positions = []
        
        try:
            for position_id, position in self.open_positions.items():
                current_price = position.get('current_price')
                if not current_price:
                    continue
                
                stop_loss = position['stop_loss']
                is_long = any(keyword in position['signal_type'].lower() 
                             for keyword in ['long', 'bounce', 'support'])
                
                should_stop = False
                
                if is_long and current_price <= stop_loss:
                    should_stop = True
                elif not is_long and current_price >= stop_loss:
                    should_stop = True
                
                if should_stop:
                    triggered_positions.append(position_id)
                    
        except Exception as e:
            self.logger.error(f"Error checking stop losses: {e}")
        
        return triggered_positions
    
    def log_risk_event(self, event_type: str, severity: RiskLevel, 
                      description: str, action_taken: str):
        """Log a risk management event"""
        try:
            metrics = self.calculate_current_metrics()
            
            event = RiskEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                severity=severity,
                description=description,
                action_taken=action_taken,
                metrics_snapshot=asdict(metrics)
            )
            
            self.risk_events.append(event)
            
            # Keep only last 100 events
            if len(self.risk_events) > 100:
                self.risk_events = self.risk_events[-100:]
            
            self.logger.warning(f"Risk Event: {event_type} - {description}")
            
        except Exception as e:
            self.logger.error(f"Error logging risk event: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        try:
            metrics = self.calculate_current_metrics()
            
            # Recent performance
            trades_24h = len([
                t for t in self.trade_history 
                if datetime.fromisoformat(t['timestamp']) > 
                datetime.now(timezone.utc) - timedelta(hours=24)
            ])
            
            return {
                'current_metrics': asdict(metrics),
                'account_health': {
                    'balance': self.current_balance,
                    'balance_change': self.current_balance - self.initial_balance,
                    'balance_change_pct': (self.current_balance - self.initial_balance) / self.initial_balance,
                    'max_drawdown': self.max_drawdown,
                    'max_drawdown_pct': self.max_drawdown / self.initial_balance
                },
                'daily_stats': {
                    'trades_today': self.daily_trades,
                    'daily_pnl': self.daily_pnl,
                    'daily_pnl_pct': self.daily_pnl / self.current_balance,
                    'trades_24h': trades_24h
                },
                'position_summary': {
                    'open_positions': len(self.open_positions),
                    'total_exposure': metrics.current_exposure,
                    'exposure_ratio': metrics.current_exposure / self.current_balance,
                    'available_buying_power': metrics.available_buying_power
                },
                'risk_status': {
                    'risk_level': metrics.risk_level.value,
                    'can_trade': self._can_take_new_position(metrics),
                    'recent_events': len([e for e in self.risk_events 
                                        if e.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk summary: {e}")
            return {}

def main():
    """Test risk manager"""
    risk_manager = AdvancedRiskManager()
    
    # Test position size calculation
    calc = risk_manager.calculate_position_size(
        signal_type="stop_hunt_long",
        entry_price=45000.0,
        stop_loss=44800.0,
        confidence=0.75,
        symbol="BTCUSDT"
    )
    
    print("Position Size Calculation:")
    print(f"  Recommended Size: {calc.recommended_size:.6f}")
    print(f"  Risk Amount: ${calc.risk_amount:.2f}")
    print(f"  Approved: {calc.approved}")
    print(f"  Reason: {calc.reason}")
    
    # Test risk summary
    summary = risk_manager.get_risk_summary()
    print(f"\nRisk Summary:")
    print(f"  Risk Level: {summary['risk_status']['risk_level']}")
    print(f"  Can Trade: {summary['risk_status']['can_trade']}")
    print(f"  Balance: ${summary['account_health']['balance']:.2f}")

if __name__ == "__main__":
    main()