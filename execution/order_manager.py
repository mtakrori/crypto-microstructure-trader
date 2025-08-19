#!/usr/bin/env python3
"""
Order Management System for Crypto Microstructure Trading
Handles trade execution, order tracking, and position management
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

from config import (
    EXECUTION_CONFIG, LEVERAGE, DATABASE_FILE, get_table_name, PATHS
)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str
    timestamp: datetime
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: float = 0.0
    error_message: Optional[str] = None
    strategy_id: Optional[str] = None
    
@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    strategy_id: str

@dataclass
class ExecutionReport:
    """Trade execution report"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float
    execution_time_ms: float

class OrderManager:
    """Advanced order management system"""
    
    def __init__(self, paper_trading: bool = True):
        self.logger = logging.getLogger('OrderManager')
        self.paper_trading = paper_trading
        self.config = EXECUTION_CONFIG
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.execution_reports: List[ExecutionReport] = []
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        
        # Performance metrics
        self.total_trades = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
        
        # Load state
        self.load_state()
        
        if self.paper_trading:
            self.logger.info("Order Manager initialized in PAPER TRADING mode")
        else:
            self.logger.info("Order Manager initialized in LIVE TRADING mode")
    
    def load_state(self):
        """Load order manager state"""
        try:
            with open(PATHS['state_file'], 'r') as f:
                state = json.load(f)
                order_state = state.get('order_manager', {})
                
                # Load metrics
                self.total_trades = order_state.get('total_trades', 0)
                self.successful_executions = order_state.get('successful_executions', 0)
                self.failed_executions = order_state.get('failed_executions', 0)
                
                # Load positions (convert from dict)
                positions_data = order_state.get('positions', {})
                for symbol, pos_data in positions_data.items():
                    self.positions[symbol] = Position(
                        symbol=pos_data['symbol'],
                        side=pos_data['side'],
                        size=pos_data['size'],
                        entry_price=pos_data['entry_price'],
                        current_price=pos_data['current_price'],
                        unrealized_pnl=pos_data['unrealized_pnl'],
                        stop_loss=pos_data.get('stop_loss'),
                        take_profit=pos_data.get('take_profit'),
                        timestamp=datetime.fromisoformat(pos_data['timestamp']),
                        strategy_id=pos_data['strategy_id']
                    )
                    
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.info("No existing order state found")
    
    def save_state(self):
        """Save order manager state"""
        try:
            # Load existing state
            state = {}
            try:
                with open(PATHS['state_file'], 'r') as f:
                    state = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            # Convert positions to serializable format
            positions_data = {}
            for symbol, position in self.positions.items():
                positions_data[symbol] = {
                    'symbol': position.symbol,
                    'side': position.side,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'timestamp': position.timestamp.isoformat(),
                    'strategy_id': position.strategy_id
                }
            
            # Update order manager state
            state['order_manager'] = {
                'total_trades': self.total_trades,
                'successful_executions': self.successful_executions,
                'failed_executions': self.failed_executions,
                'positions': positions_data
            }
            
            with open(PATHS['state_file'], 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving order state: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from database"""
        try:
            table_name = get_table_name(symbol, '1m')
            
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT close FROM {table_name}
                    WHERE is_complete = 1
                    ORDER BY open_time DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                return float(result[0]) if result else None
                
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD_{int(datetime.now().timestamp() * 1000)}"
    
    def submit_market_order(self, symbol: str, side: OrderSide, quantity: float,
                           strategy_id: str = None) -> Optional[str]:
        """Submit market order"""
        try:
            order_id = self.generate_order_id()
            current_price = self.get_current_price(symbol)
            
            if not current_price:
                self.logger.error(f"Cannot get current price for {symbol}")
                return None
            
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None,  # Market order
                stop_price=None,
                time_in_force="IOC",  # Immediate or Cancel
                timestamp=datetime.now(timezone.utc),
                status=OrderStatus.PENDING,
                strategy_id=strategy_id
            )
            
            self.pending_orders[order_id] = order
            
            # Execute immediately for paper trading
            if self.paper_trading:
                self._execute_paper_order(order, current_price)
            else:
                # In live trading, this would submit to exchange
                self._execute_live_order(order)
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error submitting market order: {e}")
            return None
    
    def submit_limit_order(self, symbol: str, side: OrderSide, quantity: float,
                          price: float, strategy_id: str = None) -> Optional[str]:
        """Submit limit order"""
        try:
            order_id = self.generate_order_id()
            
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                stop_price=None,
                time_in_force="GTC",  # Good Till Cancelled
                timestamp=datetime.now(timezone.utc),
                status=OrderStatus.PENDING,
                strategy_id=strategy_id
            )
            
            self.pending_orders[order_id] = order
            
            # For paper trading, check if can fill immediately
            if self.paper_trading:
                current_price = self.get_current_price(symbol)
                if current_price:
                    self._check_limit_order_fill(order, current_price)
            else:
                self._submit_live_limit_order(order)
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error submitting limit order: {e}")
            return None
    
    def _execute_paper_order(self, order: Order, execution_price: float):
        """Execute order in paper trading mode"""
        try:
            start_time = time.time()
            
            # Calculate slippage (simulate market impact)
            slippage = self._calculate_simulated_slippage(order.quantity, order.symbol)
            
            if order.side == OrderSide.BUY:
                fill_price = execution_price * (1 + slippage)
            else:
                fill_price = execution_price * (1 - slippage)
            
            # Calculate commission
            commission = self._calculate_commission(order.quantity, fill_price)
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = fill_price
            order.commission = commission
            
            # Move to filled orders
            self.filled_orders[order.order_id] = order
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            
            # Update position
            self._update_position(order)
            
            # Create execution report
            execution_time = (time.time() - start_time) * 1000  # ms
            
            report = ExecutionReport(
                order_id=order.order_id,
                timestamp=datetime.now(timezone.utc),
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=fill_price,
                commission=commission,
                slippage=slippage,
                execution_time_ms=execution_time
            )
            
            self.execution_reports.append(report)
            
            # Update metrics
            self.total_trades += 1
            self.successful_executions += 1
            self.total_slippage += abs(slippage)
            self.total_commission += commission
            
            self.logger.info(f"Paper order executed: {order.symbol} {order.side.value} "
                           f"{order.quantity:.6f} @ {fill_price:.6f} "
                           f"(slippage: {slippage:.4%}, commission: ${commission:.4f})")
            
            self.save_state()
            
        except Exception as e:
            self.logger.error(f"Error executing paper order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.failed_executions += 1
    
    def _execute_live_order(self, order: Order):
        """Execute order in live trading mode (placeholder)"""
        # This would integrate with actual exchange API
        self.logger.warning("Live trading not implemented - use paper trading mode")
        order.status = OrderStatus.REJECTED
        order.error_message = "Live trading not implemented"
        self.failed_executions += 1
    
    def _submit_live_limit_order(self, order: Order):
        """Submit limit order to live exchange (placeholder)"""
        # This would integrate with actual exchange API
        self.logger.warning("Live limit orders not implemented - use paper trading mode")
        order.status = OrderStatus.REJECTED
        order.error_message = "Live trading not implemented"
    
    def _check_limit_order_fill(self, order: Order, current_price: float):
        """Check if limit order can be filled at current price"""
        try:
            can_fill = False
            
            if order.side == OrderSide.BUY and current_price <= order.price:
                can_fill = True
            elif order.side == OrderSide.SELL and current_price >= order.price:
                can_fill = True
            
            if can_fill:
                self._execute_paper_order(order, order.price)
            else:
                order.status = OrderStatus.SUBMITTED
                
        except Exception as e:
            self.logger.error(f"Error checking limit order fill: {e}")
    
    def _calculate_simulated_slippage(self, quantity: float, symbol: str) -> float:
        """Calculate simulated slippage for paper trading"""
        try:
            # Base slippage from config
            base_slippage = self.config['slippage_assumption']
            
            # Add quantity-based impact (larger orders = more slippage)
            quantity_impact = min(0.001, quantity / 100000)  # Max 0.1% additional
            
            # Add random market impact (simulate real conditions)
            import random
            random_factor = random.uniform(0.5, 1.5)
            
            total_slippage = (base_slippage + quantity_impact) * random_factor
            
            return min(0.002, total_slippage)  # Cap at 0.2%
            
        except Exception as e:
            self.logger.error(f"Error calculating slippage: {e}")
            return self.config['slippage_assumption']
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission"""
        try:
            trade_value = quantity * price / LEVERAGE  # Actual capital used
            commission = trade_value * self.config['commission_rate']
            return commission
            
        except Exception as e:
            self.logger.error(f"Error calculating commission: {e}")
            return 0.0
    
    def _update_position(self, order: Order):
        """Update position after order execution"""
        try:
            symbol = order.symbol
            
            if symbol not in self.positions:
                # New position
                side = "long" if order.side == OrderSide.BUY else "short"
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    size=order.filled_quantity,
                    entry_price=order.avg_fill_price,
                    current_price=order.avg_fill_price,
                    unrealized_pnl=0.0,
                    stop_loss=None,
                    take_profit=None,
                    timestamp=order.timestamp,
                    strategy_id=order.strategy_id or "unknown"
                )
                
                self.logger.info(f"New position opened: {symbol} {side} {order.filled_quantity:.6f}")
                
            else:
                # Modify existing position
                position = self.positions[symbol]
                
                if ((position.side == "long" and order.side == OrderSide.SELL) or
                    (position.side == "short" and order.side == OrderSide.BUY)):
                    
                    # Closing/reducing position
                    if order.filled_quantity >= position.size:
                        # Position fully closed
                        self.logger.info(f"Position closed: {symbol}")
                        del self.positions[symbol]
                    else:
                        # Position reduced
                        position.size -= order.filled_quantity
                        self.logger.info(f"Position reduced: {symbol} new size {position.size:.6f}")
                        
                else:
                    # Adding to position (averaging)
                    total_value = (position.size * position.entry_price + 
                                 order.filled_quantity * order.avg_fill_price)
                    total_size = position.size + order.filled_quantity
                    
                    position.entry_price = total_value / total_size
                    position.size = total_size
                    
                    self.logger.info(f"Position increased: {symbol} new size {total_size:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    def update_position_prices(self):
        """Update current prices and PnL for all positions"""
        try:
            for symbol, position in self.positions.items():
                current_price = self.get_current_price(symbol)
                
                if current_price:
                    position.current_price = current_price
                    
                    # Calculate unrealized PnL
                    if position.side == "long":
                        pnl_points = current_price - position.entry_price
                    else:
                        pnl_points = position.entry_price - current_price
                    
                    position.unrealized_pnl = (pnl_points / position.entry_price) * (position.size * position.entry_price / LEVERAGE) * LEVERAGE
                    
        except Exception as e:
            self.logger.error(f"Error updating position prices: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            if order_id not in self.pending_orders:
                self.logger.warning(f"Order {order_id} not found in pending orders")
                return False
            
            order = self.pending_orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                self.logger.warning(f"Cannot cancel order {order_id} - status: {order.status}")
                return False
            
            order.status = OrderStatus.CANCELLED
            
            # Move to filled orders for record keeping
            self.filled_orders[order_id] = order
            del self.pending_orders[order_id]
            
            self.logger.info(f"Order cancelled: {order_id}")
            self.save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def set_stop_loss(self, symbol: str, stop_price: float):
        """Set stop loss for position"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            self.positions[symbol].stop_loss = stop_price
            self.logger.info(f"Stop loss set for {symbol}: {stop_price:.6f}")
            self.save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting stop loss: {e}")
            return False
    
    def set_take_profit(self, symbol: str, target_price: float):
        """Set take profit for position"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            self.positions[symbol].take_profit = target_price
            self.logger.info(f"Take profit set for {symbol}: {target_price:.6f}")
            self.save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting take profit: {e}")
            return False
    
    def check_stop_loss_triggers(self) -> List[str]:
        """Check for stop loss triggers"""
        triggered_positions = []
        
        try:
            for symbol, position in self.positions.items():
                if not position.stop_loss:
                    continue
                
                current_price = position.current_price
                
                should_trigger = False
                
                if position.side == "long" and current_price <= position.stop_loss:
                    should_trigger = True
                elif position.side == "short" and current_price >= position.stop_loss:
                    should_trigger = True
                
                if should_trigger:
                    triggered_positions.append(symbol)
                    
        except Exception as e:
            self.logger.error(f"Error checking stop loss triggers: {e}")
        
        return triggered_positions
    
    def check_take_profit_triggers(self) -> List[str]:
        """Check for take profit triggers"""
        triggered_positions = []
        
        try:
            for symbol, position in self.positions.items():
                if not position.take_profit:
                    continue
                
                current_price = position.current_price
                
                should_trigger = False
                
                if position.side == "long" and current_price >= position.take_profit:
                    should_trigger = True
                elif position.side == "short" and current_price <= position.take_profit:
                    should_trigger = True
                
                if should_trigger:
                    triggered_positions.append(symbol)
                    
        except Exception as e:
            self.logger.error(f"Error checking take profit triggers: {e}")
        
        return triggered_positions
    
    def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """Close position at market price"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Determine order side (opposite of position)
            order_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
            
            # Submit market order to close
            order_id = self.submit_market_order(
                symbol=symbol,
                side=order_side,
                quantity=position.size,
                strategy_id=f"close_{reason}"
            )
            
            if order_id:
                self.logger.info(f"Position close order submitted: {symbol} ({reason})")
                return True
            else:
                self.logger.error(f"Failed to submit close order for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    def get_execution_summary(self) -> Dict:
        """Get execution performance summary"""
        try:
            total_executions = self.successful_executions + self.failed_executions
            success_rate = (self.successful_executions / total_executions) if total_executions > 0 else 0
            
            avg_slippage = (self.total_slippage / self.successful_executions) if self.successful_executions > 0 else 0
            avg_commission = (self.total_commission / self.successful_executions) if self.successful_executions > 0 else 0
            
            # Recent execution times
            recent_reports = self.execution_reports[-20:] if len(self.execution_reports) > 20 else self.execution_reports
            avg_execution_time = sum(r.execution_time_ms for r in recent_reports) / len(recent_reports) if recent_reports else 0
            
            return {
                'total_trades': self.total_trades,
                'successful_executions': self.successful_executions,
                'failed_executions': self.failed_executions,
                'success_rate': success_rate,
                'avg_slippage': avg_slippage,
                'avg_commission': avg_commission,
                'avg_execution_time_ms': avg_execution_time,
                'open_positions': len(self.positions),
                'pending_orders': len(self.pending_orders)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating execution summary: {e}")
            return {}

def main():
    """Test order manager"""
    order_manager = OrderManager(paper_trading=True)
    
    # Test market order
    order_id = order_manager.submit_market_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.001,
        strategy_id="test_strategy"
    )
    
    print(f"Market order submitted: {order_id}")
    
    # Update position prices
    order_manager.update_position_prices()
    
    # Get execution summary
    summary = order_manager.get_execution_summary()
    print(f"Execution Summary: {summary}")
    
    # Show positions
    for symbol, position in order_manager.positions.items():
        print(f"Position: {symbol} {position.side} {position.size:.6f} @ {position.entry_price:.6f}")

if __name__ == "__main__":
    main()