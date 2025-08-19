#!/usr/bin/env python3
"""
Main Trading Engine for Crypto Microstructure Trading System
Orchestrates all components: strategies, risk management, execution, and performance tracking
"""

import time
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
from pathlib import Path

# Import all components
from config import (
    SYMBOLS, MAX_CONCURRENT_TRADES, LOGGING_CONFIG, PATHS,
    setup_directories, validate_config
)
from data.validator import DataValidator
from strategies.stop_hunt import StopHuntDetector
from strategies.scalping import ScalpingStrategy
from strategies.volume_profile import VolumeProfileStrategy
from execution.risk_manager import AdvancedRiskManager
from execution.order_manager import OrderManager, OrderSide
from analysis.microstructure import MicrostructureAnalyzer
from analysis.performance import PerformanceTracker

@dataclass
class TradingSignal:
    """Unified trading signal from any strategy"""
    strategy: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    timestamp: datetime
    signal_data: Dict  # Strategy-specific data

class CryptoMicrostructureTrader:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, paper_trading: bool = True):
        """Initialize the trading engine"""
        self.paper_trading = paper_trading
        self.is_running = False
        self.cycle_count = 0
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger('MicrostructureTrader')
        
        # Setup directories
        setup_directories()
        validate_config()
        
        # Initialize components
        self.logger.info("Initializing trading system components...")
        
        try:
            # Data validation
            self.data_validator = DataValidator()
            
            # Strategy engines
            self.stop_hunt_detector = StopHuntDetector()
            self.scalping_strategy = ScalpingStrategy()
            self.volume_profile_strategy = VolumeProfileStrategy()
            
            # Risk and execution
            self.risk_manager = AdvancedRiskManager()
            self.order_manager = OrderManager(paper_trading=paper_trading)
            
            # Analysis
            self.microstructure_analyzer = MicrostructureAnalyzer()
            self.performance_tracker = PerformanceTracker()
            
            # Trading state
            self.active_signals: List[TradingSignal] = []
            self.last_health_check = datetime.now(timezone.utc)
            self.last_performance_report = datetime.now(timezone.utc)
            
            self.logger.info("Trading system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading system: {e}")
            raise
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        try:
            # Create logs directory
            log_dir = Path(LOGGING_CONFIG['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure root logger with UTF-8 encoding
            logging.basicConfig(
                level=getattr(logging, LOGGING_CONFIG['level']),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(
                        log_dir / LOGGING_CONFIG['system_log_file'],
                        encoding='utf-8'
                    ),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            # Ensure console can handle Unicode
            if sys.stdout.encoding != 'UTF-8':
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            raise
    
    def run(self):
        """Main trading loop"""
        self.logger.info("Starting Crypto Microstructure Trading Engine")
        self.is_running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Initial health check
            if not self._initial_health_check():
                self.logger.error("Initial health check failed - aborting")
                return
            
            # Main trading loop
            while self.is_running:
                cycle_start = time.time()
                
                try:
                    # Execute trading cycle
                    self._execute_trading_cycle()
                    
                    # Periodic tasks
                    self._handle_periodic_tasks()
                    
                    # Cycle timing
                    cycle_duration = time.time() - cycle_start
                    self.logger.debug(f"Trading cycle {self.cycle_count} completed in {cycle_duration:.2f}s")
                    
                    # Sleep until next cycle (run every minute)
                    sleep_time = max(1, 60 - cycle_duration)
                    time.sleep(sleep_time)
                    
                    self.cycle_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}")
                    time.sleep(30)  # Wait before retrying
                    
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}")
        finally:
            self._shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum} - initiating graceful shutdown")
        self.is_running = False
    
    def _initial_health_check(self) -> bool:
        """Perform initial system health check"""
        try:
            self.logger.info("Performing initial health check...")
            
            # Check ONLY 1m and 5m data availability (microstructure trading focus)
            ready_pairs = []
            
            microstructure_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            microstructure_timeframes = ['1m', '5m']
            
            for symbol in microstructure_symbols:
                for timeframe in microstructure_timeframes:
                    report = self.data_validator.validate_symbol_timeframe(symbol, timeframe)
                    
                    if report.is_valid:
                        ready_pairs.append((symbol, timeframe))
                        self.logger.info(f"✓ {symbol} {timeframe} ready (score: {report.quality_score:.2f})")
                    else:
                        self.logger.warning(f"✗ {symbol} {timeframe} not ready: {report.issues}")
            
            if len(ready_pairs) < 4:  # Need at least 2 symbols x 2 timeframes
                self.logger.error(f"Insufficient microstructure data: {len(ready_pairs)} pairs ready, need 4")
                self.logger.error("Ready pairs: " + str(ready_pairs))
                return False
            
            # Check risk manager
            risk_summary = self.risk_manager.get_risk_summary()
            if not risk_summary['risk_status']['can_trade']:
                self.logger.error("Risk manager prevents trading")
                return False
            
            self.logger.info(f"Health check passed - {len(ready_pairs)} pairs ready for trading")
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            # 1. Update position prices and check stops
            self._update_positions()
            
            # 2. Check for exit signals
            self._process_exits()
            
            # 3. Scan for new trading opportunities
            if len(self.order_manager.positions) < MAX_CONCURRENT_TRADES:
                new_signals = self._scan_for_signals()
                
                # 4. Process new signals
                for signal in new_signals:
                    if len(self.order_manager.positions) >= MAX_CONCURRENT_TRADES:
                        break
                    self._process_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle execution: {e}")
    
    def _update_positions(self):
        """Update all position prices and unrealized PnL"""
        try:
            self.order_manager.update_position_prices()
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _process_exits(self):
        """Check and process position exits"""
        try:
            # Check stop losses
            stop_triggered = self.order_manager.check_stop_loss_triggers()
            for symbol in stop_triggered:
                self.logger.info(f"Stop loss triggered for {symbol}")
                self.order_manager.close_position(symbol, "stop_loss")
            
            # Check take profits
            profit_triggered = self.order_manager.check_take_profit_triggers()
            for symbol in profit_triggered:
                self.logger.info(f"Take profit triggered for {symbol}")
                self.order_manager.close_position(symbol, "take_profit")
            
            # Check time-based exits (for scalping strategies)
            self._check_time_exits()
            
        except Exception as e:
            self.logger.error(f"Error processing exits: {e}")
    
    def _check_time_exits(self):
        """Check for time-based position exits"""
        try:
            current_time = datetime.now(timezone.utc)
            
            for symbol, position in self.order_manager.positions.items():
                # Check if position has been open too long for its strategy
                time_open = (current_time - position.timestamp).total_seconds() / 60  # minutes
                
                max_hold_time = self._get_max_hold_time_for_strategy(position.strategy_id)
                
                if time_open > max_hold_time:
                    self.logger.info(f"Time exit triggered for {symbol} after {time_open:.1f} minutes")
                    self.order_manager.close_position(symbol, "time_limit")
                    
        except Exception as e:
            self.logger.error(f"Error checking time exits: {e}")
    
    def _get_max_hold_time_for_strategy(self, strategy_id: str) -> float:
        """Get maximum hold time for a strategy"""
        if 'stop_hunt' in strategy_id.lower():
            return 5.0  # 5 minutes
        elif 'scalp' in strategy_id.lower():
            return 3.0  # 3 minutes
        elif 'volume_profile' in strategy_id.lower():
            return 10.0  # 10 minutes
        else:
            return 15.0  # Default 15 minutes
    
    def _scan_for_signals(self) -> List[TradingSignal]:
        """Scan all symbols for trading signals"""
        signals = []
        
        try:
            for symbol in SYMBOLS:
                # Check data quality first
                if not self._is_symbol_ready(symbol):
                    continue
                
                # Get microstructure analysis
                market_structure = self.microstructure_analyzer.analyze_market_regime(symbol)
                
                # Scan each strategy
                try:
                    # Stop hunt detection
                    stop_hunt_signal = self.stop_hunt_detector.scan_symbol(symbol)
                    if stop_hunt_signal:
                        signals.append(self._convert_stop_hunt_signal(stop_hunt_signal, market_structure))
                    
                    # Scalping opportunities
                    scalp_signal = self.scalping_strategy.scan_symbol(symbol)
                    if scalp_signal:
                        signals.append(self._convert_scalp_signal(scalp_signal, market_structure))
                    
                    # Volume profile trades
                    volume_signal = self.volume_profile_strategy.scan_symbol(symbol)
                    if volume_signal:
                        signals.append(self._convert_volume_signal(volume_signal, market_structure))
                        
                except Exception as e:
                    self.logger.warning(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort signals by confidence and strategy priority
            signals.sort(key=lambda x: (x.confidence, self._get_strategy_priority(x.strategy)), reverse=True)
            
            return signals[:3]  # Return top 3 signals
            
        except Exception as e:
            self.logger.error(f"Error scanning for signals: {e}")
            return []
    
    def _is_symbol_ready(self, symbol: str) -> bool:
        """Check if symbol data is ready for microstructure trading"""
        try:
            # Only check 1m and 5m data - that's all we need for microstructure strategies
            for timeframe in ['1m', '5m']:
                report = self.data_validator.validate_symbol_timeframe(symbol, timeframe)
                if not report.is_valid:
                    # Only fail if it's a serious issue (stale data or no data)
                    serious_issues = [issue for issue in report.issues if 
                                    'Stale data:' in issue or 'No data' in issue or 'does not exist' in issue]
                    if serious_issues:
                        self.logger.debug(f"{symbol} {timeframe} not ready: {serious_issues[0]}")
                        return False
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking symbol readiness for {symbol}: {e}")
            return False
    
    def _convert_stop_hunt_signal(self, signal, market_structure) -> TradingSignal:
        """Convert stop hunt signal to unified format"""
        direction = "short" if "LONG_HUNT" in signal.hunt_type.value else "long"
        
        return TradingSignal(
            strategy="stop_hunt",
            symbol=signal.symbol,
            direction=direction,
            entry_price=signal.entry_price,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            signal_data={
                'hunt_type': signal.hunt_type.value,
                'volume_spike_ratio': signal.volume_spike_ratio,
                'price_spike_percentage': signal.price_spike_percentage,
                'market_regime': market_structure.regime.value
            }
        )
    
    def _convert_scalp_signal(self, signal, market_structure) -> TradingSignal:
        """Convert scalping signal to unified format"""
        direction = "long" if "LONG" in signal.scalp_type.value else "short"
        
        return TradingSignal(
            strategy="scalping",
            symbol=signal.symbol,
            direction=direction,
            entry_price=signal.entry_price,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            signal_data={
                'scalp_type': signal.scalp_type.value,
                'momentum_strength': signal.momentum_strength,
                'mean_reversion_score': signal.mean_reversion_score,
                'market_regime': market_structure.regime.value
            }
        )
    
    def _convert_volume_signal(self, signal, market_structure) -> TradingSignal:
        """Convert volume profile signal to unified format"""
        direction = "long" if signal.target_price > signal.entry_price else "short"
        
        return TradingSignal(
            strategy="volume_profile",
            symbol=signal.symbol,  # Use the actual symbol from the signal
            direction=direction,
            entry_price=signal.entry_price,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            signal_data={
                'signal_type': signal.signal_type.value,
                'level_strength': signal.volume_level.strength_score,
                'distance_to_level': signal.distance_to_level,
                'market_regime': market_structure.regime.value
            }
        )
    
    def _get_strategy_priority(self, strategy: str) -> float:
        """Get priority weight for strategy"""
        priorities = {
            'stop_hunt': 1.0,      # Highest priority
            'volume_profile': 0.8,  # High priority
            'scalping': 0.6        # Medium priority
        }
        return priorities.get(strategy, 0.5)
    
    def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            # Calculate position size
            position_calc = self.risk_manager.calculate_position_size(
                signal_type=f"{signal.strategy}_{signal.direction}",
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                confidence=signal.confidence,
                symbol=signal.symbol
            )
            
            if not position_calc.approved or position_calc.recommended_size <= 0:
                self.logger.debug(f"Signal rejected: {signal.symbol} {signal.strategy} - {position_calc.reason}")
                return
            
            # Execute trade
            order_side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
            
            order_id = self.order_manager.submit_market_order(
                symbol=signal.symbol,
                side=order_side,
                quantity=position_calc.recommended_size,
                strategy_id=f"{signal.strategy}_{signal.direction}"
            )
            
            if order_id:
                # Set stop loss and take profit
                self.order_manager.set_stop_loss(signal.symbol, signal.stop_loss)
                self.order_manager.set_take_profit(signal.symbol, signal.target_price)
                
                # Add to risk manager tracking
                self.risk_manager.add_position(
                    symbol=signal.symbol,
                    signal_type=f"{signal.strategy}_{signal.direction}",
                    size=position_calc.recommended_size,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    target_price=signal.target_price
                )
                
                self.logger.info(f"TRADE EXECUTED: {signal.symbol} {signal.strategy} {signal.direction} "
                               f"size={position_calc.recommended_size:.6f} @{signal.entry_price:.6f}")
            else:
                self.logger.error(f"TRADE FAILED: {signal.symbol} {signal.strategy}")
                
        except Exception as e:
            self.logger.error(f"Error processing signal {signal.symbol} {signal.strategy}: {e}")
    
    def _handle_periodic_tasks(self):
        """Handle periodic maintenance tasks"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Health check every 5 minutes
            if (current_time - self.last_health_check).total_seconds() > 300:
                self._periodic_health_check()
                self.last_health_check = current_time
            
            # Performance report every 4 hours
            if (current_time - self.last_performance_report).total_seconds() > 14400:
                self._generate_performance_report()
                self.last_performance_report = current_time
            
            # Save state periodically
            if self.cycle_count % 5 == 0:  # Every 5 cycles
                self._save_all_state()
                
        except Exception as e:
            self.logger.error(f"Error in periodic tasks: {e}")
    
    def _periodic_health_check(self):
        """Perform periodic health check"""
        try:
            # Data health
            health_report = self.data_validator.generate_health_report()
            if health_report['overall_status'] in ['critical', 'warning']:
                self.logger.warning(f"Data health: {health_report['overall_status']}")
            
            # Risk health
            risk_summary = self.risk_manager.get_risk_summary()
            risk_level = risk_summary['current_metrics']['risk_level']
            
            if risk_level in ['high', 'critical']:
                self.logger.warning(f"Risk level: {risk_level}")
            
            # Execution health
            execution_summary = self.order_manager.get_execution_summary()
            success_rate = execution_summary.get('success_rate', 0)
            
            if success_rate < 0.9:
                self.logger.warning(f"Low execution success rate: {success_rate:.1%}")
                
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
    
    def _generate_performance_report(self):
        """Generate and log performance report"""
        try:
            report = self.performance_tracker.generate_daily_report()
            
            if report:
                self.logger.info("PERFORMANCE REPORT:")
                
                daily_summary = report['daily_summary']['summary']
                self.logger.info(f"  Daily: {daily_summary['total_trades']} trades, "
                               f"{daily_summary['win_rate']:.1%} win rate, "
                               f"${daily_summary['total_pnl']:.2f} PnL")
                
                weekly_summary = report['weekly_context']['summary']
                self.logger.info(f"  Weekly: {weekly_summary['total_trades']} trades, "
                                f"{weekly_summary['win_rate']:.1%} win rate, "
                                f"${weekly_summary['total_pnl']:.2f} PnL")
                
                # Log highlights
                for highlight in report.get('highlights', []):
                    self.logger.info(f"  HIGHLIGHT: {highlight}")
                
                # Log recommendations
                for rec in report.get('recommendations', []):
                    self.logger.info(f"  RECOMMENDATION: {rec}")
                    
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
    
    def _save_all_state(self):
        """Save state for all components"""
        try:
            self.risk_manager.save_state()
            self.order_manager.save_state()
            self.performance_tracker.save_performance_data()
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def _shutdown(self):
        """Graceful shutdown procedure"""
        try:
            self.logger.info("Initiating graceful shutdown...")
            
            # Close any pending orders
            for order_id in list(self.order_manager.pending_orders.keys()):
                self.order_manager.cancel_order(order_id)
            
            # Save final state
            self._save_all_state()
            
            # Generate final report
            final_summary = self.performance_tracker.get_performance_summary()
            if final_summary['summary']['total_trades'] > 0:
                self.logger.info("FINAL PERFORMANCE SUMMARY:")
                self.logger.info(f"  Total Trades: {final_summary['summary']['total_trades']}")
                self.logger.info(f"  Win Rate: {final_summary['summary']['win_rate']:.1%}")
                self.logger.info(f"  Total PnL: ${final_summary['summary']['total_pnl']:.2f}")
            
            self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        # Create and run the trading engine
        trader = CryptoMicrostructureTrader(paper_trading=True)
        trader.run()
        
    except KeyboardInterrupt:
        print("\nTrading engine stopped by user")
    except Exception as e:
        print(f"\nCritical error: {e}")
        logging.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
