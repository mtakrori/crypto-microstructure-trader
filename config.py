#!/usr/bin/env python3
"""
Configuration settings for Crypto Microstructure Trading System
Adapted to work with existing enhanced_crypto_fetcher.py setup
"""

import os
from pathlib import Path

# === DATABASE CONFIGURATION ===
# Using your existing database structure
DATA_DIR = 'C:/Users/Sammour/Documents/Qpredict/crypto-futures-trader/data'
DATABASE_FILE = f'{DATA_DIR}/crypto_candles.sqlite'

# === TRADING SYMBOLS ===
# Matching your existing fetcher symbols
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

# === ACCOUNT CONFIGURATION ===
ACCOUNT_BALANCE = 10000  # USD
LEVERAGE = 10
MAX_CONCURRENT_TRADES = 3
MAX_DAILY_TRADES = 20

# === RISK MANAGEMENT ===
RISK_CONFIG = {
    'max_risk_per_trade': 0.02,        # 2% per trade
    'daily_loss_limit': 0.05,          # 5% daily loss limit
    'win_rate_threshold': 0.4,          # Reduce size below 40% win rate
    'confidence_multiplier': True,      # Adjust size by confidence
    'max_position_size': 0.3            # 30% of account max per position
}

# === STOP HUNT STRATEGY CONFIGURATION ===
STOP_HUNT_CONFIG = {
    'min_volume_spike': 1.5,            # 1.5x average volume required
    'min_spike_threshold': 0.002,       # 0.2% minimum price spike
    'target_profit': 0.003,             # 0.3% profit target
    'stop_loss_buffer': 0.001,          # 0.1% buffer beyond spike
    'max_hold_time': 5,                 # 5 minutes maximum hold
    'lookback_candles': 20,             # Look back 20 candles for reference
    'wick_rejection_min': 0.003,        # 0.3% minimum wick rejection
    'confidence_threshold': 0.6         # Minimum confidence to trade
}

# === SCALPING STRATEGY CONFIGURATION ===
SCALPING_CONFIG = {
    'min_move_threshold': 0.004,        # 0.4% move to trigger mean reversion
    'target_profit': 0.002,             # 0.2% profit target
    'stop_loss_ratio': 1.5,             # 1.5:1 risk/reward
    'max_hold_time': 3,                 # 3 minutes maximum hold
    'volume_confirmation': 1.3,         # 1.3x average volume
    'momentum_lookback': 3,             # 3 candles for momentum
    'mean_reversion_strength': 0.6      # Mean reversion signal strength
}

# === VOLUME PROFILE STRATEGY CONFIGURATION ===
VOLUME_PROFILE_CONFIG = {
    'lookback_hours': 4,                # 4 hours for volume profile calculation
    'min_level_strength': 0.6,         # 60% of maximum volume level
    'max_distance_to_level': 0.001,     # 0.1% maximum distance from level
    'num_bins': 50,                     # Number of price bins
    'target_profit': 0.004,             # 0.4% profit target
    'stop_buffer': 0.002,               # 0.2% stop loss buffer
    'max_hold_time': 10,                # 10 minutes maximum hold
    'volume_confirmation': True         # Require volume confirmation
}

# === DATA VALIDATION CONFIGURATION ===
DATA_VALIDATION_CONFIG = {
    'min_candles_required': 100,        # Minimum candles for analysis
    'max_data_age_minutes': 5,          # Maximum age of latest data
    'price_change_threshold': 0.1,      # 10% maximum single candle change
    'volume_spike_threshold': 10,       # 10x volume spike detection
    'missing_data_tolerance': 0.05      # 5% missing data tolerance
}

# === PERFORMANCE TRACKING ===
PERFORMANCE_CONFIG = {
    'track_all_signals': True,          # Track even non-traded signals
    'save_frequency_minutes': 15,       # Save performance every 15 minutes
    'report_frequency_hours': 4,        # Generate reports every 4 hours
    'metrics_retention_days': 30,       # Keep metrics for 30 days
    'comparison_periods': [1, 7, 30]    # Compare 1-day, 7-day, 30-day performance
}

# === LOGGING CONFIGURATION ===
LOGGING_CONFIG = {
    'level': 'INFO',
    'file_rotation_mb': 10,
    'backup_count': 5,
    'log_dir': f'{DATA_DIR}/logs/microstructure',
    'trade_log_file': 'trades.log',
    'performance_log_file': 'performance.log',
    'system_log_file': 'system.log'
}

# === API CONFIGURATION ===
API_CONFIG = {
    'rate_limit_delay': 0.1,            # 100ms between requests
    'max_retries': 3,                   # Maximum retry attempts
    'timeout_seconds': 10,              # Request timeout
    'health_check_interval': 300        # 5 minutes health check
}

# === EXECUTION CONFIGURATION ===
EXECUTION_CONFIG = {
    'paper_trading': True,              # Start with paper trading
    'slippage_assumption': 0.0005,      # 0.05% slippage assumption
    'commission_rate': 0.0004,          # 0.04% commission
    'min_order_size': 10,               # $10 minimum order
    'order_timeout_seconds': 30,        # Order timeout
    'partial_fill_tolerance': 0.9       # Accept 90%+ fills
}

# === ALERT CONFIGURATION ===
ALERT_CONFIG = {
    'enable_alerts': True,
    'win_rate_alert_threshold': 0.3,    # Alert if win rate drops below 30%
    'daily_loss_alert_threshold': 0.03, # Alert at 3% daily loss
    'system_error_alerts': True,
    'performance_report_frequency': 4    # Hours between performance reports
}

# === FILE PATHS ===
PATHS = {
    'data_dir': DATA_DIR,
    'database': DATABASE_FILE,
    'logs_dir': LOGGING_CONFIG['log_dir'],
    'metrics_file': f'{DATA_DIR}/microstructure_metrics.json',
    'state_file': f'{DATA_DIR}/microstructure_state.json',
    'performance_file': f'{DATA_DIR}/performance_history.json',
    'trades_file': f'{DATA_DIR}/trades_history.json'
}

# === STRATEGY WEIGHTS ===
# Used for signal prioritization when multiple strategies trigger
STRATEGY_WEIGHTS = {
    'stop_hunt': 1.0,           # Highest priority
    'volume_profile': 0.8,      # High priority
    'scalping': 0.6             # Medium priority
}

# === TABLE NAMING CONVENTION ===
# Matching your existing fetcher table structure
def get_table_name(symbol: str, timeframe: str) -> str:
    """Generate table name matching existing fetcher convention"""
    return f"{symbol.lower()}_{timeframe}"

# === DEVELOPMENT FLAGS ===
DEV_CONFIG = {
    'debug_mode': False,
    'verbose_logging': False,
    'save_debug_data': False,
    'mock_execution': True,             # Mock order execution for testing
    'strategy_testing': False           # Enable individual strategy testing
}

# === ENVIRONMENT SETUP ===
def setup_directories():
    """Create necessary directories"""
    for path_key, path_value in PATHS.items():
        if 'dir' in path_key:
            Path(path_value).mkdir(parents=True, exist_ok=True)

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Validate database file exists
    if not Path(DATABASE_FILE).exists():
        errors.append(f"Database file not found: {DATABASE_FILE}")
    
    # Validate risk parameters
    if RISK_CONFIG['max_risk_per_trade'] > 0.1:
        errors.append("Risk per trade too high (>10%)")
    
    # Validate strategy parameters
    if STOP_HUNT_CONFIG['target_profit'] <= 0:
        errors.append("Stop hunt target profit must be positive")
    
    if len(errors) > 0:
        raise ValueError(f"Configuration errors: {errors}")
    
    return True

# === EXPORT ALL CONFIGS ===
__all__ = [
    'SYMBOLS', 'TIMEFRAMES', 'DATABASE_FILE', 'DATA_DIR',
    'ACCOUNT_BALANCE', 'LEVERAGE', 'MAX_CONCURRENT_TRADES',
    'RISK_CONFIG', 'STOP_HUNT_CONFIG', 'SCALPING_CONFIG', 
    'VOLUME_PROFILE_CONFIG', 'DATA_VALIDATION_CONFIG',
    'PERFORMANCE_CONFIG', 'LOGGING_CONFIG', 'API_CONFIG',
    'EXECUTION_CONFIG', 'ALERT_CONFIG', 'PATHS',
    'STRATEGY_WEIGHTS', 'DEV_CONFIG',
    'get_table_name', 'setup_directories', 'validate_config'
]