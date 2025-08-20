#!/usr/bin/env python3
"""
Enhanced Cryptocurrency Data Fetcher - Fixed Single Database Version
- Single SQLite file for all timeframes
- Only fetches 1m data from Binance API
- Real-time aggregation to all other timeframes
- Proper time alignment for incomplete candles
"""
import sqlite3
import time
import logging
import json
from datetime import datetime, timedelta, timezone
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import threading
import queue
import psutil

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    from dotenv import load_dotenv
    import os
    load_dotenv()
except ImportError as e:
    print(f"Error: Required package not installed - {str(e)}. Run: pip install python-binance psutil python-dotenv")
    exit(1)

# Enhanced Configuration - SINGLE DATABASE
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT']
DATA_DIR = 'C:/Users/Sammour/Documents/Qpredict/crypto-futures-trader/data'
DATABASE_FILE = f'{DATA_DIR}/crypto_candles.sqlite'  # Single database file
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

# Aggregation configuration
AGGREGATION_FACTORS = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

# Fetch Configuration
INITIAL_DAYS_TO_FETCH = 7
INCREMENTAL_CANDLES = 2
MIN_CANDLES_FOR_INCREMENTAL = 100

MAX_RETRIES = 5
RETRY_DELAY = 3
RATE_LIMIT_DELAY = 0.15
HEALTH_CHECK_INTERVAL = 300
DATA_VALIDATION_INTERVAL = 3600
METRICS_FILE = f'{DATA_DIR}/fetcher_metrics.json'
STATE_FILE = f'{DATA_DIR}/fetcher_state.json'

@dataclass
class FetcherMetrics:
    """Track fetcher performance metrics"""
    start_time: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_candles_processed: int = 0
    last_successful_fetch: str = ""
    uptime_seconds: float = 0
    memory_usage_mb: float = 0
    cpu_usage_percent: float = 0
    data_gaps_detected: int = 0
    data_gaps_filled: int = 0
    initial_fetch_completed: bool = False
    incremental_fetches: int = 0

class EnhancedCryptoDataFetcher:
    """Enhanced cryptocurrency data fetcher with single database"""
    
    def __init__(self):
        """Initialize the enhanced fetcher with API keys"""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("Binance API keys not found in .env file")
            
        self.client = Client(api_key, api_secret)
        self.metrics = FetcherMetrics(start_time=datetime.now(timezone.utc).isoformat())
        self.setup_enhanced_logging()
        self.data_queue = queue.Queue()
        self.is_running = False
        self.health_status = "starting"
        self.state = self.load_state()
        
    def setup_enhanced_logging(self):
        """Setup comprehensive logging with rotation"""
        from logging.handlers import RotatingFileHandler
        
        log_dir = Path(DATA_DIR) / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('EnhancedFetcher')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Rotating file handler
        file_handler = RotatingFileHandler(
            log_dir / 'crypto_fetcher.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Enhanced Crypto Data Fetcher initialized with single database")

    def load_state(self) -> Dict[str, Any]:
        """Load fetcher state from file"""
        try:
            if Path(STATE_FILE).exists():
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.logger.info(f"Loaded fetcher state: {state}")
                    return state
        except Exception as e:
            self.logger.warning(f"Failed to load state: {e}")
        
        return {
            'initial_fetch_completed': False,
            'last_fetch_mode': 'initial',
            'symbols_initialized': {},
        }

    def save_state(self):
        """Save fetcher state to file"""
        try:
            Path(DATA_DIR).mkdir(exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def get_table_name(self, symbol: str, timeframe: str) -> str:
        """Generate table name for symbol/timeframe combination"""
        return f"{symbol.lower()}_{timeframe}"

    def get_timeframe_start_time(self, timestamp_ms: int, timeframe: str) -> int:
        """Get the start time for a timeframe period (proper alignment)"""
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        
        if timeframe == '1m':
            aligned = dt.replace(second=0, microsecond=0)
        elif timeframe == '5m':
            aligned = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
        elif timeframe == '15m':
            aligned = dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
        elif timeframe == '1h':
            aligned = dt.replace(minute=0, second=0, microsecond=0)
        elif timeframe == '4h':
            aligned = dt.replace(hour=(dt.hour // 4) * 4, minute=0, second=0, microsecond=0)
        elif timeframe == '1d':
            aligned = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp_ms
            
        return int(aligned.timestamp() * 1000)

    def check_initial_data_exists(self, symbol: str, timeframe: str) -> bool:
        """Check if sufficient initial data exists for a symbol/timeframe"""
        try:
            table_name = self.get_table_name(symbol, timeframe)
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                ''', (table_name,))
                
                if not cursor.fetchone():
                    return False
                
                # Count total candles
                cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
                total_candles = cursor.fetchone()[0]
                
                if total_candles < MIN_CANDLES_FOR_INCREMENTAL:
                    return False
                
                # Check recent data (within last 2 hours)
                current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                two_hours_ago_ms = current_time_ms - (2 * 60 * 60 * 1000)
                
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE open_time > ?
                ''', (two_hours_ago_ms,))
                
                recent_candles = cursor.fetchone()[0]
                
                self.logger.debug(f"Data check for {symbol} {timeframe}: "
                                f"{total_candles} total, {recent_candles} recent")
                
                return total_candles >= MIN_CANDLES_FOR_INCREMENTAL and recent_candles > 0
                
        except sqlite3.Error as e:
            self.logger.error(f"Error checking initial data for {symbol}: {e}")
            return False

    def determine_fetch_mode(self, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """Determine whether to do initial or incremental fetch (only for 1m data)"""
        # Only check 1m data since that's what we fetch from API
        if self.check_initial_data_exists(symbol, '1m'):
            return 'incremental', {'limit': INCREMENTAL_CANDLES, 'days': None}
        else:
            return 'initial', {'limit': 1500, 'days': INITIAL_DAYS_TO_FETCH}
    
    def update_system_metrics(self):
        """Update system performance metrics"""
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
            self.metrics.uptime_seconds = time.time() - datetime.fromisoformat(self.metrics.start_time).timestamp()
        except Exception as e:
            self.logger.warning(f"Failed to update system metrics: {e}")
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            self.update_system_metrics()
            with open(METRICS_FILE, 'w') as f:
                json.dump(self.metrics.__dict__, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def init_database(self) -> None:
        """Initialize single database with all symbol/timeframe tables"""
        try:
            full_path = Path(DATABASE_FILE).absolute()
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(str(full_path)) as conn:
                # Enable optimizations
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=10000')
                conn.execute('PRAGMA temp_store=memory')
                
                cursor = conn.cursor()
                
                # Create tables for all symbol/timeframe combinations
                for symbol in SYMBOLS:
                    for timeframe in TIMEFRAMES:
                        table_name = self.get_table_name(symbol, timeframe)
                        
                        cursor.execute(f'''
                            CREATE TABLE IF NOT EXISTS {table_name} (
                                open_time INTEGER PRIMARY KEY,
                                open REAL NOT NULL CHECK(open > 0),
                                high REAL NOT NULL CHECK(high > 0),
                                low REAL NOT NULL CHECK(low > 0),
                                close REAL NOT NULL CHECK(close > 0),
                                volume REAL NOT NULL CHECK(volume >= 0),
                                close_time INTEGER NOT NULL,
                                quote_volume REAL NOT NULL CHECK(quote_volume >= 0),
                                count INTEGER NOT NULL CHECK(count >= 0),
                                taker_buy_volume REAL NOT NULL CHECK(taker_buy_volume >= 0),
                                taker_buy_quote_volume REAL NOT NULL CHECK(taker_buy_quote_volume >= 0),
                                is_complete INTEGER NOT NULL DEFAULT 1 CHECK(is_complete IN (0, 1)),
                                last_updated INTEGER DEFAULT (strftime('%s', 'now')),
                                symbol TEXT NOT NULL DEFAULT '{symbol}',
                                timeframe TEXT NOT NULL DEFAULT '{timeframe}',
                                CHECK(high >= open AND high >= close AND high >= low),
                                CHECK(low <= open AND low <= close AND low <= high)
                            )
                        ''')
                        
                        # Create optimized indexes
                        cursor.execute(f'''
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_time_complete 
                            ON {table_name}(open_time, is_complete)
                        ''')
                        
                        cursor.execute(f'''
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_timeframe 
                            ON {table_name}(symbol, timeframe, open_time)
                        ''')
                
                conn.commit()
                self.logger.info(f"Database initialized: {DATABASE_FILE}")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
    
    def enhanced_validate_candle(self, candle: List) -> Tuple[bool, List[str]]:
        """Enhanced candle validation with detailed error reporting"""
        errors = []
        
        try:
            if len(candle) < 11:
                errors.append("Insufficient data fields")
                return False, errors
            
            # Extract and validate numeric values
            try:
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
                volume = float(candle[5])
                taker_buy_volume = float(candle[9])
                taker_buy_quote_volume = float(candle[10])
            except (ValueError, TypeError):
                errors.append("Invalid numeric values")
                return False, errors
            
            # Price validation
            if not all(price > 0 for price in [open_price, high_price, low_price, close_price]):
                errors.append("Non-positive prices")
            
            if high_price < max(open_price, close_price):
                errors.append("High price too low")
            
            if low_price > min(open_price, close_price):
                errors.append("Low price too high")
            
            # Volume validation
            if volume < 0:
                errors.append("Negative volume")
            
            if taker_buy_volume > volume:
                errors.append("Taker buy volume exceeds total volume")
            
            # Timestamp validation
            try:
                open_time = int(candle[0])
                close_time = int(candle[6])
                if close_time <= open_time:
                    errors.append("Invalid timestamp sequence")
            except (ValueError, TypeError):
                errors.append("Invalid timestamps")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def upsert_candle_enhanced(self, conn: sqlite3.Connection, table_name: str, candle: List, symbol: str, timeframe: str) -> bool:
        """Enhanced candle insertion with validation and error handling"""
        is_valid, validation_errors = self.enhanced_validate_candle(candle)
        
        if not is_valid:
            self.logger.warning(f"Invalid candle for {table_name}: {', '.join(validation_errors)}")
            return False
        
        try:
            cursor = conn.cursor()
            
            # Extract and prepare data
            open_time = int(candle[0])
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])
            close_time = int(candle[6])
            quote_volume = float(candle[7])
            count = int(candle[8])
            taker_buy_volume = float(candle[9])
            taker_buy_quote_volume = float(candle[10])
            
            # Determine if candle is complete
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            is_complete = 1 if close_time < current_time_ms else 0
            
            # Enhanced upsert logic
            if is_complete == 0:
                # For incomplete candles, merge data intelligently
                cursor.execute(f'''
                    INSERT INTO {table_name} (
                        open_time, open, high, low, close, volume, close_time,
                        quote_volume, count, taker_buy_volume, taker_buy_quote_volume, 
                        is_complete, symbol, timeframe
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(open_time) DO UPDATE SET
                        high = MAX(excluded.high, high),
                        low = MIN(excluded.low, low),
                        close = excluded.close,
                        volume = excluded.volume,
                        quote_volume = excluded.quote_volume,
                        count = excluded.count,
                        taker_buy_volume = excluded.taker_buy_volume,
                        taker_buy_quote_volume = excluded.taker_buy_quote_volume,
                        last_updated = strftime('%s', 'now')
                ''', (open_time, open_price, high_price, low_price, close_price, volume,
                      close_time, quote_volume, count, taker_buy_volume, 
                      taker_buy_quote_volume, is_complete, symbol, timeframe))
            else:
                # Complete candles - replace entirely
                cursor.execute(f'''
                    INSERT OR REPLACE INTO {table_name} (
                        open_time, open, high, low, close, volume, close_time,
                        quote_volume, count, taker_buy_volume, taker_buy_quote_volume, 
                        is_complete, symbol, timeframe
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (open_time, open_price, high_price, low_price, close_price, volume,
                      close_time, quote_volume, count, taker_buy_volume, 
                      taker_buy_quote_volume, is_complete, symbol, timeframe))
            
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error upserting {table_name} candle: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error upserting {table_name} candle: {e}")
            return False
    
    def aggregate_candles_realtime(self, base_candles: List[List], target_timeframe: str) -> List[List]:
        """Real-time aggregation with proper time alignment"""
        if target_timeframe == '1m' or not base_candles:
            return base_candles
            
        # Group candles by their timeframe period
        periods = {}
        
        for candle in base_candles:
            timestamp_ms = int(candle[0])
            period_start = self.get_timeframe_start_time(timestamp_ms, target_timeframe)
            
            if period_start not in periods:
                periods[period_start] = []
            periods[period_start].append(candle)
        
        # Create aggregated candles for ALL periods (complete and incomplete)
        aggregated = []
        factor_ms = AGGREGATION_FACTORS[target_timeframe] * 60 * 1000
        
        for period_start_ms, period_candles in sorted(periods.items()):
            if not period_candles:
                continue
                
            # Sort candles by time within period
            period_candles.sort(key=lambda x: x[0])
            
            # Calculate aggregated values
            open_price = float(period_candles[0][1])
            high_price = max(float(c[2]) for c in period_candles)
            low_price = min(float(c[3]) for c in period_candles)
            close_price = float(period_candles[-1][4])
            volume = sum(float(c[5]) for c in period_candles)
            
            # Calculate close time for the period
            close_time = period_start_ms + factor_ms - 1
            
            quote_volume = sum(float(c[7]) for c in period_candles)
            count = sum(int(c[8]) for c in period_candles)
            taker_buy_volume = sum(float(c[9]) for c in period_candles)
            taker_buy_quote_volume = sum(float(c[10]) for c in period_candles)
            
            aggregated_candle = [
                period_start_ms,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                close_time,
                quote_volume,
                count,
                taker_buy_volume,
                taker_buy_quote_volume
            ]
            
            aggregated.append(aggregated_candle)
        
        return aggregated

    def fetch_candles_enhanced(self, symbol: str, fetch_mode: str, fetch_params: Dict[str, Any]) -> List[List]:
        """Enhanced candle fetching for 1m data only"""
        self.metrics.total_requests += 1
        interval = '1m'  # Always fetch 1m data
        
        for attempt in range(MAX_RETRIES):
            try:
                if fetch_mode == 'initial':
                    # Paginated initial fetch
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=fetch_params['days'])
                    current_end = end_time
                    klines = []
                    
                    self.logger.info(f"Paginated initial fetch for {symbol} {interval}: "
                                   f"{fetch_params['days']} days of history")
                    
                    # Pagination loop
                    while True:
                        current_start = current_end - timedelta(minutes=fetch_params['limit'])
                        if current_start < start_time:
                            current_start = start_time
                            
                        batch = self.client.futures_klines(
                            symbol=symbol,
                            interval=interval,
                            startTime=int(current_start.timestamp() * 1000),
                            endTime=int(current_end.timestamp() * 1000),
                            limit=fetch_params['limit']
                        )
                        
                        if not batch:
                            if current_start == start_time:
                                break
                            current_end = current_start
                            continue
                            
                        if len(batch) < fetch_params['limit']:
                            current_end = current_start
                        else:
                            current_end = datetime.fromtimestamp(batch[0][0]/1000, tz=timezone.utc) - timedelta(minutes=1)
                        
                        if current_end < start_time:
                            current_end = start_time
                        
                        if not batch:
                            break
                            
                        klines.extend(batch)
                        current_end = current_start
                        
                        if current_start <= start_time:
                            break
                            
                        time.sleep(RATE_LIMIT_DELAY)
                    
                    # Sort candles by time (oldest first)
                    klines.sort(key=lambda x: x[0])
                    
                else:  # incremental mode
                    # Incremental fetch: get latest candles only
                    self.logger.debug(f"Incremental fetch for {symbol} {interval}: "
                                    f"last {fetch_params['limit']} candles")
                    
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=fetch_params['limit']
                    )
                
                self.metrics.successful_requests += 1
                self.metrics.last_successful_fetch = datetime.now(timezone.utc).isoformat()
                
                if fetch_mode == 'incremental':
                    self.metrics.incremental_fetches += 1
                
                self.logger.debug(f"Successfully fetched {len(klines)} candles for {symbol} {interval} ({fetch_mode})")
                time.sleep(RATE_LIMIT_DELAY)
                
                return klines
                
            except BinanceAPIException as e:
                self.logger.warning(f"Binance API error for {symbol} {interval} (attempt {attempt + 1}): {e}")
                if e.code == -1121:  # Invalid symbol
                    self.metrics.failed_requests += 1
                    break
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    self.metrics.failed_requests += 1
                    self.logger.error(f"Failed to fetch {symbol} {interval} after {MAX_RETRIES} attempts")
                    
            except Exception as e:
                self.logger.error(f"Unexpected error fetching {symbol} {interval}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    self.metrics.failed_requests += 1
        
        return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_report = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'issues': [],
            'metrics': self.metrics.__dict__
        }
        
        try:
            # Check database connectivity
            try:
                with sqlite3.connect(DATABASE_FILE) as conn:
                    conn.execute('SELECT 1').fetchone()
            except Exception as e:
                health_report['issues'].append(f"Database connectivity issue: {e}")
                health_report['status'] = 'degraded'
            
            # Check API connectivity
            try:
                self.client.ping()
            except Exception as e:
                health_report['issues'].append(f"API connectivity issue: {e}")
                health_report['status'] = 'degraded'
            
            # Check memory usage
            if self.metrics.memory_usage_mb > 500:
                health_report['issues'].append(f"High memory usage: {self.metrics.memory_usage_mb:.1f}MB")
                health_report['status'] = 'warning'
            
            # Check success rate
            if self.metrics.total_requests > 0:
                success_rate = self.metrics.successful_requests / self.metrics.total_requests
                if success_rate < 0.95:
                    health_report['issues'].append(f"Low success rate: {success_rate:.1%}")
                    health_report['status'] = 'warning'
            
        except Exception as e:
            health_report['issues'].append(f"Health check error: {e}")
            health_report['status'] = 'error'
        
        self.health_status = health_report['status']
        return health_report
    
    def run_enhanced(self) -> None:
        """Enhanced main execution loop with single database and real-time aggregation"""
        self.is_running = True
        self.health_status = "running"
        last_health_check = 0
        last_validation_check = 0
        
        self.logger.info("Starting enhanced cryptocurrency data fetcher with single database")
        
        try:
            while self.is_running:
                cycle_start = time.time()
                
                # Initialize single database
                self.init_database()
                
                total_processed = 0
                
                for symbol in SYMBOLS:
                    # Only fetch 1m data from API
                    fetch_mode, fetch_params = self.determine_fetch_mode(symbol)
                    base_klines = self.fetch_candles_enhanced(symbol, fetch_mode, fetch_params)
                    
                    if not base_klines:
                        continue
                    
                    # Process all timeframes in single transaction
                    with sqlite3.connect(DATABASE_FILE) as conn:
                        for timeframe in TIMEFRAMES:
                            table_name = self.get_table_name(symbol, timeframe)
                            
                            if timeframe == '1m':
                                # Store 1m data directly
                                klines = base_klines
                            else:
                                # Aggregate to higher timeframe
                                klines = self.aggregate_candles_realtime(base_klines, timeframe)
                            
                            if not klines:
                                continue
                            
                            processed_count = 0
                            for kline in klines:
                                if self.upsert_candle_enhanced(conn, table_name, kline, symbol, timeframe):
                                    processed_count += 1
                            
                            total_processed += processed_count
                            self.metrics.total_candles_processed += processed_count
                            
                            self.logger.debug(f"Processed {processed_count} candles for {symbol} {timeframe}")
                        
                        conn.commit()
                
                # Periodic health checks
                current_time = time.time()
                if current_time - last_health_check > HEALTH_CHECK_INTERVAL:
                    health_report = self.health_check()
                    self.logger.info(f"Health check: {health_report['status']} - {len(health_report['issues'])} issues")
                    if health_report['issues']:
                        for issue in health_report['issues']:
                            self.logger.warning(f"Health issue: {issue}")
                    last_health_check = current_time
                
                # Save metrics and state
                self.save_metrics()
                self.save_state()
                
                # Cycle summary
                cycle_time = time.time() - cycle_start
                self.logger.info(f"Cycle completed: {total_processed} candles processed in {cycle_time:.2f}s (5s interval)")
                
                # Sleep until next 5-second cycle (:00, :05, :10, etc.)
                now = datetime.now(timezone.utc)
                current_second = now.second
                next_5_boundary = ((current_second // 5) + 1) * 5
                if next_5_boundary >= 60:
                    next_5_boundary = 0
                    target_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                else:
                    target_time = now.replace(second=next_5_boundary, microsecond=0)

                sleep_time = max(1, (target_time - now).total_seconds())
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Graceful shutdown initiated...")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}")
            self.health_status = "error"
            raise
        finally:
            self.save_metrics()
            self.save_state()
            self.logger.info("Enhanced fetcher stopped")

def query_data(symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
    """Helper function to query data from single database"""
    table_name = f"{symbol.lower()}_{timeframe}"
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            
            cursor.execute(f'''
                SELECT open_time, open, high, low, close, volume, is_complete, symbol, timeframe
                FROM {table_name}
                WHERE is_complete = 1
                ORDER BY open_time DESC
                LIMIT ?
            ''', (limit,))
            
            columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'is_complete', 'symbol', 'timeframe']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
    except sqlite3.Error as e:
        logging.error(f"Error querying {symbol} {timeframe}: {e}")
        return []

def main():
    """Enhanced main entry point"""
    try:
        # Start enhanced fetcher
        fetcher = EnhancedCryptoDataFetcher()
        fetcher.run_enhanced()
        
    except KeyboardInterrupt:
        print("\n🛑 Fetcher stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        raise

if __name__ == "__main__":
    main()
