#!/usr/bin/env python3
"""
Historical Cryptocurrency Data Puller - FIXED VERSION
- Uses API keys for better rate limits
- Pulls only 1m data (other timeframes aggregated by fetcher)
- Single database file structure
- Increased limit to 5000 candles per request
- Bulk downloads historical 1m data from 2023 to present
"""
import sqlite3
import time
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math
from dataclasses import dataclass, asdict

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    print(f"Error: Required package not installed - {str(e)}. Run: pip install python-binance python-dotenv")
    exit(1)

# Configuration - FIXED
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT']
DATA_DIR = 'C:/Users/Sammour/Documents/Qpredict/crypto-futures-trader/data'
DATABASE_FILE = f'{DATA_DIR}/crypto_candles.sqlite'  # Single database file
TIMEFRAMES = ['1m']  # Only pull 1m data - fetcher will aggregate others
START_DATE = '2023-01-01'  # Start from 2023
MAX_KLINES_PER_REQUEST = 1500  # Testing higher limit with API keys (standard is 1000-1500)
RATE_LIMIT_DELAY = 0.1  # Faster with API keys
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

# Progress tracking file
PROGRESS_FILE = f'{DATA_DIR}/historical_download_progress.json'

@dataclass
class DownloadProgress:
    """Track download progress for resumability"""
    symbol: str
    timeframe: str
    last_completed_timestamp: int
    total_candles_downloaded: int
    start_time: str
    last_update: str
    status: str = 'in_progress'  # 'in_progress', 'completed', 'error'

class HistoricalDataPuller:
    """Bulk historical data downloader with API keys and single database"""
    
    def __init__(self):
        """Initialize the historical data puller with API keys"""
        # Setup logging first
        self.progress = {}
        self.setup_logging()
        self.ensure_data_dir()
        
        # Load API keys from environment
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            self.logger.warning("API keys not found, using public client (slower rates)")
            self.client = Client()  # Public data only
        else:
            self.logger.info("Using authenticated client with API keys")
            self.client = Client(api_key, api_secret)
        
        self.load_progress()
        
    def setup_logging(self):
        """Setup comprehensive logging with Unicode support"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # File handler for detailed logs (supports Unicode)
        file_handler = logging.FileHandler(f'{DATA_DIR}/historical_puller.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler for progress updates (no emojis to avoid encoding issues)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
        # Configure logger
        self.logger = logging.getLogger('HistoricalPuller')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Historical Data Puller initialized with single database")
        
    def ensure_data_dir(self):
        """Ensure data directory exists"""
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        
    def load_progress(self):
        """Load download progress from file"""
        try:
            if Path(PROGRESS_FILE).exists():
                with open(PROGRESS_FILE, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self.progress[key] = DownloadProgress(**value)
                self.logger.info(f"Loaded progress for {len(self.progress)} tasks")
            else:
                self.logger.info("No previous progress found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading progress: {e}")
            self.progress = {}
    
    def save_progress(self):
        """Save download progress to file"""
        try:
            progress_data = {key: asdict(value) for key, value in self.progress.items()}
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")
    
    def get_progress_key(self, symbol: str, timeframe: str) -> str:
        """Generate unique key for progress tracking"""
        return f"{symbol}_{timeframe}"
    
    def get_table_name(self, symbol: str, timeframe: str) -> str:
        """Generate table name for symbol/timeframe combination"""
        return f"{symbol.lower()}_{timeframe}"
    
    def init_database(self) -> str:
        """Initialize single database for all data"""
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                # Enable optimizations
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=10000')
                conn.execute('PRAGMA temp_store=memory')
                
                cursor = conn.cursor()
                
                # Create tables for each symbol/timeframe combination
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
                                is_complete INTEGER NOT NULL DEFAULT 1,
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
            
        return DATABASE_FILE
    
    def get_existing_data_range(self, symbol: str, timeframe: str) -> Tuple[Optional[int], Optional[int]]:
        """Get the range of existing data for a symbol/timeframe"""
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
                    return None, None
                
                cursor.execute(f'''
                    SELECT MIN(open_time), MAX(open_time) 
                    FROM {table_name} 
                    WHERE is_complete = 1
                ''')
                result = cursor.fetchone()
                return result if result[0] is not None else (None, None)
        except sqlite3.Error as e:
            self.logger.error(f"Error getting existing data range for {symbol} {timeframe}: {e}")
            return None, None
    
    def calculate_missing_ranges(self, symbol: str, timeframe: str, 
                               start_timestamp: int, end_timestamp: int) -> List[Tuple[int, int]]:
        """Calculate missing data ranges to avoid re-downloading existing data"""
        existing_min, existing_max = self.get_existing_data_range(symbol, timeframe)
        
        if existing_min is None:
            # No existing data, download everything
            return [(start_timestamp, end_timestamp)]
        
        ranges = []
        
        # Check if we need data before existing range
        if start_timestamp < existing_min:
            ranges.append((start_timestamp, existing_min - 1))
        
        # Check if we need data after existing range
        if end_timestamp > existing_max:
            ranges.append((existing_max + 1, end_timestamp))
        
        return ranges
    
    def fetch_historical_klines(self, symbol: str, interval: str, 
                              start_time: int, end_time: int) -> List[List]:
        """Fetch historical klines with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.debug(f"Fetching {symbol} {interval} from {start_time} to {end_time}")
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_time,
                    endTime=end_time,
                    limit=MAX_KLINES_PER_REQUEST
                )
                
                time.sleep(RATE_LIMIT_DELAY)
                return klines
                
            except BinanceAPIException as e:
                if e.code == -1121:  # Invalid symbol
                    self.logger.error(f"Invalid symbol {symbol}")
                    return []
                elif attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"API error for {symbol} (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to fetch {symbol} after {MAX_RETRIES} attempts: {e}")
                    raise
                    
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(f"Error fetching {symbol} (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to fetch {symbol} after {MAX_RETRIES} attempts: {e}")
                    raise
        
        return []
    
    def bulk_insert_klines(self, symbol: str, timeframe: str, klines: List[List]) -> int:
        """Bulk insert klines into database"""
        if not klines:
            return 0
            
        table_name = self.get_table_name(symbol, timeframe)
        inserted_count = 0
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                
                # Prepare data for bulk insert
                records = []
                for kline in klines:
                    try:
                        record = (
                            int(kline[0]),      # open_time
                            float(kline[1]),    # open
                            float(kline[2]),    # high
                            float(kline[3]),    # low
                            float(kline[4]),    # close
                            float(kline[5]),    # volume
                            int(kline[6]),      # close_time
                            float(kline[7]),    # quote_volume
                            int(kline[8]),      # count
                            float(kline[9]),    # taker_buy_volume
                            float(kline[10]),   # taker_buy_quote_volume
                            1,                  # is_complete (historical data is always complete)
                            symbol,             # symbol
                            timeframe           # timeframe
                        )
                        records.append(record)
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Invalid kline data for {symbol}: {e}")
                        continue
                
                # Bulk insert with conflict resolution
                cursor.executemany(f'''
                    INSERT OR REPLACE INTO {table_name} (
                        open_time, open, high, low, close, volume, close_time,
                        quote_volume, count, taker_buy_volume, taker_buy_quote_volume, 
                        is_complete, symbol, timeframe
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', records)
                
                inserted_count = cursor.rowcount
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error inserting {symbol} data: {e}")
            raise
            
        return inserted_count
    
    def download_symbol_timeframe(self, symbol: str, timeframe: str) -> bool:
        """Download all historical data for a symbol/timeframe combination"""
        progress_key = self.get_progress_key(symbol, timeframe)
        
        # Initialize progress if not exists
        if progress_key not in self.progress:
            start_dt = datetime.strptime(START_DATE, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            self.progress[progress_key] = DownloadProgress(
                symbol=symbol,
                timeframe=timeframe,
                last_completed_timestamp=int(start_dt.timestamp() * 1000),
                total_candles_downloaded=0,
                start_time=datetime.now(timezone.utc).isoformat(),
                last_update=datetime.now(timezone.utc).isoformat()
            )
        
        progress = self.progress[progress_key]
        
        if progress.status == 'completed':
            self.logger.info(f"{symbol} {timeframe} already completed")
            return True
        
        # Calculate time ranges
        start_timestamp = progress.last_completed_timestamp
        end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # Initialize database
        self.init_database()
        
        # Calculate missing ranges
        missing_ranges = self.calculate_missing_ranges(symbol, timeframe, start_timestamp, end_timestamp)
        
        if not missing_ranges:
            self.logger.info(f"{symbol} {timeframe} - no missing data")
            progress.status = 'completed'
            self.save_progress()
            return True
        
        self.logger.info(f"Downloading {symbol} {timeframe} - {len(missing_ranges)} range(s) to fill")
        
        # Convert timeframe to milliseconds for chunk calculation
        timeframe_ms = self.timeframe_to_ms(timeframe)
        chunk_size_ms = MAX_KLINES_PER_REQUEST * timeframe_ms
        
        total_downloaded = 0
        
        try:
            for range_start, range_end in missing_ranges:
                current_start = range_start
                
                while current_start < range_end:
                    # Calculate chunk end
                    current_end = min(current_start + chunk_size_ms - timeframe_ms, range_end)
                    
                    # Fetch data chunk
                    klines = self.fetch_historical_klines(symbol, timeframe, current_start, current_end)
                    
                    if klines:
                        # Insert into database
                        inserted = self.bulk_insert_klines(symbol, timeframe, klines)
                        total_downloaded += inserted
                        
                        # Update progress
                        progress.last_completed_timestamp = current_end
                        progress.total_candles_downloaded += inserted
                        progress.last_update = datetime.now(timezone.utc).isoformat()
                        
                        # Log progress
                        progress_pct = ((current_end - range_start) / (range_end - range_start)) * 100
                        self.logger.info(f"{symbol} {timeframe}: {progress_pct:.1f}% - Downloaded {inserted} candles (Total: {progress.total_candles_downloaded:,})")
                        
                        # Save progress periodically
                        if total_downloaded % 5000 == 0:  # Save every 5k candles
                            self.save_progress()
                    
                    # Move to next chunk
                    current_start = current_end + timeframe_ms
            
            # Mark as completed
            progress.status = 'completed'
            progress.last_update = datetime.now(timezone.utc).isoformat()
            self.save_progress()
            
            self.logger.info(f"{symbol} {timeframe} completed - {total_downloaded:,} total candles downloaded")
            return True
            
        except Exception as e:
            progress.status = 'error'
            self.save_progress()
            self.logger.error(f"Error downloading {symbol} {timeframe}: {e}")
            return False
    
    def timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 60 * 1000)  # Default to 1m
    
    def get_download_summary(self) -> Dict[str, Any]:
        """Get summary of download progress"""
        total_tasks = len(SYMBOLS) * len(TIMEFRAMES)
        completed_tasks = sum(1 for p in self.progress.values() if p.status == 'completed')
        error_tasks = sum(1 for p in self.progress.values() if p.status == 'error')
        in_progress_tasks = sum(1 for p in self.progress.values() if p.status == 'in_progress')
        
        total_candles = sum(p.total_candles_downloaded for p in self.progress.values())
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'error_tasks': error_tasks,
            'in_progress_tasks': in_progress_tasks,
            'completion_percentage': (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
            'total_candles_downloaded': total_candles
        }
    
    def run_bulk_download(self) -> bool:
        """Run the complete bulk download process"""
        self.logger.info("Starting bulk historical data download")
        self.logger.info(f"Date range: {START_DATE} to present")
        self.logger.info(f"Symbols: {SYMBOLS}")
        self.logger.info(f"Timeframes: {TIMEFRAMES} (1m only - fetcher will aggregate others)")
        self.logger.info(f"Max candles per request: {MAX_KLINES_PER_REQUEST}")
        self.logger.info(f"Database: {DATABASE_FILE}")
        
        start_time = time.time()
        success_count = 0
        total_tasks = len(SYMBOLS) * len(TIMEFRAMES)
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                self.logger.info(f"Processing {symbol} {timeframe}")
                
                if self.download_symbol_timeframe(symbol, timeframe):
                    success_count += 1
                
                # Progress summary
                summary = self.get_download_summary()
                self.logger.info(f"Overall progress: {summary['completion_percentage']:.1f}% "
                               f"({summary['completed_tasks']}/{total_tasks} tasks completed)")
        
        # Final summary
        elapsed_time = time.time() - start_time
        summary = self.get_download_summary()
        
        self.logger.info("BULK DOWNLOAD COMPLETE")
        self.logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        self.logger.info(f"Successful downloads: {success_count}/{total_tasks}")
        self.logger.info(f"Total candles downloaded: {summary['total_candles_downloaded']:,}")
        
        if summary['error_tasks'] > 0:
            self.logger.warning(f"{summary['error_tasks']} tasks had errors")
            self.logger.info("You can re-run this script to retry failed downloads")
        
        return success_count == total_tasks

def query_historical_data(symbol: str, timeframe: str = '1m', limit: int = 1000) -> List[Dict]:
    """Helper function to query historical data"""
    table_name = f"{symbol.lower()}_{timeframe}"
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            
            cursor.execute(f'''
                SELECT open_time, open, high, low, close, volume, symbol, timeframe
                FROM {table_name}
                WHERE is_complete = 1
                ORDER BY open_time DESC
                LIMIT ?
            ''', (limit,))
            
            columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
    except sqlite3.Error as e:
        print(f"Error querying {symbol} {timeframe}: {e}")
        return []

def main():
    """Main entry point"""
    try:
        puller = HistoricalDataPuller()
        
        print("HISTORICAL CRYPTOCURRENCY DATA PULLER (FIXED)")
        print("=" * 60)
        print(f"Date range: {START_DATE} to present")
        print(f"Symbols: {', '.join(SYMBOLS)}")
        print(f"Timeframes: {', '.join(TIMEFRAMES)} (1m only)")
        print(f"Max candles per request: {MAX_KLINES_PER_REQUEST}")
        print(f"Single database: {DATABASE_FILE}")
        print(f"Using API keys: {'Yes' if os.getenv('BINANCE_API_KEY') else 'No (public only)'}")
        print()
        
        # Check for existing progress
        if puller.progress:
            summary = puller.get_download_summary()
            print(f"Found existing progress:")
            print(f"   Completed: {summary['completed_tasks']}")
            print(f"   Errors: {summary['error_tasks']}")
            print(f"   In progress: {summary['in_progress_tasks']}")
            print(f"   Overall: {summary['completion_percentage']:.1f}% complete")
            print(f"   Total candles: {summary['total_candles_downloaded']:,}")
            print()
            
            if summary['completion_percentage'] == 100:
                response = input("All downloads appear complete. Re-download anyway? (y/N): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    return
                puller.progress = {}  # Reset progress
        
        response = input("Start bulk download? This may take several hours. (y/N): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
        
        # Start download
        success = puller.run_bulk_download()
        
        if success:
            print("\nAll downloads completed successfully!")
            print("Your fetcher will now aggregate this 1m data to other timeframes in real-time.")
        else:
            print("\nSome downloads failed. Check logs and re-run to retry.")
            
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise

if __name__ == "__main__":
    main()