#!/usr/bin/env python3
"""
Historical Data Aggregator
Aggregates ALL existing 1m data to higher timeframes (5m, 15m, 1h, 4h, 1d)
Uses the same logic as your fetcher for consistency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

# Import your existing configuration
from config import DATABASE_FILE, SYMBOLS, get_table_name

# Aggregation configuration (matching your fetcher)
AGGREGATION_FACTORS = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

TARGET_TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']  # What to aggregate to

class HistoricalAggregator:
    """Aggregate all historical 1m data to higher timeframes"""
    
    def __init__(self):
        self.setup_logging()
        self.processed_candles = 0
        self.start_time = time.time()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('historical_aggregation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HistoricalAggregator')
        self.logger.info("Historical Aggregator initialized")
    
    def get_timeframe_start_time(self, timestamp_ms: int, timeframe: str) -> int:
        """Get the start time for a timeframe period (same logic as fetcher)"""
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
    
    def check_existing_data(self, symbol: str) -> Dict[str, Dict]:
        """Check what data already exists for each timeframe"""
        data_status = {}
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                
                for timeframe in ['1m'] + TARGET_TIMEFRAMES:
                    table_name = get_table_name(symbol, timeframe)
                    
                    # Check if table exists
                    cursor.execute('''
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name=?
                    ''', (table_name,))
                    
                    if not cursor.fetchone():
                        data_status[timeframe] = {'exists': False, 'count': 0, 'range': None}
                        continue
                    
                    # Get data statistics
                    cursor.execute(f'''
                        SELECT COUNT(*) as count,
                               MIN(open_time) as min_time,
                               MAX(open_time) as max_time
                        FROM {table_name}
                        WHERE is_complete = 1
                    ''')
                    
                    result = cursor.fetchone()
                    count = result[0] if result else 0
                    min_time = result[1] if result else None
                    max_time = result[2] if result else None
                    
                    if min_time and max_time:
                        min_dt = datetime.fromtimestamp(min_time / 1000, tz=timezone.utc)
                        max_dt = datetime.fromtimestamp(max_time / 1000, tz=timezone.utc)
                        date_range = f"{min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}"
                    else:
                        date_range = "No data"
                    
                    data_status[timeframe] = {
                        'exists': True,
                        'count': count,
                        'range': date_range,
                        'min_time': min_time,
                        'max_time': max_time
                    }
        
        except Exception as e:
            self.logger.error(f"Error checking existing data for {symbol}: {e}")
            
        return data_status
    
    def get_1m_data_chunks(self, symbol: str, chunk_size: int = 10000) -> List[List]:
        """Get 1m data in chunks for memory efficiency"""
        table_name = get_table_name(symbol, '1m')
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                
                # Get total count first
                cursor.execute(f'SELECT COUNT(*) FROM {table_name} WHERE is_complete = 1')
                total_candles = cursor.fetchone()[0]
                
                self.logger.info(f"Processing {total_candles:,} 1m candles for {symbol}")
                
                # Process in chunks
                offset = 0
                while offset < total_candles:
                    cursor.execute(f'''
                        SELECT open_time, open, high, low, close, volume, close_time,
                               quote_volume, count, taker_buy_volume, taker_buy_quote_volume
                        FROM {table_name}
                        WHERE is_complete = 1
                        ORDER BY open_time
                        LIMIT {chunk_size} OFFSET {offset}
                    ''')
                    
                    chunk = cursor.fetchall()
                    if not chunk:
                        break
                    
                    yield chunk
                    offset += len(chunk)
                    
                    # Progress update
                    progress = (offset / total_candles) * 100
                    self.logger.info(f"  {symbol} 1m data: {progress:.1f}% processed ({offset:,}/{total_candles:,})")
        
        except Exception as e:
            self.logger.error(f"Error reading 1m data for {symbol}: {e}")
            return []
    
    def aggregate_candles_batch(self, candles_1m: List[List], target_timeframe: str) -> List[List]:
        """Aggregate 1m candles to target timeframe (same logic as fetcher)"""
        if not candles_1m:
            return []
        
        # Group candles by their timeframe period
        periods = {}
        
        for candle in candles_1m:
            timestamp_ms = int(candle[0])
            period_start = self.get_timeframe_start_time(timestamp_ms, target_timeframe)
            
            if period_start not in periods:
                periods[period_start] = []
            periods[period_start].append(candle)
        
        # Create aggregated candles
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
            
            # Aggregate microstructure data
            quote_volume = sum(float(c[7]) for c in period_candles)
            count = sum(int(c[8]) for c in period_candles)
            taker_buy_volume = sum(float(c[9]) for c in period_candles)
            taker_buy_quote_volume = sum(float(c[10]) for c in period_candles)
            
            aggregated_candle = [
                period_start_ms,           # open_time
                open_price,                # open
                high_price,                # high
                low_price,                 # low
                close_price,               # close
                volume,                    # volume
                close_time,                # close_time
                quote_volume,              # quote_volume
                count,                     # count
                taker_buy_volume,          # taker_buy_volume
                taker_buy_quote_volume     # taker_buy_quote_volume
            ]
            
            aggregated.append(aggregated_candle)
        
        return aggregated
    
    def create_target_table(self, symbol: str, timeframe: str):
        """Create target timeframe table if it doesn't exist"""
        table_name = get_table_name(symbol, timeframe)
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                
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
                
                # Create indexes
                cursor.execute(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_time_complete 
                    ON {table_name}(open_time, is_complete)
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error creating table {table_name}: {e}")
    
    def bulk_insert_aggregated_data(self, symbol: str, timeframe: str, aggregated_candles: List[List]):
        """Bulk insert aggregated candles"""
        if not aggregated_candles:
            return 0
        
        table_name = get_table_name(symbol, timeframe)
        
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                
                # Prepare records for bulk insert
                records = []
                for candle in aggregated_candles:
                    record = (
                        int(candle[0]),     # open_time
                        float(candle[1]),   # open
                        float(candle[2]),   # high
                        float(candle[3]),   # low
                        float(candle[4]),   # close
                        float(candle[5]),   # volume
                        int(candle[6]),     # close_time
                        float(candle[7]),   # quote_volume
                        int(candle[8]),     # count
                        float(candle[9]),   # taker_buy_volume
                        float(candle[10]),  # taker_buy_quote_volume
                        1,                  # is_complete
                        symbol,             # symbol
                        timeframe           # timeframe
                    )
                    records.append(record)
                
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
                
                return inserted_count
                
        except Exception as e:
            self.logger.error(f"Error inserting data into {table_name}: {e}")
            return 0
    
    def aggregate_symbol_timeframe(self, symbol: str, target_timeframe: str) -> bool:
        """Aggregate all 1m data for a symbol to target timeframe"""
        self.logger.info(f"Aggregating {symbol} 1m -> {target_timeframe}")
        
        # Create target table
        self.create_target_table(symbol, target_timeframe)
        
        # Process 1m data in chunks
        total_aggregated = 0
        all_aggregated_candles = []
        
        for chunk in self.get_1m_data_chunks(symbol, chunk_size=50000):  # 50k candles per chunk
            # Aggregate this chunk
            aggregated_chunk = self.aggregate_candles_batch(chunk, target_timeframe)
            all_aggregated_candles.extend(aggregated_chunk)
            
            # If we have enough aggregated candles, insert them
            if len(all_aggregated_candles) >= 10000:  # Insert every 10k aggregated candles
                inserted = self.bulk_insert_aggregated_data(symbol, target_timeframe, all_aggregated_candles)
                total_aggregated += inserted
                all_aggregated_candles = []
                
                self.logger.info(f"  {symbol} {target_timeframe}: {total_aggregated:,} candles aggregated")
        
        # Insert remaining candles
        if all_aggregated_candles:
            inserted = self.bulk_insert_aggregated_data(symbol, target_timeframe, all_aggregated_candles)
            total_aggregated += inserted
        
        self.logger.info(f"✅ {symbol} {target_timeframe}: {total_aggregated:,} total candles aggregated")
        self.processed_candles += total_aggregated
        
        return True
    
    def run_full_aggregation(self):
        """Run complete historical aggregation for all symbols and timeframes"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING HISTORICAL DATA AGGREGATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Symbols: {SYMBOLS}")
        self.logger.info(f"Target timeframes: {TARGET_TIMEFRAMES}")
        self.logger.info(f"Database: {DATABASE_FILE}")
        
        # Check existing data first
        for symbol in SYMBOLS:
            self.logger.info(f"\n--- {symbol} Data Status ---")
            data_status = self.check_existing_data(symbol)
            
            for timeframe, status in data_status.items():
                if status['exists']:
                    self.logger.info(f"  {timeframe}: {status['count']:,} candles ({status['range']})")
                else:
                    self.logger.info(f"  {timeframe}: No data")
        
        # Start aggregation
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STARTING AGGREGATION PROCESS")
        self.logger.info("=" * 60)
        
        success_count = 0
        total_tasks = len(SYMBOLS) * len(TARGET_TIMEFRAMES)
        
        for symbol in SYMBOLS:
            self.logger.info(f"\n🔄 Processing {symbol}")
            
            for target_timeframe in TARGET_TIMEFRAMES:
                try:
                    if self.aggregate_symbol_timeframe(symbol, target_timeframe):
                        success_count += 1
                    
                    # Progress update
                    progress = (success_count / total_tasks) * 100
                    self.logger.info(f"Overall progress: {progress:.1f}% ({success_count}/{total_tasks} completed)")
                    
                except Exception as e:
                    self.logger.error(f"Failed to aggregate {symbol} {target_timeframe}: {e}")
        
        # Final summary
        elapsed_time = time.time() - self.start_time
        self.logger.info("\n" + "=" * 60)
        self.logger.info("AGGREGATION COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info(f"Elapsed time: {elapsed_time/3600:.2f} hours")
        self.logger.info(f"Successful aggregations: {success_count}/{total_tasks}")
        self.logger.info(f"Total candles processed: {self.processed_candles:,}")
        
        if success_count == total_tasks:
            self.logger.info("🎉 ALL AGGREGATIONS SUCCESSFUL!")
            self.logger.info("Your trading system now has full historical data across all timeframes!")
        else:
            self.logger.warning(f"⚠️ {total_tasks - success_count} aggregations failed")
        
        return success_count == total_tasks

def main():
    """Main entry point"""
    print("HISTORICAL DATA AGGREGATION")
    print("=" * 60)
    print("This will aggregate ALL your 1m data since 2023 to higher timeframes")
    print("Timeframes: 5m, 15m, 1h, 4h, 1d")
    print("This may take 2-6 hours depending on your data volume")
    print()
    
    response = input("Continue with full historical aggregation? (y/N): ")
    if response.lower() != 'y':
        print("Aggregation cancelled.")
        return
    
    try:
        aggregator = HistoricalAggregator()
        success = aggregator.run_full_aggregation()
        
        if success:
            print("\n🎉 SUCCESS! All historical data aggregated!")
            print("You can now run your trading system with full historical context.")
            print("\nNext step: python main.py")
        else:
            print("\n⚠️ Some aggregations failed. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\nAggregation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
