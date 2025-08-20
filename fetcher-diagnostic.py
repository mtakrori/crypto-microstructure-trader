#!/usr/bin/env python3
"""
Diagnostic script to check fetcher status and data freshness
"""

import sqlite3
import time
from datetime import datetime, timezone
import json
from pathlib import Path

# Configuration
DATA_DIR = 'C:/Users/Sammour/Documents/Qpredict/crypto-futures-trader/data'
DATABASE_FILE = f'{DATA_DIR}/crypto_candles.sqlite'
STATE_FILE = f'{DATA_DIR}/fetcher_state.json'
METRICS_FILE = f'{DATA_DIR}/fetcher_metrics.json'

def check_fetcher_status():
    """Check if fetcher is running and how fresh the data is"""
    
    print("=" * 60)
    print("FETCHER STATUS CHECK")
    print("=" * 60)
    
    # 1. Check fetcher state file
    try:
        if Path(STATE_FILE).exists():
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                print(f"\n📄 Fetcher State:")
                print(f"  Initial fetch completed: {state.get('initial_fetch_completed', False)}")
                print(f"  Last fetch mode: {state.get('last_fetch_mode', 'unknown')}")
        else:
            print("\n❌ No fetcher state file found - fetcher may not be running")
    except Exception as e:
        print(f"❌ Error reading state: {e}")
    
    # 2. Check fetcher metrics
    try:
        if Path(METRICS_FILE).exists():
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
                last_fetch = metrics.get('last_successful_fetch', 'unknown')
                if last_fetch != 'unknown':
                    last_fetch_time = datetime.fromisoformat(last_fetch.replace('Z', '+00:00'))
                    age = (datetime.now(timezone.utc) - last_fetch_time).total_seconds()
                    print(f"\n📊 Fetcher Metrics:")
                    print(f"  Last successful fetch: {age:.1f} seconds ago")
                    print(f"  Total requests: {metrics.get('total_requests', 0)}")
                    print(f"  Successful: {metrics.get('successful_requests', 0)}")
                    print(f"  Failed: {metrics.get('failed_requests', 0)}")
                    
                    if age > 60:
                        print(f"  ⚠️ WARNING: Last fetch was {age:.0f} seconds ago!")
        else:
            print("\n❌ No metrics file found - fetcher may not be running")
    except Exception as e:
        print(f"❌ Error reading metrics: {e}")
    
    # 3. Check database for live candles
    print(f"\n🗄️ Database Live Candle Status:")
    print("-" * 40)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    current_time = datetime.now(timezone.utc)
    
    all_fresh = True
    
    for symbol in symbols:
        try:
            with sqlite3.connect(DATABASE_FILE) as conn:
                cursor = conn.cursor()
                table_name = f"{symbol.lower()}_1m"
                
                # Get the latest candle
                cursor.execute(f'''
                    SELECT open_time, close, volume, is_complete, last_updated
                    FROM {table_name}
                    ORDER BY open_time DESC
                    LIMIT 1
                ''')
                
                result = cursor.fetchone()
                if result:
                    open_time_ms, close_price, volume, is_complete, last_updated = result
                    candle_time = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
                    candle_age = (current_time - candle_time).total_seconds()
                    
                    # Check update time
                    if last_updated:
                        update_time = datetime.fromtimestamp(last_updated, tz=timezone.utc)
                        update_age = (current_time - update_time).total_seconds()
                    else:
                        update_age = float('inf')
                    
                    status = "✅ LIVE" if is_complete == 0 else "❌ COMPLETE"
                    
                    print(f"\n{symbol}:")
                    print(f"  Status: {status}")
                    print(f"  Candle age: {candle_age:.1f}s")
                    print(f"  Last update: {update_age:.1f}s ago")
                    print(f"  Price: {close_price:.2f}")
                    
                    if is_complete == 0:
                        if candle_age > 65:  # More than 1 minute + buffer
                            print(f"  ⚠️ STALE: Live candle is {candle_age:.0f}s old!")
                            all_fresh = False
                        elif candle_age > 10:
                            print(f"  ⚠️ TOO OLD FOR MICROSTRUCTURE: Need < 10s, got {candle_age:.1f}s")
                            all_fresh = False
                        else:
                            print(f"  ✅ FRESH: Good for microstructure trading")
                    else:
                        print(f"  ❌ NO LIVE DATA: Last candle is complete")
                        all_fresh = False
                        
        except Exception as e:
            print(f"\n{symbol}: ❌ Error - {e}")
            all_fresh = False
    
    # 4. Recommendations
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    
    if not all_fresh:
        print("❌ DATA NOT FRESH ENOUGH FOR MICROSTRUCTURE TRADING\n")
        print("The fetcher needs to update more frequently. Issues found:")
        print("1. Live candles are > 10 seconds old")
        print("2. Fetcher may not be running or is delayed\n")
        print("SOLUTION:")
        print("1. Check if fetcher.py is running")
        print("2. Modify fetcher to update every 5-10 seconds instead of 15")
        print("3. Or temporarily increase microstructure tolerance")
    else:
        print("✅ All systems operational - data is fresh!")
    
    return all_fresh

def monitor_freshness(duration_seconds=60):
    """Monitor data freshness over time"""
    print("\n" + "=" * 60)
    print(f"MONITORING DATA FRESHNESS FOR {duration_seconds} SECONDS")
    print("=" * 60)
    
    start_time = time.time()
    check_count = 0
    fresh_count = 0
    
    while time.time() - start_time < duration_seconds:
        check_count += 1
        current_time = datetime.now(timezone.utc)
        
        # Check BTCUSDT as representative
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT open_time, is_complete
                FROM btcusdt_1m
                ORDER BY open_time DESC
                LIMIT 1
            ''')
            
            result = cursor.fetchone()
            if result:
                open_time_ms, is_complete = result
                candle_age = (current_time.timestamp() * 1000 - open_time_ms) / 1000
                
                if is_complete == 0 and candle_age < 10:
                    fresh_count += 1
                    status = "✅ FRESH"
                else:
                    status = f"❌ STALE ({candle_age:.1f}s)"
                
                print(f"Check {check_count}: {status}")
        
        time.sleep(5)  # Check every 5 seconds
    
    success_rate = (fresh_count / check_count) * 100
    print(f"\nFreshness Rate: {success_rate:.1f}% ({fresh_count}/{check_count} checks)")
    
    if success_rate < 50:
        print("⚠️ Poor freshness rate - fetcher needs adjustment")

if __name__ == "__main__":
    # Run diagnostic
    is_fresh = check_fetcher_status()
    
    if not is_fresh:
        print("\n" + "=" * 60)
        print("TEMPORARY WORKAROUND")
        print("=" * 60)
        print("While you fix the fetcher, you can temporarily adjust the validator:")
        print("\nIn validator.py, change line ~180:")
        print("  FROM: if candle_age_seconds > 10:")
        print("  TO:   if candle_age_seconds > 40:  # Temporary workaround")
        print("\nBut this reduces the effectiveness of microstructure trading!")
    
    # Optionally monitor for a period
    # monitor_freshness(60)