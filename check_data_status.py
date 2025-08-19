#!/usr/bin/env python3
"""
Quick Data Status Checker
Run this to check if your enhanced_crypto_fetcher.py is working properly
"""

import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import DATABASE_FILE, SYMBOLS, get_table_name

def check_data_freshness():
    """Check how fresh the data is"""
    print("Checking data freshness...")
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            current_time = datetime.now(timezone.utc)
            
            for symbol in SYMBOLS[:3]:  # Check first 3 symbols
                for timeframe in ['1m', '5m']:
                    table_name = get_table_name(symbol, timeframe)
                    
                    # Get latest timestamp (including incomplete candles)
                    query = f"""
                    SELECT MAX(open_time) as latest_time, 
                           COUNT(*) as total_candles,
                           COUNT(CASE WHEN is_complete = 1 THEN 1 END) as complete_candles,
                           COUNT(CASE WHEN is_complete = 0 THEN 1 END) as incomplete_candles
                    FROM {table_name}
                    """
                    
                    result = pd.read_sql_query(query, conn)
                    
                    if not result.empty and result.iloc[0]['latest_time']:
                        latest_ms = result.iloc[0]['latest_time']
                        latest_time = datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc)
                        age_minutes = (current_time - latest_time).total_seconds() / 60
                        total_candles = result.iloc[0]['total_candles']
                        complete_candles = result.iloc[0]['complete_candles']
                        incomplete_candles = result.iloc[0]['incomplete_candles']
                        
                        status = "FRESH" if age_minutes < 5 else "STALE" if age_minutes < 20 else "VERY STALE"
                        print(f"  {symbol} {timeframe}: {status} ({age_minutes:.1f} min old)")
                        print(f"    Total: {total_candles}, Complete: {complete_candles}, Incomplete: {incomplete_candles}")
                    else:
                        print(f"  {symbol} {timeframe}: NO DATA")
                        
    except Exception as e:
        print(f"Error checking data: {e}")

def check_data_quality():
    """Check data quality issues"""
    print("\nChecking data quality...")
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            for symbol in SYMBOLS[:2]:  # Check first 2 symbols
                table_name = get_table_name(symbol, '1m')
                
                # Check for invalid candles
                query = f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN high < low OR high < open OR high < close 
                                 OR low > open OR low > close OR low > high
                                 OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                           THEN 1 ELSE 0 END) as invalid
                FROM {table_name}
                WHERE is_complete = 1
                """
                
                result = pd.read_sql_query(query, conn)
                
                if not result.empty:
                    total = result.iloc[0]['total']
                    invalid = result.iloc[0]['invalid']
                    invalid_pct = (invalid / total * 100) if total > 0 else 0
                    
                    status = "GOOD" if invalid_pct < 1 else "FAIR" if invalid_pct < 5 else "POOR"
                    print(f"  {symbol}: {status} ({invalid}/{total} invalid = {invalid_pct:.1f}%)")
                    
    except Exception as e:
        print(f"Error checking quality: {e}")

def check_fetcher_status():
    """Check if fetcher is likely running"""
    print("\nFetcher Status Analysis:")
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            # Check recent activity (last 10 minutes)
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            ten_min_ago_ms = current_time_ms - (10 * 60 * 1000)
            
            total_recent = 0
            
            for symbol in SYMBOLS:
                table_name = get_table_name(symbol, '1m')
                
                query = f"""
                SELECT COUNT(*) as recent_count
                FROM {table_name}
                WHERE open_time > {ten_min_ago_ms}
                """
                
                result = pd.read_sql_query(query, conn)
                if not result.empty:
                    recent_count = result.iloc[0]['recent_count']
                    total_recent += recent_count
            
            if total_recent > 50:  # Should have ~60 candles (6 symbols * 10 minutes)
                print(f"  LIKELY RUNNING: {total_recent} recent candles found")
            elif total_recent > 10:
                print(f"  POSSIBLY RUNNING: {total_recent} recent candles (may be slow)")
            else:
                print(f"  LIKELY STOPPED: Only {total_recent} recent candles")
                print("  Action: Check if enhanced_crypto_fetcher.py is running")
                
    except Exception as e:
        print(f"Error checking fetcher status: {e}")

def main():
    """Main status check"""
    print("=" * 60)
    print("CRYPTO DATA STATUS CHECK")
    print("=" * 60)
    
    # Check if database exists
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            print(f"Database found: {DATABASE_FILE}")
    except Exception as e:
        print(f"Database error: {e}")
        return
    
    check_data_freshness()
    check_data_quality()
    check_fetcher_status()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("1. If data is STALE: Restart enhanced_crypto_fetcher.py")
    print("2. If data quality is POOR: Check fetcher logs")
    print("3. If fetcher is STOPPED: Start enhanced_crypto_fetcher.py")
    print("4. Wait 5-10 minutes after starting fetcher before trading")
    print("=" * 60)

if __name__ == "__main__":
    main()