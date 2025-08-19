#!/usr/bin/env python3
"""
Analyze Fetcher Aggregation Logic
Check if the fetcher is properly aggregating 1m data to higher timeframes
"""

import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import DATABASE_FILE, get_table_name

def analyze_aggregation_logic():
    """Check if aggregation from 1m to 5m is working correctly"""
    print("Analyzing fetcher aggregation logic...")
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            symbol = 'BTCUSDT'
            
            # Get last 10 1m candles
            table_1m = get_table_name(symbol, '1m')
            query_1m = f"""
            SELECT open_time, open, high, low, close, volume, is_complete
            FROM {table_1m}
            ORDER BY open_time DESC
            LIMIT 10
            """
            df_1m = pd.read_sql_query(query_1m, conn)
            
            # Get last 3 5m candles  
            table_5m = get_table_name(symbol, '5m')
            query_5m = f"""
            SELECT open_time, open, high, low, close, volume, is_complete
            FROM {table_5m}
            ORDER BY open_time DESC
            LIMIT 3
            """
            df_5m = pd.read_sql_query(query_5m, conn)
            
            print(f"\n=== LATEST 1M CANDLES ({symbol}) ===")
            for _, row in df_1m.iterrows():
                dt = datetime.fromtimestamp(row['open_time'] / 1000, tz=timezone.utc)
                complete = "✓" if row['is_complete'] else "⚪"
                print(f"{dt.strftime('%H:%M:%S')} {complete} O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.0f}")
            
            print(f"\n=== LATEST 5M CANDLES ({symbol}) ===")
            for _, row in df_5m.iterrows():
                dt = datetime.fromtimestamp(row['open_time'] / 1000, tz=timezone.utc)
                complete = "✓" if row['is_complete'] else "⚪"
                print(f"{dt.strftime('%H:%M:%S')} {complete} O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.0f}")
            
            # Check if latest 5m candle should exist based on 1m data
            if not df_1m.empty:
                latest_1m_time = df_1m.iloc[0]['open_time']
                latest_1m_dt = datetime.fromtimestamp(latest_1m_time / 1000, tz=timezone.utc)
                
                # Calculate expected 5m candle time (should be on 5-minute boundaries)
                expected_5m_minute = (latest_1m_dt.minute // 5) * 5
                expected_5m_dt = latest_1m_dt.replace(minute=expected_5m_minute, second=0, microsecond=0)
                expected_5m_time = int(expected_5m_dt.timestamp() * 1000)
                
                print(f"\n=== AGGREGATION ANALYSIS ===")
                print(f"Latest 1m time: {latest_1m_dt.strftime('%H:%M:%S')}")
                print(f"Expected 5m time: {expected_5m_dt.strftime('%H:%M:%S')}")
                
                if not df_5m.empty:
                    latest_5m_time = df_5m.iloc[0]['open_time']
                    latest_5m_dt = datetime.fromtimestamp(latest_5m_time / 1000, tz=timezone.utc)
                    print(f"Actual 5m time: {latest_5m_dt.strftime('%H:%M:%S')}")
                    
                    time_diff_minutes = (latest_1m_dt - latest_5m_dt).total_seconds() / 60
                    print(f"Time difference: {time_diff_minutes:.1f} minutes")
                    
                    if time_diff_minutes > 5:
                        print("🚨 ISSUE: 5m aggregation is lagging behind 1m data")
                        print("   This suggests the fetcher's aggregation logic may have a problem")
                    else:
                        print("✅ GOOD: 5m aggregation is keeping up with 1m data")
                else:
                    print("🚨 ISSUE: No 5m data found at all")
                    
    except Exception as e:
        print(f"Error analyzing aggregation: {e}")

def check_incomplete_candles():
    """Check incomplete candles across timeframes"""
    print("\n" + "="*50)
    print("INCOMPLETE CANDLES ANALYSIS")
    print("="*50)
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                print(f"\n--- {symbol} ---")
                
                for timeframe in ['1m', '5m', '15m', '1h']:
                    table_name = get_table_name(symbol, timeframe)
                    
                    query = f"""
                    SELECT COUNT(*) as incomplete_count,
                           MAX(open_time) as latest_incomplete
                    FROM {table_name}
                    WHERE is_complete = 0
                    """
                    
                    result = pd.read_sql_query(query, conn)
                    
                    if not result.empty:
                        incomplete_count = result.iloc[0]['incomplete_count']
                        latest_incomplete = result.iloc[0]['latest_incomplete']
                        
                        if latest_incomplete:
                            latest_dt = datetime.fromtimestamp(latest_incomplete / 1000, tz=timezone.utc)
                            age = (datetime.now(timezone.utc) - latest_dt).total_seconds() / 60
                            print(f"  {timeframe}: {incomplete_count} incomplete, latest {age:.1f}min old")
                        else:
                            print(f"  {timeframe}: {incomplete_count} incomplete, no latest time")
                    
    except Exception as e:
        print(f"Error checking incomplete candles: {e}")

def check_aggregation_gaps():
    """Check for gaps in aggregation"""
    print("\n" + "="*50)
    print("AGGREGATION GAPS ANALYSIS")
    print("="*50)
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            symbol = 'BTCUSDT'
            
            # Check if we have the right number of 1m candles for each 5m period
            current_time = datetime.now(timezone.utc)
            five_min_ago = current_time - timedelta(minutes=5)
            
            # Get 5-minute boundary
            boundary_minute = (five_min_ago.minute // 5) * 5
            boundary_time = five_min_ago.replace(minute=boundary_minute, second=0, microsecond=0)
            boundary_ms = int(boundary_time.timestamp() * 1000)
            
            table_1m = get_table_name(symbol, '1m')
            query = f"""
            SELECT COUNT(*) as count_1m
            FROM {table_1m}
            WHERE open_time >= {boundary_ms}
            AND open_time < {boundary_ms + (5 * 60 * 1000)}
            """
            
            result = pd.read_sql_query(query, conn)
            count_1m = result.iloc[0]['count_1m'] if not result.empty else 0
            
            print(f"Last complete 5m period: {boundary_time.strftime('%H:%M:%S')}")
            print(f"1m candles in that period: {count_1m}/5 expected")
            
            if count_1m < 5:
                print("🚨 ISSUE: Missing 1m candles for aggregation")
            else:
                print("✅ GOOD: Complete 1m data for aggregation")
                
    except Exception as e:
        print(f"Error checking aggregation gaps: {e}")

def main():
    """Main analysis"""
    print("="*60)
    print("FETCHER AGGREGATION ANALYSIS")
    print("="*60)
    
    analyze_aggregation_logic()
    check_incomplete_candles()
    check_aggregation_gaps()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("If you see issues above, your fetcher's aggregation logic")
    print("may need debugging. The 1m data should automatically")
    print("create fresh 5m, 15m, 1h, 4h, 1d data in real-time.")
    print("="*60)

if __name__ == "__main__":
    main()