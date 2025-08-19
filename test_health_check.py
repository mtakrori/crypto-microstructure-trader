#!/usr/bin/env python3
"""
Force start trading system with minimal validation
Bypasses strict health checks when you know your data is good
"""

import sqlite3
from datetime import datetime, timezone
from config import DATABASE_FILE, SYMBOLS, get_table_name
from main import CryptoMicrostructureTrader

def check_minimal_requirements():
    """Check absolute minimum requirements for trading"""
    print("Checking minimal trading requirements...")
    
    ready_count = 0
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            
            for symbol in SYMBOLS[:3]:  # Check first 3 symbols
                for timeframe in ['1m', '5m']:
                    table_name = get_table_name(symbol, timeframe)
                    
                    try:
                        # Very basic check: table exists and has recent data
                        cursor.execute(f"""
                            SELECT COUNT(*) as total,
                                   MAX(open_time) as latest_time
                            FROM {table_name}
                            WHERE open_time > ?
                        """, (int((datetime.now(timezone.utc).timestamp() - 3600) * 1000),))  # Last hour
                        
                        result = cursor.fetchone()
                        total = result[0] if result else 0
                        latest_time = result[1] if result else None
                        
                        if total > 0 and latest_time:
                            latest_dt = datetime.fromtimestamp(latest_time / 1000, tz=timezone.utc)
                            age_minutes = (datetime.now(timezone.utc) - latest_dt).total_seconds() / 60
                            
                            print(f"  {symbol} {timeframe}: {total} recent candles, {age_minutes:.1f}min old")
                            
                            if total > 5 and age_minutes < 60:  # Very lenient
                                ready_count += 1
                                print(f"    ✅ READY")
                            else:
                                print(f"    ⚠️ Not ready")
                        else:
                            print(f"  {symbol} {timeframe}: ❌ No recent data")
                    
                    except Exception as e:
                        print(f"  {symbol} {timeframe}: ❌ Error: {e}")
            
            print(f"\nReady pairs: {ready_count}/6")
            return ready_count >= 2  # Need at least 2 pairs
            
    except Exception as e:
        print(f"Error checking requirements: {e}")
        return False

def create_forced_trader():
    """Create trader with bypassed health check"""
    
    class ForcedTrader(CryptoMicrostructureTrader):
        def _initial_health_check(self):
            """Override health check to always pass"""
            self.logger.info("FORCING health check to pass...")
            
            # Do minimal check
            if check_minimal_requirements():
                self.logger.info("Minimal requirements met - proceeding with trading")
                return True
            else:
                self.logger.error("Even minimal requirements not met - aborting")
                return False
        
        def _is_symbol_ready(self, symbol):
            """Override symbol readiness check to be very lenient"""
            try:
                with sqlite3.connect(DATABASE_FILE) as conn:
                    cursor = conn.cursor()
                    
                    # Just check if we have some 1m data
                    table_name = get_table_name(symbol, '1m')
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
                    
                    result = cursor.fetchone()
                    return result and result[0] > 0
                    
            except Exception:
                return False
    
    return ForcedTrader(paper_trading=True)

def main():
    """Force start the trading system"""
    print("=" * 60)
    print("FORCE STARTING CRYPTO MICROSTRUCTURE TRADING SYSTEM")
    print("⚠️ This bypasses health checks - use only when you know data is good!")
    print("=" * 60)
    
    try:
        # Create forced trader
        trader = create_forced_trader()
        
        print("\nStarting forced trading session...")
        trader.run()
        
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()