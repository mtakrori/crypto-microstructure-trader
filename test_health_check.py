#!/usr/bin/env python3
"""
Test script to check the health check logic specifically
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.validator import DataValidator
from config import SYMBOLS

def test_health_check():
    """Test the health check logic that's failing"""
    print("Testing health check logic...")
    print("=" * 50)
    
    validator = DataValidator()
    
    # Test microstructure symbols specifically
    microstructure_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    microstructure_timeframes = ['1m', '5m']
    
    ready_pairs = []
    
    for symbol in microstructure_symbols:
        for timeframe in microstructure_timeframes:
            print(f"\nChecking {symbol} {timeframe}:")
            report = validator.validate_symbol_timeframe(symbol, timeframe)
            
            print(f"  Valid: {report.is_valid}")
            print(f"  Microstructure Ready: {report.is_microstructure_ready}")
            print(f"  Has Live Data: {report.has_live_data}")
            print(f"  Live Candle Age: {report.live_candle_age_seconds:.1f}s")
            print(f"  Quality Score: {report.quality_score:.2f}")
            
            if report.is_valid:
                ready_pairs.append((symbol, timeframe))
                print(f"  ✓ READY")
            else:
                print(f"  ✗ NOT READY - Issues:")
                for issue in report.issues[:3]:  # Show first 3 issues
                    print(f"    - {issue}")
    
    print(f"\n{'='*50}")
    print(f"Ready pairs: {len(ready_pairs)}")
    for pair in ready_pairs:
        print(f"  - {pair[0]} {pair[1]}")
    
    # Test the specific check from main.py
    print(f"\nTesting main.py health check logic:")
    print(f"Need 4 pairs, have {len(ready_pairs)}")
    
    if len(ready_pairs) < 4:
        print("❌ INSUFFICIENT MICROSTRUCTURE DATA")
        print("Ready pairs: " + str(ready_pairs))
    else:
        print("✅ HEALTH CHECK PASSED")

def check_microstructure_readiness():
    """Check microstructure readiness specifically"""
    print("\n" + "="*50)
    print("MICROSTRUCTURE READINESS CHECK")
    print("="*50)
    
    validator = DataValidator()
    
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        print(f"\n{symbol}:")
        readiness = validator.get_microstructure_readiness_report(symbol)
        
        print(f"  Ready: {readiness.is_ready}")
        print(f"  Score: {readiness.readiness_score:.1%}")
        print(f"  Live 1m: {readiness.has_live_1m}")
        print(f"  Live 5m: {readiness.has_live_5m}")
        print(f"  Live Age: {readiness.live_data_age_seconds:.1f}s")
        print(f"  Complete Candles: {readiness.complete_candle_count}")
        
        if readiness.issues:
            print(f"  Issues:")
            for issue in readiness.issues[:3]:
                print(f"    - {issue}")

if __name__ == "__main__":
    test_health_check()
    check_microstructure_readiness()
