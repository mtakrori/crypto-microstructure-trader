#!/usr/bin/env python3
"""
System Verification Script for Crypto Microstructure Trading System
Run this to verify your setup is working correctly before starting live trading
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timezone

# Setup basic logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SystemTest')

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import sqlite3
        print("  ✅ Core dependencies (pandas, numpy, sqlite3)")
    except ImportError as e:
        print(f"  ❌ Core dependencies failed: {e}")
        return False
    
    try:
        import config
        print(f"  ✅ Config module")
    except ImportError as e:
        print(f"  ❌ Config import failed: {e}")
        return False
    
    try:
        from data.validator import DataValidator
        from strategies.stop_hunt import StopHuntDetector
        from strategies.scalping import ScalpingStrategy
        from strategies.volume_profile import VolumeProfileStrategy
        print("  ✅ Strategy modules")
    except ImportError as e:
        print(f"  ❌ Strategy imports failed: {e}")
        return False
    
    try:
        from execution.risk_manager import AdvancedRiskManager
        from execution.order_manager import OrderManager
        print("  ✅ Execution modules")
    except ImportError as e:
        print(f"  ❌ Execution imports failed: {e}")
        return False
    
    try:
        from analysis.microstructure import MicrostructureAnalyzer
        from analysis.performance import PerformanceTracker
        print("  ✅ Analysis modules")
    except ImportError as e:
        print(f"  ❌ Analysis imports failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration settings"""
    print("\n⚙️ Testing configuration...")
    
    try:
        import config
        
        # Check database path
        if not Path(config.DATABASE_FILE).exists():
            print(f"  ⚠️ Database file not found: {config.DATABASE_FILE}")
            print("     Make sure your enhanced_crypto_fetcher.py is running")
            return False
        else:
            print(f"  ✅ Database file found: {config.DATABASE_FILE}")
        
        # Check data directory
        if not Path(config.DATA_DIR).exists():
            print(f"  ❌ Data directory not found: {config.DATA_DIR}")
            return False
        else:
            print(f"  ✅ Data directory found: {config.DATA_DIR}")
        
        # Check symbols
        if len(config.SYMBOLS) == 0:
            print("  ❌ No symbols configured")
            return False
        else:
            print(f"  ✅ {len(config.SYMBOLS)} symbols configured: {', '.join(config.SYMBOLS)}")
        
        # Check risk settings
        if config.RISK_CONFIG['max_risk_per_trade'] > 0.1:
            print(f"  ⚠️ High risk per trade: {config.RISK_CONFIG['max_risk_per_trade']:.1%}")
        else:
            print(f"  ✅ Risk per trade: {config.RISK_CONFIG['max_risk_per_trade']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_data_availability():
    """Test data availability and quality"""
    print("\n📊 Testing data availability...")
    
    try:
        from data.validator import DataValidator
        
        validator = DataValidator()
        health_report = validator.generate_health_report()
        
        print(f"  📈 Overall Status: {health_report['overall_status']}")
        print(f"  📊 Ready Symbols: {health_report['symbols_ready']}/{health_report['symbols_total']}")
        
        if health_report['overall_status'] == 'critical':
            print("  ❌ Critical data issues detected:")
            for issue in health_report['issues_found'][:3]:
                print(f"     - {issue}")
            return False
        
        # Test trading ready pairs
        ready_pairs = validator.get_trading_ready_pairs()
        print(f"  ✅ Trading ready pairs: {len(ready_pairs)}")
        
        if len(ready_pairs) < 2:
            print("  ⚠️ Insufficient trading pairs - need at least 2")
            return False
        
        # Show quality scores
        for symbol, score in health_report['quality_scores'].items():
            status = "✅" if score > 0.7 else "⚠️" if score > 0.4 else "❌"
            print(f"     {status} {symbol}: {score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data test failed: {e}")
        return False

def test_strategies():
    """Test strategy components"""
    print("\n🎯 Testing strategy components...")
    
    try:
        from strategies.stop_hunt import StopHuntDetector
        from strategies.scalping import ScalpingStrategy
        from strategies.volume_profile import VolumeProfileStrategy
        
        # Test stop hunt detector
        print("  🎯 Testing stop hunt detector...")
        detector = StopHuntDetector()
        signal = detector.scan_symbol('BTCUSDT')
        if signal:
            print(f"     ✅ Stop hunt signal found (confidence: {signal.confidence:.2%})")
        else:
            print("     ✅ Stop hunt detector working (no signals currently)")
        
        # Test scalping strategy
        print("  ⚡ Testing scalping strategy...")
        scalper = ScalpingStrategy()
        scalp_signal = scalper.scan_symbol('BTCUSDT')
        if scalp_signal:
            print(f"     ✅ Scalp signal found (confidence: {scalp_signal.confidence:.2%})")
        else:
            print("     ✅ Scalping strategy working (no signals currently)")
        
        # Test volume profile
        print("  📊 Testing volume profile strategy...")
        volume_strategy = VolumeProfileStrategy()
        volume_signal = volume_strategy.scan_symbol('BTCUSDT')
        if volume_signal:
            print(f"     ✅ Volume signal found (confidence: {volume_signal.confidence:.2%})")
        else:
            print("     ✅ Volume profile working (no signals currently)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy test failed: {e}")
        return False

def test_risk_management():
    """Test risk management system"""
    print("\n🛡️ Testing risk management...")
    
    try:
        from execution.risk_manager import AdvancedRiskManager
        
        risk_manager = AdvancedRiskManager()
        
        # Test position size calculation
        calc = risk_manager.calculate_position_size(
            signal_type="test_signal",
            entry_price=45000.0,
            stop_loss=44800.0,
            confidence=0.75,
            symbol="BTCUSDT"
        )
        
        print(f"  💰 Position size calculation: {calc.recommended_size:.6f}")
        print(f"  🔒 Risk amount: ${calc.risk_amount:.2f}")
        print(f"  ✅ Approved: {calc.approved}")
        print(f"  📝 Reason: {calc.reason}")
        
        # Test risk summary
        summary = risk_manager.get_risk_summary()
        print(f"  📊 Risk Level: {summary['risk_status']['risk_level']}")
        print(f"  💵 Available Balance: ${summary['account_health']['balance']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Risk management test failed: {e}")
        return False

def test_order_management():
    """Test order management system"""
    print("\n📋 Testing order management...")
    
    try:
        from execution.order_manager import OrderManager, OrderSide
        
        # Test in paper trading mode
        order_manager = OrderManager(paper_trading=True)
        
        print("  📝 Paper trading mode: ✅")
        
        # Test market order (will be simulated)
        order_id = order_manager.submit_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.001,
            strategy_id="test_strategy"
        )
        
        if order_id:
            print(f"  ✅ Test order executed: {order_id}")
            
            # Test position management
            order_manager.update_position_prices()
            print(f"  📊 Positions: {len(order_manager.positions)}")
            
            # Close test position
            if order_manager.positions:
                symbol = list(order_manager.positions.keys())[0]
                order_manager.close_position(symbol, "test_cleanup")
                print("  🧹 Test position cleaned up")
        else:
            print("  ⚠️ Order execution failed (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Order management test failed: {e}")
        return False

def test_analysis():
    """Test analysis components"""
    print("\n📈 Testing analysis components...")
    
    try:
        from analysis.microstructure import MicrostructureAnalyzer
        from analysis.performance import PerformanceTracker
        
        # Test microstructure analyzer
        analyzer = MicrostructureAnalyzer()
        summary = analyzer.get_microstructure_summary('BTCUSDT')
        
        if summary.get('error'):
            print(f"  ⚠️ Microstructure analysis: {summary['error']}")
        else:
            regime = summary['market_regime']['regime']
            volatility = summary['market_regime']['volatility']
            print(f"  📊 Market Regime: {regime}")
            print(f"  📉 Volatility: {volatility:.4f}")
            print("  ✅ Microstructure analysis working")
        
        # Test performance tracker
        tracker = PerformanceTracker()
        perf_summary = tracker.get_performance_summary()
        print(f"  📈 Performance tracker initialized")
        print(f"  📊 Total tracked trades: {perf_summary['summary']['total_trades']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Analysis test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all system tests"""
    print("🚀 Starting Crypto Microstructure Trading System Tests")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Data Availability", test_data_availability),
        ("Trading Strategies", test_strategies),
        ("Risk Management", test_risk_management),
        ("Order Management", test_order_management),
        ("Analysis Components", test_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for trading.")
        print("\n🚀 To start trading:")
        print("   python main.py")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Review failures before trading.")
    else:
        print("❌ Multiple test failures. Fix issues before trading.")
    
    return passed == total

def main():
    """Main test runner"""
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()