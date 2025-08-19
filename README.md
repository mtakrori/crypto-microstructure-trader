# Crypto Microstructure Trader

A Python-based algorithmic trading system focused on cryptocurrency market microstructure analysis and execution.

## Overview

This project implements various trading strategies that analyze market microstructure patterns in cryptocurrency markets, with a focus on high-frequency trading concepts like order book dynamics, volume profile analysis, and stop hunting detection.

## Features

- **Market Data Fetcher**: Real-time and historical data collection
- **Volume Profile Strategy**: Identifies key support/resistance levels based on volume concentration
- **Scalping Strategy**: High-frequency trading with tight spreads
- **Stop Hunt Detection**: Identifies potential stop-loss hunting patterns
- **Risk Management**: Comprehensive position sizing and risk controls
- **Order Execution**: Efficient order management system

## Project Structure

```
crypto-microstructure-trader/
├── analysis/           # Market analysis modules
│   ├── microstructure.py
│   └── performance.py
├── data/              # Data handling modules
│   ├── fetcher.py
│   ├── historical_aggregator.py
│   ├── historical_data_puller.py
│   └── validator.py
├── execution/         # Order execution modules
│   ├── order_manager.py
│   └── risk_manager.py
├── strategies/        # Trading strategies
│   ├── scalping.py
│   ├── stop_hunt.py
│   └── volume_profile.py
├── tests/            # Test suite
│   └── test_strategies.py
├── logs/             # Application logs
├── main.py           # Main application entry point
├── config.py         # Configuration settings
├── requirements.txt  # Python dependencies
└── .env             # Environment variables
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/crypto-microstructure-trader.git
cd crypto-microstructure-trader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

Create a `.env` file with the following variables:

```
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_SECRET_KEY=your_exchange_secret_key
EXCHANGE_PASSPHRASE=your_exchange_passphrase
TRADING_PAIR=BTC-USDT
INITIAL_CAPITAL=1000
MAX_RISK_PER_TRADE=0.02  # 2% of capital
```

## Usage

Run the main trading application:

```bash
python main.py
```

Run specific strategies:

```bash
# Volume profile strategy
python -m strategies.volume_profile

# Scalping strategy  
python -m strategies.scalping

# Stop hunt detection
python -m strategies.stop_hunt
```

## Strategies

### Volume Profile Strategy
Identifies key price levels where significant trading volume has occurred, using these as potential support/resistance zones for trade entries and exits.

### Scalping Strategy
High-frequency trading approach that capitalizes on small price movements, typically holding positions for seconds to minutes.

### Stop Hunt Detection
Analyzes order book dynamics to identify potential stop-loss hunting patterns where large players might be triggering retail traders' stops.

## Risk Management

- Position sizing based on account equity and risk tolerance
- Maximum drawdown limits
- Daily loss limits
- Correlation-aware position management

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Logging

Application logs are stored in the `logs/` directory with rotating file handlers to manage disk space.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves significant risk and may not be suitable for all investors. Past performance is not indicative of future results. The authors are not responsible for any financial losses incurred while using this software.
