# 🚀 Babilon Trade Bot - Enhanced AI Trading Assistant

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/your-username/babilon-trade-bot)

An advanced AI-powered trading bot that combines sentiment analysis using FinBERT with sophisticated risk management and technical analysis to make informed trading decisions.

## ✨ What's New in This Enhanced Version

### 🔧 Major Improvements
- **Robust Error Handling**: Comprehensive error handling with retry logic and graceful failures
- **Enhanced Risk Management**: Built-in position sizing, stop losses, and portfolio risk controls
- **Technical Analysis**: Moving averages, volatility analysis, and trend detection
- **Security Improvements**: Environment-based configuration and secure API key handling
- **Comprehensive Testing**: Extensive test suite covering edge cases and error conditions
- **Docker Support**: Complete containerization with Docker and Docker Compose
- **CI/CD Pipeline**: Automated testing, security scanning, and deployment
- **Enhanced Documentation**: Comprehensive documentation with examples and best practices

### 🛡️ Risk Management Features
- Automatic position sizing based on portfolio value
- Configurable stop loss and take profit levels
- Portfolio risk limits to prevent overexposure
- Wash trade prevention
- Market volatility analysis

### 🏗️ Architecture Improvements
- Modular, extensible design
- Enhanced AlpacaClient with retry logic
- Structured TradeSignal objects
- Comprehensive logging and monitoring
- Multiple interfaces (CLI, API, Web UI)

## 🚀 Quick Start

### 1. Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/babilon-trade-bot.git
cd babilon-trade-bot

# Run automated setup
python setup.py
```

### 2. Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env_template.txt .env
# Edit .env with your API keys and preferences
```

### 3. Configure Your Environment
Edit `.env` file with your Alpaca API keys:
```bash
ALPACA_API_KEY=your_paper_trading_api_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_key_here
STOCK_SYMBOL=AAPL
POSITIVE_THRESHOLD=0.65
NEGATIVE_THRESHOLD=0.70
```

### 4. Run the Bot
```bash
# Basic usage
python main.py

# Web interface
streamlit run streamlit_app.py

# API server
python api.py
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t babilon-trade-bot .
docker run --env-file .env babilon-trade-bot
```

## 📊 Usage Examples

### Basic Trading
```python
from main import BabilonTradeBot

# Initialize bot
bot = BabilonTradeBot("AAPL")

# Analyze news and trade
news = "Apple reports record quarterly revenue exceeding expectations"
result = bot.analyze_and_trade(news)

print(f"Signal: {result['trade_signal']['action']}")
print(f"Confidence: {result['trade_signal']['confidence']:.2f}")
```

### Advanced Configuration
```python
# Custom risk parameters
from config import *
MAX_POSITION_SIZE = 2000.0  # $2000 max position
MAX_PORTFOLIO_RISK = 0.01   # 1% max risk per trade
STOP_LOSS_PERCENTAGE = 0.03 # 3% stop loss
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_trade_decision_enhanced.py -v
```

## 📖 Documentation

- **[Enhanced Documentation](README_ENHANCED.md)** - Comprehensive guide with examples
- **[Environment Template](env_template.txt)** - Configuration template with explanations
- **[API Reference](README_ENHANCED.md#api-reference)** - Complete API documentation
- **[Deployment Guide](README_ENHANCED.md#deployment)** - Docker and production deployment

## 🛡️ Security Features

- Environment-based API key management
- Comprehensive .gitignore for security
- Input validation and sanitization
- Rate limiting and retry logic
- Paper trading by default

## 🔧 Configuration Options

### Trading Modes
```bash
# Conservative (fewer, more confident trades)
POSITIVE_THRESHOLD=0.75
MAX_PORTFOLIO_RISK=0.01

# Aggressive (more frequent trades)
POSITIVE_THRESHOLD=0.55
MAX_PORTFOLIO_RISK=0.03
```

## 📈 Key Features

- **AI Sentiment Analysis**: FinBERT for financial text analysis
- **Technical Analysis**: Moving averages, volatility, trend detection
- **Risk Management**: Position sizing, stop losses, portfolio limits
- **Paper Trading**: Safe testing with Alpaca paper trading
- **Multiple Interfaces**: CLI, Web UI, REST API
- **Comprehensive Logging**: Detailed logs for monitoring
- **Docker Support**: Easy deployment and scaling
- **CI/CD Pipeline**: Automated testing and deployment

## ⚠️ Important Notes

- **Paper Trading Only**: This bot is configured for paper trading by default
- **Educational Purpose**: This software is for educational and research purposes
- **Risk Disclaimer**: Trading involves substantial risk of loss
- **API Keys**: Keep your API keys secure and never commit them to version control

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for the trading API
- [Hugging Face](https://huggingface.co/) for the FinBERT model
- [Streamlit](https://streamlit.io/) for the web interface

---

**Made with ❤️ for the trading community**

> ⚠️ **Disclaimer**: This software is for educational purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.
