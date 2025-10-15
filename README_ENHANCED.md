# Babilon Trade Bot - Enhanced AI Trading Assistant

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

An advanced AI-powered trading bot that combines sentiment analysis using FinBERT with sophisticated risk management and technical analysis to make informed trading decisions.

## 🚀 Features

### Core Functionality
- **AI Sentiment Analysis**: Uses FinBERT (financial BERT) for accurate sentiment analysis of news and text
- **Intelligent Trading Signals**: Combines sentiment with technical indicators for robust decision making
- **Paper Trading**: Safe testing environment using Alpaca's paper trading API
- **Risk Management**: Built-in position sizing, stop losses, and portfolio risk controls
- **Real-time Analysis**: Processes market data and news in real-time

### Enhanced Features
- **Technical Analysis**: Moving averages, volatility analysis, and trend detection
- **Error Handling**: Comprehensive error handling with retry logic and graceful failures
- **Portfolio Management**: Advanced position sizing and risk controls
- **Multiple Interfaces**: CLI, Streamlit web app, and REST API
- **Extensive Logging**: Detailed logging for monitoring and debugging
- **Configurable**: Environment-based configuration for all parameters

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Risk Management](#risk-management)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Alpaca account (paper trading recommended)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/babilon-trade-bot.git
   cd babilon-trade-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env_template.txt .env
   # Edit .env with your API keys and preferences
   ```

## ⚙️ Configuration

### Environment Variables

Copy `env_template.txt` to `.env` and configure:

```bash
# Required: Alpaca API credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# Trading parameters
POSITIVE_THRESHOLD=0.65
NEGATIVE_THRESHOLD=0.70
STOCK_SYMBOL=AAPL

# Risk management
MAX_POSITION_SIZE=1000.0
MAX_PORTFOLIO_RISK=0.02
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.10
```

### Trading Modes

#### Conservative Mode
```bash
POSITIVE_THRESHOLD=0.75
NEGATIVE_THRESHOLD=0.80
MAX_PORTFOLIO_RISK=0.01
STOP_LOSS_PERCENTAGE=0.03
```

#### Aggressive Mode
```bash
POSITIVE_THRESHOLD=0.55
NEGATIVE_THRESHOLD=0.60
MAX_PORTFOLIO_RISK=0.03
STOP_LOSS_PERCENTAGE=0.08
```

## 🚀 Quick Start

### 1. Basic Usage

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

### 2. Command Line Interface

```bash
# Run with default settings
python main.py

# Run CLI with custom parameters
python cli.py --symbol TSLA --news "Tesla announces new factory"
```

### 3. Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

## 📖 Usage

### Programmatic Usage

```python
from main import BabilonTradeBot
from broker.alpaca_client import AlpacaClient
from logic.trade_decision import generate_trade_signal

# Initialize components
bot = BabilonTradeBot("NVDA")
client = AlpacaClient()

# Get market data
current_price = bot.get_current_price()
price_history = bot.get_price_history()

# Analyze sentiment
sentiment, score = analyze_sentiment("Nvidia's AI chips in high demand")

# Generate trade signal
signal = generate_trade_signal(
    sentiment=sentiment,
    score=score,
    current_price=current_price,
    price_history=price_history
)

print(f"Action: {signal.action}")
print(f"Confidence: {signal.confidence:.2f}")
print(f"Stop Loss: ${signal.stop_loss:.2f}")
```

### REST API Usage

Start the API server:
```bash
python api.py
```

Make requests:
```bash
# Analyze sentiment
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple stock surges on strong earnings", "symbol": "AAPL"}'

# Get portfolio status
curl "http://localhost:8000/portfolio"
```

## 🏗 Architecture

```
babilon-trade-bot/
├── broker/           # Trading API integration
│   ├── alpaca_client.py
│   └── __init__.py
├── logic/            # Trading logic and decisions
│   ├── trade_decision.py
│   └── __init__.py
├── model/            # AI/ML models
│   ├── sentiment_model.py
│   └── __init__.py
├── data/             # Data storage
├── tests/            # Test suite
├── visualization/    # Charts and plots
├── main.py           # Main application
├── api.py            # REST API
├── streamlit_app.py  # Web interface
└── config.py         # Configuration
```

### Key Components

1. **AlpacaClient**: Enhanced broker client with error handling
2. **TradeSignal**: Structured trading signals with risk management
3. **BabilonTradeBot**: Main bot class orchestrating all components
4. **Sentiment Analysis**: FinBERT-based sentiment analysis
5. **Risk Management**: Position sizing and portfolio controls

## 🛡 Risk Management

### Built-in Protections

1. **Position Sizing**: Automatic calculation based on portfolio value
2. **Stop Losses**: Configurable stop loss levels for all trades
3. **Take Profits**: Automatic profit-taking at specified levels
4. **Portfolio Limits**: Maximum risk per trade and total exposure
5. **Wash Trade Prevention**: Avoids buying when already holding position

### Risk Parameters

```python
# Maximum position size in dollars
MAX_POSITION_SIZE = 1000.0

# Maximum portfolio risk per trade (2%)
MAX_PORTFOLIO_RISK = 0.02

# Stop loss percentage (5%)
STOP_LOSS_PERCENTAGE = 0.05

# Take profit percentage (10%)
TAKE_PROFIT_PERCENTAGE = 0.10
```

## 📊 API Reference

### BabilonTradeBot Class

```python
class BabilonTradeBot:
    def __init__(self, symbol: str = "AAPL")
    def analyze_and_trade(self, news_text: str) -> dict
    def get_current_price(self) -> Optional[float]
    def get_price_history(self, limit: int = 20) -> list
    def get_portfolio_status(self) -> dict
```

### TradeSignal Class

```python
class TradeSignal:
    def __init__(self, action: str, confidence: float, reasoning: str, 
                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None)
    def to_dict(self) -> Dict[str, Any]
```

### AlpacaClient Class

```python
class AlpacaClient:
    def __init__(self)
    def place_order(self, symbol: str, side: str, qty: Optional[int] = None, 
                   current_price: Optional[float] = None) -> Optional[Dict[str, Any]]
    def get_account_info(self) -> Dict[str, Any]
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]
    def calculate_position_size(self, symbol: str, current_price: float) -> int
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_trade_decision.py

# Run with verbose output
pytest -v
```

### Test Structure

```bash
tests/
├── test_trade_decision.py    # Trading logic tests
├── test_alpaca_client.py     # Broker integration tests
├── test_sentiment_model.py   # AI model tests
└── test_integration.py       # End-to-end tests
```

### Writing Tests

```python
def test_buy_signal():
    signal = generate_trade_signal(
        sentiment="positive",
        score=0.8,
        current_price=100,
        reference_price=110
    )
    assert signal.action == "BUY"
    assert signal.confidence > 0.7
```

## 🚀 Deployment

### Docker Deployment

1. **Build Docker image**
   ```bash
   docker build -t babilon-trade-bot .
   ```

2. **Run container**
   ```bash
   docker run -d --env-file .env babilon-trade-bot
   ```

3. **Docker Compose**
   ```bash
   docker-compose up -d
   ```

### Production Deployment

1. **Environment Setup**
   ```bash
   # Use production environment variables
   cp .env.production .env
   
   # Set up logging
   export LOG_LEVEL=INFO
   ```

2. **Process Management**
   ```bash
   # Using systemd
   sudo systemctl start babilon-trade-bot
   
   # Using PM2
   pm2 start main.py --name babilon-bot
   ```

3. **Monitoring**
   ```bash
   # Check logs
   tail -f logs/babilon-bot.log
   
   # Monitor system resources
   htop
   ```

## 🤝 Contributing

### Development Setup

1. **Fork and clone**
   ```bash
   git fork https://github.com/your-username/babilon-trade-bot.git
   cd babilon-trade-bot
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install black flake8 mypy pytest
   ```

3. **Run pre-commit checks**
   ```bash
   black .
   flake8 .
   mypy .
   pytest
   ```

### Contributing Guidelines

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Use meaningful commit messages
5. Submit pull requests for review

## 🔒 Security

### API Key Management

1. **Never commit API keys**
   ```bash
   # .gitignore already includes .env
   echo ".env" >> .gitignore
   ```

2. **Use environment variables**
   ```python
   import os
   api_key = os.getenv("ALPACA_API_KEY")
   ```

3. **Rotate keys regularly**
   - Change API keys monthly
   - Monitor for unauthorized usage
   - Use different keys for different environments

### Security Best Practices

1. **Paper Trading Only**: Use paper trading for testing
2. **Limit Permissions**: Use minimal required API permissions
3. **Monitor Activity**: Regularly check trading activity
4. **Secure Deployment**: Use HTTPS and secure hosting
5. **Regular Updates**: Keep dependencies updated

## 🔧 Troubleshooting

### Common Issues

#### 1. API Connection Errors
```bash
# Check API keys
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY

# Test connection
python -c "from broker.alpaca_client import AlpacaClient; AlpacaClient()"
```

#### 2. Sentiment Analysis Failures
```bash
# Check model download
python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert')"
```

#### 3. Trading Errors
```bash
# Check account status
python -c "from broker.alpaca_client import AlpacaClient; print(AlpacaClient().get_account_info())"
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

### Log Files

Check log files for detailed error information:
```bash
tail -f logs/babilon-bot.log
grep "ERROR" logs/babilon-bot.log
```

## 📈 Performance Optimization

### Optimization Tips

1. **Caching**: Enable model caching for faster sentiment analysis
2. **Batch Processing**: Process multiple symbols in batches
3. **Async Operations**: Use async/await for I/O operations
4. **Connection Pooling**: Reuse API connections

### Monitoring

```python
# Monitor performance
import time
start_time = time.time()
result = bot.analyze_and_trade(news)
execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.2f}s")
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always do your own research and consider your risk tolerance before trading.**

## 🙏 Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for the trading API
- [Hugging Face](https://huggingface.co/) for the FinBERT model
- [Streamlit](https://streamlit.io/) for the web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/babilon-trade-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/babilon-trade-bot/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/babilon-trade-bot/wiki)

---

**Made with ❤️ for the trading community**