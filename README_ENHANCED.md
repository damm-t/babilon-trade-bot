# 🚀 Babilon AI Stock Trader - Enhanced Version

An advanced AI-powered stock sentiment trading system with multiple interfaces and real-time analysis capabilities.

## ✨ New Features

### 🎯 **Dynamic Stock Selection**
- Search and select any stock symbol
- Real-time stock data fetching via yfinance
- Comprehensive stock information display

### 🔌 **RESTful API Backend**
- FastAPI-based backend with full documentation
- Multiple endpoints for different analysis types
- CORS-enabled for frontend integration

### 🖥️ **Multiple User Interfaces**
1. **Enhanced Streamlit App** - Interactive dashboard with API integration
2. **Modern Web App** - Pure HTML/JS frontend with real-time updates
3. **Command Line Interface** - For automated trading and testing

### 📊 **Advanced Analytics**
- Real-time sentiment analysis using FinBERT
- Interactive candlestick charts with trading signals
- News article integration and analysis
- Configurable trading thresholds

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  Streamlit UI   │    │   CLI Interface │
│   (HTML/JS)     │    │   (Python)      │    │   (Python)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     FastAPI Backend       │
                    │   (RESTful API Server)    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Analysis Engine        │
                    │  ┌─────────────────────┐  │
                    │  │   FinBERT Model     │  │
                    │  │  (Sentiment Analysis)│  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  Trading Logic      │  │
                    │  │  (BUY/SELL/HOLD)    │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Data Sources          │
                    │  ┌─────────────────────┐  │
                    │  │   yfinance API      │  │
                    │  │  (Stock Data)       │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │   News API          │  │
                    │  │  (News Articles)    │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │   Alpaca API        │  │
                    │  │  (Paper Trading)    │  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a `.env` file:
```bash
# Alpaca Trading API (for paper trading)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# News API (optional - for real news fetching)
NEWS_API_KEY=your_news_api_key

# Trading Thresholds
POSITIVE_THRESHOLD=0.65
NEGATIVE_THRESHOLD=0.7
```

### 3. Start All Services
```bash
python start_app.py
```

This will start:
- FastAPI backend on http://localhost:8000
- Streamlit UI on http://localhost:8501
- Web app on http://localhost:8080

### 4. Test the System
```bash
python test_api.py
```

## 📖 API Documentation

### Available Endpoints

#### 🔍 Stock Search
```http
GET /stocks/search?query=AAPL
```
Returns list of matching stock symbols.

#### 📊 Stock Data
```http
GET /stocks/{symbol}/data
```
Returns comprehensive stock data including prices, volume, and chart data.

#### 📰 Stock News
```http
GET /stocks/{symbol}/news?days=7
```
Returns recent news articles for the stock.

#### 🤖 Stock Analysis
```http
POST /stocks/analyze
Content-Type: application/json

{
    "symbol": "AAPL",
    "positive_threshold": 0.65,
    "negative_threshold": 0.7,
    "enable_trading": false
}
```

#### 📝 News Analysis
```http
POST /news/analyze
Content-Type: application/json

{
    "symbol": "AAPL",
    "news_text": "Apple reports record earnings...",
    "positive_threshold": 0.65,
    "negative_threshold": 0.7,
    "enable_trading": false
}
```

### Interactive API Documentation
Visit http://localhost:8000/docs for Swagger UI documentation.

## 🖥️ User Interfaces

### 1. Enhanced Streamlit App
```bash
streamlit run streamlit_enhanced.py
```
Features:
- Stock search and selection
- Real-time analysis
- Interactive charts
- News integration
- Trading controls

### 2. Modern Web App
Open `web_app/index.html` in your browser or serve it:
```bash
cd web_app
python -m http.server 8080
```
Features:
- Responsive design
- Real-time updates
- Interactive charts
- Mobile-friendly

### 3. Command Line Interface
```bash
python main.py
```
For automated trading and testing.

## 🔧 Configuration

### Trading Thresholds
- **Positive Threshold**: Minimum confidence for BUY signals (default: 0.65)
- **Negative Threshold**: Minimum confidence for SELL signals (default: 0.7)

### API Keys
- **Alpaca**: Required for paper trading
- **News API**: Optional, for real news fetching (uses sample data if not provided)

## 📊 Analysis Workflow

1. **Stock Selection**: User searches and selects a stock symbol
2. **Data Fetching**: System fetches real-time stock data and recent news
3. **Sentiment Analysis**: FinBERT analyzes news sentiment and confidence
4. **Trading Decision**: Rule-based engine generates BUY/SELL/HOLD signal
5. **Execution**: Optional paper trade execution via Alpaca
6. **Visualization**: Interactive charts with trading signals
7. **Logging**: All decisions and trades are logged

## 🧪 Testing

### Run API Tests
```bash
python test_api.py
```

### Manual Testing
1. Start the API: `python api.py`
2. Test endpoints using curl or Postman
3. Use the web interfaces for interactive testing

## 📁 Project Structure

```
babilon-trade-bot/
├── api.py                      # FastAPI backend server
├── streamlit_enhanced.py       # Enhanced Streamlit UI
├── start_app.py               # Startup script for all services
├── test_api.py                # API testing script
├── web_app/                   # Modern web frontend
│   ├── index.html
│   └── app.js
├── broker/                    # Trading integration
│   └── alpaca_client.py
├── model/                     # AI models
│   └── sentiment_model.py
├── logic/                     # Trading logic
│   └── trade_decision.py
├── visualization/             # Chart generation
│   ├── chart_data.py
│   └── export_plot.py
├── data/                      # Logs and data
│   └── logs.csv
└── requirements.txt           # Dependencies
```

## 🔒 Security Notes

- All trading is done via Alpaca's paper trading API (no real money)
- API keys should be stored in environment variables
- CORS is enabled for development (configure properly for production)
- Input validation is implemented for all API endpoints

## 🚀 Production Deployment

For production deployment:

1. **Configure CORS** properly in `api.py`
2. **Set up proper authentication** for API endpoints
3. **Use environment variables** for all sensitive data
4. **Set up logging** and monitoring
5. **Configure reverse proxy** (nginx) for web serving
6. **Use HTTPS** for all communications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with all applicable regulations when using for actual trading.

## 🆘 Support

For issues and questions:
1. Check the API documentation at http://localhost:8000/docs
2. Run the test suite: `python test_api.py`
3. Check the logs in `data/logs.csv`
4. Review the console output for error messages

---

**Happy Trading! 📈🤖**
