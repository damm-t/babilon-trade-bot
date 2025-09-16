from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
import requests
from datetime import datetime, timedelta
import json
import os

from model.sentiment_model import analyze_sentiment
from logic.trade_decision import generate_trade_signal
from broker.alpaca_client import get_alpaca_client, place_order
from visualization.chart_data import format_chart_data, to_json

app = FastAPI(title="Babilon Trade Bot API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class StockAnalysisRequest(BaseModel):
    symbol: str
    positive_threshold: float = 0.65
    negative_threshold: float = 0.7
    enable_trading: bool = False

class NewsAnalysisRequest(BaseModel):
    symbol: str
    news_text: str
    positive_threshold: float = 0.65
    negative_threshold: float = 0.7
    enable_trading: bool = False

class AnalysisResponse(BaseModel):
    symbol: str
    sentiment: str
    score: float
    decision: str
    current_price: Optional[float] = None
    reference_price: Optional[float] = None
    chart_data: Optional[dict] = None
    news_articles: Optional[List[dict]] = None
    timestamp: str

class StockDataResponse(BaseModel):
    symbol: str
    current_price: float
    reference_price: float
    price_change: float
    price_change_percent: float
    volume: int
    market_cap: Optional[float] = None
    chart_data: dict

# News API configuration (using NewsAPI as example - you can replace with other sources)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_stock_data(symbol: str, period: str = "3mo") -> dict:
    """Fetch comprehensive stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        hist = ticker.history(period=period)
        if hist.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Get current info
        info = ticker.info
        
        current_price = float(hist["Close"].iloc[-1])
        reference_price = float(hist["Close"].mean())
        price_change = current_price - hist["Close"].iloc[-2] if len(hist) > 1 else 0
        price_change_percent = (price_change / hist["Close"].iloc[-2]) * 100 if len(hist) > 1 else 0
        
        # Prepare chart data
        recent_hist = hist.tail(60)  # Last 60 days
        prices = []
        for idx, row in recent_hist.iterrows():
            prices.append({
                "timestamp": idx.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        
        chart_data = format_chart_data(prices)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "reference_price": reference_price,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "volume": int(hist["Volume"].iloc[-1]),
            "market_cap": info.get("marketCap"),
            "chart_data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {symbol}: {str(e)}")

def fetch_news_articles(symbol: str, days: int = 7) -> List[dict]:
    """Fetch recent news articles for a stock symbol"""
    if not NEWS_API_KEY:
        # Return sample news if no API key
        return [
            {
                "title": f"{symbol} shows strong performance in recent trading",
                "description": f"Analysis of {symbol} market trends and investor sentiment",
                "url": f"https://example.com/news/{symbol.lower()}",
                "publishedAt": datetime.now().isoformat(),
                "source": "Sample News"
            }
        ]
    
    try:
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        params = {
            "q": f"{symbol} stock",
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("articles", [])
        
        # Format articles
        formatted_articles = []
        for article in articles[:10]:  # Limit to 10 most recent
            if article.get("title") and article.get("description"):
                formatted_articles.append({
                    "title": article["title"],
                    "description": article["description"],
                    "url": article["url"],
                    "publishedAt": article["publishedAt"],
                    "source": article["source"]["name"]
                })
        
        return formatted_articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

@app.get("/")
async def root():
    return {"message": "Babilon Trade Bot API", "version": "1.0.0"}

@app.get("/stocks/search")
async def search_stocks(query: str):
    """Search for stock symbols (basic implementation)"""
    # This is a simple implementation - you might want to use a more comprehensive stock search API
    common_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC",
        "ORCL", "CRM", "ADBE", "PYPL", "UBER", "LYFT", "SNAP", "TWTR", "SQ", "ROKU"
    ]
    
    matching_stocks = [stock for stock in common_stocks if query.upper() in stock]
    return {"stocks": matching_stocks}

@app.get("/stocks/{symbol}/data")
async def get_stock_data(symbol: str):
    """Get comprehensive stock data including price, chart data, and basic info"""
    try:
        data = fetch_stock_data(symbol)
        return StockDataResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stocks/{symbol}/news")
async def get_stock_news(symbol: str, days: int = 7):
    """Get recent news articles for a stock"""
    articles = fetch_news_articles(symbol, days)
    return {"symbol": symbol, "articles": articles}

@app.post("/stocks/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a stock with news sentiment and generate trading signals"""
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(request.symbol)
        
        # Fetch news articles
        news_articles = fetch_news_articles(request.symbol)
        
        # Combine news articles for sentiment analysis
        combined_news = " ".join([article["title"] + " " + article["description"] for article in news_articles])
        
        if not combined_news.strip():
            combined_news = f"{request.symbol} stock analysis and market trends"
        
        # Analyze sentiment
        sentiment, score = analyze_sentiment(combined_news)
        
        # Generate trading decision
        decision = generate_trade_signal(
            sentiment.lower(),
            score,
            positive_threshold=request.positive_threshold,
            negative_threshold=request.negative_threshold,
            current_price=stock_data["current_price"],
            reference_price=stock_data["reference_price"]
        )
        
        # Execute trade if enabled
        if request.enable_trading and decision in ["BUY", "SELL"]:
            try:
                client = get_alpaca_client()
                place_order(client, request.symbol, decision)
            except Exception as e:
                print(f"Trading error: {e}")
        
        # Add signals to chart data
        signals = []
        if decision in ["BUY", "SELL"]:
            signals.append({
                "timestamp": datetime.now().isoformat(),
                "price": stock_data["current_price"],
                "type": decision
            })
        
        chart_data = stock_data["chart_data"].copy()
        chart_data["signals"] = signals
        
        return AnalysisResponse(
            symbol=request.symbol,
            sentiment=sentiment,
            score=score,
            decision=decision,
            current_price=stock_data["current_price"],
            reference_price=stock_data["reference_price"],
            chart_data=chart_data,
            news_articles=news_articles,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/news/analyze", response_model=AnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest):
    """Analyze custom news text for a specific stock"""
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(request.symbol)
        
        # Analyze sentiment
        sentiment, score = analyze_sentiment(request.news_text)
        
        # Generate trading decision
        decision = generate_trade_signal(
            sentiment.lower(),
            score,
            positive_threshold=request.positive_threshold,
            negative_threshold=request.negative_threshold,
            current_price=stock_data["current_price"],
            reference_price=stock_data["reference_price"]
        )
        
        # Execute trade if enabled
        if request.enable_trading and decision in ["BUY", "SELL"]:
            try:
                client = get_alpaca_client()
                place_order(client, request.symbol, decision)
            except Exception as e:
                print(f"Trading error: {e}")
        
        # Add signals to chart data
        signals = []
        if decision in ["BUY", "SELL"]:
            signals.append({
                "timestamp": datetime.now().isoformat(),
                "price": stock_data["current_price"],
                "type": decision
            })
        
        chart_data = stock_data["chart_data"].copy()
        chart_data["signals"] = signals
        
        return AnalysisResponse(
            symbol=request.symbol,
            sentiment=sentiment,
            score=score,
            decision=decision,
            current_price=stock_data["current_price"],
            reference_price=stock_data["reference_price"],
            chart_data=chart_data,
            news_articles=None,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
