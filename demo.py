#!/usr/bin/env python3
"""
Demonstration script for Babilon Trade Bot
Shows the complete workflow from stock selection to analysis
"""

import requests
import json
import time
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def demo_complete_workflow():
    """Demonstrate the complete trading workflow"""
    
    print_header("BABILON AI STOCK TRADER - COMPLETE WORKFLOW DEMO")
    
    print("\n🚀 This demo will show you:")
    print("   1. Stock search and selection")
    print("   2. Real-time data fetching")
    print("   3. News analysis and sentiment scoring")
    print("   4. Trading decision generation")
    print("   5. Chart visualization with signals")
    print("   6. API integration examples")
    
    input("\nPress Enter to start the demo...")
    
    # Step 1: Stock Search
    print_step(1, "Stock Search and Selection")
    
    print("🔍 Searching for popular tech stocks...")
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/search", params={"query": "tech"})
        if response.status_code == 200:
            stocks = response.json()["stocks"]
            print(f"✅ Found stocks: {', '.join(stocks)}")
            selected_stock = stocks[0] if stocks else "AAPL"
            print(f"🎯 Selected: {selected_stock}")
        else:
            print("❌ Search failed, using default: AAPL")
            selected_stock = "AAPL"
    except Exception as e:
        print(f"❌ Error: {e}, using default: AAPL")
        selected_stock = "AAPL"
    
    # Step 2: Stock Data Fetching
    print_step(2, "Real-time Stock Data Fetching")
    
    print(f"📊 Fetching data for {selected_stock}...")
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/{selected_stock}/data")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stock Data Retrieved:")
            print(f"   💰 Current Price: ${data['current_price']:.2f}")
            print(f"   📈 Reference Price: ${data['reference_price']:.2f}")
            print(f"   📊 Price Change: ${data['price_change']:.2f} ({data['price_change_percent']:.2f}%)")
            print(f"   📦 Volume: {data['volume']:,}")
            if data.get('market_cap'):
                print(f"   🏢 Market Cap: ${data['market_cap']:,}")
        else:
            print(f"❌ Data fetch failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # Step 3: News Analysis
    print_step(3, "News Analysis and Sentiment Scoring")
    
    print(f"📰 Fetching recent news for {selected_stock}...")
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/{selected_stock}/news")
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data.get('articles', [])
            print(f"✅ Found {len(articles)} news articles")
            
            if articles:
                print("\n📄 Recent News Headlines:")
                for i, article in enumerate(articles[:3], 1):
                    print(f"   {i}. {article['title']}")
                    print(f"      Source: {article['source']}")
                    print(f"      Published: {article['publishedAt']}")
                    print()
        else:
            print("⚠️  News fetch failed, proceeding with analysis...")
    except Exception as e:
        print(f"⚠️  News error: {e}, proceeding with analysis...")
    
    # Step 4: AI Analysis
    print_step(4, "AI Sentiment Analysis and Trading Decision")
    
    print("🤖 Running AI analysis...")
    try:
        payload = {
            "symbol": selected_stock,
            "positive_threshold": 0.65,
            "negative_threshold": 0.7,
            "enable_trading": False  # Demo mode - no actual trading
        }
        
        response = requests.post(f"{API_BASE_URL}/stocks/analyze", json=payload)
        if response.status_code == 200:
            analysis = response.json()
            
            print(f"✅ Analysis Complete for {analysis['symbol']}:")
            print(f"   🧠 Sentiment: {analysis['sentiment'].upper()}")
            print(f"   📊 Confidence: {analysis['score']:.2f} ({analysis['score']*100:.1f}%)")
            print(f"   🎯 Decision: {analysis['decision']}")
            print(f"   💰 Current Price: ${analysis['current_price']:.2f}")
            print(f"   📈 Reference Price: ${analysis['reference_price']:.2f}")
            
            # Display decision reasoning
            if analysis['decision'] == 'BUY':
                print(f"   💡 Reasoning: Positive sentiment ({analysis['score']:.2f}) with price below reference")
            elif analysis['decision'] == 'SELL':
                print(f"   💡 Reasoning: Negative sentiment ({analysis['score']:.2f}) with price above reference")
            else:
                print(f"   💡 Reasoning: Sentiment confidence below threshold or price conditions not met")
            
            # Show news articles if available
            if analysis.get('news_articles'):
                print(f"\n📰 Analyzed {len(analysis['news_articles'])} news articles")
            
            # Show chart data
            if analysis.get('chart_data'):
                candles = analysis['chart_data'].get('candles', [])
                signals = analysis['chart_data'].get('signals', [])
                print(f"📊 Chart Data: {len(candles)} price points, {len(signals)} trading signals")
            
        else:
            print(f"❌ Analysis failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return
    
    # Step 5: Custom News Analysis
    print_step(5, "Custom News Analysis Example")
    
    custom_news = f"{selected_stock} reports strong quarterly earnings and positive outlook for next quarter"
    print(f"📝 Analyzing custom news: '{custom_news}'")
    
    try:
        payload = {
            "symbol": selected_stock,
            "news_text": custom_news,
            "positive_threshold": 0.65,
            "negative_threshold": 0.7,
            "enable_trading": False
        }
        
        response = requests.post(f"{API_BASE_URL}/news/analyze", json=payload)
        if response.status_code == 200:
            analysis = response.json()
            print(f"✅ Custom News Analysis:")
            print(f"   🧠 Sentiment: {analysis['sentiment'].upper()}")
            print(f"   📊 Confidence: {analysis['score']:.2f}")
            print(f"   🎯 Decision: {analysis['decision']}")
        else:
            print(f"❌ Custom analysis failed: {response.text}")
    except Exception as e:
        print(f"❌ Custom analysis error: {e}")
    
    # Step 6: API Integration Examples
    print_step(6, "API Integration Examples")
    
    print("🔌 Here are the API endpoints you can use:")
    print(f"   • Stock Search: GET {API_BASE_URL}/stocks/search?query=AAPL")
    print(f"   • Stock Data: GET {API_BASE_URL}/stocks/AAPL/data")
    print(f"   • Stock News: GET {API_BASE_URL}/stocks/AAPL/news")
    print(f"   • Stock Analysis: POST {API_BASE_URL}/stocks/analyze")
    print(f"   • News Analysis: POST {API_BASE_URL}/news/analyze")
    print(f"   • API Docs: {API_BASE_URL}/docs")
    
    # Final Summary
    print_header("DEMO COMPLETE - NEXT STEPS")
    
    print("\n🎉 Congratulations! You've seen the complete workflow.")
    print("\n🚀 To use the system:")
    print("   1. Start the API: python api.py")
    print("   2. Open Streamlit UI: streamlit run streamlit_enhanced.py")
    print("   3. Open Web App: Open web_app/index.html in browser")
    print("   4. Run tests: python test_api.py")
    
    print("\n📚 Available Interfaces:")
    print("   • 🌐 Web App: http://localhost:8080 (if serving)")
    print("   • 📊 Streamlit: http://localhost:8501")
    print("   • 🔌 API Docs: http://localhost:8000/docs")
    
    print("\n🔧 Configuration:")
    print("   • Set up .env file with your API keys")
    print("   • Adjust trading thresholds as needed")
    print("   • Enable live trading when ready")
    
    print("\n⚠️  Remember: This is for educational purposes only!")
    print("   Always use paper trading for testing.")

def main():
    """Main demo function"""
    print("🎯 Babilon AI Stock Trader - Interactive Demo")
    print("This demo requires the API to be running on localhost:8000")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding correctly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start it first:")
        print("   python api.py")
        return
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return
    
    print("✅ API is running and ready!")
    
    # Run the demo
    demo_complete_workflow()

if __name__ == "__main__":
    main()
