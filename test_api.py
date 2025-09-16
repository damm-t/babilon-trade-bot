#!/usr/bin/env python3
"""
Test script for Babilon Trade Bot API
Tests the complete workflow from stock selection to analysis
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def test_stock_search():
    """Test stock search functionality"""
    print("\n🔍 Testing stock search...")
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/search", params={"query": "AAPL"})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['stocks'])} stocks: {data['stocks']}")
            return True
        else:
            print(f"❌ Search failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_stock_data():
    """Test stock data fetching"""
    print("\n📊 Testing stock data fetching...")
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/AAPL/data")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stock data for AAPL:")
            print(f"   Current Price: ${data['current_price']:.2f}")
            print(f"   Reference Price: ${data['reference_price']:.2f}")
            print(f"   Price Change: ${data['price_change']:.2f} ({data['price_change_percent']:.2f}%)")
            print(f"   Volume: {data['volume']:,}")
            return True
        else:
            print(f"❌ Data fetch failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return False

def test_stock_analysis():
    """Test complete stock analysis"""
    print("\n🤖 Testing stock analysis...")
    try:
        payload = {
            "symbol": "AAPL",
            "positive_threshold": 0.65,
            "negative_threshold": 0.7,
            "enable_trading": False
        }
        
        response = requests.post(f"{API_BASE_URL}/stocks/analyze", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis completed for {data['symbol']}:")
            print(f"   Sentiment: {data['sentiment']}")
            print(f"   Confidence: {data['score']:.2f}")
            print(f"   Decision: {data['decision']}")
            print(f"   Current Price: ${data['current_price']:.2f}")
            print(f"   Reference Price: ${data['reference_price']:.2f}")
            
            if data.get('news_articles'):
                print(f"   News Articles: {len(data['news_articles'])} found")
            
            if data.get('chart_data'):
                candles = data['chart_data'].get('candles', [])
                signals = data['chart_data'].get('signals', [])
                print(f"   Chart Data: {len(candles)} candles, {len(signals)} signals")
            
            return True
        else:
            print(f"❌ Analysis failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def test_news_analysis():
    """Test custom news analysis"""
    print("\n📰 Testing news analysis...")
    try:
        payload = {
            "symbol": "TSLA",
            "news_text": "Tesla reports record quarterly earnings and strong delivery numbers",
            "positive_threshold": 0.65,
            "negative_threshold": 0.7,
            "enable_trading": False
        }
        
        response = requests.post(f"{API_BASE_URL}/news/analyze", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ News analysis completed for {data['symbol']}:")
            print(f"   Sentiment: {data['sentiment']}")
            print(f"   Confidence: {data['score']:.2f}")
            print(f"   Decision: {data['decision']}")
            return True
        else:
            print(f"❌ News analysis failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ News analysis error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Babilon Trade Bot API Test Suite")
    print("=" * 40)
    
    # Test API connection
    if not test_api_connection():
        print("\n❌ Cannot proceed - API is not running")
        print("Please start the API with: python api.py")
        sys.exit(1)
    
    # Run all tests
    tests = [
        test_stock_search,
        test_stock_data,
        test_stock_analysis,
        test_news_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
        print("\n🌐 You can now use:")
        print("   • Streamlit UI: streamlit run streamlit_enhanced.py")
        print("   • Web App: Open web_app/index.html in browser")
        print("   • API Docs: http://localhost:8000/docs")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
