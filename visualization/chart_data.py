import json
from datetime import datetime


def format_chart_data(prices, signals=None):
    """
    Convert OHLCV price data + AI signals into KLineChart JSON format.
    
    Args:
        prices (list[dict]): List of dicts with keys: timestamp, open, high, low, close, volume
        signals (list[dict]): Optional list of dicts with keys: timestamp, price, type (BUY/SELL)

    Returns:
        dict: JSON-serializable chart data
    """
    candles = []
    for p in prices:
        candles.append({
            "timestamp": int(datetime.fromisoformat(p["timestamp"]).timestamp() * 1000),
            "open": p["open"],
            "high": p["high"],
            "low": p["low"],
            "close": p["close"],
            "volume": p["volume"],
        })
    
    return {
        "candles": candles,
        "signals": signals or []
    }

def to_json(prices, signals=None):
    """Return chart data as JSON string."""
    return json.dumps(format_chart_data(prices, signals))
