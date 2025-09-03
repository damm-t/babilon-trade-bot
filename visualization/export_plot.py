from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def export_candlestick(prices, signals=None, filename="chart.png"):
    """
    Export candlestick chart with optional BUY/SELL signals.

    Args:
        prices (list[dict]): List of dicts with keys: timestamp, open, high, low, close
        signals (list[dict]): Optional list of dicts with keys: timestamp, price, type
        filename (str): Output PNG file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert timestamps
    dates = [datetime.fromisoformat(p["timestamp"]) for p in prices]
    opens = [p["open"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    closes = [p["close"] for p in prices]

    # Plot candlesticks manually
    for i in range(len(dates)):
        color = "green" if closes[i] >= opens[i] else "red"
        ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color=color)
        ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=6)

    # Plot signals
    if signals:
        for s in signals:
            d = datetime.fromisoformat(s["timestamp"])
            ax.scatter(
                d, s["price"],
                color="green" if s["type"] == "BUY" else "red",
                marker="o", s=60, zorder=5
            )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.set_title("Babilon Candlestick Chart")
    ax.set_ylabel("Price")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename