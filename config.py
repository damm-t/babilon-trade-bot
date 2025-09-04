# config.py

import os

from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Threshold defaults (can be overridden via .env or Streamlit sliders)
POSITIVE_THRESHOLD = float(os.getenv("POSITIVE_THRESHOLD", 0.65))
NEGATIVE_THRESHOLD = float(os.getenv("NEGATIVE_THRESHOLD", 0.7))

# Default stock symbols for analysis
STOCK_SYMBOLS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]