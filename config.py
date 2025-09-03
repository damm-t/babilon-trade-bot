import os

from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "NVDA")
TRADE_THRESHOLD = 0.7