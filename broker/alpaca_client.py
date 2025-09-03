import logging
import os

import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY


def get_alpaca_client():
    return tradeapi.REST(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        base_url="https://paper-api.alpaca.markets"
    )

def place_order(client, symbol, side, qty=1):
    try:
        position = client.get_position(symbol)
        if side.lower() == 'buy' and float(position.qty) > 0:
            logging.warning(f"Already holding {symbol}, skipping BUY to avoid wash trade.")
            return None
        if side.lower() == 'sell' and float(position.qty) == 0:
            logging.warning(f"No holdings of {symbol}, skipping SELL.")
            return None
    except:
        if side.lower() == 'sell':
            logging.warning(f"No holdings of {symbol}, skipping SELL.")
            return None

    return client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side.lower(),
        type='market',
        time_in_force='gtc'
    )
