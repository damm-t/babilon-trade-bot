import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf


LOGS_FILE = os.path.join("data", "logs.csv")


@dataclass
class TradeEvent:
    timestamp: pd.Timestamp
    symbol: str
    decision: str  # BUY/SELL/HOLD


def _load_logs() -> pd.DataFrame:
    if not os.path.exists(LOGS_FILE):
        return pd.DataFrame(columns=["timestamp", "stock_symbol", "decision"])  # empty
    df = pd.read_csv(LOGS_FILE)
    # Normalize columns if needed
    if "stock_symbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "stock_symbol"})
    # Parse time
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])  # drop unparsable rows
    return df


def _fetch_close_series(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.Series]:
    closes: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            data = yf.download(sym, start=start.date(), end=(end + pd.Timedelta(days=1)).date(), progress=False)
            if not data.empty and "Close" in data.columns:
                closes[sym] = data["Close"].astype(float)
        except Exception:
            continue
    return closes


def simulate_from_logs(lookback_days: int = 30) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simulate a simple strategy from historical BUY/SELL logs.

    Rules:
    - BUY switches position to long 1 unit at next day's close
    - SELL closes any long position at next day's close (go flat)
    - HOLD does nothing

    Returns:
    - trades_df: list of executed simulated trades with P&L
    - summary: dict with win_rate, avg_return_pct, total_return_pct
    """
    df = _load_logs()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "action", "price", "pnl_pct"]), {
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
            "num_trades": 0,
        }

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "action", "price", "pnl_pct"]), {
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
            "num_trades": 0,
        }

    df = df.sort_values("timestamp")
    symbols = sorted(df["stock_symbol"].dropna().unique().tolist())
    start_time = df["timestamp"].min() - pd.Timedelta(days=2)
    end_time = pd.Timestamp.utcnow()

    close_series = _fetch_close_series(symbols, start_time, end_time)

    executed_rows: List[dict] = []
    trade_returns: List[float] = []

    # Track per-symbol position
    for sym in symbols:
        sym_events = df[df["stock_symbol"] == sym]
        position_open_price = None

        # For next-day execution, map timestamps to next available close
        sym_close = close_series.get(sym)
        if sym_close is None or sym_close.empty:
            continue

        for _, r in sym_events.iterrows():
            action = str(r.get("decision", "")).upper()
            ts = pd.to_datetime(r["timestamp"]) + pd.Timedelta(days=1)
            # find nearest index in close series at or after ts date
            exec_price = None
            for idx_ts, price in sym_close.items():
                if pd.Timestamp(idx_ts) >= ts.normalize():
                    exec_price = float(price)
                    exec_time = pd.Timestamp(idx_ts)
                    break
            if exec_price is None:
                continue

            if action == "BUY":
                if position_open_price is None:
                    position_open_price = exec_price
                    executed_rows.append({
                        "timestamp": exec_time,
                        "symbol": sym,
                        "action": "BUY",
                        "price": exec_price,
                        "pnl_pct": None,
                    })
            elif action == "SELL":
                if position_open_price is not None:
                    ret = (exec_price - position_open_price) / position_open_price * 100.0
                    trade_returns.append(ret)
                    executed_rows.append({
                        "timestamp": exec_time,
                        "symbol": sym,
                        "action": "SELL",
                        "price": exec_price,
                        "pnl_pct": ret,
                    })
                    position_open_price = None

        # If a position remains open, mark current unrealized P&L against last close
        if position_open_price is not None and not sym_close.empty:
            last_time = sym_close.index[-1]
            last_price = float(sym_close.iloc[-1])
            ret = (last_price - position_open_price) / position_open_price * 100.0
            executed_rows.append({
                "timestamp": pd.Timestamp(last_time),
                "symbol": sym,
                "action": "UNREALIZED",
                "price": last_price,
                "pnl_pct": ret,
            })

    trades_df = pd.DataFrame(executed_rows)
    realized = trades_df[trades_df["action"] == "SELL"]["pnl_pct"].dropna()
    win_rate = float((realized > 0).mean() * 100.0) if not realized.empty else 0.0
    avg_return = float(realized.mean()) if not realized.empty else 0.0
    total_return = float(realized.sum()) if not realized.empty else 0.0

    summary = {
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "total_return_pct": total_return,
        "num_trades": int(realized.shape[0]),
    }
    return trades_df, summary


