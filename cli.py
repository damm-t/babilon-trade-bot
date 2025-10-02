import argparse
import os
from typing import List

import pandas as pd
import yfinance as yf

from config import STOCK_SYMBOLS


def fetch_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist is None or hist.empty:
        raise ValueError(f"No data returned for {symbol}")
    # Ensure index is a column for saving
    hist = hist.reset_index()
    return hist


def save_history_csv(df: pd.DataFrame, symbol: str, out_dir: str = "data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol.upper()}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def cmd_fetch(tickers: List[str], period: str, interval: str) -> None:
    symbols = [t.strip().upper() for t in tickers if t.strip()]
    for symbol in symbols:
        print(f"Fetching {symbol} ({period}, {interval})...")
        df = fetch_history(symbol, period=period, interval=interval)
        path = save_history_csv(df, symbol)
        print(f"Saved: {path} ({len(df)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Babilon Trade Bot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch historical OHLCV for tickers and save to data/")
    fetch.add_argument(
        "--tickers",
        type=str,
        default=",".join(STOCK_SYMBOLS),
        help="Comma-separated list of tickers. Defaults to config.STOCK_SYMBOLS",
    )
    fetch.add_argument(
        "--period",
        type=str,
        default="1y",
        help="yfinance period (e.g., 1mo, 3mo, 6mo, 1y, 2y, max)",
    )
    fetch.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="yfinance interval (e.g., 1d, 1h, 30m, 15m)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fetch":
        tickers = args.tickers.split(",") if args.tickers else STOCK_SYMBOLS
        cmd_fetch(tickers, args.period, args.interval)


if __name__ == "__main__":
    main()

import argparse
import json
from typing import Optional

import pandas as pd

from logic.backtest import simulate_from_logs
from model.model_manager import aggregate_decision, train_technical_predictor


def cmd_analyze(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    news = args.news or ""
    decision, sent_score, tech_proba = aggregate_decision(symbol, news)
    print(json.dumps({
        "symbol": symbol,
        "decision": decision,
        "sentiment_score": round(sent_score, 4),
        "technical_up_probability": round(tech_proba, 4),
    }, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    predictor = train_technical_predictor(symbol, period=args.period)
    print(json.dumps({
        "symbol": symbol,
        "status": "trained",
        "weights": predictor.model.weights.tolist(),
        "bias": predictor.model.bias,
    }, indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    trades_df, summary = simulate_from_logs(args.days)
    print(json.dumps({
        "summary": summary,
        "trades": trades_df.to_dict(orient="records")[:50],
    }, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Babilon CLI Agent")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Analyze a symbol with optional news text")
    a.add_argument("symbol", type=str)
    a.add_argument("--news", type=str, default="")
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("train", help="Train technical predictor for a symbol")
    t.add_argument("symbol", type=str)
    t.add_argument("--period", type=str, default="6mo")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("backtest", help="Simulate trades from logs.csv")
    b.add_argument("--days", type=int, default=30)
    b.set_defaults(func=cmd_backtest)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


