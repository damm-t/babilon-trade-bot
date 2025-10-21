import os
import sys
import glob
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd


def _lazy_import_backtrader():
    try:
        import backtrader as bt  # type: ignore
        return bt
    except Exception as exc:  # pragma: no cover
        msg = (
            "Backtrader is required for Phase 2 backtesting.\n"
            "Install it via: pip install backtrader\n"
            "Alternatively, enable the 'backtest' extra in pyproject if configured.\n"
            f"Underlying error: {exc}"
        )
        raise RuntimeError(msg)


DATA_DIR = os.path.join("data")
LOGS_PATH = os.path.join("data", "logs.csv")
REPORTS_DIR = os.path.join("reports")


def _discover_symbols() -> List[str]:
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbols: List[str] = []
    for path in csvs:
        base = os.path.basename(path)
        sym = os.path.splitext(base)[0].upper()
        if sym not in {"LOGS"}:
            symbols.append(sym)
    return sorted(list(set(symbols)))


def _load_price_csv(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Normalize columns expected by GenericCSVData
    # Expected: Datetime, Open, High, Low, Close, Volume
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    elif "Date" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("CSV must contain 'Datetime' or 'Date' column")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df


def _load_logs() -> pd.DataFrame:
    if not os.path.exists(LOGS_PATH):
        return pd.DataFrame(columns=["timestamp", "stock_symbol", "decision"])  # empty
    df = pd.read_csv(LOGS_PATH)
    # Normalize columns
    if "stock_symbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "stock_symbol"})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])  # drop unparsable rows
    if "decision" in df.columns:
        df["decision"] = df["decision"].astype(str).str.upper()
    return df


def _filter_logs_last_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp.utcnow() - pd.DateOffset(months=months)
    return df[df["timestamp"] >= cutoff].copy()


def _build_events_index(df_logs: pd.DataFrame) -> Dict[str, List[pd.Timestamp]]:
    # Map symbol -> list of timestamps for BUY and SELL
    events: Dict[str, List[pd.Timestamp]] = {}
    for sym, g in df_logs.groupby("stock_symbol"):
        # We will store all events with their action encoded as part of timestamp mapping
        # Strategy will read full dataframe for actions; this index is to fast-check presence
        events[sym] = sorted(g["timestamp"].tolist())
    return events


def run_backtest(months: int = 12, starting_cash: float = 100000.0,
                 commission_pct: float = 0.0001, slippage_pct: float = 0.0005,
                 symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    bt = _lazy_import_backtrader()

    class LogSignalStrategy(bt.Strategy):
        params = dict(
            symbol=None,
            logs_df=None,
        )

        def __init__(self):
            self.data_symbol = self.p.symbol
            # Filter logs relevant to this symbol and shift to next bar execution
            logs = self.p.logs_df
            if logs is None or logs.empty:
                self.my_events = pd.DataFrame(columns=["timestamp", "decision"])  # empty
            else:
                ldf = logs[logs["stock_symbol"] == self.data_symbol].copy()
                ldf = ldf.sort_values("timestamp")
                # Execute on next bar: we simply check >= current datetime
                self.my_events = ldf[["timestamp", "decision"]]
            self._consumed = 0  # index pointer in my_events

        def next(self):
            # Current bar datetime
            cur_dt = self.datas[0].datetime.datetime(0)
            # Execute all events whose timestamp is before or equal current bar date
            while self._consumed < len(self.my_events):
                evt_ts = self.my_events.iloc[self._consumed]["timestamp"]
                decision = self.my_events.iloc[self._consumed]["decision"]
                # Compare dates ignoring intrabar time zone nuances
                if pd.Timestamp(cur_dt) >= pd.Timestamp(evt_ts):
                    if decision == "BUY":
                        if not self.position:
                            # Market buy: size = all-in for simplicity
                            size = math.floor(self.broker.getcash() / self.data.close[0])
                            if size > 0:
                                self.buy(size=size)
                    elif decision == "SELL":
                        if self.position:
                            self.close()  # market sell close
                    # HOLD is a no-op
                    self._consumed += 1
                else:
                    break

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load logs and filter to last N months
    df_logs = _load_logs()
    df_logs = _filter_logs_last_months(df_logs, months)

    # Select symbols
    chosen_symbols = symbols or _discover_symbols()
    if not chosen_symbols:
        raise RuntimeError("No symbols discovered under data/*.csv")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(starting_cash)
    # Commission and slippage
    cerebro.broker.setcommission(commission=commission_pct)
    cerebro.broker.set_slippage_perc(perc=slippage_pct)

    # Add datas & strategies per symbol
    datas_map: Dict[str, any] = {}
    for sym in chosen_symbols:
        try:
            df = _load_price_csv(sym)
        except Exception:
            continue
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=df.columns.get_loc("Datetime"),
            open=df.columns.get_loc("Open"),
            high=df.columns.get_loc("High"),
            low=df.columns.get_loc("Low"),
            close=df.columns.get_loc("Close"),
            volume=df.columns.get_loc("Volume"),
            openinterest=-1,
        )
        cerebro.adddata(data, name=sym)
        datas_map[sym] = data
        cerebro.addstrategy(LogSignalStrategy, symbol=sym, logs_df=df_logs)

    if not datas_map:
        raise RuntimeError("No valid price data feeds added. Ensure data/*.csv exist with OHLCV.")

    # Run
    results = cerebro.run()

    # Collect trades and equity curve
    # Backtrader doesn't directly expose trade list; we approximate from broker value over time
    equity = []
    for i in range(len(datas_map[next(iter(datas_map))])):
        # Use the first data's datetime as canonical timeline
        dt = list(datas_map.values())[0].datetime.datetime(i)
        value = cerebro.broker.getvalue()
        equity.append({"timestamp": dt, "equity": value})
    df_equity = pd.DataFrame(equity)

    # Write reports
    eq_path = os.path.join(REPORTS_DIR, "equity_curve.csv")
    df_equity.to_csv(eq_path, index=False)

    # Plot equity curve to PNG if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(10, 5))
        if not df_equity.empty:
            plt.plot(df_equity["timestamp"], df_equity["equity"], label="Equity")
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Account Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        png_path = os.path.join(REPORTS_DIR, "equity_curve.png")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
    except Exception:
        # Silently skip plotting if matplotlib is not installed or plotting fails
        pass

    # Basic summary
    summary = {
        "start_cash": starting_cash,
        "end_value": float(df_equity["equity"].iloc[-1]) if not df_equity.empty else starting_cash,
        "return_pct": (float(df_equity["equity"].iloc[-1]) - starting_cash) / starting_cash * 100.0 if not df_equity.empty else 0.0,
        "months": months,
        "symbols": ",".join(chosen_symbols),
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(REPORTS_DIR, "backtest_summary.csv"), index=False)

    return {"equity": df_equity, "summary": pd.DataFrame([summary])}


def main():
    months = int(os.environ.get("BACKTEST_MONTHS", "12"))
    try:
        run_backtest(months=months)
        print(f"Backtest completed. Reports written to '{REPORTS_DIR}'.")
    except Exception as e:
        print(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


