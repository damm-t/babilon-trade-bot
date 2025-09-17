import os
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import pandas as pd


PORTFOLIO_DIR = "data"
PORTFOLIO_FILE = os.path.join(PORTFOLIO_DIR, "portfolio.csv")


@dataclass
class Holding:
    symbol: str
    quantity: float
    avg_price: float
    notes: str = ""


def ensure_storage() -> None:
    if not os.path.exists(PORTFOLIO_DIR):
        os.makedirs(PORTFOLIO_DIR)
    if not os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "quantity", "avg_price", "notes"])  # header


def load_portfolio() -> List[Holding]:
    ensure_storage()
    df = pd.read_csv(PORTFOLIO_FILE)
    holdings: List[Holding] = []
    for _, row in df.iterrows():
        try:
            holdings.append(
                Holding(
                    symbol=str(row["symbol"]).upper(),
                    quantity=float(row["quantity"]),
                    avg_price=float(row["avg_price"]),
                    notes=str(row.get("notes", "")),
                )
            )
        except Exception:
            continue
    return holdings


def save_portfolio(holdings: List[Holding]) -> None:
    ensure_storage()
    df = pd.DataFrame([asdict(h) for h in holdings])
    df.to_csv(PORTFOLIO_FILE, index=False)


def upsert_holding(symbol: str, quantity: float, price: float, notes: str = "") -> List[Holding]:
    symbol = symbol.upper()
    holdings = load_portfolio()
    found = False
    for h in holdings:
        if h.symbol == symbol:
            # Recalculate volume-weighted average price
            total_qty = h.quantity + quantity
            if total_qty <= 0:
                # Remove position if quantity becomes zero or negative
                holdings = [x for x in holdings if x.symbol != symbol]
            else:
                new_cost = h.avg_price * h.quantity + price * quantity
                h.quantity = total_qty
                h.avg_price = new_cost / total_qty
                if notes:
                    h.notes = notes
            found = True
            break
    if not found and quantity > 0:
        holdings.append(Holding(symbol=symbol, quantity=quantity, avg_price=price, notes=notes))
    save_portfolio(holdings)
    return holdings


def remove_holding(symbol: str) -> List[Holding]:
    symbol = symbol.upper()
    holdings = [h for h in load_portfolio() if h.symbol != symbol]
    save_portfolio(holdings)
    return holdings


def compute_portfolio_metrics(prices_by_symbol: Dict[str, float]) -> pd.DataFrame:
    """Return a DataFrame with P&L metrics per holding.

    Columns: symbol, quantity, avg_price, current_price, market_value, cost_basis,
             unrealized_pl, unrealized_pl_pct
    """
    holdings = load_portfolio()
    rows: List[Dict] = []
    for h in holdings:
        current_price = float(prices_by_symbol.get(h.symbol, 0.0) or 0.0)
        market_value = h.quantity * current_price
        cost_basis = h.quantity * h.avg_price
        unrealized_pl = market_value - cost_basis
        unrealized_pl_pct = (unrealized_pl / cost_basis * 100.0) if cost_basis > 0 else 0.0
        rows.append(
            {
                "symbol": h.symbol,
                "quantity": h.quantity,
                "avg_price": h.avg_price,
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pct": unrealized_pl_pct,
                "notes": h.notes,
            }
        )
    return pd.DataFrame(rows)


