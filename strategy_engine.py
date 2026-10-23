from __future__ import annotations

from typing import Dict

import pandas as pd
import yfinance as yf


def fetch_market_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No market data returned for symbol {symbol}.")

    df = data.reset_index().copy()
    if "Date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("Date").reset_index(drop=True)


def build_market_snapshot(df: pd.DataFrame) -> Dict[str, float | int]:
    if df.empty:
        return {"last_price": 0.0, "change": 0.0, "pct_change": 0.0, "volume": 0, "high_52w": 0.0, "low_52w": 0.0}

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    close = float(latest.get("Close", 0.0) or 0.0)
    prev_close = float(previous.get("Close", 0.0) or 0.0)
    change = close - prev_close
    pct_change = (change / prev_close) * 100 if prev_close else 0.0

    return {
        "last_price": round(close, 2),
        "change": round(change, 2),
        "pct_change": round(pct_change, 2),
        "volume": int(latest.get("Volume", 0) or 0),
        "high_52w": round(float(df["High"].max()), 2) if "High" in df.columns else 0.0,
        "low_52w": round(float(df["Low"].min()), 2) if "Low" in df.columns else 0.0,
    }


if __name__ == "__main__":
    sample = fetch_market_data("AAPL", period="1mo")
    print(sample.head())


