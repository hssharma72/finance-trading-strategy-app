from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


DEFAULT_TIMEOUT = 30


def safe_series(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return series.replace([float("inf"), float("-inf")], float("nan")).fillna(fill_value)


def fetch_market_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for symbol {symbol}.")

    df = data.reset_index().copy()
    if "Date" not in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("Date").reset_index(drop=True)


def build_market_snapshot(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    close = float(latest["Close"])
    prev_close = float(prev["Close"])
    change = close - prev_close
    pct_change = (change / prev_close) * 100 if prev_close else 0.0

    return {
        "last_price": round(close, 2),
        "change": round(change, 2),
        "pct_change": round(pct_change, 2),
        "volume": int(latest.get("Volume", 0) or 0),
        "high_52w": round(df["High"].max(), 2),
        "low_52w": round(df["Low"].min(), 2),
    }


def get_ohlcv(symbol: str, period: str = "1y") -> pd.DataFrame:
    return fetch_market_data(symbol=symbol, period=period, interval="1d")

