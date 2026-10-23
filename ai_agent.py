from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class StrategyEngine:
    def __init__(self, strategy_name: str = "Momentum"):
        self.strategy_name = strategy_name

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if out.empty:
            return out

        out["SMA_20"] = out["Close"].rolling(20).mean()
        out["SMA_50"] = out["Close"].rolling(50).mean()
        out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
        out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = out["EMA_12"] - out["EMA_26"]
        out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

        delta = out["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out["RSI_14"] = 100 - (100 / (1 + rs))
        out["RSI_14"] = out["RSI_14"].fillna(50)

        std = out["Close"].rolling(20).std()
        out["BOLL_UPPER"] = out["SMA_20"] + (std * 2)
        out["BOLL_LOWER"] = out["SMA_20"] - (std * 2)

        high_low = out["High"] - out["Low"]
        prev_close = out["Close"].shift(1)
        atr = (high_low + (out["High"] - prev_close).abs() + (out["Low"] - prev_close).abs()).rolling(14).mean() / 3
        out["ATR_14"] = atr.fillna(0)

        return out

    def _score_signal(self, row: pd.Series) -> float:
        score = 0.0
        close = float(row.get("Close", 0.0) or 0.0)
        sma20 = float(row.get("SMA_20", np.nan) or np.nan)
        sma50 = float(row.get("SMA_50", np.nan) or np.nan)
        ema12 = float(row.get("EMA_12", np.nan) or np.nan)
        ema26 = float(row.get("EMA_26", np.nan) or np.nan)
        rsi = float(row.get("RSI_14", 50.0) or 50.0)
        macd_hist = float(row.get("MACD_HIST", 0.0) or 0.0)
        upper = float(row.get("BOLL_UPPER", np.nan) or np.nan)
        lower = float(row.get("BOLL_LOWER", np.nan) or np.nan)

        if pd.notna(sma20) and pd.notna(sma50):
            score += 2.0 if close > sma20 else -1.5
            score += 1.5 if sma20 > sma50 else -1.5

        if pd.notna(ema12) and pd.notna(ema26):
            score += 1.5 if ema12 > ema26 else -1.5

        if 45 <= rsi <= 65:
            score += 1.0
        elif rsi < 30:
            score += 1.5
        elif rsi > 70:
            score -= 1.8

        if macd_hist > 0:
            score += 1.5
        else:
            score -= 1.5

        if pd.notna(upper) and pd.notna(lower):
            if close > upper:
                score -= 1.5
            elif close < lower:
                score += 1.5

        return score

    def _signal_label(self, score: float) -> str:
        if score >= 3.5:
            return "BUY"
        if score >= 1.0:
            return "BUY"
        if score <= -3.5:
            return "SELL"
        if score <= -1.0:
            return "SELL"
        return "HOLD"

    def _build_trade_levels(self, row: pd.Series) -> Dict[str, float]:
        close = float(row.get("Close", 0.0) or 0.0)
        atr = float(row.get("ATR_14", 0.0) or 0.0)
        if close == 0:
            return {"entry_price": 0.0, "target_price": 0.0, "stop_loss": 0.0, "risk_reward": 0.0, "risk_pct": 0.0, "reward_pct": 0.0}

        if atr > 0:
            target = close + (atr * 1.8)
            stop = close - (atr * 1.2)
            rr = (target - close) / max(close - stop, 1e-6)
        else:
            target = close * 1.06
            stop = close * 0.94
            rr = 2.0

        return {
            "entry_price": close,
            "target_price": float(target),
            "stop_loss": float(stop),
            "risk_reward": float(rr),
            "risk_pct": float((close - stop) / close * 100),
            "reward_pct": float((target - close) / close * 100),
        }

    def _strategy_specific_adjustment(self, row: pd.Series) -> float:
        if self.strategy_name == "Trend Following":
            return 1.5 if float(row.get("SMA_20", 0) or 0) > float(row.get("SMA_50", 0) or 0) else -1.5
        if self.strategy_name == "Breakout":
            return 1.0 if float(row.get("Close", 0) or 0) > float(row.get("BOLL_UPPER", 0) or 0) * 0.995 else -1.0
        if self.strategy_name == "Mean Reversion":
            return 1.5 if float(row.get("Close", 0) or 0) < float(row.get("BOLL_LOWER", 0) or 0) * 1.005 else -1.0
        if self.strategy_name == "Multi Factor":
            return 1.5 if float(row.get("RSI_14", 50) or 50) in range(45, 65) else 0.5
        return 0.0

    def _calculate_backtest_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

        returns = []
        closes = df["Close"].to_numpy(dtype=float)
        for i in range(1, len(closes)):
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

        if not returns:
            returns = [0.0]

        arr = np.array(returns, dtype=float)
        cumulative = np.cumprod(1 + arr)
        cagr = (cumulative[-1] ** (1 / max(len(arr), 1)) - 1) * 100 if len(cumulative) else 0.0
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(len(arr))) if np.std(arr) else 0.0
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / peak - 1) * 100
        max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0
        win_rate = float(np.mean(arr > 0) * 100) if len(arr) else 0.0

        return {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
        }

    def analyze(self, df: pd.DataFrame) -> Dict[str, object]:
        prepared = self._compute_indicators(df)
        prepared["StrategyScore"] = prepared.apply(self._score_signal, axis=1)
        prepared["StrategyAdjustment"] = prepared.apply(self._strategy_specific_adjustment, axis=1)
        prepared["CombinedScore"] = prepared["StrategyScore"] + prepared["StrategyAdjustment"]
        prepared["Signal"] = prepared["CombinedScore"].apply(self._signal_label)

        latest = prepared.iloc[-1]
        trade = self._build_trade_levels(latest)
        trade["signal"] = latest["Signal"]
        trade["confidence"] = max(0, min(99, int(abs(latest["CombinedScore"]) * 12)))
        trade["trend"] = "Bullish" if latest["SMA_20"] > latest["SMA_50"] else "Neutral"
        trade["momentum"] = "Positive" if latest["MACD_HIST"] > 0 else "Negative"
        trade["volatility"] = "Expanded" if latest["ATR_14"] > prepared["ATR_14"].median() else "Compressed"

        metrics = self._calculate_backtest_metrics(prepared)

        return {
            "latest": trade,
            "metrics": metrics,
            "signals": prepared["Signal"].tail(10).tolist(),
            "prepared": prepared,
        }


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "Open": np.linspace(100, 110, 60),
            "High": np.linspace(101, 111, 60),
            "Low": np.linspace(99, 109, 60),
            "Close": np.linspace(100, 115, 60),
            "Volume": np.ones(60) * 1000,
        }
    )
    result = StrategyEngine("Momentum").analyze(sample)
    print(result["latest"])


