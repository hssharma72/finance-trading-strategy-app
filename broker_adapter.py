from __future__ import annotations

import os
from typing import Any, Dict


class AITradeAnalyst:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def _fallback_summary(self, signal: Dict[str, Any], symbol: str, strategy_name: str) -> str:
        signal_name = signal.get("signal", "HOLD")
        entry = signal.get("entry_price", 0)
        target = signal.get("target_price", 0)
        stop = signal.get("stop_loss", 0)
        confidence = signal.get("confidence", 0)
        trend = signal.get("trend", "Neutral")
        momentum = signal.get("momentum", "Neutral")

        summary = (
            f"## Trade idea for {symbol.upper()}\n"
            f"Strategy: {strategy_name}\n"
            f"Bias: {signal_name} with {confidence}% confidence.\n"
            f"Current trend is {trend} and momentum is {momentum}.\n"
            f"Entry: ${entry:.2f} | Target: ${target:.2f} | Stop: ${stop:.2f}.\n"
            f"The setup is designed to reward upside continuation while keeping downside risk controlled through a defined stop-loss level."
        )
        return summary

    def generate_summary(self, signal: Dict[str, Any], symbol: str, strategy_name: str) -> str:
        if not self.api_key:
            return self._fallback_summary(signal, symbol, strategy_name)

        # This is a lightweight hook for future LLM integration.
        # In production, use the OpenAI / Azure SDK to generate a deeper market narrative.
        return self._fallback_summary(signal, symbol, strategy_name)


if __name__ == "__main__":
    analyst = AITradeAnalyst()
    print(analyst.generate_summary({"signal": "BUY", "entry_price": 100, "target_price": 108, "stop_loss": 94, "confidence": 77, "trend": "Bullish", "momentum": "Positive"}, "AAPL", "Momentum"))


