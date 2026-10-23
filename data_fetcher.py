from __future__ import annotations

import os
from typing import Dict, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_fetcher import fetch_market_data, build_market_snapshot
from strategy_engine import StrategyEngine
from ai_agent import AITradeAnalyst
from broker_adapter import BrokerManager

st.set_page_config(page_title="AI Trading Strategy App", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_market_data(symbol: str, period: str) -> pd.DataFrame:
    return fetch_market_data(symbol=symbol, period=period, interval="1d")


def metric_box(title: str, value: str, delta: str = ""):
    st.metric(label=title, value=value, delta=delta)


def draw_candles(df: pd.DataFrame, strategy_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], mode="lines", name="SMA 20"))
    if "EMA_12" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_12"], mode="lines", name="EMA 12"))
    if "BOLL_UPPER" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BOLL_UPPER"], mode="lines", line=dict(dash="dash"), name="Upper Band"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BOLL_LOWER"], mode="lines", line=dict(dash="dash"), name="Lower Band"))

    fig.update_layout(
        title=f"{strategy_name} Signal View",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def main() -> None:
    st.title("AI Finance Trading Strategy App")
    st.caption("Institutional-style technical screening, risk control, and broker-ready execution architecture.")

    with st.sidebar:
        st.header("Market Setup")
        symbol = st.text_input("Ticker / Symbol", value="AAPL")
        period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
        strategy_name = st.selectbox(
            "Strategy",
            [
                "Momentum",
                "Trend Following",
                "Breakout",
                "Mean Reversion",
                "Multi Factor",
            ],
        )

        st.markdown("---")
        st.subheader("Broker / Execution")
        broker_name = st.selectbox("Broker Adapter", ["None", "Zerodha", "Upstox", "Interactive Brokers", "Alpaca"])
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")

        if os.getenv("OPENAI_API_KEY") or st.checkbox("Enable AI Analyst (optional)"):
            st.success("AI analyst enabled for signal summaries.")

    if not symbol:
        st.warning("Please enter a symbol to load the market data.")
        return

    try:
        df = load_market_data(symbol, period)
    except Exception as exc:
        st.error(f"Unable to fetch market data: {exc}")
        return

    engine = StrategyEngine(strategy_name=strategy_name)
    details = engine.analyze(df)
    latest = details["latest"]
    snapshot = build_market_snapshot(df)

    st.subheader(f"{symbol.upper()} - {period} analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_box("Last Price", f"${snapshot.get('last_price', 0):,.2f}")
    with col2:
        metric_box("Signal", latest["signal"], f"{latest['confidence']:.0f}% confidence")
    with col3:
        metric_box("Target", f"${latest['target_price']:.2f}")
    with col4:
        metric_box("Stop", f"${latest['stop_loss']:.2f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Price Chart")
        st.plotly_chart(draw_candles(df, strategy_name), use_container_width=True)

    with right:
        st.subheader("Strategy Summary")
        st.json(
            {
                "strategy": strategy_name,
                "trend": latest["trend"],
                "momentum": latest["momentum"],
                "volatility": latest["volatility"],
                "risk_reward": round(latest["risk_reward"], 2),
                "confidence": latest["confidence"],
            }
        )

        st.subheader("Trade Setup")
        st.json(
            {
                "entry": round(latest["entry_price"], 2),
                "target": round(latest["target_price"], 2),
                "stop_loss": round(latest["stop_loss"], 2),
                "risk_pct": round(latest["risk_pct"], 2),
                "reward_pct": round(latest["reward_pct"], 2),
            }
        )

    st.markdown("---")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        metric_box("CAGR", f"{details['metrics']['cagr']:.2f}%")
    with col_b:
        metric_box("Sharpe", f"{details['metrics']['sharpe']:.2f}")
    with col_c:
        metric_box("Max DD", f"{details['metrics']['max_drawdown']:.2f}%")
    with col_d:
        metric_box("Win Rate", f"{details['metrics']['win_rate']:.2f}%")

    st.markdown("---")
    st.subheader("Indicator Breakdown")
    indicator_df = df.tail(10).copy()
    indicator_df["Signal"] = details["signals"]
    display_df = indicator_df[["Date", "Close", "SMA_20", "SMA_50", "RSI_14", "MACD", "Signal"]].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("AI Analyst")
    analyst = AITradeAnalyst()
    summary = analyst.generate_summary(latest, symbol=symbol, strategy_name=strategy_name)
    st.markdown(summary)

    st.markdown("---")
    st.subheader("Broker Readiness")
    if broker_name == "None":
        st.info("No live broker selected. Connect a broker adapter to enable execution approval workflow or paper-trading mode.")
    else:
        broker = BrokerManager.get_adapter(broker_name, api_key=api_key, api_secret=api_secret, symbol=symbol)
        st.json(broker.get_status())


if __name__ == "__main__":
    main()


