import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import json
import os
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Black Gold AI Market Scanner",
    layout="wide",
    page_icon="📈"
)

# =====================================================
# BLACK + GOLD THEME
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: #FFD700;
}
section[data-testid="stSidebar"] {
    background-color: #111111;
}
h1, h2, h3, h4, h5, h6, p, div, label, span {
    color: #FFD700 !important;
}
.stMetric {
    background-color: #111111;
    border: 1px solid #FFD700;
    padding: 10px;
    border-radius: 12px;
}
.stButton>button {
    background-color: #FFD700;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}
.stSelectbox div[data-baseweb="select"] {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# AUTO REFRESH
# =====================================================
st_autorefresh(interval=15000, key="refresh")

# =====================================================
# STORAGE
# =====================================================
DATA_FILE = "scanner_results.json"

def load_results():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_results(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

results = load_results()

# =====================================================
# TITLE
# =====================================================
st.title("🖤 Gold Elite AI Stock Scanner")
st.caption("Live Trend Dashboard • AI Confidence Engine • Multi Stock Scanner")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Scanner Controls")

watchlist = st.sidebar.multiselect(
    "Choose Stocks",
    ["SPY","QQQ","TSLA","NVDA","AAPL","MSFT","META","AMD","AMZN"],
    default=["SPY","TSLA","NVDA"]
)

main_ticker = st.sidebar.selectbox("Main Chart Ticker", watchlist)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m","5m","15m","30m","60m","1d"]
)

show_ema9 = st.sidebar.checkbox("EMA 9", True)
show_ema20 = st.sidebar.checkbox("EMA 20", True)
show_ema50 = st.sidebar.checkbox("EMA 50", False)
show_vwap = st.sidebar.checkbox("VWAP", True)

# =====================================================
# PERIOD MAP
# =====================================================
period_map = {
    "1m":"1d",
    "5m":"5d",
    "15m":"5d",
    "30m":"1mo",
    "60m":"1mo",
    "1d":"6mo"
}

# =====================================================
# FUNCTIONS
# =====================================================
def add_indicators(df):
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["Typical"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["CumVol"] = df["Volume"].cumsum()
    df["CumPV"] = (df["Typical"] * df["Volume"]).cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def ai_score(df):
    latest = df.iloc[-1]
    score = 50

    rsi = latest["RSI"] if pd.notna(latest["RSI"]) else 50

    if 55 <= rsi <= 70:
        score += 10
    elif rsi < 40:
        score += 5

    avg_vol = df["Volume"].tail(20).mean()
    vol_ratio = latest["Volume"] / avg_vol if avg_vol else 1

    score += min(vol_ratio * 5, 15)

    if latest["Close"] > latest["VWAP"]:
        score += 10

    if latest["EMA9"] > latest["EMA20"]:
        score += 10

    if latest["Close"] > df["Close"].iloc[-2]:
        score += 8

    return max(1, min(99, round(score)))

def badge(score):
    if score >= 80:
        return "🟢 STRONG"
    elif score >= 65:
        return "🟡 MEDIUM"
    return "🔴 WEAK"

# =====================================================
# MULTI STOCK LIVE SCANNER
# =====================================================
st.subheader("🔥 Live Scanner Rankings")

scan_data = []

for symbol in watchlist:
    try:
        df = yf.download(
            symbol,
            period=period_map[timeframe],
            interval=timeframe,
            auto_adjust=True,
            progress=False
        )

        if len(df) < 20:
            continue

        df = add_indicators(df)
        score = ai_score(df)
        price = round(df["Close"].iloc[-1], 2)

        scan_data.append({
            "Ticker": symbol,
            "Price": price,
            "Confidence": score,
            "Signal": badge(score)
        })

    except:
        pass

scan_df = pd.DataFrame(scan_data).sort_values("Confidence", ascending=False)
st.dataframe(scan_df, use_container_width=True)

# =====================================================
# MAIN CHART
# =====================================================
st.subheader(f"📊 {main_ticker} Interactive Chart")

data = yf.download(
    main_ticker,
    period=period_map[timeframe],
    interval=timeframe,
    auto_adjust=True,
    progress=False
)

data = add_indicators(data)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.75,0.25]
)

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
), row=1, col=1)

if show_ema9:
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA9"], name="EMA9"), row=1, col=1)

if show_ema20:
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], name="EMA20"), row=1, col=1)

if show_ema50:
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], name="EMA50"), row=1, col=1)

if show_vwap:
    fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], name="VWAP"), row=1, col=1)

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["RSI"],
    name="RSI"
), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="gold"),
    height=850,
    dragmode="pan",
    hovermode="x unified",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ALERT PANEL
# =====================================================
score = ai_score(data)
signal = badge(score)

c1,c2,c3 = st.columns(3)

with c1:
    st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")

with c2:
    st.metric("AI Confidence", f"{score}%")

with c3:
    st.metric("Signal", signal)

if score >= 80:
    st.success(f"🔥 High Probability Setup Detected in {main_ticker}")

# =====================================================
# SAVE RESULTS
# =====================================================
outcome = st.selectbox("Trade Result", ["pending","win","loss"])

if st.button("Save Result"):
    results.append({
        "ticker": main_ticker,
        "confidence": score,
        "outcome": outcome,
        "time": str(datetime.now())
    })
    save_results(results)
    st.success("Saved")

# =====================================================
# DASHBOARD
# =====================================================
st.subheader("🏆 Performance Dashboard")

wins = len([x for x in results if x["outcome"]=="win"])
losses = len([x for x in results if x["outcome"]=="loss"])
total = len(results)

winrate = round((wins/total)*100,1) if total else 0

d1,d2,d3,d4 = st.columns(4)

d1.metric("Trades", total)
d2.metric("Wins", wins)
d3.metric("Losses", losses)
d4.metric("Win Rate", f"{winrate}%")

# =====================================================
# FOOTER
# =====================================================
st.caption("Black Gold Elite Scanner v2")
