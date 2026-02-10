import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="VIX Volatility Dashboard", layout="wide")

st.title("VIX Volatility Dashboard")
st.markdown("Real-time CBOE Volatility Index data and analysis powered by Yahoo Finance.")

# --- Sidebar controls ---
st.sidebar.header("Settings")

period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Max": "max",
}
selected_period = st.sidebar.selectbox("Time Period", list(period_options.keys()), index=3)
period = period_options[selected_period]

sma_window = st.sidebar.slider("Moving Average Window (days)", 5, 100, 20)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_sp500 = st.sidebar.checkbox("Overlay S&P 500 (inverted scale)", value=True)


@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


# --- Fetch data ---
with st.spinner("Fetching VIX data..."):
    vix = fetch_data("^VIX", period)

if vix.empty:
    st.error("Failed to fetch VIX data. Please try again later.")
    st.stop()

if show_sp500:
    with st.spinner("Fetching S&P 500 data..."):
        sp500 = fetch_data("^GSPC", period)

# --- Compute indicators ---
vix["SMA"] = vix["Close"].rolling(window=sma_window).mean()

if show_bollinger:
    rolling_std = vix["Close"].rolling(window=sma_window).std()
    vix["BB_Upper"] = vix["SMA"] + 2 * rolling_std
    vix["BB_Lower"] = vix["SMA"] - 2 * rolling_std

vix["Daily_Change"] = vix["Close"].pct_change() * 100

# --- Key metrics row ---
latest = vix.iloc[-1]
prev = vix.iloc[-2] if len(vix) > 1 else vix.iloc[-1]

close_val = float(latest["Close"])
prev_close = float(prev["Close"])
daily_change = close_val - prev_close
daily_change_pct = (daily_change / prev_close) * 100 if prev_close != 0 else 0
high_val = float(vix["High"].max())
low_val = float(vix["Low"].min())
avg_val = float(vix["Close"].mean())
current_sma = float(latest["SMA"]) if pd.notna(latest["SMA"]) else avg_val

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("VIX Close", f"{close_val:.2f}", f"{daily_change:+.2f} ({daily_change_pct:+.1f}%)")
col2.metric(f"{sma_window}-Day SMA", f"{current_sma:.2f}")
col3.metric("Period High", f"{high_val:.2f}")
col4.metric("Period Low", f"{low_val:.2f}")
col5.metric("Period Average", f"{avg_val:.2f}")

# --- Volatility regime ---
st.markdown("---")
if close_val < 15:
    regime = "Low Volatility"
    regime_color = "green"
    regime_desc = "Markets are calm. Historically low fear levels."
elif close_val < 20:
    regime = "Normal"
    regime_color = "blue"
    regime_desc = "Volatility is within the normal historical range."
elif close_val < 30:
    regime = "Elevated"
    regime_color = "orange"
    regime_desc = "Heightened market uncertainty. Investors are hedging."
else:
    regime = "High Volatility"
    regime_color = "red"
    regime_desc = "Extreme fear in markets. Significant turbulence expected."

st.markdown(
    f"**Current Regime:** :{regime_color}[{regime}] — {regime_desc}"
)

# --- Volatility regime range reference ---
regime_col1, regime_col2, regime_col3, regime_col4 = st.columns(4)
with regime_col1:
    st.markdown(
        '<div style="background-color:#c8e6c9;padding:12px 16px;border-radius:8px;border-left:5px solid #388e3c;">'
        '<strong style="color:#2e7d32;">Low Volatility</strong><br>'
        '<span style="font-size:1.3em;font-weight:bold;color:#1b5e20;">0 – 15</span><br>'
        '<span style="color:#33691e;font-size:0.85em;">Markets calm, low fear</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with regime_col2:
    st.markdown(
        '<div style="background-color:#bbdefb;padding:12px 16px;border-radius:8px;border-left:5px solid #1976d2;">'
        '<strong style="color:#1565c0;">Normal</strong><br>'
        '<span style="font-size:1.3em;font-weight:bold;color:#0d47a1;">15 – 20</span><br>'
        '<span style="color:#0d47a1;font-size:0.85em;">Typical market conditions</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with regime_col3:
    st.markdown(
        '<div style="background-color:#ffe0b2;padding:12px 16px;border-radius:8px;border-left:5px solid #f57c00;">'
        '<strong style="color:#e65100;">Elevated</strong><br>'
        '<span style="font-size:1.3em;font-weight:bold;color:#bf360c;">20 – 30</span><br>'
        '<span style="color:#bf360c;font-size:0.85em;">Uncertainty rising, hedging up</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with regime_col4:
    st.markdown(
        '<div style="background-color:#ffcdd2;padding:12px 16px;border-radius:8px;border-left:5px solid #d32f2f;">'
        '<strong style="color:#c62828;">High Volatility</strong><br>'
        '<span style="font-size:1.3em;font-weight:bold;color:#b71c1c;">30+</span><br>'
        '<span style="color:#b71c1c;font-size:0.85em;">Extreme fear, major turbulence</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# --- Main VIX chart ---
st.markdown("---")
st.subheader("VIX Index")

if show_sp500 and not sp500.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
else:
    fig = make_subplots()

# Candlestick chart
fig.add_trace(
    go.Candlestick(
        x=vix.index,
        open=vix["Open"],
        high=vix["High"],
        low=vix["Low"],
        close=vix["Close"],
        name="VIX",
    )
)

# SMA line
fig.add_trace(
    go.Scatter(
        x=vix.index,
        y=vix["SMA"],
        mode="lines",
        name=f"{sma_window}-Day SMA",
        line=dict(color="orange", width=2),
    )
)

# Bollinger Bands
if show_bollinger and "BB_Upper" in vix.columns:
    fig.add_trace(
        go.Scatter(
            x=vix.index,
            y=vix["BB_Upper"],
            mode="lines",
            name="Upper BB",
            line=dict(color="rgba(150,150,150,0.5)", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=vix.index,
            y=vix["BB_Lower"],
            mode="lines",
            name="Lower BB",
            line=dict(color="rgba(150,150,150,0.5)", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(150,150,150,0.1)",
        )
    )

# Volatility regime shading with labels
fig.add_hrect(y0=0, y1=15, fillcolor="green", opacity=0.06, line_width=0,
              annotation_text="Low (0–15)", annotation_position="top left",
              annotation=dict(font_size=11, font_color="green"))
fig.add_hrect(y0=15, y1=20, fillcolor="blue", opacity=0.06, line_width=0,
              annotation_text="Normal (15–20)", annotation_position="top left",
              annotation=dict(font_size=11, font_color="blue"))
fig.add_hrect(y0=20, y1=30, fillcolor="orange", opacity=0.06, line_width=0,
              annotation_text="Elevated (20–30)", annotation_position="top left",
              annotation=dict(font_size=11, font_color="orange"))
fig.add_hrect(y0=30, y1=100, fillcolor="red", opacity=0.06, line_width=0,
              annotation_text="High (30+)", annotation_position="top left",
              annotation=dict(font_size=11, font_color="red"))

# Dashed threshold lines at regime boundaries
for level, color in [(15, "green"), (20, "blue"), (30, "red")]:
    fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1, opacity=0.5)

# S&P 500 overlay
if show_sp500 and not sp500.empty:
    fig.add_trace(
        go.Scatter(
            x=sp500.index,
            y=sp500["Close"],
            mode="lines",
            name="S&P 500",
            line=dict(color="steelblue", width=1.5),
            opacity=0.7,
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="S&P 500", secondary_y=True, autorange="reversed")

fig.update_layout(
    height=550,
    xaxis_rangeslider_visible=False,
    yaxis_title="VIX Level",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=30, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# --- Daily change distribution & volume ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Daily % Change Distribution")
    changes = vix["Daily_Change"].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=changes,
            nbinsx=50,
            marker_color=np.where(changes >= 0, "rgba(239,83,80,0.7)", "rgba(38,166,91,0.7)").tolist()
            if len(changes) < 5000
            else "steelblue",
            name="Daily Change %",
        )
    )
    fig_hist.update_layout(
        xaxis_title="Daily Change (%)",
        yaxis_title="Frequency",
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("Trading Volume")
    if "Volume" in vix.columns and vix["Volume"].sum() > 0:
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Bar(
                x=vix.index,
                y=vix["Volume"],
                marker_color="steelblue",
                name="Volume",
            )
        )
        fig_vol.update_layout(
            yaxis_title="Volume",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("Volume data is not available for VIX index. Showing rolling volatility of VIX instead.")
        vix["VIX_Vol"] = vix["Close"].rolling(window=sma_window).std()
        fig_vvol = go.Figure()
        fig_vvol.add_trace(
            go.Scatter(
                x=vix.index,
                y=vix["VIX_Vol"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(70,130,180,0.2)",
                line=dict(color="steelblue"),
                name="Rolling Std Dev",
            )
        )
        fig_vvol.update_layout(
            yaxis_title=f"{sma_window}-Day Rolling Std Dev",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_vvol, use_container_width=True)

# --- VIX Term Structure ---
st.markdown("---")
st.subheader("VIX Futures Term Structure (Spot vs Short/Mid-Term)")

term_tickers = {
    "VIX Spot": "^VIX",
    "VIX 9-Day (VIX9D)": "^VIX9D",
    "VIX 3-Month (VIX3M)": "^VIX3M",
    "VIX 6-Month (VIX6M)": "^VIX6M",
}


@st.cache_data(ttl=300)
def fetch_term_structure(tickers: dict) -> dict:
    results = {}
    for label, ticker in tickers.items():
        try:
            d = yf.download(ticker, period="5d", progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if not d.empty:
                results[label] = float(d["Close"].iloc[-1])
        except Exception:
            pass
    return results


with st.spinner("Fetching term structure data..."):
    term_data = fetch_term_structure(term_tickers)

if len(term_data) >= 2:
    fig_term = go.Figure()
    fig_term.add_trace(
        go.Bar(
            x=list(term_data.keys()),
            y=list(term_data.values()),
            marker_color=["#ef5350", "#ff9800", "#42a5f5", "#66bb6a"][: len(term_data)],
            text=[f"{v:.2f}" for v in term_data.values()],
            textposition="outside",
        )
    )
    fig_term.update_layout(
        yaxis_title="VIX Level",
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
    )

    if len(term_data) >= 2:
        values = list(term_data.values())
        if values[-1] > values[0]:
            structure = "Contango (normal) — longer-term volatility expectations exceed spot."
        else:
            structure = "Backwardation — spot volatility exceeds longer-term expectations, suggesting near-term stress."
        st.markdown(f"**Term Structure:** {structure}")

    st.plotly_chart(fig_term, use_container_width=True)
else:
    st.warning("Unable to retrieve enough term structure data.")

# --- Regime breakdown ---
st.markdown("---")
st.subheader("Time Spent in Each Regime")

total_days = len(vix)
days_low = int((vix["Close"] < 15).sum())
days_normal = int(((vix["Close"] >= 15) & (vix["Close"] < 20)).sum())
days_elevated = int(((vix["Close"] >= 20) & (vix["Close"] < 30)).sum())
days_high = int((vix["Close"] >= 30).sum())

pct_low = days_low / total_days * 100 if total_days else 0
pct_normal = days_normal / total_days * 100 if total_days else 0
pct_elevated = days_elevated / total_days * 100 if total_days else 0
pct_high = days_high / total_days * 100 if total_days else 0

fig_regime = go.Figure()
fig_regime.add_trace(go.Bar(
    x=["Low (0–15)", "Normal (15–20)", "Elevated (20–30)", "High (30+)"],
    y=[days_low, days_normal, days_elevated, days_high],
    marker_color=["#4caf50", "#2196f3", "#ff9800", "#f44336"],
    text=[f"{d} days ({p:.1f}%)" for d, p in
          [(days_low, pct_low), (days_normal, pct_normal),
           (days_elevated, pct_elevated), (days_high, pct_high)]],
    textposition="outside",
))
fig_regime.update_layout(
    yaxis_title="Number of Trading Days",
    height=320,
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_regime, use_container_width=True)

# --- Statistics table ---
st.markdown("---")
st.subheader("Summary Statistics")

stats = {
    "Current Close": f"{close_val:.2f}",
    f"{sma_window}-Day SMA": f"{current_sma:.2f}",
    "Period High": f"{high_val:.2f}",
    "Period Low": f"{low_val:.2f}",
    "Period Mean": f"{avg_val:.2f}",
    "Period Median": f"{float(vix['Close'].median()):.2f}",
    "Std Deviation": f"{float(vix['Close'].std()):.2f}",
    "Avg Daily Change": f"{float(changes.mean()):.2f}%",
    "Max Daily Spike": f"{float(changes.max()):.2f}%",
    "Max Daily Drop": f"{float(changes.min()):.2f}%",
    "Days Above 30": str(int((vix["Close"] > 30).sum())),
    "Days Below 15": str(int((vix["Close"] < 15).sum())),
}

stat_col1, stat_col2, stat_col3 = st.columns(3)
stat_items = list(stats.items())
for i, (k, v) in enumerate(stat_items):
    target_col = [stat_col1, stat_col2, stat_col3][i % 3]
    target_col.metric(k, v)

# --- Footer ---
st.markdown("---")
st.caption(
    "Data sourced from Yahoo Finance via yfinance. "
    "The VIX index measures 30-day expected volatility of the S&P 500 implied by option prices. "
    "Dashboard refreshes data every 5 minutes."
)
