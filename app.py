import hashlib
import hmac
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import pandas_ta as ta

# ----------------------------- Page and theme -----------------------------
st.set_page_config(page_title="HR Trading AI", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background: #0b1220; color: #edf2f7; }
[data-testid="stHeader"] { background: rgba(11,18,32,0.95); }
[data-testid="stMetric"] { background: #121d30; border: 1px solid #26364d; padding: 14px 16px; border-radius: 14px; }
.hero { padding: 18px; border-radius: 16px; margin-bottom: 12px; background: linear-gradient(120deg, #132a43, #162039); border: 1px solid #29435d; }
.muted { color: #a9b8cc; font-size: 0.92rem; }
.signal { padding: 14px; border-radius: 12px; font-weight: 700; border: 1px solid #334155; margin: 6px 0 14px 0; }
.buy { background: #10382e; color: #7bf0c2; }
.sell { background: #401e2a; color: #ff9ca9; }
.wait { background: #3a3018; color: #ffdb8a; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📈 HR Trading AI</h1><div class="muted">NSE stock analysis • technical indicators • paper trading</div></div>', unsafe_allow_html=True)

# ----------------------------- Login -----------------------------
def get_login_secrets():
    try:
        username = str(st.secrets["auth"]["username"])
        password = str(st.secrets["auth"]["password"])
        return username, password
    except Exception:
        return None, None

def show_login():
    configured_user, configured_password = get_login_secrets()
    if not configured_user or not configured_password:
        st.error("Login is not configured yet. Please set up secrets.")
        st.stop()
    
    with st.form("login_form"):
        st.subheader("🔐 Private login")
        entered_user = st.text_input("Username")
        entered_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    
    if submitted:
        if hmac.compare_digest(entered_user, configured_user) and hmac.compare_digest(entered_password, configured_password):
            st.session_state["authenticated"] = True
            st.session_state["login_user"] = entered_user
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    st.stop()

if not st.session_state.get("authenticated", False):
    show_login()

with st.sidebar:
    st.markdown("### HR Trading AI")
    st.caption(f"Signed in as: {st.session_state.get('login_user', 'user')}")
    if st.button("Log out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# ----------------------------- Helpers -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False, timeout=20)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in raw.columns for col in required):
        return pd.DataFrame()
    raw = raw[required].copy()
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw.dropna(subset=["High", "Low", "Close"])

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"] = macd.iloc[:, 0]
        df["MACD_SIGNAL"] = macd.iloc[:, 1]
    else:
        df["MACD"] = float("nan")
        df["MACD_SIGNAL"] = float("nan")
        
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None and not adx.empty:
        adx_col = next((c for c in adx.columns if c.startswith("ADX_")), None)
        df["ADX14"] = adx[adx_col] if adx_col else float("nan")
    else:
        df["ADX14"] = float("nan")
        
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    return df

def safe_float(value, default=float("nan")):
    try:
        result = float(value)
        return result if pd.notna(result) else default
    except (TypeError, ValueError):
        return default

def format_inr(value):
    return "—" if pd.isna(value) else f"₹{value:,.2f}"

def analyse_stock(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest
    close = safe_float(latest["Close"])
    previous_close = safe_float(previous["Close"])
    daily_change = ((close / previous_close) - 1) * 100 if previous_close else float("nan")
    
    ema20, ema50, ema200 = safe_float(latest["EMA20"]), safe_float(latest["EMA50"]), safe_float(latest["EMA200"])
    rsi, atr, adx = safe_float(latest["RSI14"]), safe_float(latest["ATR14"]), safe_float(latest["ADX14"])
    macd, macd_signal = safe_float(latest["MACD"]), safe_float(latest["MACD_SIGNAL"])
    volume, volume_avg = safe_float(latest["Volume"]), safe_float(latest["VOL_AVG20"])
    
    if all(pd.notna(x) for x in [ema20, ema50, ema200]) and ema20 > ema50 > ema200:
        trend = "Bullish"
    elif all(pd.notna(x) for x in [ema20, ema50, ema200]) and ema20 < ema50 < ema200:
        trend = "Bearish"
    else:
        trend = "Sideways / mixed"
        
    volume_ratio = volume / volume_avg if pd.notna(volume_avg) and volume_avg > 0 else float("nan")
    volume_status = "High" if pd.notna(volume_ratio) and volume_ratio >= 1.5 else ("Low" if pd.notna(volume_ratio) and volume_ratio < 0.7 else "Normal")
    
    score, reasons = 0, []
    if trend == "Bullish": score += 1; reasons.append("EMA20 > EMA50 > EMA200 supports uptrend.")
    elif trend == "Bearish": score -= 1; reasons.append("EMA20 < EMA50 < EMA200 supports downtrend.")
    else: reasons.append("EMA alignment is mixed.")
    
    if pd.notna(rsi):
        if 50 <= rsi < 70: score += 1; reasons.append(f"RSI {rsi:.1f} shows positive momentum.")
        elif 30 < rsi < 50: score -= 1; reasons.append(f"RSI {rsi:.1f} shows weaker momentum.")
        elif rsi >= 70: reasons.append(f"RSI {rsi:.1f} is overbought.")
        else: reasons.append(f"RSI {rsi:.1f} is oversold.")
        
    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal: score += 1; reasons.append("MACD is above signal line.")
        elif macd < macd_signal: score -= 1; reasons.append("MACD is below signal line.")
        
    if pd.notna(volume_ratio): reasons.append(f"Volume is {volume_ratio:.2f}x average.")
    if pd.notna(adx): reasons.append(f"ADX {adx:.1f} indicates trend strength.")
    
    signal = "BUY" if score >= 2 else ("SELL" if score <= -2 else "WAIT")
    confidence = min(90, 50 + abs(score) * 10) if signal != "WAIT" else max(50, 60 - abs(score) * 5)
    
    trade_type = "Intraday candidate*" if pd.notna(adx) and pd.notna(atr) and adx >= 25 and (atr/close)*100 <= 2.0 else ("Swing candidate" if pd.notna(adx) and pd.notna(atr) and adx >= 18 and (atr/close)*100 <= 4.0 else "Positional / wait")
    
    if pd.isna(atr) or atr <= 0:
        entry = stop = target1 = target2 = float("nan")
    else:
        entry = close
        if signal == "SELL":
            stop, target1, target2 = entry + 1.5 * atr, entry - 1.5 * atr, entry - 3.0 * atr
        else:
            stop, target1, target2 = entry - 1.5 * atr, entry + 1.5 * atr, entry + 3.0 * atr
            
    return {"close": close, "daily_change": daily_change, "trend": trend, "rsi": rsi, "atr": atr, "adx": adx,
            "macd": macd, "macd_signal": macd_signal, "volume_ratio": volume_ratio, "volume_status": volume_status,
            "signal": signal, "confidence": confidence, "reasons": reasons, "trade_type": trade_type,
            "entry": entry, "stop": stop, "target1": target1, "target2": target2}

def make_chart(df: pd.DataFrame, symbol: str):
    chart = go.Figure()
    chart.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    for col, label in [("EMA20", "EMA 20"), ("EMA50", "EMA 50"), ("EMA200", "EMA 200")]:
        chart.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=label))
    chart.update_layout(title=f"{symbol} — daily candles", template="plotly_dark", height=460, margin=dict(l=8, r=8, t=50, b=8), xaxis_rangeslider_visible=False)
    return chart

# ----------------------------- Main app -----------------------------
st.markdown("### 🔎 Analyse an NSE stock")
col1, col2 = st.columns([3, 1])
with col1:
    entered_symbol = st.text_input("NSE symbol", value="TCS", max_chars=20).strip().upper()
with col2:
    period_choice = st.selectbox("History", ["1y", "2y", "5y"], index=0)

if st.button("Analyse stock", type="primary", use_container_width=True):
    clean_symbol = "".join(ch for ch in entered_symbol if ch.isalnum() or ch in ".-_")
    if clean_symbol:
        if not clean_symbol.endswith(".NS"): clean_symbol += ".NS"
        st.session_state["last_symbol"] = clean_symbol
        st.session_state["last_period"] = period_choice

symbol = st.session_state.get("last_symbol", "TCS.NS")
period = st.session_state.get("last_period", "1y")

with st.spinner(f"Fetching data for {symbol}..."):
    history = fetch_history(symbol, period)

if history.empty:
    st.warning(f"No data for {symbol}. Check symbol.")
else:
    data = add_indicators(history)
    usable = data.dropna(subset=["Close"])
    if len(usable) < 2:
        st.error("Not enough data.")
    else:
        info = analyse_stock(usable)
        st.caption(f"Symbol: {symbol} · Latest: {usable.index[-1].date()}")
        
        c1, c2 = st.columns(2)
        c1.metric("Latest close", format_inr(info["close"]))
        c2.metric("Daily change", f"{info['daily_change']:+.2f}%" if pd.notna(info["daily_change"]) else "—")
        
        signal_class = {"BUY": "buy", "SELL": "sell", "WAIT": "wait"}[info["signal"]]
        st.markdown(f'<div class="signal {signal_class}">Signal: {info["signal"]} · Confidence: {info["confidence"]}%</div>', unsafe_allow_html=True)
        
        st.subheader("🎯 Trade plan (ATR-based)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry", format_inr(info["entry"]))
        p2.metric("Stop-loss", format_inr(info["stop"]))
        p3.metric("Target 1", format_inr(info["target1"]))
        p4.metric("Main Target", format_inr(info["target2"]))
        
        if info["signal"] == "BUY": st.success("Action: BUY setup suggested.")
        elif info["signal"] == "SELL": st.error("Action: SELL setup suggested.")
        else: st.warning("Action: WAIT. Indicators not aligned.")
        
        i1, i2, i3 = st.columns(3)
        i1.metric("Trend", info["trend"])
        i2.metric("RSI (14)", f"{info['rsi']:.1f}" if pd.notna(info["rsi"]) else "—")
        i3.metric("ADX (14)", f"{info['adx']:.1f}" if pd.notna(info["adx"]) else "—")
        
        st.markdown("**Reasons:**")
        for reason in info["reasons"]: st.write(f"- {reason}")
        
        st.subheader("📈 Price chart")
        st.plotly_chart(make_chart(usable.tail(260), symbol), use_container_width=True)

st.divider()
st.markdown("## 📝 Paper trading tracker")
if "paper_trades" not in st.session_state: st.session_state["paper_trades"] = []

with st.form("paper_trade_form", clear_on_submit=True):
    t1, t2 = st.columns(2)
    with t1:
        trade_date = st.date_input("Date", value=date.today())
        trade_symbol = st.text_input("Symbol", value="TCS").strip().upper()
        direction = st.selectbox("Direction", ["LONG / BUY", "SHORT / SELL"])
    with t2:
        entry_price = st.number_input("Entry (₹)", min_value=0.01, value=100.0, step=0.5)
        exit_price = st.number_input("Exit (₹)", min_value=0.01, value=101.0, step=0.5)
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
    
    if st.form_submit_button("Save Trade", use_container_width=True):
        stock = "".join(ch for ch in trade_symbol if ch.isalnum() or ch in ".-_")
        if stock:
            if not stock.endswith(".NS"): stock += ".NS"
            multiplier = 1 if direction == "LONG / BUY" else -1
            pnl = (float(exit_price) - float(entry_price)) * int(quantity) * multiplier
            st.session_state["paper_trades"].append({"Date": str(trade_date), "Stock": stock, "Direction": direction, "Entry": float(entry_price), "Exit": float(exit_price), "Qty": int(quantity), "P&L (₹)": round(pnl, 2)})
            st.success(f"Saved! P&L: {format_inr(pnl)}")

if st.session_state["paper_trades"]:
    trade_df = pd.DataFrame(st.session_state["paper_trades"])
    total_pnl = float(trade_df["P&L (₹)"].sum())
    winners = int((trade_df["P&L (₹)"] > 0).sum())
    win_rate = (winners / len(trade_df) * 100) if len(trade_df) else 0.0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total P&L", format_inr(total_pnl))
    m2.metric("Trades", len(trade_df))
    m3.metric("Win Rate", f"{win_rate:.1f}%")
    
    st.dataframe(trade_df.iloc[::-1], use_container_width=True, hide_index=True)
    
    csv_bytes = trade_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="trades.csv", mime="text/csv", use_container_width=True)
    
    if st.button("Clear all", type="secondary"):
        st.session_state["paper_trades"] = []
        st.rerun()
else:
    st.info("No trades logged yet.")

st.caption("⚠️ Educational analysis only. Not investment advice.")