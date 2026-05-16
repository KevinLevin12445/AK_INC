import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
import time as _time
import os
import warnings
warnings.filterwarnings("ignore")

from translations import LANGS, get_lang

def t(key: str) -> str:
    return get_lang(st.session_state).get(key, key)

# ─── REAL-TIME GOLD WEBSOCKET — start background thread once per process ──────
try:
    from engines import ws_gold as _ws_gold
    _ws_gold.start()   # idempotent: safe to call on every Streamlit re-render
except Exception:
    _ws_gold = None    # graceful degradation if module missing

# ─── MARKET HOURS + AUTO-REFRESH CONFIG ───────────────────────────────────────
REFRESH_INTERVALS = {"15m": 120, "1h": 300, "4h": 600, "1d": 900}

def _is_gold_open() -> bool:
    """Gold futures trade Sun 18:00 – Fri 17:00 ET (≈ UTC-4 summer)."""
    now_et = datetime.now(timezone(timedelta(hours=-4)))
    dow, hour = now_et.weekday(), now_et.hour
    if dow == 5: return False                    # Saturday always closed
    if dow == 6 and hour < 18: return False      # Sunday before 18:00 ET
    if dow == 4 and hour >= 17: return False     # Friday after 17:00 ET
    return True

st.set_page_config(
    page_title="AK-INC TERMINAL",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── BLOOMBERG STYLE ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    background-color: #000000 !important;
    color: #00ff41 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}
.stApp { background-color: #000000 !important; }
.block-container { padding-top: 0.5rem !important; padding-bottom: 1rem; }
/* Fix overflow so title text is never clipped */
.stMarkdown, .element-container,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdown"] {
    overflow: visible !important;
}

/* Header bar */
.terminal-header {
    background: #000000;
    border-bottom: 2px solid #00ff41;
    border-top: 1px solid #003300;
    padding: 10px 24px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 32px rgba(0,255,65,0.12);
}
.terminal-logo-wrap {
    display: flex;
    align-items: center;
    gap: 18px;
}
.terminal-logo-img {
    height: 72px;
    width: auto;
    filter: invert(1) sepia(1) saturate(18) hue-rotate(86deg) brightness(1.1);
    opacity: 0.95;
    flex-shrink: 0;
    drop-shadow: 0 0 10px #00ff41;
}
.terminal-title-block {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
}
.terminal-title {
    color: #00ff41;
    font-size: 30px;
    font-weight: bold;
    letter-spacing: 6px;
    text-shadow: 0 0 18px #00ff41, 0 0 40px rgba(0,255,65,0.4);
    line-height: 1.1;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
}
.terminal-subtitle {
    color: #00aa33;
    font-size: 10px;
    letter-spacing: 4px;
    margin-top: 0px;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
}
.terminal-time {
    color: #ffd700;
    font-size: 12px;
    text-align: right;
    line-height: 1.6;
}

/* Metric cards */
.metric-card {
    background: #050505;
    border: 1px solid #00ff41;
    border-radius: 2px;
    padding: 10px 14px;
    margin: 4px 0;
}
.metric-label {
    color: #007722;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.metric-value {
    color: #00ff41;
    font-size: 22px;
    font-weight: bold;
    text-shadow: 0 0 8px #00ff41;
}
.metric-value.red { color: #ff4444; text-shadow: 0 0 8px #ff4444; }
.metric-value.gold { color: #ffd700; text-shadow: 0 0 8px #ffd700; }

/* Command terminal */
.cmd-output {
    background: #000000;
    border: 1px solid #00ff41;
    border-radius: 2px;
    padding: 12px;
    font-family: monospace;
    font-size: 12px;
    color: #00ff41;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}
.cmd-prompt { color: #ffd700; }
.cmd-header { color: #00ff41; border-bottom: 1px solid #003300; padding-bottom: 4px; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #007722 !important;
    background: #000000 !important;
    border: 1px solid #003300 !important;
    font-family: monospace !important;
}
.stTabs [aria-selected="true"] {
    color: #00ff41 !important;
    border-color: #00ff41 !important;
    background: #001100 !important;
}

/* Inputs */
.stTextInput > div > div > input {
    background-color: #000000 !important;
    color: #00ff41 !important;
    border: 1px solid #00ff41 !important;
    font-family: monospace !important;
    caret-color: #00ff41;
}
.stSelectbox > div > div {
    background-color: #000000 !important;
    color: #00ff41 !important;
    border: 1px solid #00ff41 !important;
}
.stSlider > div > div > div { background-color: #00ff41 !important; }

/* Buttons */
.stButton > button {
    background-color: #000000 !important;
    color: #00ff41 !important;
    border: 1px solid #00ff41 !important;
    font-family: monospace !important;
    letter-spacing: 2px;
    transition: all 0.1s;
}
.stButton > button:hover {
    background-color: #001a00 !important;
    box-shadow: 0 0 10px #00ff41;
}

/* Section headers */
.section-header {
    color: #00ff41;
    font-size: 11px;
    letter-spacing: 3px;
    border-bottom: 1px solid #003300;
    padding-bottom: 4px;
    margin: 8px 0;
}

/* Status dot */
.status-live { color: #00ff41; animation: blink 1s infinite; }
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; background: #000; }
::-webkit-scrollbar-thumb { background: #003300; }

/* Divider */
hr { border-color: #003300 !important; }

/* ── Hide ALL Streamlit chrome: hamburger, toolbar, deploy btn, footer ── */
#MainMenu,
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
footer,
footer * { display: none !important; visibility: hidden !important; height: 0 !important; }

/* Remove the top padding Streamlit reserves for its header */
.stApp > header { display: none !important; }
.block-container { padding-top: 0.3rem !important; }

/* ── Language toggle buttons — compact pill row ── */
div[data-testid="stHorizontalBlock"]:has(button[kind="primary"][key^="lang_btn"]),
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"][key^="lang_btn"]) {
    gap: 4px !important;
}
button[data-testid="baseButton-primary"][kind="primary"],
button[data-testid="baseButton-secondary"][kind="secondary"] {
    padding: 2px 10px !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    min-height: 28px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "data_engine" not in st.session_state:
    st.session_state.data_engine = None
if "feature_engine" not in st.session_state:
    st.session_state.feature_engine = None
if "hmm_engine" not in st.session_state:
    st.session_state.hmm_engine = None
if "bayesian_model" not in st.session_state:
    st.session_state.bayesian_model = None
if "insider_engine" not in st.session_state:
    st.session_state.insider_engine = None
if "portfolio_optimizer" not in st.session_state:
    st.session_state.portfolio_optimizer = None
if "risk_engine" not in st.session_state:
    st.session_state.risk_engine = None
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "cmd_history" not in st.session_state:
    st.session_state.cmd_history = []
if "signal_result" not in st.session_state:
    st.session_state.signal_result = {}
if "risk_metrics" not in st.session_state:
    st.session_state.risk_metrics = {}
if "period" not in st.session_state:
    st.session_state.period = "6mo"
if "interval" not in st.session_state:
    st.session_state.interval = "1d"
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "last_auto_refresh" not in st.session_state:
    st.session_state.last_auto_refresh = 0.0


# ─── ENGINE LOADER ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_engines(period="6mo", interval="1d"):
    from engines.data_engine import DataEngine
    from engines.feature_engine import FeatureEngine, KalmanFilter
    from engines.state_space_engine import HMMEngine
    from engines.bayesian_model import BayesianSignalModel
    from engines.insider_engine import InsiderEngine
    from engines.portfolio_optimizer import PortfolioOptimizer
    from engines.risk_engine import RiskEngine

    de = DataEngine(period=period, interval=interval)
    ok = de.load()
    if not ok:
        return None

    fe = FeatureEngine()
    features = fe.build(de.prices, de.returns, de.vwap)

    hmm = HMMEngine(n_components=3)
    xau_ret = de.get_xau_returns()
    rv = de.realized_vol.reindex(xau_ret.index).fillna(xau_ret.std() * np.sqrt(252))
    hmm.fit(xau_ret, rv)

    bm = BayesianSignalModel()
    fwd = xau_ret.shift(-1).dropna()
    feat_aligned = features.reindex(fwd.index).dropna()
    fwd_aligned = fwd.reindex(feat_aligned.index)
    bm.fit(feat_aligned, fwd_aligned)

    ie = InsiderEngine()
    ie.run()

    po = PortfolioOptimizer()
    po.run(de.returns)

    re = RiskEngine()
    xau_price = de.get_xau_price()
    re.run(xau_ret, xau_price)

    return {
        "data_engine": de,
        "feature_engine": fe,
        "hmm_engine": hmm,
        "bayesian_model": bm,
        "insider_engine": ie,
        "portfolio_optimizer": po,
        "risk_engine": re,
    }


def initialize_engines():
    with st.spinner("▶ LOADING MARKET DATA..."):
        result = load_all_engines(st.session_state.period, st.session_state.interval)
    if result:
        for k, v in result.items():
            st.session_state[k] = v
        st.session_state.loaded = True

        bm = st.session_state.bayesian_model
        fe = st.session_state.feature_engine
        if bm.fitted and not fe.features.empty:
            x = fe.features.iloc[-1].values
            st.session_state.signal_result = bm.predict_proba(x)

        re = st.session_state.risk_engine
        st.session_state.risk_metrics = re.metrics
    else:
        st.session_state.loaded = False
        load_all_engines.clear()   # clear cache so next retry hits network fresh


# ─── LIVE PRICE — reads from ws_gold thread (no cache needed, already in memory) ──
@st.cache_data(ttl=3, show_spinner=False)
def _cached_live_price():
    from engines.data_engine import fetch_live_price
    return fetch_live_price()


# ─── CACHED MAP DATA (5 min TTL) ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _load_map_data(period: str = "5d"):
    from engines.map_engine import MapEngine
    me = MapEngine()
    me.load_sector_data(period)
    me.load_geo_data(period)
    me.load_news()
    return me.sector_data, me.geo_data, me.news_items, me.gold_change_pct, me.gold_price

# ─── LIVE TICKER + AUTO-REFRESH FRAGMENT ──────────────────────────────────────
@st.fragment(run_every=3)
def _live_ticker_fragment():
    """Runs every 3 s — reads price from ws_gold thread (no network call here)."""

    interval      = st.session_state.get("interval", "1d")
    refresh_s     = REFRESH_INTERVALS.get(interval, 300)
    market_open   = _is_gold_open()
    now_ts        = _time.time()
    elapsed       = now_ts - st.session_state.get("last_auto_refresh", 0.0)
    remaining     = max(0, int(refresh_s - elapsed))

    # ── Trigger full data refresh when due ────────────────────────────────────
    if (st.session_state.get("auto_refresh", True)
            and st.session_state.get("loaded", False)
            and market_open
            and elapsed >= refresh_s):
        st.session_state.last_auto_refresh = now_ts
        load_all_engines.clear()
        st.session_state.loaded = False
        st.rerun(scope="app")

    # ── Fetch live spot price ──────────────────────────────────────────────────
    live     = _cached_live_price()
    live_px  = live.get("last_price", float("nan"))
    live_chg = live.get("change", float("nan"))
    live_pct = live.get("change_pct", float("nan"))
    live_src = live.get("source", "—")
    live_age = live.get("age_s", float("nan"))

    nan = float("nan")
    chg_color   = "#ff4444" if (live_chg == live_chg and live_chg < 0) else "#00ff41"
    chg_sign    = "+" if (live_chg == live_chg and live_chg >= 0) else ""
    px_display  = f"{live_px:,.2f}"           if live_px  == live_px  else "—"
    chg_display = f"{chg_sign}{live_chg:,.2f}" if live_chg == live_chg else "—"
    pct_display = f"{chg_sign}{live_pct:.3f}%" if live_pct == live_pct else "—"
    age_display = f"{live_age:.1f}s ago" if (live_age == live_age and live_age < 60) else ""

    mkt_color = "#00ff41" if market_open else "#ff6600"
    mkt_label = "● MKT OPEN" if market_open else "● MKT CLOSED"
    auto_on   = st.session_state.get("auto_refresh", True)

    if auto_on and market_open:
        refresh_info = f"AUTO ↺ <span style='color:#ffd700'>{remaining}s</span>"
    elif auto_on and not market_open:
        refresh_info = "AUTO — awaiting open"
    else:
        refresh_info = "MANUAL MODE"

    now_str  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    tf_label = f"{interval} · {st.session_state.get('period','6mo')}"
    age_html = f"&nbsp;·&nbsp; <span style='color:#00aa55'>{age_display}</span>" if age_display else ""

    st.markdown(f"""
    <div style="background:#000;border:1px solid #00ff41;border-radius:2px;padding:10px 20px;
                margin-bottom:10px;display:flex;align-items:center;gap:32px;flex-wrap:wrap;
                box-shadow:0 0 14px rgba(0,255,65,0.12);">
      <div>
        <div style="color:#007722;font-size:9px;letter-spacing:3px;">XAUUSD CFD — REAL-TIME</div>
        <div style="color:#00ff41;font-size:36px;font-weight:bold;letter-spacing:2px;
                    text-shadow:0 0 16px #00ff41;line-height:1.1;">{px_display}</div>
      </div>
      <div>
        <div style="color:#007722;font-size:9px;letter-spacing:2px;">CHANGE / %</div>
        <div style="color:{chg_color};font-size:18px;font-weight:bold;">{chg_display} &nbsp; {pct_display}</div>
      </div>
      <div style="margin-left:auto;text-align:right;">
        <div style="color:{mkt_color};font-size:10px;letter-spacing:1px;">{mkt_label}</div>
        <div style="color:#ffd700;font-size:10px;margin-top:2px;">{live_src}{age_html} &nbsp;|&nbsp; {tf_label}</div>
        <div style="color:#444;font-size:9px;margin-top:2px;">{now_str} &nbsp;|&nbsp; {refresh_info}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── COMMAND PROCESSOR ────────────────────────────────────────────────────────
def process_command(cmd: str) -> str:
    cmd = cmd.strip().lower()
    if not st.session_state.loaded:
        return "[ERROR] — Engines not loaded. Click INITIALIZE first."

    de = st.session_state.data_engine
    fe = st.session_state.feature_engine
    hmm = st.session_state.hmm_engine
    bm = st.session_state.bayesian_model
    ie = st.session_state.insider_engine
    po = st.session_state.portfolio_optimizer
    re = st.session_state.risk_engine

    if cmd == "/signal":
        sig = st.session_state.signal_result
        if not sig:
            return "[ERROR] — Signal model not fitted."
        lines = [
            "═══════════════════════════════",
            "   BAYESIAN SIGNAL ENGINE",
            "═══════════════════════════════",
            f"  P(LONG)   : {sig.get('P_long', 0):.4f}  [{sig.get('P_long', 0)*100:.1f}%]",
            f"  P(SHORT)  : {sig.get('P_short', 0):.4f}  [{sig.get('P_short', 0)*100:.1f}%]",
            f"  SIGNAL    : *** {sig.get('signal', 'NEUTRAL')} ***",
            f"  CONFIDENCE: {sig.get('confidence', 0):.4f}",
            "───────────────────────────────",
        ]
        return "\n".join(lines)

    elif cmd == "/kalman":
        kf = fe.kf
        price = de.get_xau_price()
        trend = fe.kalman_trend
        noise = fe.kalman_noise
        if trend.empty:
            return "[ERROR] — Kalman filter not computed."
        last_price = float(price.iloc[-1])
        last_trend = float(trend.iloc[-1])
        last_noise = float(noise.iloc[-1])
        direction = "UP" if last_price > last_trend else "DOWN"
        lines = [
            "═══════════════════════════════",
            "   KALMAN FILTER — TREND EXTRACTION",
            "═══════════════════════════════",
            f"  Market Price : {last_price:.4f}",
            f"  Kalman Trend : {last_trend:.4f}",
            f"  Noise Resid  : {last_noise:.6f}",
            f"  Direction    : {direction}",
            f"  Process Noise: {kf.Q:.2e}",
            f"  Obs Noise    : {kf.R:.2e}",
            "───────────────────────────────",
        ]
        return "\n".join(lines)

    elif cmd == "/hmm":
        if not hmm.fitted:
            return "[ERROR] — HMM not fitted."
        state = hmm.current_state()
        probs = hmm.current_probs()
        from engines.state_space_engine import STATE_LABELS
        label = STATE_LABELS.get(state, "UNKNOWN")
        tm = hmm.transition_matrix()
        lines = [
            "═══════════════════════════════",
            "   HIDDEN MARKOV MODEL — REGIME",
            "═══════════════════════════════",
            f"  Current State: [{state}] {label}",
            "  State Probabilities:",
        ]
        for s, p in probs.items():
            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
            lines.append(f"    {s:<18}: {bar} {p:.4f}")
        lines.append("  Transition Matrix:")
        for i, row in enumerate(tm):
            lines.append(f"    {STATE_LABELS[i]:<16}: " + " ".join(f"{v:.3f}" for v in row))
        lines.append("───────────────────────────────")
        return "\n".join(lines)

    elif cmd == "/cointegration":
        cr = fe.coint_result
        if not cr:
            return "[ERROR] — DXY data unavailable. Cointegration not computed."
        spread = cr.get("spread", pd.Series())
        zscore = cr.get("zscore", pd.Series())
        beta = cr.get("beta", np.nan)
        r2 = cr.get("r_squared", np.nan)
        lines = [
            "═══════════════════════════════",
            "   COINTEGRATION — GOLD vs DXY",
            "═══════════════════════════════",
            f"  Beta (hedge ratio): {beta:.6f}",
            f"  R²                : {r2:.4f}",
        ]
        if not spread.empty:
            lines.append(f"  Current Spread    : {float(spread.iloc[-1]):.4f}")
        if not zscore.empty:
            z = float(zscore.iloc[-1])
            signal = "LONG GOLD" if z < -2 else "SHORT GOLD" if z > 2 else "NEUTRAL"
            lines.append(f"  Spread Z-Score    : {z:.4f}")
            lines.append(f"  Signal            : {signal}")
        lines.append("  Spread = Gold - β × DXY")
        lines.append("───────────────────────────────")
        return "\n".join(lines)

    elif cmd == "/insider":
        summary = ie.summary()
        score_series = ie.score_series
        lines = [
            "═══════════════════════════════",
            "   INSIDER ENGINE — FLOW ANALYSIS",
            "═══════════════════════════════",
            f"  Current Score    : {summary['current_score']:+.2f}",
            f"  dS/dt (momentum) : {summary['momentum']:+.2f}",
            f"  Total Trxn       : {summary['n_transactions']}",
            f"  Buy Clusters     : {summary['n_buy_clusters']}",
            f"  Signal           : {summary['signal']}",
        ]
        if not score_series.empty:
            ma5 = score_series.rolling(5).mean().iloc[-1]
            lines.append(f"  Score MA(5)      : {ma5:+.2f}")
        lines.append("  [Live SEC EDGAR data — real Form 4 filings]")
        lines.append("───────────────────────────────")
        return "\n".join(lines)

    elif cmd == "/markowitz":
        res = po.result
        if not res or not res.get("weights"):
            return "[ERROR] — Optimizer not run."
        lines = [
            "═══════════════════════════════",
            "   MARKOWITZ PORTFOLIO OPTIMIZER",
            "═══════════════════════════════",
            f"  Expected Return : {res['expected_return']:.4f} ({res['expected_return']*100:.2f}% p.a.)",
            f"  Portfolio Vol   : {res['portfolio_vol']:.4f} ({res['portfolio_vol']*100:.2f}% p.a.)",
            f"  Sharpe Ratio    : {res['sharpe']:.4f}",
            f"  Converged       : {res.get('success', False)}",
            "  Weights:",
        ]
        for asset, w in res["weights"].items():
            bar = "█" * int(w * 30) + "░" * (30 - int(w * 30))
            lines.append(f"    {asset:<10}: {bar} {w:.4f}")
        lines.append("  Objective: max μᵀw − λwᵀΣw − γ||w−w_prev||²")
        lines.append("───────────────────────────────")
        return "\n".join(lines)

    elif cmd == "/heatmap":
        return "[INFO] — Switch to HEATMAP tab to view Order Flow visualization."

    elif cmd == "/gc3d":
        return "[INFO] — Switch to GC3D tab to view the 3D Alpha Surface."

    elif cmd == "/risk":
        m = st.session_state.risk_metrics
        if not m:
            return "[ERROR] — Risk metrics not computed."
        lines = [
            "═══════════════════════════════",
            "   RISK ENGINE — INSTITUTIONAL",
            "═══════════════════════════════",
            f"  VaR 95% (param)  : {m.get('VaR_95_param', np.nan):.6f}",
            f"  VaR 95% (hist)   : {m.get('VaR_95_hist', np.nan):.6f}",
            f"  Expected Shortfall: {m.get('ES_95', np.nan):.6f}",
            f"  Max Drawdown     : {m.get('max_drawdown', np.nan)*100:.2f}%",
            f"  Current Drawdown : {m.get('current_drawdown', np.nan)*100:.2f}%",
            f"  Kelly Fraction   : {m.get('kelly_fraction', np.nan):.4f}",
            f"  Info Ratio       : {m.get('information_ratio', np.nan):.4f}",
            f"  Sharpe Ratio     : {m.get('sharpe_ratio', np.nan):.4f}",
            f"  Daily Vol        : {m.get('vol_daily', np.nan):.6f}",
            f"  Annual Vol       : {m.get('vol_annual', np.nan)*100:.2f}%",
            "───────────────────────────────",
        ]
        return "\n".join(lines)

    elif cmd in ("/help", "help", "/?"):
        lines = [
            "═══════════════════════════════",
            "   AK-INC TERMINAL — COMMANDS",
            "═══════════════════════════════",
            "  /signal       → Bayesian signal probabilities",
            "  /kalman       → Kalman filter trend extraction",
            "  /hmm          → Hidden Markov Model regimes",
            "  /cointegration → Gold/DXY cointegration spread",
            "  /insider      → Insider flow analysis",
            "  /markowitz    → Portfolio optimization",
            "  /risk         → Risk metrics (VaR, ES, DD)",
            "  /heatmap      → Order flow heatmap (tab)",
            "  /gc3d         → 3D alpha surface (tab)",
            "  /help         → This menu",
            "───────────────────────────────",
        ]
        return "\n".join(lines)

    else:
        return f"[ERROR] Unknown command: '{cmd}'. Type /help for commands."


# ─── HEADER ───────────────────────────────────────────────────────────────────
now_str = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")

_LOGO_B64 = open(os.path.join(os.path.dirname(__file__), "static", "logo_small_b64.txt")).read().strip()

# ── Language selector (rendered BEFORE header HTML so state is set first) ──
_lang_cols = st.columns([6, 1, 1, 1])
with _lang_cols[1]:
    if st.button("EN", key="lang_btn_en", use_container_width=True,
                 help="English", type="primary" if st.session_state.lang == "en" else "secondary"):
        st.session_state.lang = "en"
        st.rerun()
with _lang_cols[2]:
    if st.button("ES", key="lang_btn_es", use_container_width=True,
                 help="Español", type="primary" if st.session_state.lang == "es" else "secondary"):
        st.session_state.lang = "es"
        st.rerun()
with _lang_cols[3]:
    if st.button("AR", key="lang_btn_ar", use_container_width=True,
                 help="العربية", type="primary" if st.session_state.lang == "ar" else "secondary"):
        st.session_state.lang = "ar"
        st.rerun()

_is_rtl = st.session_state.lang == "ar"
_dir    = 'dir="rtl"' if _is_rtl else ''
_txt_align = "right" if _is_rtl else "left"
_flex_dir  = "row-reverse" if _is_rtl else "row"

st.markdown(f"""
<div style="display:flex;flex-direction:{_flex_dir};align-items:center;
            justify-content:space-between;
            background:#000;border-bottom:2px solid #00ff41;
            padding:14px 24px;margin-bottom:6px;
            box-shadow:0 4px 24px rgba(0,255,65,0.14);">
  <div style="display:flex;flex-direction:{_flex_dir};align-items:center;gap:18px;">
    <img src="data:image/png;base64,{_LOGO_B64}"
         style="height:72px;width:auto;display:block;flex-shrink:0;"/>
    <div {_dir} style="text-align:{_txt_align};">
      <div style="color:#00ff41;font-size:32px;font-weight:900;
                  letter-spacing:8px;line-height:1.25;
                  text-shadow:0 0 22px #00ff41,0 0 48px rgba(0,255,65,0.5);
                  font-family:'Share Tech Mono','Courier New',monospace;">
        AK-INC TERMINAL
      </div>
      <div style="color:#00cc44;font-size:10px;letter-spacing:3px;margin-top:4px;
                  font-family:'Share Tech Mono','Courier New',monospace;">
        {t("subtitle")}
      </div>
    </div>
  </div>
  <div style="text-align:right;font-family:'Share Tech Mono','Courier New',monospace;flex-shrink:0;">
    <div style="color:#00ff41;font-size:14px;letter-spacing:3px;font-weight:bold;">{t("live")}</div>
    <div style="color:#ffd700;font-size:10px;margin-top:5px;letter-spacing:1px;">{now_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── CONTROL ROW ──────────────────────────────────────────────────────────────
from engines.data_engine import INTERVAL_PERIOD_MAP, DEFAULT_PERIOD

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1.4, 1.6, 1, 1, 1])

with ctrl_col1:
    st.markdown(f'<div class="section-header">{t("timeframe")}</div>', unsafe_allow_html=True)
    interval_opt = st.selectbox(
        "interval",
        ["15m", "1h", "4h", "1d"],
        index=["15m", "1h", "4h", "1d"].index(st.session_state.interval),
        label_visibility="collapsed",
        key="interval_select",
    )
    if interval_opt != st.session_state.interval:
        st.session_state.interval = interval_opt
        # Reset period to a valid default for the new interval
        st.session_state.period = DEFAULT_PERIOD[interval_opt]
        st.session_state.loaded = False
        load_all_engines.clear()

with ctrl_col2:
    st.markdown(f'<div class="section-header">{t("period")}</div>', unsafe_allow_html=True)
    period_choices = INTERVAL_PERIOD_MAP[st.session_state.interval]
    cur_period = st.session_state.period
    if cur_period not in period_choices:
        cur_period = period_choices[0]
        st.session_state.period = cur_period
    period_opt = st.selectbox(
        "period", period_choices,
        index=period_choices.index(cur_period),
        label_visibility="collapsed",
        key="period_select",
    )
    if period_opt != st.session_state.period:
        st.session_state.period = period_opt
        st.session_state.loaded = False
        load_all_engines.clear()

with ctrl_col3:
    st.markdown(f'<div class="section-header">{t("engine")}</div>', unsafe_allow_html=True)
    if st.button(t("initialize"), use_container_width=True):
        initialize_engines()
        st.rerun()

with ctrl_col4:
    st.markdown(f'<div class="section-header">{t("status")}</div>', unsafe_allow_html=True)
    mkt_open_now = _is_gold_open()
    mkt_color_now = "#00ff41" if mkt_open_now else "#ff6600"
    mkt_lbl_now   = t("mkt_open") if mkt_open_now else t("mkt_closed")
    if st.session_state.loaded:
        tf_label = f"{st.session_state.interval} · {st.session_state.period}"
        st.markdown(
            f'<div style="color:#00ff41;font-size:11px;letter-spacing:1px;">'
            f'{t("online")}<br>'
            f'<span style="color:{mkt_color_now};font-size:9px;">{mkt_lbl_now}</span><br>'
            f'<span style="color:#007722;font-size:9px;">{tf_label}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="color:#ff4444;font-size:12px;letter-spacing:2px;">'
            f'{t("offline")}<br>'
            f'<span style="color:{mkt_color_now};font-size:9px;">{mkt_lbl_now}</span></div>',
            unsafe_allow_html=True,
        )

with ctrl_col5:
    st.markdown(f'<div class="section-header">{t("auto_refresh")}</div>', unsafe_allow_html=True)
    c5a, c5b = st.columns(2)
    with c5a:
        if st.button("↺", use_container_width=True, help=t("manual_reload")):
            load_all_engines.clear()
            st.session_state.loaded = False
            st.session_state.last_auto_refresh = 0.0
            initialize_engines()
            st.rerun()
    with c5b:
        auto_lbl = t("on") if st.session_state.get("auto_refresh", True) else t("off")
        if st.button(auto_lbl, use_container_width=True, help=t("toggle_refresh")):
            st.session_state.auto_refresh = not st.session_state.get("auto_refresh", True)
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# Auto-load on first visit
if not st.session_state.loaded:
    initialize_engines()
    if st.session_state.loaded:
        st.rerun()
    else:
        st.error(
            "⚠ **Market data unavailable** — Yahoo Finance is rate-limiting this server's IP.  \n"
            "The live gold price (gold-api.com) still works. Click **INITIALIZE** in the sidebar "
            "to retry, or wait a few minutes and refresh the page.",
            icon="⚠",
        )
        st.stop()

# ─── LIVE TICKER (auto-refreshes every 30 s, triggers data reload when due) ───
_live_ticker_fragment()

# ─── INSIDER LIVE PANEL (auto-refreshes every 5 min, direct Form 4 fetch) ─────
@st.fragment(run_every=30)
def _insider_live_panel():
    """Auto-refreshes every 30 s — fetches latest SEC EDGAR Form 4 filings."""
    from engines.insider_engine import fetch_real_transactions, InsiderEngine
    import plotly.graph_objects as _go

    fetch_ts = datetime.utcnow().strftime("%H:%M:%S UTC")

    # ── Cache last valid result in session_state so UI never goes blank ────────
    cache_key = "_insider_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {"txns": None, "src": None, "ts": None}

    with st.spinner(""):
        try:
            txns, src = fetch_real_transactions(lookback_days=90)
            if txns is not None and not txns.empty:
                # Only update cache when we get real new data
                st.session_state[cache_key] = {"txns": txns, "src": src, "ts": fetch_ts}
        except Exception as e:
            src  = f"ERROR: {e}"
            txns = None

    # Use cached data if live fetch returned nothing
    cached = st.session_state[cache_key]
    if (txns is None or (hasattr(txns, "empty") and txns.empty)) and cached["txns"] is not None:
        txns = cached["txns"]
        src  = cached["src"] + f"  ·  cached {cached['ts']}"

    ie = InsiderEngine()
    if txns is not None and not txns.empty:
        ie.data_source = src
        ie.load(txns)
        ie.build_score_series()
        ie.compute_momentum()
        ie.detect_clusters()
    else:
        ie.data_source = "LIVE DATA UNAVAILABLE (SEC EDGAR unreachable)"
        ie.load(None)
        ie.build_score_series()
        ie.compute_momentum()

    ins_sum = ie.summary()
    src = ins_sum.get("data_source", "—")
    src_color = "#ffd700" if "EDGAR" in src else "#ff4444"
    src_label = t("src_real") if "EDGAR" in src else "⚠ LIVE DATA UNAVAILABLE"

    st.markdown(
        f'<div class="section-header">{t("sh_insider")} &nbsp;'
        f'<span style="color:{src_color};font-size:9px;">{src_label}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:#444;font-size:9px;margin-bottom:4px;letter-spacing:1px;">'
        f'{src}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:#003300;font-size:8px;letter-spacing:1px;margin-bottom:6px;">'
        f'{t("last_fetch")}: {fetch_ts} &nbsp;·&nbsp; {t("auto_refresh_lbl")}</div>',
        unsafe_allow_html=True,
    )

    if not ins_sum.get("data_available", True):
        st.markdown(
            "<div style='background:#1a0000;border:1px solid #ff4444;border-radius:4px;"
            "padding:8px 14px;font-family:monospace;font-size:10px;color:#ff4444;"
            "letter-spacing:1px;margin-bottom:8px;'>"
            "⚠ LIVE DATA UNAVAILABLE — SEC EDGAR (Form 4) is currently unreachable.<br>"
            "<span style='color:#555;font-size:9px;'>Metrics are zeroed. No synthetic data is displayed. "
            "Try reinitializing in a few minutes.</span></div>",
            unsafe_allow_html=True,
        )

    sig = ins_sum["signal"]
    sig_color = "#00ff41" if sig == "BULLISH" else ("#ff4444" if sig == "BEARISH" else "#666666")
    ie_metrics = [
        (t("signal"),       sig,                               sig_color),
        (t("score"),        f"{ins_sum['current_score']:+.2f}" if ins_sum.get("data_available") else "—", sig_color),
        (t("momentum"),     f"{ins_sum['momentum']:+.2f}"      if ins_sum.get("data_available") else "—", "#ffd700"),
        (t("buys"),         str(ins_sum["n_buys"])             if ins_sum.get("data_available") else "—", "#00ff41"),
        (t("sells"),        str(ins_sum["n_sells"])            if ins_sum.get("data_available") else "—", "#ff4444"),
        (t("buy_clusters"), str(ins_sum["n_buy_clusters"])     if ins_sum.get("data_available") else "—", "#ffd700"),
    ]
    mc1, mc2 = st.columns(2)
    for i, (label, val, col) in enumerate(ie_metrics):
        target = mc1 if i % 2 == 0 else mc2
        target.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="font-size:15px;color:{col}">{val}</div></div>',
            unsafe_allow_html=True,
        )

    if not ie.score_series.empty:
        fig_ins = _go.Figure()
        fig_ins.add_trace(_go.Scatter(
            x=ie.score_series.index, y=ie.score_series.values,
            mode="lines", line=dict(color="#00ff41", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,255,65,0.08)",
        ))
        fig_ins.add_hline(y=0, line=dict(color="#ffd700", width=1, dash="dash"))
        fig_ins.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            xaxis=dict(color="#00ff41", gridcolor="#0a2a0a", showticklabels=False),
            yaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
            height=110, margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_ins, width="stretch")

    # ── Recent Form 4 transactions table ──────────────────────────────────────
    recent_txns = ie.recent_transactions(10)
    if not recent_txns.empty:
        st.markdown(
            f'<div style="color:#00aa33;font-size:9px;letter-spacing:2px;margin-top:4px;'
            f'border-top:1px solid #003300;padding-top:6px;">{t("recent_form4")}</div>',
            unsafe_allow_html=True,
        )
        rows_html = ""
        for _, row in recent_txns.iterrows():
            ttype  = row.get("type", "—")
            tcol   = "#00ff41" if ttype == "BUY" else "#ff4444"
            ticker = row.get("ticker", "—")
            ins    = str(row.get("insider", "—"))[:20]
            role   = str(row.get("role", "—"))[:12]
            val    = row.get("value", 0)
            val_s  = f"${val/1e6:.2f}M" if val >= 1e6 else f"${val/1e3:.0f}K"
            ts     = row.get("timestamp", "")
            ts_s   = pd.Timestamp(ts).strftime("%m-%d") if ts != "" else "—"
            rows_html += (
                f'<tr style="border-bottom:1px solid #081808;">'
                f'<td style="color:#ffd700;padding:3px 5px;white-space:nowrap;">{ts_s}</td>'
                f'<td style="color:#00aaff;padding:3px 5px;">{ticker}</td>'
                f'<td style="color:#cccccc;padding:3px 5px;font-size:9px;">{ins}</td>'
                f'<td style="color:#666;padding:3px 5px;font-size:9px;">{role}</td>'
                f'<td style="color:{tcol};font-weight:bold;padding:3px 5px;">{ttype}</td>'
                f'<td style="color:#00ff41;padding:3px 5px;text-align:right;">{val_s}</td>'
                f'</tr>'
            )
        st.markdown(
            '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:10px;">'
            '<thead><tr style="color:#007722;border-bottom:1px solid #00ff41;">'
            '<th style="padding:3px 5px;text-align:left;">DATE</th>'
            '<th style="text-align:left;">TKR</th>'
            '<th style="text-align:left;">INSIDER</th>'
            '<th style="text-align:left;">ROLE</th>'
            '<th style="text-align:left;">TYPE</th>'
            '<th style="text-align:right;">VALUE</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#444;font-size:9px;margin-top:8px;">No Form 4 data available.</div>',
            unsafe_allow_html=True,
        )


# ─── COMMAND TAB LIVE OUTPUT FRAGMENT ────────────────────────────────────────
# refresh rates per command (seconds); None = no auto-refresh
_CMD_REFRESH = {
    "/signal":        30,
    "/kalman":        30,
    "/hmm":           60,
    "/cointegration": 60,
    "/insider":       90,
    "/markowitz":    120,
    "/risk":          60,
    "/heatmap":      None,
    "/gc3d":         None,
    "/help":         None,
}

@st.fragment(run_every=30)
def _cmd_live_output():
    """Auto-refreshes every 30 s — re-runs the active command and shows output."""
    active = st.session_state.get("active_cmd")
    if not active:
        st.markdown(
            '<div class="cmd-output">'
            '<span style="color:#333;font-family:monospace;">▷  Select a command above or type one below</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    refresh_s = _CMD_REFRESH.get(active)
    if refresh_s and st.session_state.get("loaded"):
        output = process_command(active)
        st.session_state.cmd_output = output
    else:
        output = st.session_state.get("cmd_output", "")

    # ── Status bar ──────────────────────────────────────────────────────────
    lbl = f"⟳ auto {refresh_s}s" if refresh_s else "static"
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"font-family:monospace;font-size:10px;color:#444;"
        f"border-top:1px solid #0a2a0a;border-bottom:1px solid #0a2a0a;"
        f"padding:3px 6px;margin-bottom:6px;'>"
        f"<span style='color:#00ff41;letter-spacing:1px;'>▶ {active}</span>"
        f"<span style='color:#335533;'>{lbl}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    hist_text = f"<span class='cmd-prompt'>AK-INC >> {active}</span>\n{output}\n"
    st.markdown(f'<div class="cmd-output">{hist_text}</div>', unsafe_allow_html=True)


# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab_quant, tab_cmd, tab_kalman, tab_hmm, tab_heat, tab_gc3d, tab_risk, tab_portfolio, tab_map, tab_vwap, tab_anomaly, tab_volprofile = st.tabs([
    t("tab_quant"),
    t("tab_cmd"),
    t("tab_kalman"),
    t("tab_hmm"),
    t("tab_heat"),
    t("tab_gc3d"),
    t("tab_risk"),
    t("tab_portfolio"),
    t("tab_map"),
    t("tab_vwap"),
    t("tab_anomaly"),
    t("tab_volprofile"),
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUANT PANEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_quant:
    if not st.session_state.loaded:
        st.markdown(f'<div class="cmd-output">{t("offline_msg")}</div>', unsafe_allow_html=True)
    else:
        de = st.session_state.data_engine
        fe = st.session_state.feature_engine
        hmm = st.session_state.hmm_engine
        bm = st.session_state.bayesian_model
        ie = st.session_state.insider_engine
        re = st.session_state.risk_engine

        ds = de.summary()
        sig = st.session_state.signal_result
        rm = st.session_state.risk_metrics

        # Row 1 — secondary metrics
        st.markdown(f'<div class="section-header">{t("sh_market")}</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        last_px = ds.get("last_price", np.nan)
        chg = ds.get("price_change_pct", np.nan)
        chg_class = "red" if (not np.isnan(chg) and chg < 0) else ""
        cs = "+" if (not np.isnan(chg) and chg > 0) else ""

        prev_close = ds.get("previous_close", ds.get("last_price", np.nan))
        c1.markdown(f'<div class="metric-card"><div class="metric-label">{t("prev_close")}</div><div class="metric-value">{prev_close:,.2f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">{t("change_pct")}</div><div class="metric-value {chg_class}">{cs}{chg:.3f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">{t("real_vol")}</div><div class="metric-value">{ds.get("realized_vol", 0)*100:.2f}%</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">{t("sharpe_ann")}</div><div class="metric-value">{ds.get("sharpe_approx", 0):.3f}</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="metric-card"><div class="metric-label">{t("n_obs")}</div><div class="metric-value">{ds.get("n_obs", 0)}</div></div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Row 2 — Bayesian signal + HMM
        col_sig, col_hmm = st.columns(2)

        with col_sig:
            st.markdown(f'<div class="section-header">{t("sh_bayesian")}</div>', unsafe_allow_html=True)
            p_long = sig.get("P_long", 0.5)
            p_short = sig.get("P_short", 0.5)
            signal = sig.get("signal", "NEUTRAL")
            conf = sig.get("confidence", 0)

            sig_color = "#00ff41" if signal == "LONG" else "#ff4444" if signal == "SHORT" else "#ffd700"
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("active_signal")}</div><div class="metric-value" style="color:{sig_color};text-shadow: 0 0 12px {sig_color};">{signal}</div></div>', unsafe_allow_html=True)

            sc1, sc2, sc3 = st.columns(3)
            sc1.markdown(f'<div class="metric-card"><div class="metric-label">{t("p_long")}</div><div class="metric-value">{p_long:.4f}</div></div>', unsafe_allow_html=True)
            sc2.markdown(f'<div class="metric-card"><div class="metric-label">{t("p_short")}</div><div class="metric-value red">{p_short:.4f}</div></div>', unsafe_allow_html=True)
            sc3.markdown(f'<div class="metric-card"><div class="metric-label">{t("confidence")}</div><div class="metric-value gold">{conf:.4f}</div></div>', unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_long * 100,
                title={"text": "P(LONG) %", "font": {"color": "#00ff41", "family": "monospace", "size": 12}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#00ff41"},
                    "bar": {"color": "#00ff41"},
                    "bgcolor": "#000000",
                    "bordercolor": "#00ff41",
                    "steps": [
                        {"range": [0, 40], "color": "#1a0000"},
                        {"range": [40, 60], "color": "#1a1a00"},
                        {"range": [60, 100], "color": "#001a00"},
                    ],
                    "threshold": {"line": {"color": "#ffd700", "width": 2}, "thickness": 0.75, "value": 50},
                },
                number={"font": {"color": "#00ff41", "family": "monospace"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#000000", font_color="#00ff41",
                height=200, margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, width="stretch")

        with col_hmm:
            st.markdown(f'<div class="section-header">{t("sh_hmm_detector")}</div>', unsafe_allow_html=True)
            if hmm.fitted:
                state = hmm.current_state()
                probs = hmm.current_probs()
                from engines.state_space_engine import STATE_LABELS, STATE_COLORS
                state_label = STATE_LABELS.get(state, "UNKNOWN")
                state_col = STATE_COLORS.get(state, "#00ff41")

                st.markdown(f'<div class="metric-card"><div class="metric-label">{t("current_regime")}</div><div class="metric-value" style="color:{state_col};text-shadow: 0 0 12px {state_col};">{state_label}</div></div>', unsafe_allow_html=True)

                cats = list(probs.keys())
                vals = list(probs.values())
                colors = [STATE_COLORS.get(i, "#00ff41") for i in range(len(cats))]

                fig_bar = go.Figure(go.Bar(
                    x=vals,
                    y=cats,
                    orientation="h",
                    marker=dict(
                        color=colors,
                        line=dict(color="#000000", width=1),
                    ),
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(color="#00ff41", family="monospace", size=11),
                ))
                fig_bar.update_layout(
                    paper_bgcolor="#000000",
                    plot_bgcolor="#000000",
                    xaxis=dict(range=[0, 1], color="#00ff41", gridcolor="#0a2a0a"),
                    yaxis=dict(color="#00ff41"),
                    font=dict(color="#00ff41", family="monospace"),
                    height=220,
                    margin=dict(l=10, r=60, t=10, b=20),
                )
                st.plotly_chart(fig_bar, width="stretch")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Row 3 — Risk + Insider + Features
        r3c1, r3c2, r3c3 = st.columns(3)

        with r3c1:
            st.markdown(f'<div class="section-header">{t("sh_risk")}</div>', unsafe_allow_html=True)
            rm_metrics = [
                (t("var95_hist"),    f"{rm.get('VaR_95_hist', np.nan):.6f}"),
                (t("exp_shortfall"), f"{rm.get('ES_95', np.nan):.6f}"),
                (t("max_drawdown"),  f"{rm.get('max_drawdown', np.nan)*100:.2f}%"),
                (t("curr_drawdown"), f"{rm.get('current_drawdown', np.nan)*100:.2f}%"),
                (t("annual_vol"),    f"{rm.get('vol_annual', np.nan)*100:.2f}%"),
                (t("sharpe_ratio"),  f"{rm.get('sharpe_ratio', np.nan):.4f}"),
            ]
            for label, val in rm_metrics:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:16px">{val}</div></div>', unsafe_allow_html=True)

        with r3c2:
            _insider_live_panel()

        with r3c3:
            st.markdown(f'<div class="section-header">{t("sh_factor")}</div>', unsafe_allow_html=True)
            latest = fe.get_latest()
            for feat, val in list(latest.items())[:8]:
                bar_len = min(int(abs(val) * 10), 20) if not np.isnan(val) else 0
                color = "#00ff41" if val >= 0 else "#ff4444"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{feat}</div>'
                    f'<div class="metric-value" style="font-size:14px;color:{color}">{val:.4f}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Price chart
        st.markdown(f'<div class="section-header">{t("sh_price_kalman")}</div>', unsafe_allow_html=True)
        xau_price = de.get_xau_price()
        kalman_trend = fe.kalman_trend

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=xau_price.index, y=xau_price.values,
            mode="lines", name="XAUUSD",
            line=dict(color="#00ff41", width=1.5),
        ))
        if not kalman_trend.empty:
            fig_price.add_trace(go.Scatter(
                x=kalman_trend.index, y=kalman_trend.values,
                mode="lines", name="Kalman Trend",
                line=dict(color="#ffd700", width=2, dash="dot"),
            ))
        fig_price.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            xaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
            yaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
            legend=dict(font=dict(color="#00ff41", family="monospace"), bgcolor="#000000"),
            font=dict(color="#00ff41", family="monospace"),
            height=280, margin=dict(l=40, r=20, t=10, b=30),
        )
        st.plotly_chart(fig_price, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMMAND TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cmd:
    st.markdown(f'<div class="section-header">{t("sh_cmd")}</div>', unsafe_allow_html=True)

    # ── Session state init ─────────────────────────────────────────────────
    if "active_cmd" not in st.session_state:
        st.session_state.active_cmd = None
    if "cmd_output" not in st.session_state:
        st.session_state.cmd_output = ""

    def _select_cmd(cmd: str):
        """Switch active command — clears output and re-executes."""
        if cmd != st.session_state.active_cmd:
            st.session_state.active_cmd = cmd
            st.session_state.cmd_output = ""
            st.session_state.cmd_history = []
        if st.session_state.get("loaded"):
            st.session_state.cmd_output = process_command(cmd)

    # ── Command palette ────────────────────────────────────────────────────
    # Category definitions: (label, color, commands)
    CMD_PALETTE = [
        ("SIGNAL",    "#00cc44", ["/signal", "/kalman", "/hmm", "/cointegration"]),
        ("FLOW",      "#ffd700", ["/insider"]),
        ("PORTFOLIO", "#00aaff", ["/markowitz"]),
        ("RISK",      "#ff7700", ["/risk"]),
        ("CHARTS",    "#aa44ff", ["/heatmap", "/gc3d"]),
        ("HELP",      "#666666", ["/help"]),
    ]

    # Build label + button rows per category
    for cat_label, cat_color, cat_cmds in CMD_PALETTE:
        row_cols = st.columns([0.85] + [1.5] * len(cat_cmds) + [10 - 0.85 - 1.5 * len(cat_cmds)])
        row_cols[0].markdown(
            f"<div style='font-family:monospace;font-size:9px;color:{cat_color};"
            f"letter-spacing:2px;padding-top:8px;text-align:right;"
            f"border-right:2px solid {cat_color}22;padding-right:6px;'>"
            f"{cat_label}</div>",
            unsafe_allow_html=True,
        )
        for idx, cmd in enumerate(cat_cmds):
            is_active = st.session_state.active_cmd == cmd
            btn_style = "primary" if is_active else "secondary"
            if row_cols[idx + 1].button(
                cmd,
                key=f"cmd_btn_{cmd.replace('/', '_')}",
                use_container_width=True,
                type=btn_style,
            ):
                _select_cmd(cmd)
                st.rerun()

    st.markdown(
        "<div style='border-bottom:1px solid #0a2a0a;margin:8px 0 10px;'></div>",
        unsafe_allow_html=True,
    )

    # ── Manual text input ──────────────────────────────────────────────────
    inp_col, btn_col, clr_col = st.columns([6, 1, 1])
    manual_cmd = inp_col.text_input(
        "manual_cmd",
        placeholder="▷  /command  or  /help",
        key="cmd_manual_input",
        label_visibility="collapsed",
    )
    exec_clicked = btn_col.button("EXEC ↵", key="cmd_exec_btn", use_container_width=True)
    clear_clicked = clr_col.button("⌫ CLR", key="cmd_clr_btn", use_container_width=True)

    if exec_clicked and manual_cmd.strip():
        _select_cmd(manual_cmd.strip().lower())
        st.rerun()

    if clear_clicked:
        st.session_state.active_cmd = None
        st.session_state.cmd_output = ""
        st.session_state.cmd_history = []
        st.rerun()

    # ── Live output panel (auto-refresh fragment) ──────────────────────────
    _cmd_live_output()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_kalman:
    if not st.session_state.loaded:
        st.info(t("init_first"))
    else:
        de = st.session_state.data_engine
        fe = st.session_state.feature_engine

        st.markdown(f'<div class="section-header">{t("sh_kalman")}</div>', unsafe_allow_html=True)

        kc1, kc2 = st.columns(2)
        with kc1:
            kf_q = st.slider("Process Noise (Q × 1e-4)", 1, 100, 10, key="kf_q")
            kf_r = st.slider("Obs Noise (R × 1e-2)", 1, 100, 10, key="kf_r")

        xau_price = de.get_xau_price()

        if xau_price.empty:
            st.warning("⚠ No XAUUSD price data for this interval/period. Click INITIALIZE or switch to 1d.")
        else:
            from engines.feature_engine import KalmanFilter as KF
            kf_custom = KF(Q=kf_q * 1e-4, R=kf_r * 1e-2)
            custom_trend = kf_custom.smooth_series(xau_price)
            custom_noise = kf_custom.residual_noise(xau_price)

            fig_kf = go.Figure()
            fig_kf.add_trace(go.Scatter(
                x=xau_price.index, y=xau_price.values,
                mode="lines", name="Observed Price",
                line=dict(color="#004400", width=1),
            ))
            fig_kf.add_trace(go.Scatter(
                x=custom_trend.index, y=custom_trend.values,
                mode="lines", name="Kalman Trend (x_t)",
                line=dict(color="#00ff41", width=2.5),
            ))
            fig_kf.update_layout(
                paper_bgcolor="#000000", plot_bgcolor="#000000",
                xaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
                yaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="Price"),
                legend=dict(font=dict(color="#00ff41", family="monospace"), bgcolor="#000000"),
                font=dict(color="#00ff41", family="monospace"),
                title=dict(text="x_t = x_{t-1} + w_t  |  y_t = x_t + v_t", font=dict(color="#007722", size=12)),
                height=320, margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_kf, width="stretch")

            fig_noise = go.Figure()
            fig_noise.add_trace(go.Scatter(
                x=custom_noise.index, y=custom_noise.values,
                mode="lines", name="Residual Noise v_t",
                line=dict(color="#ffd700", width=1),
                fill="tozeroy",
                fillcolor="rgba(255,215,0,0.05)",
            ))
            fig_noise.add_hline(y=0, line=dict(color="#00ff41", width=1, dash="dash"))
            fig_noise.update_layout(
                paper_bgcolor="#000000", plot_bgcolor="#000000",
                xaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
                yaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="Noise"),
                font=dict(color="#00ff41", family="monospace"),
                title=dict(text="Kalman Residual Noise (v_t = y_t − x_t)", font=dict(color="#007722", size=12)),
                height=200, margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_noise, width="stretch")

            k_col1, k_col2, k_col3 = st.columns(3)
            k_col1.markdown(f'<div class="metric-card"><div class="metric-label">{t("last_trend")}</div><div class="metric-value">{float(custom_trend.iloc[-1]):.4f}</div></div>', unsafe_allow_html=True)
            k_col2.markdown(f'<div class="metric-card"><div class="metric-label">{t("last_price")}</div><div class="metric-value">{float(xau_price.iloc[-1]):.4f}</div></div>', unsafe_allow_html=True)
            k_col3.markdown(f'<div class="metric-card"><div class="metric-label">{t("noise_std")}</div><div class="metric-value">{float(custom_noise.std()):.6f}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HMM REGIMES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hmm:
    if not st.session_state.loaded:
        st.info(t("init_first"))
    else:
        de = st.session_state.data_engine
        fe = st.session_state.feature_engine
        hmm = st.session_state.hmm_engine

        st.markdown(f'<div class="section-header">{t("sh_hmm")}</div>', unsafe_allow_html=True)

        if hmm.fitted:
            from engines.state_space_engine import STATE_LABELS, STATE_COLORS

            state_series = hmm.state_series()
            prob_df = hmm.state_prob_df()
            xau_price = de.get_xau_price()

            fig_regimes = go.Figure()
            fig_regimes.add_trace(go.Scatter(
                x=xau_price.index, y=xau_price.values,
                mode="lines", name="XAUUSD",
                line=dict(color="#002200", width=1),
            ))

            if not state_series.empty:
                # Align price to state index; fall back to nearest-value ffill
                # for any minor timestamp mismatch (daily normalize vs intraday)
                price_aligned = xau_price.reindex(state_series.index)
                if price_aligned.isna().all():
                    combined = pd.concat(
                        [xau_price.rename("p"), state_series.rename("s")], axis=1
                    ).sort_index().ffill().bfill().dropna()
                    price_aligned = combined["p"].reindex(state_series.index).ffill()
                for s_id, s_label in STATE_LABELS.items():
                    mask = state_series == s_id
                    if mask.sum() > 0:
                        fig_regimes.add_trace(go.Scatter(
                            x=state_series[mask].index,
                            y=price_aligned[mask].values,
                            mode="markers",
                            name=s_label,
                            marker=dict(color=STATE_COLORS[s_id], size=5, opacity=0.8),
                        ))

            fig_regimes.update_layout(
                paper_bgcolor="#000000", plot_bgcolor="#000000",
                xaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
                yaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="Price"),
                legend=dict(font=dict(color="#00ff41", family="monospace"), bgcolor="#000000"),
                font=dict(color="#00ff41", family="monospace"),
                title=dict(text="Price Colored by HMM Regime State", font=dict(color="#007722", size=12)),
                height=300, margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_regimes, width="stretch")

            if not prob_df.empty:
                fig_probs = go.Figure()
                colors_list = [STATE_COLORS[i] for i in range(3)]
                for i, col in enumerate(prob_df.columns):
                    fig_probs.add_trace(go.Scatter(
                        x=prob_df.index,
                        y=prob_df[col].values,
                        mode="lines",
                        name=col,
                        line=dict(color=colors_list[i], width=1.5),
                        fill="tonexty" if i > 0 else "tozeroy",
                        fillcolor="rgba({},{},{},0.12)".format(
                            int(colors_list[i][1:3], 16),
                            int(colors_list[i][3:5], 16),
                            int(colors_list[i][5:7], 16),
                        ),
                    ))
                fig_probs.update_layout(
                    paper_bgcolor="#000000", plot_bgcolor="#000000",
                    xaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
                    yaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="P(state)", range=[0, 1]),
                    legend=dict(font=dict(color="#00ff41", family="monospace"), bgcolor="#000000"),
                    font=dict(color="#00ff41", family="monospace"),
                    title=dict(text="P(state_i | data_t)", font=dict(color="#007722", size=12)),
                    height=250, margin=dict(l=50, r=20, t=40, b=30),
                )
                st.plotly_chart(fig_probs, width="stretch")

            hm_cols = st.columns(3)
            for i, (s_id, s_label) in enumerate(STATE_LABELS.items()):
                probs_dict = hmm.current_probs()
                p = probs_dict.get(s_label, 0)
                hm_cols[i].markdown(
                    f'<div class="metric-card"><div class="metric-label">{s_label}</div>'
                    f'<div class="metric-value" style="color:{STATE_COLORS[s_id]}">{p:.4f}</div></div>',
                    unsafe_allow_html=True
                )

            tm = hmm.transition_matrix()
            st.markdown(f'<div class="section-header">{t("sh_transition")}</div>', unsafe_allow_html=True)
            fig_tm = px.imshow(
                tm,
                x=[STATE_LABELS[i] for i in range(3)],
                y=[STATE_LABELS[i] for i in range(3)],
                color_continuous_scale=[[0, "#000000"], [0.5, "#003300"], [1, "#00ff41"]],
                text_auto=".3f",
            )
            fig_tm.update_layout(
                paper_bgcolor="#000000", plot_bgcolor="#000000",
                font=dict(color="#00ff41", family="monospace"),
                height=260, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_tm, width="stretch")
        else:
            st.warning("HMM not fitted — insufficient data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ORDER FLOW HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_heat:
    if not st.session_state.loaded:
        st.info(t("init_first"))
    else:
        de = st.session_state.data_engine
        st.markdown(f'<div class="section-header">{t("sh_heatmap")}</div>', unsafe_allow_html=True)

        h_col1, h_col2 = st.columns(2)
        with h_col1:
            n_bins = st.slider(t("price_bins"), 30, 120, 60, key="hmap_bins")
        with h_col2:
            hmap_window = st.slider(t("vol_window"), 5, 60, 20, key="hmap_window")

        from visualization.heatmap import build_order_flow_heatmap
        xau_price = de.get_xau_price()
        xau_ret = de.get_xau_returns()

        fig_hmap = build_order_flow_heatmap(xau_price, xau_ret, n_bins=n_bins, window=hmap_window)
        st.plotly_chart(fig_hmap, width="stretch")

        st.markdown("""
        <div style="color:#007722;font-size:11px;letter-spacing:1px;">
        HIGH DENSITY → Institutional accumulation zones (order blocks) &nbsp;|&nbsp;
        GOLD LINE → Current price &nbsp;|&nbsp;
        Kernel smoothing applied (Gaussian KDE)
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — GC3D ALPHA SURFACE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gc3d:
    if not st.session_state.loaded:
        st.info(t("init_first"))
    else:
        de = st.session_state.data_engine
        fe = st.session_state.feature_engine
        st.markdown(f'<div class="section-header">{t("sh_gc3d")}</div>', unsafe_allow_html=True)

        from visualization.gc3d import build_gc3d, build_volatility_surface, FEATURE_LABELS

        # Only keep the 3 valid GC3D features — zscore/carry/OI removed
        available_features = [c for c in fe.features.columns if c in FEATURE_LABELS]
        if not available_features:
            available_features = list(FEATURE_LABELS.keys())[:1]

        gc_col1, gc_col2 = st.columns(2)
        with gc_col1:
            selected_feat = st.selectbox(
                t("z_feature"),
                available_features,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="gc3d_feat",
            )
        with gc_col2:
            view_type = st.radio(t("view"), ["Scatter 3D", "Multi-Feature Surface"], horizontal=True, key="gc3d_view")

        xau_price = de.get_xau_price()

        if view_type == "Scatter 3D":
            fig_gc3d = build_gc3d(xau_price, fe.features, feature_col=selected_feat)
        else:
            fig_gc3d = build_volatility_surface(fe.features)

        st.plotly_chart(fig_gc3d, width="stretch")
        st.markdown("""
        <div style="color:#007722;font-size:11px;letter-spacing:1px;">
        Z = f(time, price, feature) &nbsp;|&nbsp; COLOR = feature intensity &nbsp;|&nbsp; GOLD = current position
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — RISK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    if not st.session_state.loaded:
        st.info(t("init_first"))
    else:
        de = st.session_state.data_engine
        re = st.session_state.risk_engine
        rm = re.metrics

        st.markdown(f'<div class="section-header">{t("sh_risk_engine")}</div>', unsafe_allow_html=True)

        r_cols = st.columns(3)
        risk_groups = [
            [(t("var95_param"),    f"{rm.get('VaR_95_param', np.nan):.6f}"),
             (t("var95_hist"),     f"{rm.get('VaR_95_hist', np.nan):.6f}"),
             (t("exp_shortfall"),  f"{rm.get('ES_95', np.nan):.6f}")],
            [(t("max_drawdown"),   f"{rm.get('max_drawdown', np.nan)*100:.2f}%"),
             (t("curr_drawdown"),  f"{rm.get('current_drawdown', np.nan)*100:.2f}%"),
             (t("kelly"),          f"{rm.get('kelly_fraction', np.nan):.4f}")],
            [(t("sharpe_ratio"),   f"{rm.get('sharpe_ratio', np.nan):.4f}"),
             (t("info_ratio"),     f"{rm.get('information_ratio', np.nan):.4f}"),
             (t("annual_vol"),     f"{rm.get('vol_annual', np.nan)*100:.2f}%")],
        ]
        for i, group in enumerate(risk_groups):
            for label, val in group:
                r_cols[i].markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:18px">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        xau_ret = de.get_xau_returns()
        xau_price = de.get_xau_price()
        dd_series = rm.get("drawdown_series", pd.Series(dtype=float))

        if not dd_series.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values * 100,
                mode="lines", name="Drawdown %",
                line=dict(color="#ff4444", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,68,68,0.08)",
            ))
            fig_dd.update_layout(
                paper_bgcolor="#000000", plot_bgcolor="#000000",
                xaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
                yaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="Drawdown %"),
                font=dict(color="#00ff41", family="monospace"),
                title=dict(text="Drawdown: (Peak − Current) / Peak", font=dict(color="#007722", size=12)),
                height=250, margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_dd, width="stretch")

        from engines.risk_engine import rolling_var
        rv_series = rolling_var(xau_ret, window=20)
        if not rv_series.empty:
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(
                x=rv_series.index, y=rv_series.values,
                mode="lines", name="Rolling VaR 95% (20d)",
                line=dict(color="#ffd700", width=1.5),
            ))
            fig_rv.update_layout(
                paper_bgcolor="#000000", plot_bgcolor="#000000",
                xaxis=dict(color="#00ff41", gridcolor="#0a2a0a"),
                yaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="VaR"),
                font=dict(color="#00ff41", family="monospace"),
                title=dict(text="Rolling VaR 95% — 20-day Window", font=dict(color="#007722", size=12)),
                height=220, margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_rv, width="stretch")

        st.markdown("""
        <div style="color:#007722;font-size:11px;letter-spacing:1px;">
        VaR = μ − z_α σ &nbsp;|&nbsp; ES = E[loss | loss > VaR] &nbsp;|&nbsp; DD = (peak − current) / peak
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — PORTFOLIO OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    if not st.session_state.loaded:
        st.info(t("init_first"))
    else:
        de = st.session_state.data_engine
        po = st.session_state.portfolio_optimizer

        st.markdown(f'<div class="section-header">{t("sh_portfolio")}</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#007722;font-size:11px;">Objective: max μᵀw − λ wᵀΣw − γ ||w − w_prev||²</div>', unsafe_allow_html=True)

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            risk_aversion = st.slider("Risk Aversion (λ)", 0.5, 10.0, 2.0, 0.5, key="po_lambda")
            turnover_pen = st.slider("Turnover Penalty (γ)", 0.0, 1.0, 0.1, 0.05, key="po_gamma")
        with p_col2:
            max_w = st.slider("Max Weight per Asset", 0.1, 1.0, 0.4, 0.05, key="po_maxw")

        if st.button("▶ RUN OPTIMIZER", key="run_optimizer"):
            from engines.portfolio_optimizer import PortfolioOptimizer as PO2
            po2 = PO2(risk_aversion=risk_aversion, turnover_penalty=turnover_pen)
            res = po2.run(de.returns)
            st.session_state.portfolio_optimizer = po2
            po = po2

        res = po.result
        if res and res.get("weights"):
            p_m1, p_m2, p_m3, p_m4 = st.columns(4)
            p_m1.markdown(f'<div class="metric-card"><div class="metric-label">{t("exp_return")}</div><div class="metric-value">{res["expected_return"]*100:.2f}%</div></div>', unsafe_allow_html=True)
            p_m2.markdown(f'<div class="metric-card"><div class="metric-label">{t("portfolio_vol")}</div><div class="metric-value">{res["portfolio_vol"]*100:.2f}%</div></div>', unsafe_allow_html=True)
            p_m3.markdown(f'<div class="metric-card"><div class="metric-label">{t("portfolio_sharpe")}</div><div class="metric-value gold">{res["sharpe"]:.4f}</div></div>', unsafe_allow_html=True)
            p_m4.markdown(f'<div class="metric-card"><div class="metric-label">CONVERGED</div><div class="metric-value">{str(res.get("success", False))}</div></div>', unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            pw_col1, pw_col2 = st.columns(2)

            with pw_col1:
                st.markdown(f'<div class="section-header">{t("weights")}</div>', unsafe_allow_html=True)
                weights = res["weights"]
                fig_w = go.Figure(go.Bar(
                    x=list(weights.keys()),
                    y=list(weights.values()),
                    marker=dict(
                        color=list(weights.values()),
                        colorscale=[[0, "#001100"], [0.5, "#00aa33"], [1, "#00ff41"]],
                        line=dict(color="#000000", width=1),
                    ),
                    text=[f"{v:.4f}" for v in weights.values()],
                    textposition="outside",
                    textfont=dict(color="#00ff41", family="monospace"),
                ))
                fig_w.update_layout(
                    paper_bgcolor="#000000", plot_bgcolor="#000000",
                    xaxis=dict(color="#00ff41"),
                    yaxis=dict(color="#00ff41", range=[0, max(list(weights.values())) * 1.3]),
                    font=dict(color="#00ff41", family="monospace"),
                    height=280, margin=dict(l=30, r=20, t=10, b=30),
                )
                st.plotly_chart(fig_w, width="stretch")

            with pw_col2:
                st.markdown('<div class="section-header">EFFICIENT FRONTIER</div>', unsafe_allow_html=True)

                frontier = po.frontier
                if not frontier.empty:
                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=frontier["vol"] * 100,
                        y=frontier["ret"] * 100,
                        mode="lines",
                        name="Efficient Frontier",
                        line=dict(color="#00ff41", width=2),
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[res["portfolio_vol"] * 100],
                        y=[res["expected_return"] * 100],
                        mode="markers",
                        name="Optimal Portfolio",
                        marker=dict(color="#ffd700", size=12, symbol="star"),
                    ))
                    fig_ef.update_layout(
                        paper_bgcolor="#000000", plot_bgcolor="#000000",
                        xaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="Vol (%)"),
                        yaxis=dict(color="#00ff41", gridcolor="#0a2a0a", title="Return (%)"),
                        legend=dict(font=dict(color="#00ff41", family="monospace"), bgcolor="#000000"),
                        font=dict(color="#00ff41", family="monospace"),
                        height=280, margin=dict(l=50, r=20, t=10, b=40),
                    )
                    st.plotly_chart(fig_ef, width="stretch")

        else:
            st.warning("Run the optimizer to see results.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — MAP + ALERT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    from visualization.map_module import render_map_tab
    from alerts.alert_engine import AlertEngine

    map_top = st.container()

    with map_top:
        mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
        with mcol1:
            map_period = st.selectbox(
                "Period", ["5d", "1mo", "3mo"], index=0,
                key="map_period",
                label_visibility="collapsed",
            )
        with mcol2:
            if st.button("⟳ REFRESH MAP", key="map_refresh"):
                _load_map_data.clear()
                st.rerun()
        with mcol3:
            st.markdown(
                "<div style='font-family:monospace;font-size:9px;color:#444;"
                "padding-top:8px;text-align:right;'>auto-refresh every 5 min</div>",
                unsafe_allow_html=True,
            )

    with st.spinner("▶ LOADING MARKET MAP…"):
        sector_df, geo_df, news_items, _gold_chg, _gold_px = _load_map_data(map_period)

    # ── Gather real price data for alert engine ──────────────────────────────
    _live_px    = None
    _prev_close = None
    _xau_ret    = None

    if st.session_state.get("loaded"):
        try:
            _price_data = _cached_live_price()          # returns a dict
            _live_px    = float(_price_data["last_price"])
            _prev_close = float(_price_data.get("previous_close") or 0) or None
        except Exception:
            pass
        try:
            _xau_ret = st.session_state.data_engine.get_xau_returns()
        except Exception:
            pass

    ae = AlertEngine()
    alerts = ae.generate(
        live_price=_live_px,
        prev_close=_prev_close,
        returns=_xau_ret,
        insider_engine=st.session_state.get("insider_engine") if st.session_state.get("loaded") else None,
        news_items=news_items,
    )

    # ── Toast only when alert set changes (avoid repeat on every re-render) ───
    _alert_hash = hash(tuple(a.message for a in alerts))
    if st.session_state.get("_last_alert_hash") != _alert_hash:
        st.session_state["_last_alert_hash"] = _alert_hash
        for a in alerts[:2]:
            if a.severity == "HIGH":
                st.toast(f"🔴 {a.asset} — {a.message}", icon="🔴")
            elif a.severity == "MEDIUM":
                st.toast(f"🟡 {a.asset} — {a.message}", icon="🟡")

    render_map_tab(sector_df, geo_df, alerts, news_items,
                   gold_change_pct=_gold_chg, gold_price=_gold_px)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10 — VWAP + SD BANDS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def _load_vwap_data(ticker: str, interval: str, period: str):
    from visualization.vwap_module import _download_intraday, compute_vwap
    df = _download_intraday(ticker, interval, period)
    if not df.empty:
        df = compute_vwap(df)
    return df


with tab_vwap:
    st.markdown(
        '<div class="section-header">VWAP + BANDAS DE DESVIACIÓN ESTÁNDAR — XAUUSD</div>',
        unsafe_allow_html=True,
    )

    vcol1, vcol2, vcol3 = st.columns([1, 1, 1])
    with vcol1:
        vwap_ticker   = st.selectbox("Ticker", ["GC=F", "GLD", "IAU", "XAUUSD=X"], key="vwap_ticker")
    with vcol2:
        vwap_interval = st.selectbox("Intervalo", ["5m", "15m", "1h"], index=0, key="vwap_interval")
    with vcol3:
        vwap_period   = st.selectbox("Período", ["1d", "2d", "5d"], index=1, key="vwap_period")

    vcol_btn1, vcol_btn2 = st.columns([1, 5])
    with vcol_btn1:
        if st.button("↺ REFRESH", key="vwap_refresh"):
            _load_vwap_data.clear()

    with st.spinner("▶ LOADING VWAP DATA…"):
        from visualization.vwap_module import build_vwap_chart
        fig_vwap = build_vwap_chart(vwap_ticker, vwap_interval, vwap_period)

    st.plotly_chart(fig_vwap, width="stretch")
    st.markdown(
        "<div style='color:#444;font-family:monospace;font-size:10px;'>"
        "VWAP Sesión = Σ(TP×Vol) / Σ(Vol) por día  ·  "
        "Bandas = VWAP ± n × σ(precio típico)  ·  "
        "Cache 5 min  ·  Fuente: Yahoo Finance"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 11 — YFINANCE ANOMALY DR
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def _load_anomaly_data(ticker: str, period: str, window: int, threshold: float):
    from visualization.anomaly_module import build_anomaly_chart
    fig, log = build_anomaly_chart(ticker, period, window, threshold)
    return fig, log


with tab_anomaly:
    st.markdown(
        '<div class="section-header">YFINANCE ANOMALY DR — DETECCIÓN ESTADÍSTICA DE ANOMALÍAS</div>',
        unsafe_allow_html=True,
    )

    acol1, acol2, acol3, acol4 = st.columns([1, 1, 1, 1])
    with acol1:
        anom_ticker    = st.selectbox("Ticker", ["GC=F", "GLD", "^SPX", "^GSPC", "XAUUSD=X"], key="anom_ticker")
    with acol2:
        anom_period    = st.selectbox("Período", ["2d", "5d", "10d"], index=1, key="anom_period")
    with acol3:
        anom_window    = st.slider("Ventana rolling", 5, 50, 20, 5, key="anom_window")
    with acol4:
        anom_threshold = st.slider("Umbral (σ)", 1.0, 4.0, 2.0, 0.25, key="anom_thresh")

    acol_btn1, _ = st.columns([1, 5])
    with acol_btn1:
        if st.button("↺ REFRESH", key="anom_refresh"):
            _load_anomaly_data.clear()

    with st.spinner("▶ LOADING ANOMALY DATA…"):
        fig_anom, anom_log = _load_anomaly_data(
            anom_ticker, anom_period, anom_window, anom_threshold
        )

    anom_chart_col, anom_log_col = st.columns([4, 1])

    with anom_chart_col:
        st.plotly_chart(fig_anom, width="stretch")

    with anom_log_col:
        st.markdown(
            "<div style='font-family:monospace;font-size:11px;color:#00ff41;"
            "letter-spacing:1px;padding:4px 0;border-bottom:1px solid #222;'>"
            "REGISTRO</div>",
            unsafe_allow_html=True,
        )
        if anom_log:
            for item in anom_log:
                color  = "#ff00ff" if item["direction"] == "UP" else "#00eeff"
                arrow  = "▲" if item["direction"] == "UP" else "▼"
                st.markdown(
                    f"<div style='font-family:monospace;font-size:9px;"
                    f"border-bottom:1px solid #111;padding:3px 0;line-height:1.5;'>"
                    f"<span style='color:#444;'>{item['timestamp']}</span><br>"
                    f"<span style='color:#00ff41;font-weight:bold;'>{item['tf']}</span> "
                    f"<span style='color:{color};'>{arrow} {item['ret']:+.4f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#333;font-family:monospace;font-size:9px;'>Sin anomalías</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div style='color:#444;font-family:monospace;font-size:10px;'>"
        "Retorno = pct_change(Close)  ·  Anomalía = |ret| > media ± umbral×σ  ·  "
        "▲ magenta = UP  ▼ cyan = DOWN  ·  Señales de reversión a la media  ·  Cache 5 min"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 12 — VOLUME PROFILE SESSIONS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def _load_volprofile_data(ticker: str, interval: str, period: str, n_bins: int):
    from visualization.volprofile_module import build_volprofile_chart
    return build_volprofile_chart(ticker, interval, period, n_bins)


with tab_volprofile:
    st.markdown(
        '<div class="section-header">VOLUME PROFILE SESSIONS — ANÁLISIS DE PERFIL DE VOLUMEN POR SESIÓN</div>',
        unsafe_allow_html=True,
    )

    vpcol1, vpcol2, vpcol3, vpcol4 = st.columns([1, 1, 1, 1])
    with vpcol1:
        vp_ticker   = st.selectbox("Ticker", ["GC=F", "GLD", "IAU", "^GSPC", "QQQ"], key="vp_ticker")
    with vpcol2:
        vp_interval = st.selectbox("Intervalo", ["5m", "15m", "1h"], index=1, key="vp_interval")
    with vpcol3:
        vp_period   = st.selectbox("Período", ["2d", "5d", "10d"], index=1, key="vp_period")
    with vpcol4:
        vp_bins     = st.slider("Bins de precio", 10, 40, 22, 2, key="vp_bins")

    vpcol_btn1, _ = st.columns([1, 5])
    with vpcol_btn1:
        if st.button("↺ REFRESH", key="vp_refresh"):
            _load_volprofile_data.clear()

    with st.spinner("▶ LOADING VOLUME PROFILE…"):
        fig_vp = _load_volprofile_data(vp_ticker, vp_interval, vp_period, vp_bins)

    st.plotly_chart(fig_vp, width="stretch")
    st.markdown(
        "<div style='color:#444;font-family:monospace;font-size:10px;'>"
        "POC = precio con mayor volumen (naranja)  ·  "
        "VAH/VAL = límites del 70% del volumen (blanco)  ·  "
        "Verde = dominancia compradores  ·  Rojo = dominancia vendedores  ·  Cache 5 min"
        "</div>",
        unsafe_allow_html=True,
    )
