"""
ws_gold.py — Real-time Gold CFD price via WebSocket background thread.

Priority chain:
  1. Finnhub WebSocket (OANDA:XAU_USD/C) — true CFD, tick-by-tick
     Requires free API key at https://finnhub.io  →  set env var FINNHUB_API_KEY
  2. gold-api.com polling every 2 s (no key needed, spot price fallback)

The background thread writes to _GOLD_STATE (module-level dict, thread-safe).
Call get_live_gold() from any thread or Streamlit fragment to read latest data.
"""

import os
import time
import json
import math
import threading
import requests

# ─── Shared state (written by bg thread, read by Streamlit fragment) ──────────
_LOCK = threading.Lock()
_GOLD_STATE: dict = {
    "price":      float("nan"),
    "prev_close": float("nan"),
    "source":     "initializing",
    "ts":         0.0,          # epoch seconds of last update
}

_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ─── Public API ───────────────────────────────────────────────────────────────

def get_live_gold() -> dict:
    """
    Returns latest gold price dict (thread-safe).
    Keys: price, prev_close, change, change_pct, source, age_s
    """
    with _LOCK:
        state = dict(_GOLD_STATE)

    price = state["price"]
    prev  = state["prev_close"]
    now   = time.time()
    age_s = now - state["ts"] if state["ts"] > 0 else float("nan")

    change     = price - prev if not (math.isnan(price) or math.isnan(prev)) else float("nan")
    change_pct = (change / prev * 100) if not math.isnan(prev) and prev != 0 else float("nan")

    return {
        "last_price":     price,
        "previous_close": prev,
        "change":         change,
        "change_pct":     change_pct,
        "source":         state["source"],
        "age_s":          age_s,
    }


def start(force_restart: bool = False):
    """
    Start the background thread (idempotent — safe to call multiple times).
    Call once at app startup.
    """
    global _thread, _stop_event

    if _thread is not None and _thread.is_alive() and not force_restart:
        return  # already running

    _stop_event.set()   # signal old thread to stop (if any)
    time.sleep(0.1)
    _stop_event = threading.Event()

    finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()

    if finnhub_key:
        _thread = threading.Thread(
            target=_run_finnhub_ws,
            args=(finnhub_key,),
            daemon=True,
            name="ws_gold_finnhub",
        )
    else:
        _thread = threading.Thread(
            target=_run_polling,
            daemon=True,
            name="ws_gold_polling",
        )

    _thread.start()


def is_alive() -> bool:
    return _thread is not None and _thread.is_alive()


# ─── Previous close helper ─────────────────────────────────────────────────────

def _fetch_prev_close() -> float:
    """Fetch previous session close from yfinance (cached in yf_cache)."""
    try:
        from engines.yf_cache import yf_download
        import pandas as pd
        df = yf_download("GC=F", period="5d", interval="1d", auto_adjust=True)
        if df.empty:
            return float("nan")
        closes = df["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if len(closes) >= 2:
            return float(closes.iloc[-2])
    except Exception:
        pass
    return float("nan")


# ─── Strategy 1: Finnhub WebSocket (OANDA:XAU_USD/C) ─────────────────────────

def _run_finnhub_ws(api_key: str):
    """
    Connects to Finnhub WebSocket and subscribes to OANDA:XAU_USD/C (Gold CFD).
    Reconnects automatically on disconnect.
    """
    try:
        import websocket  # websocket-client package
    except ImportError:
        # Fallback to polling if websocket-client not installed
        _run_polling()
        return

    symbol    = "OANDA:XAU_USD/C"
    ws_url    = f"wss://ws.finnhub.io?token={api_key}"
    prev      = _fetch_prev_close()
    reconnect_delay = 3

    def on_open(ws):
        ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
        with _LOCK:
            _GOLD_STATE["source"] = "Finnhub WS (OANDA CFD)"

    def on_message(ws, message):
        nonlocal prev
        try:
            data = json.loads(message)
            if data.get("type") == "trade" and data.get("data"):
                tick = data["data"][-1]  # most recent tick
                price = float(tick["p"])
                if price > 0:
                    with _LOCK:
                        _GOLD_STATE["price"]      = price
                        _GOLD_STATE["prev_close"] = prev
                        _GOLD_STATE["source"]     = "Finnhub WS · OANDA:XAU_USD/C"
                        _GOLD_STATE["ts"]         = time.time()
        except Exception:
            pass

    def on_error(ws, error):
        pass  # reconnect loop handles it

    def on_close(ws, *args):
        pass

    while not _stop_event.is_set():
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception:
            pass

        if _stop_event.is_set():
            break
        # Refresh prev_close on reconnect
        new_prev = _fetch_prev_close()
        if not math.isnan(new_prev):
            prev = new_prev
        _stop_event.wait(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 30)  # exponential backoff


# ─── Strategy 2: Fast REST polling (gold-api.com, no key) ─────────────────────

_POLL_INTERVAL = 2.0   # seconds between polls
_POLL_SOURCES  = [
    ("https://api.gold-api.com/price/XAU", lambda r: float(r.json().get("price", 0))),
]


def _run_polling():
    """
    Polls gold-api.com every ~2.5 s and writes to _GOLD_STATE.
    Used when FINNHUB_API_KEY is not set.
    """
    prev     = _fetch_prev_close()
    session  = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    fail_count = 0

    with _LOCK:
        _GOLD_STATE["source"] = "gold-api.com · polling 2.5s"

    while not _stop_event.is_set():
        fetched = False
        for url, extractor in _POLL_SOURCES:
            try:
                r = session.get(url, timeout=4)
                if r.status_code == 200:
                    price = extractor(r)
                    if price and price > 0:
                        # Refresh prev close once per minute
                        if time.time() - _GOLD_STATE.get("ts", 0) > 60:
                            new_prev = _fetch_prev_close()
                            if not math.isnan(new_prev):
                                prev = new_prev
                        with _LOCK:
                            _GOLD_STATE["price"]      = price
                            _GOLD_STATE["prev_close"] = prev
                            _GOLD_STATE["source"]     = "gold-api.com · ~2.5s refresh"
                            _GOLD_STATE["ts"]         = time.time()
                        fetched = True
                        fail_count = 0
                        break
            except Exception:
                pass

        if not fetched:
            fail_count += 1
            with _LOCK:
                _GOLD_STATE["source"] = f"gold-api.com · retrying (fails={fail_count})"

        _stop_event.wait(_POLL_INTERVAL)
