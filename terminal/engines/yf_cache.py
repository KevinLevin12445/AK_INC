"""
yfinance cache wrapper — bare yf.download() + Ticker.history() fallback.
yfinance 1.x manages curl_cffi internally — do NOT pass session=.
5-min in-process cache prevents repeated API calls on Streamlit re-renders.
"""

import time
import threading
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

_CACHE: dict = {}
_LOCK         = threading.Lock()
_DEFAULT_TTL  = 300   # 5 minutes


def _cache_key(tickers, period: str, interval: str, **extra) -> str:
    t = tickers if isinstance(tickers, str) else ",".join(sorted(tickers))
    kw = "&".join(f"{k}={v}" for k, v in sorted(extra.items())
                  if k not in ("progress",))
    return f"{t}|{period}|{interval}|{kw}"


def yf_download(
    tickers,
    *,
    period:   str = "1mo",
    interval: str = "1d",
    ttl:      int = _DEFAULT_TTL,
    **kwargs,
) -> pd.DataFrame:
    """
    Thread-safe yf.download with 5-min cache.
    Falls back to Ticker.history() for single tickers if download fails.
    DO NOT pass session= — yfinance 1.x handles its own curl_cffi session.
    Returns empty DataFrame on failure (never raises).
    """
    single = isinstance(tickers, str)
    key = _cache_key(tickers, period, interval, **kwargs)
    now = time.time()

    # Cache hit
    with _LOCK:
        if key in _CACHE:
            df_c, ts = _CACHE[key]
            if now - ts < ttl and not df_c.empty:
                return df_c.copy()

    df = pd.DataFrame()

    # Primary: bare yf.download (no custom session)
    try:
        df = yf.download(
            tickers,
            period=period,
            interval=interval,
            progress=False,
            **kwargs,
        )
    except Exception:
        pass

    # Fallback for single ticker: Ticker.history()
    if df.empty and single:
        try:
            df = yf.Ticker(tickers).history(
                period=period,
                interval=interval,
            )
        except Exception:
            pass

    if not df.empty:
        with _LOCK:
            _CACHE[key] = (df.copy(), time.time())
        return df

    # Stale cache fallback
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key][0].copy()

    return pd.DataFrame()


def yf_ticker_fast_info(symbol: str) -> dict:
    """Fetch fast_info. Returns {} on failure."""
    try:
        fi = yf.Ticker(symbol).fast_info
        return {
            "last_price":     getattr(fi, "last_price",     None),
            "previous_close": getattr(fi, "previous_close", None),
        }
    except Exception:
        return {}
