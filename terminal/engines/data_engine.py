import time
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from engines.yf_cache import yf_download, yf_ticker_fast_info


# Tickers downloaded in a SINGLE batch call to avoid rate limiting
SYMBOLS = {
    "XAUUSD": "GC=F",
    "DXY":    "DX-Y.NYB",
    "VIX":    "^VIX",
    "US10Y":  "^TNX",
}
_TICKERS = list(SYMBOLS.values())   # ['GC=F', 'DX-Y.NYB', '^VIX', '^TNX']
_LABEL   = {v: k for k, v in SYMBOLS.items()}

INTERVAL_PERIOD_MAP = {
    "15m": ["1d", "5d", "1mo", "2mo"],
    "1h":  ["1mo", "3mo", "6mo", "1y"],
    "4h":  ["1mo", "3mo", "6mo", "1y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y"],
}
DEFAULT_PERIOD = {
    "15m": "1mo",
    "1h":  "3mo",
    "4h":  "3mo",
    "1d":  "6mo",
}


def _batch_download(period: str, interval: str) -> pd.DataFrame:
    """
    Download ALL tickers in ONE request via the cached/retry wrapper.
    Returns a raw multi-column DataFrame from yf.download(group_by='ticker').
    """
    return yf_download(
        _TICKERS,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
    )


def _extract_series(raw: pd.DataFrame, ticker: str, field: str = "Close") -> pd.Series:
    """Extract a single field/ticker from a grouped multi-column DataFrame.
    yfinance 1.x returns (ticker, field) MultiIndex columns.
    """
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance 1.x: (ticker, field)
            if (ticker, field) in raw.columns:
                return raw[(ticker, field)].dropna()
            # yfinance 0.2.x fallback: (field, ticker)
            if (field, ticker) in raw.columns:
                return raw[(field, ticker)].dropna()
            # Case-insensitive scan
            for col in raw.columns:
                a, b = str(col[0]), str(col[1])
                if (a == ticker and b.lower() == field.lower()) or \
                   (b == ticker and a.lower() == field.lower()):
                    return raw[col].dropna()
        else:
            if field in raw.columns:
                return raw[field].dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def _normalise_index(s: pd.Series, daily: bool = True) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    s = s.copy()
    s.index = idx.normalize() if daily else idx
    return s[~s.index.duplicated(keep="last")]


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {}
    col_map = {str(c).lower(): c for c in df.columns}
    for key, col in col_map.items():
        if   "open"   in key: agg[col] = "first"
        elif "high"   in key: agg[col] = "max"
        elif "low"    in key: agg[col] = "min"
        elif "close"  in key: agg[col] = "last"
        elif "volume" in key: agg[col] = "sum"
    return df.resample(rule).agg(agg).dropna(how="all") if agg else df.resample(rule).last()


def fetch_with_volume(period="6mo", interval="1d"):
    fetch_interval = "1h" if interval == "4h" else interval
    resample_to_4h = (interval == "4h")
    is_intraday    = fetch_interval in ("1m","2m","5m","15m","30m","60m","90m","1h")

    raw = _batch_download(period, fetch_interval)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    frames_price  = {}
    frames_volume = {}

    for ticker, label in _LABEL.items():
        try:
            close = _extract_series(raw, ticker, "Close")
            if close.empty:
                continue
            if resample_to_4h:
                close = close.resample("4h").last().dropna()
            frames_price[label] = _normalise_index(close.rename(label), daily=not is_intraday)

            vol = _extract_series(raw, ticker, "Volume")
            if not vol.empty:
                if resample_to_4h:
                    vol = vol.resample("4h").sum().dropna()
                frames_volume[label] = _normalise_index(vol.rename(label), daily=not is_intraday)
        except Exception:
            pass

    prices  = pd.concat(frames_price.values(),  axis=1).dropna(how="all") if frames_price  else pd.DataFrame()
    volumes = pd.concat(frames_volume.values(), axis=1).dropna(how="all") if frames_volume else pd.DataFrame()
    return prices, volumes


def fetch_raw(period="6mo", interval="1d"):
    prices, _ = fetch_with_volume(period, interval)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def winsorize_returns(returns: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    result = returns.copy()
    for col in result.columns:
        sigma = result[col].std()
        mu    = result[col].mean()
        result[col] = result[col].clip(mu - threshold * sigma, mu + threshold * sigma)
    return result


def compute_vwap(prices: pd.DataFrame, volumes: pd.DataFrame = None, window: int = 20) -> pd.Series:
    vol   = volumes["XAUUSD"] if (volumes is not None and "XAUUSD" in volumes.columns) \
            else pd.Series(np.ones(len(prices)), index=prices.index)
    price = prices["XAUUSD"] if "XAUUSD" in prices.columns else prices.iloc[:, 0]
    return ((price * vol).rolling(window).sum() / vol.rolling(window).sum()).rename("VWAP")


def compute_realized_volatility(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    if "XAUUSD" not in returns.columns:
        return pd.Series(dtype=float)
    return (returns["XAUUSD"].rolling(window).std() * np.sqrt(252)).rename("RealizedVol")


def fetch_live_price(ticker: str = "XAUUSD=X") -> dict:
    """
    Real-time gold CFD / spot price — priority chain:
      1. ws_gold background thread (WebSocket Finnhub CFD or 2.5s polling)
         → used if data is fresh (< 10 s old)
      2. gold-api.com direct REST call (fallback, no auth)
      3. GC=F fast_info via yfinance (last resort)
    """
    import requests as _req

    # ── 1. WebSocket / fast-polling thread (preferred) ────────────────────────
    try:
        from engines.ws_gold import get_live_gold, is_alive
        if is_alive():
            ws_data = get_live_gold()
            age = ws_data.get("age_s", float("nan"))
            if not (age != age) and age < 10 and ws_data.get("last_price", 0) > 0:
                ws_data["ticker"] = "XAUUSD CFD"
                return ws_data
    except Exception:
        pass

    # ── 2. REST fallback: gold-api.com ────────────────────────────────────────
    last, prev = np.nan, np.nan
    source = "none"

    try:
        r = _req.get("https://api.gold-api.com/price/XAU",
                     timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data       = r.json()
            spot       = float(data.get("price", 0))
            prev_api   = float(data.get("prev_close_price", 0)
                               or data.get("previousClose", 0)
                               or data.get("prev_price", 0) or 0)
            if spot > 0:
                last   = spot
                source = "gold-api.com (REST)"
            if prev_api > 0:
                prev   = prev_api   # mismo origen → cierre consistente
    except Exception:
        pass

    # Previous close
    if np.isnan(prev):
        try:
            df = yf_download("GC=F", period="5d", interval="1d", auto_adjust=True)
            if not df.empty:
                closes = df["Close"]
                if isinstance(closes, pd.DataFrame):
                    closes = closes.iloc[:, 0]
                closes = closes.dropna()
                if len(closes) >= 2:
                    prev = float(closes.iloc[-2])
        except Exception:
            pass

    # ── 3. yfinance fast_info last resort ─────────────────────────────────────
    if np.isnan(last) or last == 0:
        try:
            fi = yf_ticker_fast_info("GC=F")
            lp = fi.get("last_price")
            pc = fi.get("previous_close")
            if lp and float(lp) > 0:
                last   = float(lp)
                prev   = float(pc) if pc else prev
                source = "GC=F futures (yfinance)"
        except Exception:
            pass

    change     = last - prev if not (np.isnan(last) or np.isnan(prev)) else np.nan
    change_pct = (change / prev * 100) if not np.isnan(prev) and prev != 0 else np.nan
    return {
        "last_price":     last,
        "previous_close": prev,
        "change":         change,
        "change_pct":     change_pct,
        "ticker":         "XAUUSD CFD",
        "source":         source,
    }


class DataEngine:
    def __init__(self, period="6mo", interval="1d"):
        self.period       = period
        self.interval     = interval
        self.prices       = pd.DataFrame()
        self.volumes      = pd.DataFrame()
        self.returns      = pd.DataFrame()
        self.vwap         = pd.Series(dtype=float)
        self.realized_vol = pd.Series(dtype=float)
        self.loaded       = False

    def load(self):
        self.prices, self.volumes = fetch_with_volume(self.period, self.interval)
        if self.prices.empty:
            self.loaded = False
            return False
        raw_returns       = compute_log_returns(self.prices)
        self.returns      = winsorize_returns(raw_returns, threshold=4.0)
        self.vwap         = compute_vwap(self.prices, self.volumes)
        self.realized_vol = compute_realized_volatility(self.returns)
        self.loaded       = True
        return True

    def get_xau_price(self) -> pd.Series:
        return self.prices["XAUUSD"].dropna()  if "XAUUSD" in self.prices.columns  else pd.Series(dtype=float)

    def get_xau_returns(self) -> pd.Series:
        return self.returns["XAUUSD"].dropna() if "XAUUSD" in self.returns.columns else pd.Series(dtype=float)

    def summary(self) -> dict:
        if not self.loaded:
            return {}
        xau     = self.get_xau_price()
        xau_ret = self.get_xau_returns()
        rv      = self.realized_vol.dropna()
        return {
            "last_price":       float(xau.iloc[-1])  if len(xau)      else np.nan,
            "previous_close":   float(xau.iloc[-2])  if len(xau) >= 2 else (float(xau.iloc[-1]) if len(xau) else np.nan),
            "price_change_pct": float((xau.iloc[-1] / xau.iloc[-2] - 1) * 100) if len(xau) >= 2 else np.nan,
            "realized_vol":     float(rv.iloc[-1])   if len(rv)       else np.nan,
            "mean_return":      float(xau_ret.mean()) if len(xau_ret) else np.nan,
            "sharpe_approx":    float(xau_ret.mean() / xau_ret.std() * np.sqrt(252)) if len(xau_ret) else np.nan,
            "n_obs":            len(xau_ret),
        }
