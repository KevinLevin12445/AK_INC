import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")

# ─── GOLD-PROXY COMPANIES (SEC EDGAR Form 4 via yfinance) ─────────────────────
# These are the largest publicly-traded gold miners whose insider activity
# is the best available proxy for institutional "smart money" in gold.
GOLD_PROXIES = ["NEM", "GOLD", "AEM", "KGC"]

PROXY_NAMES = {
    "NEM":  "Newmont Corp",
    "GOLD": "Barrick Gold",
    "AEM":  "Agnico Eagle",
    "KGC":  "Kinross Gold",
}

ROLE_WEIGHTS = {
    "CEO": 1.0,
    "CFO": 0.9,
    "Director": 0.7,
    "VP": 0.6,
    "10pct_owner": 0.5,
    "Other": 0.3,
}

DECAY_LAMBDA = 0.05


def _decay_weight(delta_days: float, lam: float = DECAY_LAMBDA) -> float:
    return np.exp(-lam * max(delta_days, 0))


def _map_role(position: str) -> str:
    p = str(position).lower()
    if "chief executive" in p or p.startswith("ceo"): return "CEO"
    if "chief financial" in p or p.startswith("cfo"): return "CFO"
    if "director" in p:                                return "Director"
    if "vice president" in p or " vp" in p:           return "VP"
    if "10%" in p or "10 percent" in p or "beneficial owner" in p: return "10pct_owner"
    return "Other"


def _parse_txn_type(text: str) -> str:
    t = str(text).lower()
    if any(w in t for w in ["purchase", "acquisition", "buy", "grant", "award"]):
        return "BUY"
    return "SELL"


def fetch_real_transactions(
    tickers: list = None,
    lookback_days: int = 180,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch real SEC EDGAR Form 4 insider transactions for gold-proxy miners
    via yfinance (no API key required).

    Returns (DataFrame, source_label).
    Falls back to synthetic data if all fetches fail.
    """
    if tickers is None:
        tickers = GOLD_PROXIES

    frames = []
    fetched_from = []
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback_days)

    for sym in tickers:
        try:
            tk = yf.Ticker(sym)
            df = tk.insider_transactions
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue

            df = df.copy().reset_index(drop=True)

            # ── Normalise date column ──────────────────────────────────────────
            date_col = next(
                (c for c in df.columns if "date" in c.lower() or "start" in c.lower()),
                None,
            )
            if date_col is None:
                continue
            df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
            df = df.dropna(subset=["timestamp"])
            df = df[df["timestamp"] >= cutoff]
            if df.empty:
                continue

            # ── Transaction type (BUY / SELL) ──────────────────────────────────
            txn_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ["transaction", "text"])),
                None,
            )
            df["type"] = df[txn_col].apply(_parse_txn_type) if txn_col else "BUY"

            # ── Role mapping ───────────────────────────────────────────────────
            pos_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ["position", "title"])),
                None,
            )
            df["role"] = df[pos_col].apply(_map_role) if pos_col else "Other"

            # ── Insider name ───────────────────────────────────────────────────
            name_col = next(
                (c for c in df.columns if "insider" in c.lower() or "name" in c.lower()),
                None,
            )
            df["insider"] = df[name_col] if name_col else sym

            # ── Dollar value ───────────────────────────────────────────────────
            val_col = next(
                (c for c in df.columns if "value" in c.lower()),
                None,
            )
            shr_col = next(
                (c for c in df.columns if "share" in c.lower()),
                None,
            )
            if val_col:
                df["value"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0).abs()
            elif shr_col:
                df["value"] = pd.to_numeric(df[shr_col], errors="coerce").fillna(0).abs() * 35.0
            else:
                df["value"] = 1_000_000.0

            df["value"] = df["value"].clip(lower=1.0)
            df["ticker"] = sym
            df["company"] = PROXY_NAMES.get(sym, sym)

            frames.append(df[["timestamp", "type", "role", "value", "ticker", "company", "insider"]].copy())
            fetched_from.append(PROXY_NAMES.get(sym, sym))

        except Exception:
            pass

    if not frames:
        return _empty_transactions(), "LIVE DATA UNAVAILABLE (SEC EDGAR unreachable)"

    result = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    source = f"SEC EDGAR Form 4 — {', '.join(fetched_from)}"
    return result, source


def _empty_transactions() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema when live fetch fails."""
    return pd.DataFrame(columns=[
        "timestamp", "type", "role", "value", "ticker", "company", "insider"
    ])


# ─── INSIDER ENGINE ────────────────────────────────────────────────────────────

class InsiderEngine:
    def __init__(self, decay_lambda: float = DECAY_LAMBDA):
        self.decay_lambda = decay_lambda
        self.transactions: pd.DataFrame = pd.DataFrame()
        self.score_series: pd.Series = pd.Series(dtype=float)
        self.clusters: np.ndarray = np.array([])
        self.momentum: float = 0.0
        self.current_score: float = 0.0
        self.data_source: str = "—"

    def load(self, transactions: pd.DataFrame = None):
        if transactions is not None and not transactions.empty:
            self.transactions = transactions.copy()
        else:
            self.transactions = _empty_transactions()
            self.data_source = "LIVE DATA UNAVAILABLE"

    def compute_score(self, reference_time: datetime = None) -> float:
        if self.transactions.empty:
            return 0.0
        ref = reference_time or datetime.utcnow()
        score = 0.0
        for _, row in self.transactions.iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            delta_days = (ref - ts).total_seconds() / 86400
            w = ROLE_WEIGHTS.get(row["role"], 0.3)
            decay = _decay_weight(delta_days, self.decay_lambda)
            sign = 1.0 if row["type"] == "BUY" else -1.0
            score += sign * row["value"] * decay * w
        return score

    def build_score_series(self) -> pd.Series:
        if self.transactions.empty:
            return pd.Series(dtype=float)
        dates = pd.date_range(
            start=self.transactions["timestamp"].min().date(),
            end=datetime.utcnow().date(),
            freq="D",
        )
        scores = [
            self.compute_score(datetime(d.year, d.month, d.day, 23, 59, 59))
            for d in dates
        ]
        self.score_series = pd.Series(scores, index=dates, name="InsiderScore")
        self.current_score = float(self.score_series.iloc[-1])
        return self.score_series

    def compute_momentum(self, window: int = 5) -> float:
        if len(self.score_series) < window + 1:
            return 0.0
        recent = self.score_series.iloc[-window:]
        prev   = self.score_series.iloc[-2 * window: -window]
        if len(prev) == 0:
            return 0.0
        self.momentum = float(recent.mean() - prev.mean())
        return self.momentum

    def detect_clusters(self, eps_days: float = 7.0, min_samples: int = 2) -> pd.DataFrame:
        if self.transactions.empty:
            return pd.DataFrame()
        buys = self.transactions[self.transactions["type"] == "BUY"].copy()
        if len(buys) < min_samples:
            return pd.DataFrame()
        base = buys["timestamp"].min()
        X = ((buys["timestamp"] - base).dt.total_seconds() / 86400).values.reshape(-1, 1)
        db = DBSCAN(eps=eps_days, min_samples=min_samples)
        labels = db.fit_predict(X)
        buys = buys.copy()
        buys["cluster"] = labels
        self.clusters = labels
        return buys

    def recent_transactions(self, n: int = 8) -> pd.DataFrame:
        if self.transactions.empty:
            return pd.DataFrame()
        df = self.transactions.sort_values("timestamp", ascending=False).head(n)
        cols = [c for c in ["timestamp", "ticker", "insider", "role", "type", "value"] if c in df.columns]
        return df[cols].copy()

    def run(self):
        txns, source = fetch_real_transactions()
        self.data_source = source
        self.load(txns)
        self.build_score_series()
        self.compute_momentum()
        self.detect_clusters()

    def summary(self) -> dict:
        clusters_df = self.detect_clusters()
        n_clusters = int((clusters_df["cluster"] >= 0).sum()) if not clusters_df.empty else 0
        buys  = len(self.transactions[self.transactions["type"] == "BUY"])  if not self.transactions.empty else 0
        sells = len(self.transactions[self.transactions["type"] == "SELL"]) if not self.transactions.empty else 0
        no_data = self.transactions.empty
        return {
            "current_score":   self.current_score,
            "momentum":        self.momentum,
            "n_transactions":  len(self.transactions),
            "n_buy_clusters":  n_clusters,
            "n_buys":          buys,
            "n_sells":         sells,
            "signal":          "N/A" if no_data else ("BULLISH" if self.current_score > 0 else "BEARISH"),
            "data_available":  not no_data,
            "data_source":     self.data_source,
        }
