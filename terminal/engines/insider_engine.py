import requests
import pandas as pd
import numpy as np
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# SEC CONFIG
# ─────────────────────────────────────────────

HEADERS = {
    "User-Agent": "AK_INC Insider Engine (contact: kevinagarciag27@gmail.com)"
}

BASE_URL = "https://data.sec.gov"

TICKER_TO_CIK = {
    "NEM":  "1164727",
    "GOLD": "756894",
    "AEM":  "783325",
    "KGC":  "701818",
}

ROLE_WEIGHT = {
    "CEO": 1.0,
    "CFO": 0.9,
    "Director": 0.7,
    "Other": 0.4,
}

# ─────────────────────────────────────────────
# SEC HELPERS
# ─────────────────────────────────────────────

def get_filings(cik, limit=10):
    try:
        url = f"{BASE_URL}/submissions/CIK{cik.zfill(10)}.json"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        results = []
        for i in range(len(recent.get("form", []))):
            if recent["form"][i] != "4":
                continue
            results.append({
                "accession": recent["accessionNumber"][i],
                "date": recent["filingDate"][i],
                "cik": cik,
            })
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


def get_xml(cik, accession):
    try:
        acc = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}.xml"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def parse_form4(xml_text):
    try:
        text = xml_text.lower()
        buy  = "p " in text or "purchase" in text
        sell = "s " in text or "sale" in text
        shares = text.count("<transactionshares>")
        return {"buy": buy, "sell": sell, "volume": shares}
    except Exception:
        return None


# ─────────────────────────────────────────────
# fetch_real_transactions — compatible alias
# ─────────────────────────────────────────────

def fetch_real_transactions(lookback_days=90):
    """
    Fetch Form 4 transactions from SEC EDGAR.
    Returns (DataFrame | None, source_string).
    """
    all_rows = []
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    for ticker, cik in TICKER_TO_CIK.items():
        filings = get_filings(cik, limit=20)
        for f in filings:
            try:
                filing_date = datetime.strptime(f["date"], "%Y-%m-%d")
            except Exception:
                continue
            if filing_date < cutoff:
                continue

            xml = get_xml(cik, f["accession"])
            if not xml:
                continue

            parsed = parse_form4(xml)
            if not parsed:
                continue

            if parsed["buy"] or parsed["sell"]:
                all_rows.append({
                    "ticker":    ticker,
                    "timestamp": f["date"],
                    "type":      "BUY" if parsed["buy"] else "SELL",
                    "value":     parsed["volume"] * 1000,
                    "insider":   "SEC Filing",
                    "role":      "Director",
                })
            time.sleep(0.15)

    if not all_rows:
        return None, "SEC EDGAR — no data"

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, "SEC EDGAR (Form 4 — real filings)"


# ─────────────────────────────────────────────
# ENGINE CORE
# ─────────────────────────────────────────────

class InsiderEngine:

    def __init__(self):
        self.data        = pd.DataFrame()
        self.score       = 0.0
        self.score_series = pd.Series(dtype=float)
        self.momentum    = 0.0
        self.data_source = "—"
        self._clusters   = 0

    # ── called from load_all_engines ──────────────────────────────────────
    def run_once(self):
        """Fetch data from SEC and store internally."""
        txns, src = fetch_real_transactions(lookback_days=90)
        self.data_source = src
        self.load(txns)
        self.build_score_series()
        self.compute_momentum()
        self.detect_clusters()

    # alias kept for backward compat
    def run(self):
        self.run_once()

    # ── called from _insider_live_panel ───────────────────────────────────
    def load(self, df):
        if df is None or (hasattr(df, "empty") and df.empty):
            self.data  = pd.DataFrame()
            self.score = 0.0
        else:
            self.data  = df.copy()
            buys  = df[df["type"] == "BUY"]["value"].sum()  if "type" in df.columns else 0
            sells = df[df["type"] == "SELL"]["value"].sum() if "type" in df.columns else 0
            self.score = float(buys - sells)

    def build_score_series(self):
        if self.data.empty or "timestamp" not in self.data.columns:
            self.score_series = pd.Series(dtype=float)
            return
        df = self.data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df["signed"] = df.apply(
            lambda r: r["value"] if r.get("type") == "BUY" else -r.get("value", 0), axis=1
        )
        self.score_series = df["signed"].resample("D").sum().cumsum()

    def compute_momentum(self):
        if len(self.score_series) >= 2:
            self.momentum = float(self.score_series.iloc[-1] - self.score_series.iloc[-2])
        else:
            self.momentum = 0.0

    def detect_clusters(self):
        if self.data.empty or "type" not in self.data.columns:
            self._clusters = 0
            return
        buys = self.data[self.data["type"] == "BUY"]
        self._clusters = max(0, len(buys) // 2)

    def recent_transactions(self, n=10):
        if self.data.empty:
            return pd.DataFrame()
        df = self.data.copy()
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False)
        return df.head(n)

    def summary(self):
        has_data = not self.data.empty
        n_buys  = int((self.data["type"] == "BUY").sum())  if has_data and "type" in self.data.columns else 0
        n_sells = int((self.data["type"] == "SELL").sum()) if has_data and "type" in self.data.columns else 0

        return {
            "status":          "ACTIVE" if has_data else "NO DATA",
            "data_available":  has_data,
            "data_source":     self.data_source,
            "current_score":   self.score,
            "momentum":        self.momentum,
            "n_transactions":  len(self.data),
            "n_buys":          n_buys,
            "n_sells":         n_sells,
            "n_buy_clusters":  self._clusters,
            "signal":          "BULLISH" if self.score > 0 else ("BEARISH" if self.score < 0 else "NEUTRAL"),
        }