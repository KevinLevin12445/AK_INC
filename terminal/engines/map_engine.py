import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

SECTOR_UNIVERSE: dict[str, dict[str, tuple]] = {
    "Gold & Precious": {
        "GC=F":  ("★ XAUUSD Spot / Futures",  12000),   # TOP — most prominent box
        "GLD":   ("SPDR Gold Trust",            3000),
        "IAU":   ("iShares Gold ETF",           1500),
        "GDX":   ("VanEck Gold Miners",         1200),
        "NEM":   ("Newmont Corp",                450),
        "GOLD":  ("Barrick Gold",                350),
        "AEM":   ("Agnico Eagle",                320),
        "WPM":   ("Wheaton Precious",            230),
        "FNV":   ("Franco-Nevada",               200),
        "KGC":   ("Kinross Gold",                 90),
    },
    "Energy": {
        "XOM":  ("ExxonMobil",    450),
        "CVX":  ("Chevron",       280),
        "COP":  ("ConocoPhillips",150),
        "OXY":  ("Occidental",    100),
        "SLB":  ("SLB",            60),
    },
    "Financials": {
        "JPM":  ("JPMorgan Chase",  550),
        "BAC":  ("Bank of America", 300),
        "GS":   ("Goldman Sachs",   150),
        "MS":   ("Morgan Stanley",  160),
    },
    "Technology": {
        "AAPL":  ("Apple",    3100),
        "MSFT":  ("Microsoft",3000),
        "NVDA":  ("NVIDIA",   2800),
        "GOOGL": ("Alphabet", 2100),
    },
    "Consumer": {
        "AMZN": ("Amazon",   2000),
        "WMT":  ("Walmart",   700),
        "PG":   ("P&G",       380),
        "KO":   ("Coca-Cola", 260),
    },
    "Macro ETFs": {
        "GC=F":  ("Gold Futures (XAUUSD)",  5000),   # Gold first in macro
        "SPY":   ("S&P 500 ETF",             550),
        "QQQ":   ("Nasdaq 100 ETF",          220),
        "TLT":   ("20Y Treasury",             40),
        "UUP":   ("US Dollar ETF",            20),
    },
}

COUNTRY_ETFS: dict[str, tuple] = {
    "United States":  ("SPY",  "USA"),
    "Japan":          ("EWJ",  "JPN"),
    "China":          ("FXI",  "CHN"),
    "United Kingdom": ("EWU",  "GBR"),
    "Germany":        ("EWG",  "DEU"),
    "France":         ("EWQ",  "FRA"),
    "Brazil":         ("EWZ",  "BRA"),
    "India":          ("INDA", "IND"),
    "Canada":         ("EWC",  "CAN"),
    "Australia":      ("EWA",  "AUS"),
    "South Korea":    ("EWY",  "KOR"),
    "Mexico":         ("EWW",  "MEX"),
    "Switzerland":    ("EWL",  "CHE"),
    "South Africa":   ("EZA",  "ZAF"),
    "Saudi Arabia":   ("KSA",  "SAU"),
}

# Top gold-producing nations (ISO3, approx tonnes/yr weight) — for geo overlay
GOLD_PRODUCER_NATIONS = {
    "CHN": ("China",         380),
    "AUS": ("Australia",     310),
    "RUS": ("Russia",        300),
    "CAN": ("Canada",        180),
    "USA": ("United States", 170),
    "ZAF": ("South Africa",  100),
    "GHA": ("Ghana",          80),
    "PER": ("Peru",          100),
    "IDN": ("Indonesia",      80),
    "UZB": ("Uzbekistan",     60),
}


def _extract_close(raw: pd.DataFrame, sym: str, all_syms: list) -> pd.Series:
    if len(all_syms) == 1:
        return raw.get("Close", pd.Series(dtype=float))
    key = (sym, "Close")
    if key in raw.columns:
        return raw[key]
    return pd.Series(dtype=float)


def _chg_pct(close: pd.Series) -> float:
    c = close.dropna()
    if len(c) < 2:
        return 0.0
    return float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100)


class MapEngine:
    def __init__(self):
        self.sector_data: pd.DataFrame = pd.DataFrame()
        self.geo_data: pd.DataFrame    = pd.DataFrame()
        self.news_items: list          = []
        self.gold_change_pct: float    = 0.0   # GC=F daily % for geo overlay
        self.gold_price: float         = 0.0

    def _load_gcf(self, period: str = "5d") -> tuple[float, float]:
        """Download GC=F and return (last_price, change_pct)."""
        try:
            raw = yf.download("GC=F", period=period, interval="1d",
                              auto_adjust=True, progress=False)
            close = raw.get("Close", pd.Series(dtype=float))
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            c = close.dropna()
            if len(c) >= 2:
                return float(c.iloc[-1]), float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100)
            elif len(c) == 1:
                return float(c.iloc[-1]), 0.0
        except Exception:
            pass
        return 0.0, 0.0

    def load_sector_data(self, period: str = "5d") -> pd.DataFrame:
        all_tickers = list({t for sec in SECTOR_UNIVERSE.values() for t in sec})
        try:
            raw = yf.download(
                all_tickers, period=period, interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
        except Exception:
            self.sector_data = pd.DataFrame()
            return self.sector_data

        rows = []
        for sector, tickers in SECTOR_UNIVERSE.items():
            for sym, (name, mktcap) in tickers.items():
                close = _extract_close(raw, sym, all_tickers)
                c     = close.dropna()
                chg   = _chg_pct(close)
                price = round(float(c.iloc[-1]), 2) if len(c) > 0 else 0.0

                # Store gold change for geo overlay
                if sym == "GC=F":
                    self.gold_change_pct = chg
                    self.gold_price      = price

                rows.append({
                    "sector":     sector,
                    "ticker":     sym,
                    "name":       name,
                    "market_cap": mktcap,
                    "change_pct": round(chg, 3),
                    "price":      price,
                    "is_gold":    sym == "GC=F",
                })

        self.sector_data = pd.DataFrame(rows)
        return self.sector_data

    def load_geo_data(self, period: str = "5d") -> pd.DataFrame:
        syms = [v[0] for v in COUNTRY_ETFS.values()]
        try:
            raw = yf.download(
                syms, period=period, interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
        except Exception:
            self.geo_data = pd.DataFrame()
            return self.geo_data

        rows = []
        for country, (sym, iso3) in COUNTRY_ETFS.items():
            close = _extract_close(raw, sym, syms)
            rows.append({
                "country":    country,
                "iso3":       iso3,
                "ticker":     sym,
                "change_pct": round(_chg_pct(close), 3),
                "gold_prod":  GOLD_PRODUCER_NATIONS.get(iso3, (None, 0))[1],
            })

        self.geo_data = pd.DataFrame(rows)
        return self.geo_data

    def load_news(self) -> list:
        try:
            ticker          = yf.Ticker("GC=F")
            self.news_items = (ticker.news or [])[:15]
        except Exception:
            self.news_items = []
        return self.news_items

    def load_all(self, period: str = "5d") -> bool:
        try:
            self.load_sector_data(period)
            self.load_geo_data(period)
            self.load_news()
            return True
        except Exception:
            return False
