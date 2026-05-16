from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import numpy as np
import pandas as pd

# Gold-core keywords — at least ONE must appear in the headline for a news alert to fire
GOLD_CORE = {"gold", "xau", "bullion", "precious metal", "gold price", "spot gold", "gold miner"}

# Supporting weight keywords — add severity when gold-core is already present
GOLD_SUPPORT = [
    "inflation", "fed", "federal reserve", "rates", "cpi",
    "war", "geopolit", "yield", "dollar", "treasury",
    "hike", "rate cut", "rate hike", "refuge", "safe haven",
    "tariff", "sanction", "recession", "fomc", "powell",
    "central bank", "silver", "commodity", "commodities",
]

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

NEWS_MAX_AGE_HOURS = 48


@dataclass
class Alert:
    timestamp: str
    asset: str
    alert_type: str
    severity: str
    message: str
    score: float = 0.0

    @property
    def color(self) -> str:
        return {"HIGH": "#ff4444", "MEDIUM": "#ffd700", "LOW": "#00aaff"}.get(self.severity, "#888888")

    @property
    def icon(self) -> str:
        return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(self.severity, "⚪")


class AlertEngine:
    def __init__(self):
        self.alerts: List[Alert] = []

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().strftime("%H:%M:%S")

    # ── PRICE ALERT ────────────────────────────────────────────────────────────
    def check_price_moves(
        self,
        live_price: Optional[float],
        prev_close: Optional[float],
    ) -> List[Alert]:
        """Alert only when live_price vs previous close moves significantly.
        Both values must be real numbers — never use historical series iloc[-1]."""
        out = []
        if not live_price or not prev_close or prev_close == 0:
            return out

        pct = (live_price - prev_close) / prev_close * 100

        if abs(pct) >= 2.0:
            sev = "HIGH"
        elif abs(pct) >= 1.0:
            sev = "MEDIUM"
        else:
            return out  # sub-1% intraday move is noise — skip

        direction = "SURGE ▲" if pct > 0 else "DROP ▼"
        out.append(Alert(
            timestamp=self._now(),
            asset="GOLD",
            alert_type="price",
            severity=sev,
            message=(
                f"{direction} {pct:+.2f}%  "
                f"live=${live_price:,.2f}  prev close=${prev_close:,.2f}"
            ),
            score=abs(pct),
        ))
        return out

    # ── VOLATILITY ALERT ───────────────────────────────────────────────────────
    def check_volatility(self, returns: pd.Series) -> List[Alert]:
        """Volatility spike: 5-day realized vol vs 20-day baseline."""
        out = []
        if returns is None or len(returns) < 21:
            return out

        vol_20 = float(returns.tail(20).std() * np.sqrt(252) * 100)
        vol_5  = float(returns.tail(5).std()  * np.sqrt(252) * 100)
        ratio  = vol_5 / vol_20 if vol_20 > 0 else 1.0

        if ratio >= 2.0:
            sev = "HIGH"
        elif ratio >= 1.5:
            sev = "MEDIUM"
        else:
            return out

        out.append(Alert(
            timestamp=self._now(),
            asset="GOLD",
            alert_type="volatility",
            severity=sev,
            message=f"Vol spike: 5d={vol_5:.1f}%  20d={vol_20:.1f}%  ratio={ratio:.1f}×",
            score=ratio,
        ))
        return out

    # ── INSIDER ALERT ──────────────────────────────────────────────────────────
    def check_insider(self, insider_engine) -> List[Alert]:
        """Insider accumulation / distribution signal from score_series.
        Score is dollar-weighted (value × decay × role_weight) — expressed in $K.
        Alert only fires when net flow is significant relative to rolling range."""
        out = []
        try:
            score_series = getattr(insider_engine, "score_series", None)
            if score_series is None or (hasattr(score_series, "empty") and score_series.empty):
                return out
            if len(score_series) < 3:
                return out

            s       = float(score_series.iloc[-1])
            s_abs   = abs(s)
            s_max   = float(score_series.abs().max())

            # Skip if no meaningful flow or baseline is zero
            if s_max == 0 or s_abs < 1_000:          # less than $1K net flow — skip
                return out

            # Relative strength: how extreme is today vs historical range?
            rel = s_abs / s_max                       # 0.0 → 1.0
            if rel < 0.25:                            # bottom 25% of historical range — noise
                return out

            sev       = "HIGH" if rel >= 0.70 else "MEDIUM"
            direction = "ACCUMULATION ▲" if s > 0 else "DISTRIBUTION ▼"

            # Human-readable dollar amount
            if s_abs >= 1_000_000:
                flow_str = f"${s_abs/1_000_000:.1f}M"
            elif s_abs >= 1_000:
                flow_str = f"${s_abs/1_000:.0f}K"
            else:
                flow_str = f"${s_abs:.0f}"

            out.append(Alert(
                timestamp=self._now(),
                asset="GOLD",
                alert_type="insider",
                severity=sev,
                message=f"Insider {direction}  net flow ≈ {flow_str}  (rel. strength {rel:.0%})",
                score=rel,
            ))
        except Exception:
            pass
        return out

    # ── NEWS ALERT ─────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_news_ts(item: dict) -> Optional[datetime]:
        """Parse UTC publish time from a yfinance news dict.
        yfinance ≥1.1 structure: {'id':..., 'content': {'pubDate': '2026-05-02T19:17:10Z'}}
        Older yfinance: {'providerPublishTime': <unix int>}
        """
        ts = (
            (item.get("content") or {}).get("pubDate")
            or item.get("providerPublishTime")
        )
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return None

    def check_news(self, news_items: list) -> List[Alert]:
        """Fire alerts ONLY when headline contains a gold-core keyword.
        Supporting keywords add severity weight but cannot trigger an alert alone."""
        out = []
        seen: set = set()
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=NEWS_MAX_AGE_HOURS)

        for item in (news_items or [])[:20]:
            if not isinstance(item, dict):
                continue

            title = (
                (item.get("content") or {}).get("title")
                or item.get("title")
                or item.get("headline", "")
            )
            if not title or title in seen:
                continue
            seen.add(title)

            # ── Recency guard: skip news older than NEWS_MAX_AGE_HOURS ──
            pub_dt = self._parse_news_ts(item)
            if pub_dt is not None and pub_dt < cutoff:
                continue

            lower = title.lower()

            # ── GATE: gold-core keyword must be present ──
            if not any(kw in lower for kw in GOLD_CORE):
                continue

            # ── Severity: count supporting keywords ──
            support_hits = sum(1 for kw in GOLD_SUPPORT if kw in lower)
            score = 1.0 + support_hits

            if support_hits >= 2:
                sev = "HIGH"
            elif support_hits >= 1:
                sev = "MEDIUM"
            else:
                sev = "LOW"

            short = title[:85] + "…" if len(title) > 85 else title
            out.append(Alert(
                timestamp=self._now(),
                asset="GOLD",
                alert_type="news",
                severity=sev,
                message=short,
                score=score,
            ))

        out.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 3), -a.score))
        return out[:6]

    # ── GENERATE ───────────────────────────────────────────────────────────────
    def generate(
        self,
        live_price: Optional[float]       = None,
        prev_close: Optional[float]       = None,
        returns: Optional[pd.Series]      = None,
        insider_engine                    = None,
        news_items: Optional[list]        = None,
    ) -> List[Alert]:
        self.alerts = []
        self.alerts += self.check_price_moves(live_price, prev_close)
        if returns is not None:
            self.alerts += self.check_volatility(returns)
        if insider_engine is not None:
            self.alerts += self.check_insider(insider_engine)
        if news_items:
            self.alerts += self.check_news(news_items)

        self.alerts.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 3), -a.score))
        return self.alerts
