import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def value_at_risk(returns: pd.Series, confidence: float = 0.95, method: str = "parametric") -> float:
    """VaR = μ - z_α * σ  (parametric) or empirical quantile."""
    clean = returns.dropna()
    if len(clean) < 5:
        return np.nan
    if method == "historical":
        return float(-np.percentile(clean, (1 - confidence) * 100))
    mu = clean.mean()
    sigma = clean.std()
    z = stats.norm.ppf(confidence)
    return float(-(mu - z * sigma))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """ES = E[loss | loss > VaR]"""
    clean = returns.dropna()
    if len(clean) < 5:
        return np.nan
    var = value_at_risk(clean, confidence=confidence, method="historical")
    losses = -clean
    tail = losses[losses > var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def max_drawdown(price: pd.Series) -> dict:
    """Peak-to-trough drawdown metrics."""
    clean = price.dropna()
    if len(clean) < 2:
        return {"max_dd": np.nan, "current_dd": np.nan, "peak": np.nan}

    rolling_max = clean.cummax()
    dd = (clean - rolling_max) / rolling_max
    max_dd = float(dd.min())
    current_dd = float(dd.iloc[-1])
    peak = float(rolling_max.iloc[-1])

    return {
        "max_dd": max_dd,
        "current_dd": current_dd,
        "peak": peak,
        "drawdown_series": dd,
    }


def kelly_criterion(returns: pd.Series) -> float:
    """Full Kelly fraction: f* = μ / σ²"""
    clean = returns.dropna()
    mu = clean.mean()
    sigma2 = clean.var()
    if sigma2 <= 0:
        return 0.0
    return float(mu / sigma2)


def rolling_var(returns: pd.Series, window: int = 20, confidence: float = 0.95) -> pd.Series:
    def _var(x):
        return value_at_risk(pd.Series(x), confidence=confidence, method="historical")
    return returns.rolling(window=window).apply(_var, raw=True).rename("RollingVaR")


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series = None, annualize: bool = True) -> float:
    clean = returns.dropna()
    if benchmark_returns is not None:
        excess = clean - benchmark_returns.reindex(clean.index).fillna(0)
    else:
        excess = clean
    if excess.std() == 0:
        return 0.0
    ir = excess.mean() / excess.std()
    if annualize:
        ir *= np.sqrt(252)
    return float(ir)


class RiskEngine:
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self.metrics: dict = {}

    def run(self, returns: pd.Series, price: pd.Series) -> dict:
        var_param = value_at_risk(returns, confidence=self.confidence, method="parametric")
        var_hist = value_at_risk(returns, confidence=self.confidence, method="historical")
        es = expected_shortfall(returns, confidence=self.confidence)
        dd = max_drawdown(price)
        kelly = kelly_criterion(returns)
        ir = information_ratio(returns, annualize=True)

        sigma_daily = float(returns.std())
        sigma_annual = sigma_daily * np.sqrt(252)
        mu_annual = float(returns.mean()) * 252
        sharpe = mu_annual / sigma_annual if sigma_annual > 0 else 0.0

        self.metrics = {
            "VaR_95_param": var_param,
            "VaR_95_hist": var_hist,
            "ES_95": es,
            "max_drawdown": dd["max_dd"],
            "current_drawdown": dd["current_dd"],
            "kelly_fraction": kelly,
            "information_ratio": ir,
            "sharpe_ratio": sharpe,
            "vol_daily": sigma_daily,
            "vol_annual": sigma_annual,
            "drawdown_series": dd.get("drawdown_series", pd.Series(dtype=float)),
        }
        return self.metrics
