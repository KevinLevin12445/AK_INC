import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─── KALMAN FILTER ────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    Local-level model:
        x_t = x_{t-1} + w_t      (state, Q = process noise)
        y_t = x_t + v_t          (obs,   R = obs noise)
    """

    def __init__(self, Q: float = 1e-4, R: float = 1e-2):
        self.Q = Q
        self.R = R

    def filter(self, observations: np.ndarray):
        n = len(observations)
        if n == 0:
            return np.array([]), np.array([]), np.array([])
        x_hat = np.zeros(n)
        P = np.zeros(n)
        K = np.zeros(n)

        x_hat[0] = observations[0]
        P[0] = 1.0

        for t in range(1, n):
            x_pred = x_hat[t - 1]
            P_pred = P[t - 1] + self.Q

            K[t] = P_pred / (P_pred + self.R)
            x_hat[t] = x_pred + K[t] * (observations[t] - x_pred)
            P[t] = (1 - K[t]) * P_pred

        return x_hat, P, K

    def smooth_series(self, series: pd.Series) -> pd.Series:
        clean = series.dropna()
        if clean.empty:
            return pd.Series([], dtype=float, name="Kalman_Trend")
        obs = clean.values
        smoothed, _, _ = self.filter(obs)
        return pd.Series(smoothed, index=clean.index, name="Kalman_Trend")

    def residual_noise(self, series: pd.Series) -> pd.Series:
        obs = series.dropna()
        if obs.empty:
            return pd.Series([], dtype=float, name="Kalman_Noise")
        smoothed = self.smooth_series(obs)
        noise = obs - smoothed
        return noise.rename("Kalman_Noise")


# ─── Z-SCORE ──────────────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mu = series.rolling(window=window).mean()
    sigma = series.rolling(window=window).std()
    z = (series - mu) / sigma
    return z.rename(f"zscore_{window}")


# ─── VWAP DEVIATION ───────────────────────────────────────────────────────────

def vwap_deviation(price: pd.Series, vwap: pd.Series) -> pd.Series:
    dev = (price - vwap) / vwap
    return dev.rename("VWAP_Dev")


# ─── STOCHASTIC VOLATILITY (GARCH-like) ───────────────────────────────────────

def stochastic_volatility(returns: pd.Series, alpha: float = 0.1, beta: float = 0.85, gamma: float = 0.05) -> pd.Series:
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return pd.Series(dtype=float, name="StochVol")
    sigma2 = np.zeros(n)
    r = returns.values
    sigma2[0] = r[0] ** 2
    omega = max((1 - alpha - beta - gamma) * np.var(r), 1e-10)
    for t in range(1, n):
        sigma2[t] = omega + alpha * sigma2[t - 1] + beta * (r[t - 1] ** 2) + gamma * (r[t - 1] ** 2)
    sigma2 = np.maximum(sigma2, 1e-10)
    vol = np.sqrt(sigma2) * np.sqrt(252)
    return pd.Series(vol, index=returns.index, name="StochVol")


# ─── COINTEGRATION (Engle-Granger) ────────────────────────────────────────────

def cointegration_spread(gold: pd.Series, dxy: pd.Series, window: int = 60) -> dict:
    df = pd.concat([gold, dxy], axis=1).dropna()
    df.columns = ["gold", "dxy"]

    if len(df) < 30:
        return {"spread": pd.Series(dtype=float), "zscore": pd.Series(dtype=float),
                "beta": np.nan, "pvalue": np.nan}

    from scipy.stats import linregress
    slope, intercept, r, p, se = linregress(df["dxy"].values, df["gold"].values)
    spread = df["gold"] - slope * df["dxy"] - intercept

    mu = spread.rolling(window=window).mean()
    sigma = spread.rolling(window=window).std()
    zscore = (spread - mu) / sigma

    return {
        "spread": spread.rename("Coint_Spread"),
        "zscore": zscore.rename("Coint_ZScore"),
        "beta": slope,
        "pvalue": p,
        "r_squared": r ** 2,
    }


# ─── ORDER IMBALANCE (synthetic microstructure) ───────────────────────────────

def order_imbalance(returns: pd.Series, window: int = 5) -> pd.Series:
    sign_flow = np.sign(returns)
    oi = sign_flow.rolling(window=window).sum() / window
    return oi.rename("OrderImbalance")


# ─── YIELD ANOMALY ────────────────────────────────────────────────────────────

def yield_anomaly(us10y: pd.Series, window: int = 20) -> pd.Series:
    z = rolling_zscore(us10y, window=window)
    return z.rename("YieldAnomaly")


# ─── CARRY (rate differential proxy) ─────────────────────────────────────────

def carry_proxy(us10y: pd.Series, window: int = 20) -> pd.Series:
    carry = us10y - us10y.rolling(window=window).mean()
    return carry.rename("Carry")


# ─── FEATURE MATRIX BUILDER ───────────────────────────────────────────────────

class FeatureEngine:
    def __init__(self, kalman_Q=1e-4, kalman_R=1e-2):
        self.kf = KalmanFilter(Q=kalman_Q, R=kalman_R)
        self.features: pd.DataFrame = pd.DataFrame()
        self.kalman_trend: pd.Series = pd.Series(dtype=float)
        self.kalman_noise: pd.Series = pd.Series(dtype=float)
        self.coint_result: dict = {}
        self.stoch_vol: pd.Series = pd.Series(dtype=float)

    def build(self, prices: pd.DataFrame, returns: pd.DataFrame, vwap: pd.Series) -> pd.DataFrame:
        gold_price = prices["XAUUSD"] if "XAUUSD" in prices.columns else prices.iloc[:, 0]
        gold_ret = returns["XAUUSD"] if "XAUUSD" in returns.columns else returns.iloc[:, 0]

        self.kalman_trend = self.kf.smooth_series(gold_price)
        self.kalman_noise = self.kf.residual_noise(gold_price)

        zscore_20 = rolling_zscore(gold_ret, window=20)
        zscore_60 = rolling_zscore(gold_ret, window=60)

        vwap_dev = vwap_deviation(gold_price, vwap) if not vwap.empty else pd.Series(0, index=gold_price.index)

        self.stoch_vol = stochastic_volatility(gold_ret)

        oi = order_imbalance(gold_ret, window=5)

        feat_list = [zscore_20, zscore_60, vwap_dev, self.stoch_vol, oi]

        if "DXY" in prices.columns:
            dxy = prices["DXY"]
            self.coint_result = cointegration_spread(gold_price, dxy)
            feat_list.append(self.coint_result["zscore"])
        else:
            self.coint_result = {}

        if "US10Y" in prices.columns:
            us10y = prices["US10Y"]
            feat_list.append(yield_anomaly(us10y))
            feat_list.append(carry_proxy(us10y))

        self.features = pd.concat(feat_list, axis=1).dropna()
        return self.features

    def get_latest(self) -> dict:
        if self.features.empty:
            return {}
        row = self.features.iloc[-1].to_dict()
        return row
