import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


class BayesianSignalModel:
    """
    P(long | X) ∝ P(X | long) * P(long)
    
    Gaussian likelihood per class, updated in an online fashion.
    Classes: long (1), short (-1)
    """

    def __init__(self, prior_long: float = 0.5):
        self.prior_long = prior_long
        self.prior_short = 1 - prior_long
        self.mu_long: np.ndarray = None
        self.mu_short: np.ndarray = None
        self.sigma_long: np.ndarray = None
        self.sigma_short: np.ndarray = None
        self.fitted = False

    def fit(self, features: pd.DataFrame, returns_fwd: pd.Series):
        X = features.dropna()
        y = returns_fwd.reindex(X.index).dropna()
        X = X.reindex(y.index)

        long_mask = y > 0
        short_mask = y <= 0

        X_long = X[long_mask].values
        X_short = X[short_mask].values

        if len(X_long) < 5 or len(X_short) < 5:
            self.fitted = False
            return self

        self.mu_long = X_long.mean(axis=0)
        self.mu_short = X_short.mean(axis=0)
        self.sigma_long = np.maximum(X_long.std(axis=0), 1e-6)
        self.sigma_short = np.maximum(X_short.std(axis=0), 1e-6)

        self.prior_long = long_mask.mean()
        self.prior_short = 1 - self.prior_long

        self.fitted = True
        return self

    def predict_proba(self, x: np.ndarray) -> dict:
        if not self.fitted:
            return {"P_long": 0.5, "P_short": 0.5, "signal": "NEUTRAL", "confidence": 0.0}

        log_lik_long = np.sum(norm.logpdf(x, loc=self.mu_long, scale=self.sigma_long))
        log_lik_short = np.sum(norm.logpdf(x, loc=self.mu_short, scale=self.sigma_short))

        log_post_long = log_lik_long + np.log(self.prior_long + 1e-10)
        log_post_short = log_lik_short + np.log(self.prior_short + 1e-10)

        max_log = max(log_post_long, log_post_short)
        p_long = np.exp(log_post_long - max_log)
        p_short = np.exp(log_post_short - max_log)
        total = p_long + p_short

        p_long /= total
        p_short /= total

        if p_long > 0.55:
            signal = "LONG"
        elif p_short > 0.55:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        confidence = float(abs(p_long - p_short))

        return {
            "P_long": float(p_long),
            "P_short": float(p_short),
            "signal": signal,
            "confidence": confidence,
        }

    def predict_series(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or features.empty:
            return pd.DataFrame()

        results = []
        for _, row in features.iterrows():
            r = self.predict_proba(row.values)
            results.append(r)

        return pd.DataFrame(results, index=features.index)
