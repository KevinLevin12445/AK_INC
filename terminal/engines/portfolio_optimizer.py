import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


def markowitz_optimize(
    returns: pd.DataFrame,
    risk_aversion: float = 2.0,
    turnover_penalty: float = 0.1,
    w_prev: np.ndarray = None,
    l2_reg: float = 1e-3,
    max_weight: float = 0.4,
    min_weight: float = 0.0,
) -> dict:
    """
    Maximize: μᵀw - λ wᵀΣw - γ ||w - w_prev||²
    Subject to: sum(w) = 1, 0 ≤ w ≤ max_weight
    """
    if returns.empty or len(returns) < 5:
        return {"weights": {}, "expected_return": np.nan, "portfolio_vol": np.nan, "sharpe": np.nan}

    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252
    n = len(mu)

    if w_prev is None:
        w_prev = np.ones(n) / n

    def neg_utility(w):
        port_ret = mu @ w
        port_var = w @ Sigma @ w
        turnover = turnover_penalty * np.sum((w - w_prev) ** 2)
        l2 = l2_reg * np.sum(w ** 2)
        return -(port_ret - risk_aversion * port_var - turnover - l2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n

    w0 = w_prev.copy()
    result = minimize(
        neg_utility,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if result.success:
        w_opt = result.x
    else:
        w_opt = np.ones(n) / n

    w_opt = np.clip(w_opt, 0, max_weight)
    w_opt /= w_opt.sum()

    port_ret = float(mu @ w_opt)
    port_vol = float(np.sqrt(w_opt @ Sigma @ w_opt))
    sharpe = port_ret / port_vol if port_vol > 0 else 0.0

    return {
        "weights": dict(zip(returns.columns, w_opt)),
        "expected_return": port_ret,
        "portfolio_vol": port_vol,
        "sharpe": sharpe,
        "success": result.success,
    }


def efficient_frontier(returns: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    if returns.empty or len(returns) < 5:
        return pd.DataFrame()

    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252
    n = len(mu)

    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    vols = []
    rets_out = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n

        res = minimize(
            lambda w: w @ Sigma @ w,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )
        if res.success:
            vols.append(float(np.sqrt(res.x @ Sigma @ res.x)))
            rets_out.append(float(mu @ res.x))

    return pd.DataFrame({"vol": vols, "ret": rets_out})


class PortfolioOptimizer:
    def __init__(self, risk_aversion: float = 2.0, turnover_penalty: float = 0.1):
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.result: dict = {}
        self.frontier: pd.DataFrame = pd.DataFrame()
        self.w_prev: np.ndarray = None

    def run(self, returns: pd.DataFrame) -> dict:
        self.result = markowitz_optimize(
            returns,
            risk_aversion=self.risk_aversion,
            turnover_penalty=self.turnover_penalty,
            w_prev=self.w_prev,
        )
        if "weights" in self.result and self.result["weights"]:
            self.w_prev = np.array(list(self.result["weights"].values()))
        self.frontier = efficient_frontier(returns)
        return self.result
