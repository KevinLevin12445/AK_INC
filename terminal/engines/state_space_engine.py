import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


STATE_LABELS = {0: "TREND", 1: "MEAN_REVERSION", 2: "HIGH_VOLATILITY"}
STATE_COLORS = {0: "#00ff41", 1: "#ffd700", 2: "#ff4444"}


class HMMEngine:
    """
    Gaussian HMM with 3 hidden states trained on return + volatility features.
    Falls back to a rule-based classifier if hmmlearn is unavailable.
    """

    def __init__(self, n_components: int = 3, n_iter: int = 200, random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.states: np.ndarray = np.array([])
        self.state_probs: np.ndarray = np.array([])
        self.fitted = False
        self._use_hmmlearn = True

    def _build_obs(self, returns: pd.Series, vol: pd.Series) -> np.ndarray:
        df = pd.concat([returns.rename("ret"), vol.rename("vol")], axis=1).dropna()
        self._obs_index = df.index
        return df.values

    def fit(self, returns: pd.Series, vol: pd.Series):
        obs = self._build_obs(returns, vol)
        if len(obs) < 20:
            self.fitted = False
            return self

        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(obs)
            self.model = model
            self.states = model.predict(obs)
            self.state_probs = model.predict_proba(obs)
            self._use_hmmlearn = True
        except Exception:
            self._use_hmmlearn = False
            self._rule_based(obs)

        self._relabel_states(returns)
        self.fitted = True
        return self

    def _rule_based(self, obs: np.ndarray):
        ret = obs[:, 0]
        vol = obs[:, 1]
        vol_high = np.percentile(vol, 75)
        ret_pos = np.percentile(ret, 60)
        ret_neg = np.percentile(ret, 40)

        states = np.zeros(len(obs), dtype=int)
        for i in range(len(obs)):
            if vol[i] >= vol_high:
                states[i] = 2
            elif ret[i] >= ret_pos:
                states[i] = 0
            elif ret[i] <= ret_neg:
                states[i] = 1
            else:
                states[i] = 1

        n = len(obs)
        probs = np.zeros((n, self.n_components))
        for i in range(n):
            probs[i, states[i]] = 0.8
            others = [j for j in range(self.n_components) if j != states[i]]
            for j in others:
                probs[i, j] = 0.1

        self.states = states
        self.state_probs = probs

    def _relabel_states(self, returns: pd.Series):
        """
        Re-order states: state with highest mean return → 0 (TREND),
        state with lowest volatility among remaining → 1 (MEAN_REVERSION),
        remaining → 2 (HIGH_VOLATILITY).
        """
        ret_vals = returns.reindex(self._obs_index).values
        n_states = self.n_components

        state_means = []
        state_vols = []
        for s in range(n_states):
            mask = self.states == s
            if mask.sum() == 0:
                state_means.append(-999)
                state_vols.append(999)
            else:
                state_means.append(ret_vals[mask].mean())
                state_vols.append(ret_vals[mask].std())

        trend_state = int(np.argmax(state_means))
        hv_state = int(np.argmax(state_vols))
        remaining = [s for s in range(n_states) if s not in [trend_state, hv_state]]
        mr_state = remaining[0] if remaining else (1 if trend_state != 1 else 0)

        mapping = {trend_state: 0, mr_state: 1, hv_state: 2}
        if len(set(mapping.values())) < n_states:
            return

        new_states = np.zeros_like(self.states)
        new_probs = np.zeros_like(self.state_probs)
        for old, new in mapping.items():
            new_states[self.states == old] = new
            new_probs[:, new] = self.state_probs[:, old]

        self.states = new_states
        self.state_probs = new_probs

    def current_state(self) -> int:
        if not self.fitted or len(self.states) == 0:
            return -1
        return int(self.states[-1])

    def current_probs(self) -> dict:
        if not self.fitted or len(self.state_probs) == 0:
            return {label: 0.0 for label in STATE_LABELS.values()}
        probs = self.state_probs[-1]
        return {STATE_LABELS[i]: float(probs[i]) for i in range(len(probs))}

    def state_series(self) -> pd.Series:
        if not self.fitted:
            return pd.Series(dtype=int)
        return pd.Series(self.states, index=self._obs_index, name="HMM_State")

    def state_prob_df(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()
        df = pd.DataFrame(self.state_probs, index=self._obs_index,
                          columns=[STATE_LABELS[i] for i in range(self.n_components)])
        return df

    def transition_matrix(self) -> np.ndarray:
        if self._use_hmmlearn and self.model is not None:
            return self.model.transmat_
        n = self.n_components
        tm = np.full((n, n), 1.0 / n)
        return tm
