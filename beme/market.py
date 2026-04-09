import numpy as np
import copy
import scipy.sparse
from scipy.special import logit, expit

from .pricing   import get_market_prices
from .evolution import evolve as _evolve
from .bounty    import DEFAULT_BOUNTY_MAP, get_bounty
from .utils     import portfolio_summary


class BemeMarket:
    """
    Bounty-driven Evolutionary Market Ensemble.

    A multi-label ensemble that wraps sklearn classifiers in a prediction market.
    Each classifier (fund) earns or loses balance based on prediction accuracy.
    Minority classes are assigned higher bounty multipliers to counteract imbalance.
    Low-balance funds are periodically replaced by copies of high-balance funds.

    Parameters
    ----------
    n_funds : int
        Number of hedge fund agents. Default: 50.
    leverage : float
        Scales profit/loss magnitude. Higher = faster convergence, higher bankruptcy risk.
        Default: 25.0.
    temperature : float
        Controls consensus softness. Low T = richest agent dominates.
        High T = all agents contribute equally. Default: 1.5.
    bounty_map : dict, 'auto', or None
        Per-class reward multipliers {class_index: float}.
        'auto' computes multipliers from class frequencies in y_train:
            beta_c = 1.0 + log(max_count / count_c)
        Pass {} to disable bounty entirely (all classes treated equally).
        Pass None to use the default map {1: 1.5, 2: 2.0, 7: 1.5, 8: 2.0}.
        Default: None.
    rebalance_period : int
        How many trading days between evolutionary rebalancing cycles. Default: 100.
    sector_map : dict or None
        Reserved for future custom sector definitions. Currently unused.
    """

    def __init__(
        self,
        n_funds=50,
        leverage=25.0,
        temperature=1.5,
        bounty_map=None,
        rebalance_period=100,
        sector_map=None,
    ):
        self.n_funds          = n_funds
        self.leverage         = leverage
        self.T                = temperature
        self.rebalance_period = rebalance_period
        self.sector_map       = sector_map
        self.funds            = []
        self._day             = 0
        self._bounty_config   = bounty_map  # store raw value; resolved in initialize_market

        if bounty_map == 'auto':
            self.bounty_map = {}  # filled later in initialize_market
        elif bounty_map is None:
            self.bounty_map = dict(DEFAULT_BOUNTY_MAP)
        else:
            self.bounty_map = bounty_map

    def initialize_market(self, base_model, X_train, y_train):
        """
        Fit n_funds agents on bootstrap subsets of training data.

        If bounty_map='auto' was passed at construction, this method computes
        per-class bounty multipliers from y_train before spawning agents.

        Parameters
        ----------
        base_model : sklearn estimator
            Must implement fit() and predict_proba(). Will be deep-copied n_funds times.
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
        y_train : array-like or sparse matrix of shape (n_samples, n_classes)
        """
        from sklearn.model_selection import train_test_split

        # ── Auto-Bounty ──────────────────────────────────────────────
        if self._bounty_config == 'auto':
            if scipy.sparse.issparse(y_train):
                y_arr = np.asarray(y_train.todense())
            else:
                y_arr = np.asarray(y_train)
            counts     = np.maximum(y_arr.sum(axis=0), 1).astype(float)
            max_count  = counts.max()
            self.bounty_map = {
                c: float(1.0 + np.log(max_count / counts[c]))
                for c in range(len(counts))
            }

        # ── Spawn agents ─────────────────────────────────────────────
        self.funds = []
        for i in range(self.n_funds):
            X_in, _, y_in, _ = train_test_split(
                X_train, y_train, test_size=0.3, random_state=i)
            # sklearn estimators generally require dense y
            if scipy.sparse.issparse(y_in):
                y_in = y_in.toarray()
            m = copy.deepcopy(base_model)
            try:
                m.fit(X_in, y_in)
            except ValueError:
                # Bootstrap subset may lack both label values for some class;
                # fall back to the full training set for this agent.
                y_full = y_train.toarray() if scipy.sparse.issparse(y_train) else y_train
                m.fit(X_train, y_full)
            sector = "COMMODITY" if i < (self.n_funds // 2) else "FINANCE"
            active_classes = (
                [1, 2, 4, 7, 9] if sector == "COMMODITY" else [0, 3, 5, 6, 8])
            self.funds.append({
                'model':          m,
                'balance':        100.0,
                'risk':           np.random.uniform(0.5, 1.0),
                'sector':         sector,
                'active_classes': active_classes,
            })

    def fit(self, X, y, base_model=None):
        """
        High-level scikit-learn compatible training entry point.

        Initialises the market (if not already done) and runs the full
        online training loop over (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
        y : array-like or sparse matrix of shape (n_samples, n_classes)
        base_model : sklearn estimator or None
            Required if the market has not been initialised yet.
            Ignored if initialize_market() was already called.

        Returns
        -------
        self
            Allows method chaining: model.fit(X, y, base_model=m).predict(X_test)
        """
        if not self.funds:
            if base_model is None:
                raise ValueError(
                    "base_model must be provided when the market has not yet "
                    "been initialised. Call initialize_market() first, or pass "
                    "base_model to fit()."
                )
            self.initialize_market(base_model, X, y)

        n_samples = X.shape[0]  # works for both dense and sparse

        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc="BEME training", unit="day")
        except ImportError:
            iterator = range(n_samples)
            print(f"BEME: running {n_samples} trading days "
                  f"(install tqdm for a progress bar)...")

        for i in iterator:
            x_i = X[i] if scipy.sparse.issparse(X) else X[i:i+1]
            y_i = y[i]
            if hasattr(y_i, "toarray"):
                y_i = y_i.toarray().flatten()
            self.run_trading_day(x_i, y_i)

        return self

    def run_trading_day(self, x_i, y_true):
        """
        Process one labelled sample. Updates all agent balances.
        Triggers evolution automatically every rebalance_period calls.

        Parameters
        ----------
        x_i    : array-like of shape (1, n_features), or sparse row
        y_true : array-like of shape (n_classes,), binary ground truth

        Returns
        -------
        y_hat : np.ndarray of shape (n_classes,), binary prediction
        """
        # ── Sparse y_true guard ───────────────────────────────────────
        if hasattr(y_true, "toarray"):
            y_true = y_true.toarray().flatten()
        y_true = np.asarray(y_true, dtype=float)

        preds = np.array([
            np.clip(f['model'].predict_proba(x_i)[0], 0.01, 0.99)
            for f in self.funds
        ])

        market_prices = get_market_prices(self.funds, preds, self.T)

        for idx, f in enumerate(self.funds):
            profit_day = 0.0
            for c in f['active_classes']:
                beta = get_bounty(self.bounty_map, c)
                profit_day += (
                    (y_true[c] - market_prices[c])
                  * (preds[idx, c] - market_prices[c])
                  * f['risk']
                  * self.leverage
                  * beta
                )
            f['balance'] = max(1.0, f['balance'] + profit_day)

        self._day += 1
        if self._day % self.rebalance_period == 0:
            self.evolve()

        return (market_prices > 0.5).astype(int)

    def evolve(self):
        """Manually trigger one evolutionary rebalancing cycle."""
        _evolve(self.funds)

    def predict(self, X):
        """
        Batch inference without updating balances or triggering evolution.
        Use this on validation and test sets.

        Each agent's predict_proba is called once on the full X matrix
        (rather than per-sample), then market prices are computed per row.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        y_preds : np.ndarray of shape (n_samples, n_classes), binary
        """
        n_samples = X.shape[0]

        # One predict_proba call per agent over all samples — O(n_funds) calls
        # instead of O(n_samples * n_funds)
        all_preds = np.stack([
            np.clip(f['model'].predict_proba(X), 0.01, 0.99)
            for f in self.funds
        ])  # (n_funds, n_samples, n_classes)

        results = []
        for i in range(n_samples):
            preds_i       = all_preds[:, i, :]   # (n_funds, n_classes)
            market_prices = get_market_prices(self.funds, preds_i, self.T)
            results.append((market_prices > 0.5).astype(int))

        return np.array(results)

    def portfolio_summary(self):
        """
        Return a DataFrame with the current state of every agent.

        Columns: agent_id, sector, balance, risk, active_classes
        """
        return portfolio_summary(self.funds)
