import numpy as np
import copy
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
    bounty_map : dict or None
        Per-class reward multipliers {class_index: float}.
        Pass {} or None to disable bounty (all classes treated equally).
        Default: {1: 1.5, 2: 2.0, 7: 1.5, 8: 2.0}.
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
        self.bounty_map       = bounty_map if bounty_map is not None else dict(DEFAULT_BOUNTY_MAP)
        self.rebalance_period = rebalance_period
        self.sector_map       = sector_map
        self.funds            = []
        self._day             = 0

    def initialize_market(self, base_model, X_train, y_train):
        """
        Fit n_funds agents on bootstrap subsets of training data.

        Parameters
        ----------
        base_model : sklearn estimator
            Must implement fit() and predict_proba(). Will be deep-copied n_funds times.
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples, n_classes)
        """
        from sklearn.model_selection import train_test_split
        self.funds = []
        for i in range(self.n_funds):
            X_in, _, y_in, _ = train_test_split(
                X_train, y_train, test_size=0.3, random_state=i)
            m = copy.deepcopy(base_model)
            m.fit(X_in, y_in)
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

    def run_trading_day(self, x_i, y_true):
        """
        Process one labelled sample. Updates all agent balances.
        Triggers evolution automatically every rebalance_period calls.

        Parameters
        ----------
        x_i    : array-like of shape (1, n_features)
        y_true : array-like of shape (n_classes,), binary ground truth

        Returns
        -------
        y_hat : np.ndarray of shape (n_classes,), binary prediction
        """
        preds = np.array([
            np.clip(f['model'].predict_proba(x_i)[0], 0.01, 0.99)
            for f in self.funds
        ])

        market_prices = get_market_prices(self.funds, preds, self.T)
        y_true = np.asarray(y_true, dtype=float)

        for idx, f in enumerate(self.funds):
            profit_day = 0
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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_preds : np.ndarray of shape (n_samples, n_classes), binary
        """
        X = np.asarray(X)
        results = []
        for i in range(len(X)):
            x_i = X[i:i+1]
            preds = np.array([
                np.clip(f['model'].predict_proba(x_i)[0], 0.01, 0.99)
                for f in self.funds
            ])
            market_prices = get_market_prices(self.funds, preds, self.T)
            results.append((market_prices > 0.5).astype(int))
        return np.array(results)

    def portfolio_summary(self):
        """
        Return a DataFrame with the current state of every agent.

        Columns: agent_id, sector, balance, risk, active_classes
        """
        return portfolio_summary(self.funds)
