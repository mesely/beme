import copy
import numpy as np
import scipy.sparse
from scipy.special import logit, expit  # noqa: F401 (used by submodules)

from .pricing import get_market_prices
from .bounty  import DEFAULT_BOUNTY_MAP, get_bounty
from .utils   import portfolio_summary


class BemeMarket:
    """
    Bounty-driven Evolutionary Market Ensemble — base class.

    Uses the **Template Method Pattern**: the core trading-day loop is fixed,
    but four lifecycle hooks let subclasses override specific moments without
    touching the orchestration logic.

    Lifecycle hooks (override in subclasses):
        _on_day_start(x_i, y_true)          — before predictions are gathered
        _calculate_pnl(agent, market_price,
                       agent_pred, y_true_c,
                       bounty)              — per-class P&L delta (returns float)
        _on_day_end()                       — after all balances are updated
        _on_rebalance(sector_name,
                      parent, child)        — inside evolution, before model copy

    Parameters
    ----------
    n_funds : int
        Number of hedge fund agents. Default: 50.
    leverage : float
        Global P&L scale. Higher = faster convergence, higher bankruptcy risk.
        Default: 25.0.
    temperature : float
        Consensus softness. Low T = richest agent dominates.
        High T = equal vote. Default: 1.5.
    bounty_map : dict, 'auto', or None
        Per-class reward multipliers.
        'auto'  — computed from y_train: β_c = 1 + log(max_count / count_c)
        None    — default map {1:1.5, 2:2.0, 7:1.5, 8:2.0}
        {}      — disables bounty entirely
    rebalance_period : int
        Trading days between evolutionary rebalancing cycles. Default: 100.
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
        self._bounty_config   = bounty_map  # raw value; resolved in initialize_market

        if bounty_map == 'auto':
            self.bounty_map = {}            # filled in initialize_market
        elif bounty_map is None:
            self.bounty_map = dict(DEFAULT_BOUNTY_MAP)
        else:
            self.bounty_map = bounty_map

    # ──────────────────────────────────────────────────────────────
    # Lifecycle hooks — base implementations (no-ops / defaults)
    # ──────────────────────────────────────────────────────────────

    def _on_day_start(self, x_i, y_true):
        """Called before agent predictions are gathered.

        Override to inject pre-processing, logging, or dynamic parameter
        adjustments at the start of each trading day.
        """

    def _calculate_pnl(self, agent, market_price, agent_pred, y_true_c, bounty):
        """Compute the P&L contribution for one agent on one class.

        Returns
        -------
        float
            delta_B contribution for class c.
        """
        return (
            (y_true_c - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * self.leverage
            * bounty
        )

    def _on_day_end(self):
        """Called after all agent balances have been updated for the day.

        Override to add decay, taxes, logging, or any post-day housekeeping.
        """

    def _on_rebalance(self, sector_name, parent, child):
        """Called inside the evolution cycle, just before the child receives
        the parent's model.

        Parameters
        ----------
        sector_name : str
            'COMMODITY' or 'FINANCE'.
        parent : dict
            Highest-balance agent in the sector (the winner).
        child : dict
            Lowest-balance agent in the sector (to be replaced).
        """

    # ──────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────

    def initialize_market(self, base_model, X_train, y_train):
        """Fit n_funds agents on bootstrap subsets of training data.

        If bounty_map='auto' was set, per-class multipliers are computed
        from y_train label frequencies before agents are spawned.

        Parameters
        ----------
        base_model : sklearn estimator
            Must implement fit() and predict_proba(). Deep-copied n_funds times.
        X_train : array-like or sparse (n_samples, n_features)
        y_train : array-like or sparse (n_samples, n_classes)
        """
        from sklearn.model_selection import train_test_split

        # ── Auto-Bounty ──────────────────────────────────────────
        if self._bounty_config == 'auto':
            y_arr  = y_train.toarray() if scipy.sparse.issparse(y_train) else np.asarray(y_train)
            counts = np.maximum(y_arr.sum(axis=0), 1).astype(float)
            self.bounty_map = {
                c: float(1.0 + np.log(counts.max() / counts[c]))
                for c in range(len(counts))
            }

        # ── Spawn agents ─────────────────────────────────────────
        self.funds = []
        for i in range(self.n_funds):
            X_in, _, y_in, _ = train_test_split(
                X_train, y_train, test_size=0.3, random_state=i)

            # sklearn estimators require dense y
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

            sector         = "COMMODITY" if i < (self.n_funds // 2) else "FINANCE"
            active_classes = [1, 2, 4, 7, 9] if sector == "COMMODITY" else [0, 3, 5, 6, 8]
            self.funds.append({
                'model':          m,
                'balance':        100.0,
                'risk':           np.random.uniform(0.5, 1.0),
                'sector':         sector,
                'active_classes': active_classes,
            })

    def fit(self, X, y, base_model=None):
        """High-level sklearn-compatible training entry point.

        Initialises the market (if not done yet) then runs the full online
        trading loop over (X, y). Returns self for method chaining.

        Parameters
        ----------
        X : array-like or sparse (n_samples, n_features)
        y : array-like or sparse (n_samples, n_classes)
        base_model : sklearn estimator or None
            Required on first call; ignored if market is already initialised.

        Returns
        -------
        self
        """
        if not self.funds:
            if base_model is None:
                raise ValueError(
                    "base_model must be provided when the market has not yet "
                    "been initialised. Call initialize_market() first, or pass "
                    "base_model to fit()."
                )
            self.initialize_market(base_model, X, y)

        n_samples = X.shape[0]  # works for dense and sparse

        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc="BEME training", unit="day")
        except ImportError:
            iterator = range(n_samples)
            print(f"BEME: running {n_samples} trading days "
                  "(install tqdm for a progress bar)...")

        for i in iterator:
            x_i = X[i] if scipy.sparse.issparse(X) else X[i:i+1]
            y_i = y[i]
            if hasattr(y_i, "toarray"):
                y_i = y_i.toarray().flatten()
            self.run_trading_day(x_i, y_i)

        return self

    def run_trading_day(self, x_i, y_true):
        """Process one labelled sample. Updates all agent balances.

        Hook call order:
            1. _on_day_start(x_i, y_true)
            2. [collect predictions, compute market prices]
            3. _calculate_pnl(...)  ← called once per (agent, class) pair
            4. [update balances]
            5. [optional] evolve()  ← if day % rebalance_period == 0
            6. _on_day_end()

        Parameters
        ----------
        x_i    : array-like (1, n_features) or sparse row
        y_true : array-like (n_classes,), binary ground truth

        Returns
        -------
        y_hat : np.ndarray (n_classes,), binary prediction
        """
        # ── Sparse y_true guard ───────────────────────────────────
        if hasattr(y_true, "toarray"):
            y_true = y_true.toarray().flatten()
        y_true = np.asarray(y_true, dtype=float)

        # ── Hook 1 ───────────────────────────────────────────────
        self._on_day_start(x_i, y_true)

        # ── Collect agent predictions ─────────────────────────────
        preds = np.array([
            np.clip(f['model'].predict_proba(x_i)[0], 0.01, 0.99)
            for f in self.funds
        ])

        market_prices = get_market_prices(self.funds, preds, self.T)

        # ── Hook 3: per-class P&L (via _calculate_pnl) ───────────
        for idx, f in enumerate(self.funds):
            profit_day = 0.0
            for c in f['active_classes']:
                bounty = get_bounty(self.bounty_map, c)
                profit_day += self._calculate_pnl(
                    agent=f,
                    market_price=market_prices[c],
                    agent_pred=preds[idx, c],
                    y_true_c=y_true[c],
                    bounty=bounty,
                )
            f['balance'] = max(1.0, f['balance'] + profit_day)

        # ── Evolution trigger ─────────────────────────────────────
        self._day += 1
        if self._day % self.rebalance_period == 0:
            self.evolve()

        # ── Hook 4 ───────────────────────────────────────────────
        self._on_day_end()

        return (market_prices > 0.5).astype(int)

    def evolve(self):
        """Run one evolutionary rebalancing cycle across all sectors.

        For each sector: the weakest agent (lowest balance) is replaced by
        a deep copy of the strongest (highest balance). A 10% balance transfer
        is made from parent to child.

        The _on_rebalance(sector_name, parent, child) hook fires just before
        the model copy, allowing subclasses to intercept or veto the transfer.
        """
        for sector_name in ("COMMODITY", "FINANCE"):
            sector_funds = [f for f in self.funds if f['sector'] == sector_name]
            if len(sector_funds) < 2:
                continue
            sector_funds.sort(key=lambda f: f['balance'], reverse=True)
            parent = sector_funds[0]
            child  = sector_funds[-1]

            # ── Hook ─────────────────────────────────────────────
            self._on_rebalance(sector_name, parent, child)

            child['model']     = copy.deepcopy(parent['model'])
            transfer           = parent['balance'] * 0.1
            parent['balance'] -= transfer
            child['balance']   = max(100.0, transfer)

    def predict(self, X):
        """Batch inference — does not update balances or trigger evolution.

        Each agent's predict_proba is called once on the full X matrix
        (O(n_funds) calls instead of O(n_samples × n_funds)).

        Parameters
        ----------
        X : array-like or sparse (n_samples, n_features)

        Returns
        -------
        np.ndarray (n_samples, n_classes), binary
        """
        n_samples = X.shape[0]

        all_preds = np.stack([
            np.clip(f['model'].predict_proba(X), 0.01, 0.99)
            for f in self.funds
        ])  # (n_funds, n_samples, n_classes)

        results = []
        for i in range(n_samples):
            preds_i       = all_preds[:, i, :]
            market_prices = get_market_prices(self.funds, preds_i, self.T)
            results.append((market_prices > 0.5).astype(int))

        return np.array(results)

    def portfolio_summary(self):
        """Return a DataFrame snapshot of all agents.

        Columns: agent_id, sector, balance, risk, active_classes
        """
        return portfolio_summary(self.funds)
