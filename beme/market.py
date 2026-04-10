import copy
import numpy as np
import scipy.sparse

from .pricing import get_market_prices
from .bounty  import DEFAULT_BOUNTY_MAP, get_bounty
from .utils   import portfolio_summary, MultiLabelWrapper

# Adaptive rebalancing thresholds (baked into the trading loop)
_WARMUP_THRESHOLD  = 250   # days before the stable phase
_WARMUP_REBALANCE  = 20    # rebalance period during warm-up


class BemeMarket:
    """
    Bounty-driven Evolutionary Market Ensemble — v1.0 Sovereign Market.

    Core v1.0 mechanisms (always active in the base class):
        • Adaptive Rebalancing — warm-up (day < 250 → every 20 days) then
          stable (day ≥ 250 → every `rebalance_period` days).
        • Confidence-weighted P&L — agents earn more when they are certain.
        • FutureValue bonus — poor agents get amplified rewards on correct calls.

    Template Method hooks (override in subclasses):
        _on_day_start(day_index, x_i)
        _calculate_pnl(agent, market_price, y_true_c, agent_pred, bounty)
        _on_day_end(day_index, x_i, y_true, y_pred)
        _on_rebalance(sector_name, parent, child)

    Parameters
    ----------
    n_funds : int
        Number of hedge fund agents. Default: 50.
    leverage : float
        Global P&L scale. Default: 25.0.
    temperature : float
        Consensus softness. Low T = richest dominates; High T = equal vote.
        Default: 1.5.
    bounty_map : dict, 'auto', or None
        Per-class reward multipliers.
        'auto' — β_c = 1 + log(max_count / count_c) from y_train
        None   — default map {1:1.5, 2:2.0, 7:1.5, 8:2.0}
        {}     — disables bounty entirely
    rebalance_period : int
        Stable-phase rebalance interval (used after day 250). Default: 100.
    engine_type : str or None
        'logistic' — all agents use LogisticRegression (default when None)
        'svm'      — all agents use CalibratedClassifierCV(LinearSVC)
        'hybrid'   — COMMODITY sector: LR, FINANCE sector: SVM
        Ignored when base_model is passed to initialize_market / fit.
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
        engine_type=None,
        sector_map=None,
    ):
        self.n_funds          = n_funds
        self.leverage         = leverage
        self.T                = temperature
        self.rebalance_period = rebalance_period
        self.engine_type      = engine_type
        self.sector_map       = sector_map
        self.funds            = []
        self._day             = 0
        self._bounty_config   = bounty_map

        if bounty_map == 'auto':
            self.bounty_map = {}
        elif bounty_map is None:
            self.bounty_map = dict(DEFAULT_BOUNTY_MAP)
        else:
            self.bounty_map = bounty_map

    # ──────────────────────────────────────────────────────────────
    # Lifecycle hooks — base implementations
    # ──────────────────────────────────────────────────────────────

    def _on_day_start(self, day_index: int, x_i):
        """Called before agent predictions are gathered.

        Parameters
        ----------
        day_index : int   Current day counter (before increment).
        x_i               Feature vector for this sample.
        """

    def _calculate_pnl(
        self,
        agent: dict,
        market_price: float,
        y_true_c: float,
        agent_pred: float,
        bounty: float,
    ) -> float:
        """v1.0 unified P&L formula: Base × (1 + Confidence) × FutureValue.

        Base        = (y_true_c - P_c) × (p_i,c - P_c) × risk_i × L × β_c
        Confidence  = |p_i,c − 0.5| × 2   ∈ [0, 1]
        FutureValue = 1 + 100/(B_i + 1)  if B_i < 100 else 1
        Final       = Base × (1 + Confidence) × FutureValue
        """
        base = (
            (y_true_c    - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * self.leverage
            * bounty
        )
        confidence   = abs(agent_pred - 0.5) * 2.0
        future_value = (
            1.0 + 100.0 / (agent['balance'] + 1.0)
            if agent['balance'] < 100.0 else 1.0
        )
        return base * (1.0 + confidence) * future_value

    def _on_day_end(self, day_index: int, x_i, y_true, y_pred):
        """Called after all balances are updated and (optionally) evolution runs.

        Parameters
        ----------
        day_index : int   Day counter at the time of this call.
        x_i               Feature vector used this day.
        y_true            Ground-truth label vector.
        y_pred            Binary prediction made by the market this day.
        """

    def _on_rebalance(self, sector_name: str, parent: dict, child: dict):
        """Called inside evolve(), just before the child receives the parent's model."""

    # ──────────────────────────────────────────────────────────────
    # Model factory helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_lr(**kwargs):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput  import MultiOutputClassifier
        defaults = dict(solver='liblinear', max_iter=1000, class_weight='balanced')
        defaults.update(kwargs)
        return MultiLabelWrapper(MultiOutputClassifier(LogisticRegression(**defaults)))

    @staticmethod
    def _make_svm(**kwargs):
        from sklearn.svm          import LinearSVC
        from sklearn.calibration  import CalibratedClassifierCV
        from sklearn.multioutput  import MultiOutputClassifier
        defaults = dict(max_iter=2000, class_weight='balanced')
        defaults.update(kwargs)
        return MultiLabelWrapper(
            MultiOutputClassifier(CalibratedClassifierCV(LinearSVC(**defaults)))
        )

    # ──────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────

    def initialize_market(
        self,
        base_model,
        X_train,
        y_train,
        engine_type: str = None,
    ):
        """Fit n_funds agents on bootstrap subsets of training data.

        Parameters
        ----------
        base_model : sklearn estimator or None
            Ignored when engine_type is provided (or set in constructor).
        X_train : array-like or sparse (n_samples, n_features)
        y_train : array-like or sparse (n_samples, n_classes)
        engine_type : str or None
            Overrides self.engine_type for this call.
            'logistic' | 'svm' | 'hybrid'
        """
        from sklearn.model_selection import train_test_split

        effective_engine = engine_type or self.engine_type

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

            if scipy.sparse.issparse(y_in):
                y_in = y_in.toarray()

            # Build model: engine_type overrides base_model
            sector = "COMMODITY" if i < (self.n_funds // 2) else "FINANCE"
            if effective_engine == 'svm':
                m = self._make_svm()
            elif effective_engine == 'hybrid':
                m = self._make_lr() if sector == "COMMODITY" else self._make_svm()
            elif effective_engine == 'logistic':
                m = self._make_lr()
            else:
                m = copy.deepcopy(base_model)

            try:
                m.fit(X_in, y_in)
            except ValueError:
                try:
                    y_full = y_train.toarray() if scipy.sparse.issparse(y_train) else y_train
                    m.fit(X_train, y_full)
                except ValueError:
                    from sklearn.dummy       import DummyClassifier
                    from sklearn.multioutput import MultiOutputClassifier
                    y_full = y_train.toarray() if scipy.sparse.issparse(y_train) else np.asarray(y_train)
                    m = MultiLabelWrapper(
                        MultiOutputClassifier(DummyClassifier(strategy='most_frequent'))
                    )
                    m.fit(X_train, y_full)

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

        Returns self for method chaining.

        Parameters
        ----------
        X : array-like or sparse (n_samples, n_features)
        y : array-like or sparse (n_samples, n_classes)
        base_model : sklearn estimator or None
            Required unless engine_type was set in the constructor.
        """
        if not self.funds:
            if base_model is None and self.engine_type is None:
                raise ValueError(
                    "Provide base_model or set engine_type in the constructor."
                )
            self.initialize_market(base_model, X, y)

        n_samples = X.shape[0]
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

        v1.0 loop order:
            1. _on_day_start(day_index, x_i)
            2. collect predictions → market prices → y_pred
            3. _calculate_pnl(...)  for each (agent, class)
            4. update balances
            5. adaptive evolution trigger (warm-up: every 20d, stable: every rebalance_period)
            6. _on_day_end(day_index, x_i, y_true, y_pred)

        Returns
        -------
        y_pred : np.ndarray (n_classes,), binary market consensus
        """
        if hasattr(y_true, "toarray"):
            y_true = y_true.toarray().flatten()
        y_true = np.asarray(y_true, dtype=float)

        current_day = self._day

        # Hook 1
        self._on_day_start(current_day, x_i)

        # Predictions & market prices
        preds = np.array([
            np.clip(f['model'].predict_proba(x_i)[0], 0.01, 0.99)
            for f in self.funds
        ])
        market_prices = get_market_prices(self.funds, preds, self.T)
        y_pred        = (market_prices > 0.5).astype(int)

        # Hook 2: _calculate_pnl per (agent, class)
        for idx, f in enumerate(self.funds):
            profit_day = 0.0
            for c in f['active_classes']:
                bounty = get_bounty(self.bounty_map, c)
                profit_day += self._calculate_pnl(
                    agent=f,
                    market_price=market_prices[c],
                    y_true_c=y_true[c],
                    agent_pred=preds[idx, c],
                    bounty=bounty,
                )
            f['balance'] = max(1.0, f['balance'] + profit_day)

        # Adaptive evolution trigger
        warmup_active   = current_day < _WARMUP_THRESHOLD
        rebalance_now   = _WARMUP_REBALANCE if warmup_active else self.rebalance_period
        self._day      += 1
        if self._day % rebalance_now == 0:
            self.evolve()

        # Hook 3
        self._on_day_end(current_day, x_i, y_true, y_pred)

        return y_pred

    def evolve(self):
        """Run one evolutionary rebalancing cycle across all sectors."""
        for sector_name in ("COMMODITY", "FINANCE"):
            sector_funds = [f for f in self.funds if f['sector'] == sector_name]
            if len(sector_funds) < 2:
                continue
            sector_funds.sort(key=lambda f: f['balance'], reverse=True)
            parent = sector_funds[0]
            child  = sector_funds[-1]

            self._on_rebalance(sector_name, parent, child)

            child['model']     = copy.deepcopy(parent['model'])
            transfer           = parent['balance'] * 0.1
            parent['balance'] -= transfer
            child['balance']   = max(100.0, transfer)

    def predict(self, X):
        """Batch inference — does not update balances or trigger evolution.

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
