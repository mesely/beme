"""
beme/templates.py
=================
BemeMarket subclasses built on the Template Method Pattern.

v1.0 note
---------
ConfidenceMarket, FutureValueMarket, and AdaptiveEvolutionMarket are now
**core behaviours** of BemeMarket v1.0. Their template classes are kept
for backwards API compatibility but no longer override any hook — they
are functionally identical to the base class.

Active templates (override hooks):
    DecayMarket            — _on_day_end:      daily wealth decay
    AlphaMarket            — _calculate_pnl:   sector-specific leverage
    PruningMarket          — evolve():          permanent agent culling
    HybridEngineMarket     — initialize_market: mixed LR + SVM agents

Legacy templates (no-op wrappers, kept for compatibility):
    ConfidenceMarket       — now in BemeMarket base
    FutureValueMarket      — now in BemeMarket base
    AdaptiveEvolutionMarket — now in BemeMarket base

──────────────────────────────────────────────────────────────────────
Template Builder
──────────────────────────────────────────────────────────────────────
Create your own template anywhere:

    from beme import BemeMarket

    class Template4_TaxMarket(BemeMarket):
        \"\"\"Tax the richest agent every day.\"\"\"
        def _on_day_end(self, day_index, x_i, y_true, y_pred):
            richest = max(self.funds, key=lambda f: f['balance'])
            if richest['balance'] > 500:
                richest['balance'] *= 0.8

    model = Template4_TaxMarket().fit(X_train, y_train, base_model=base)
──────────────────────────────────────────────────────────────────────
"""

import copy
import numpy as np
import scipy.sparse

from .market import BemeMarket
from .utils  import MultiLabelWrapper


# ══════════════════════════════════════════════════════════════════════
# Active Template 1 — DecayMarket  (Eskime Payı)
# ══════════════════════════════════════════════════════════════════════

class DecayMarket(BemeMarket):
    """Daily wealth decay — agents that stop earning slowly rust away.

    After each trading day (_on_day_end hook):

        B_{t+1} = max(1.0,  B_t × (1 − λ))

    Parameters
    ----------
    decay_rate : float
        Daily wealth decay fraction λ. Default: 0.001.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = DecayMarket(decay_rate=0.001).fit(X_train, y_train, base_model=base)
    """

    def __init__(self, decay_rate: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate

    def _on_day_end(self, day_index: int, x_i, y_true, y_pred):
        multiplier = 1.0 - self.decay_rate
        for f in self.funds:
            f['balance'] = max(1.0, f['balance'] * multiplier)


# ══════════════════════════════════════════════════════════════════════
# Active Template 2 — AlphaMarket  (Sektörel Volatilite / The Champion)
# ══════════════════════════════════════════════════════════════════════

class AlphaMarket(BemeMarket):
    """Sector-specific leverage and per-trade loss caps.

    Replaces the base Confidence+FutureValue formula with a sector-tuned
    leverage while keeping the same directional P&L signal.

    Default sector configs:
        COMMODITY — leverage: 35.0,  max_loss: -10.0
        FINANCE   — leverage: 15.0,  max_loss: -5.0

    Parameters
    ----------
    sector_configs : dict or None
        Override per-sector params::

            {'COMMODITY': {'leverage': float, 'max_loss': float},
             'FINANCE':   {'leverage': float, 'max_loss': float}}

    **kwargs
        All BemeMarket constructor parameters.
    """

    DEFAULT_SECTOR_CONFIGS = {
        'COMMODITY': {'leverage': 35.0, 'max_loss': -10.0},
        'FINANCE':   {'leverage': 15.0, 'max_loss': -5.0},
    }

    def __init__(self, sector_configs: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.sector_configs = sector_configs or dict(self.DEFAULT_SECTOR_CONFIGS)

    def _calculate_pnl(
        self, agent, market_price, y_true_c, agent_pred, bounty
    ) -> float:
        cfg      = self.sector_configs.get(agent['sector'], {})
        leverage = cfg.get('leverage', self.leverage)
        max_loss = cfg.get('max_loss', -float('inf'))

        base = (
            (y_true_c    - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * leverage
            * bounty
        )
        confidence   = abs(agent_pred - 0.5) * 2.0
        future_value = (
            1.0 + 100.0 / (agent['balance'] + 1.0)
            if agent['balance'] < 100.0 else 1.0
        )
        return max(base * (1.0 + confidence) * future_value, max_loss)


# ══════════════════════════════════════════════════════════════════════
# Active Template 3 — PruningMarket  (Dinamik İnfaz)
# ══════════════════════════════════════════════════════════════════════

class PruningMarket(BemeMarket):
    """Permanently remove agents whose balance falls below a sector threshold.

    After each standard evolution cycle, any agent whose balance is below
    `prune_threshold × mean(sector_balances)` is removed from self.funds.
    At least `min_per_sector` agents are always kept per sector.

    Parameters
    ----------
    prune_threshold : float
        Fraction of sector mean balance below which an agent is culled.
        Default: 0.6.
    min_per_sector : int
        Minimum agents to keep alive per sector. Default: 2.
    **kwargs
        All BemeMarket constructor parameters.
    """

    def __init__(
        self,
        prune_threshold: float = 0.6,
        min_per_sector: int    = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prune_threshold = prune_threshold
        self.min_per_sector  = min_per_sector

    def evolve(self):
        """Standard evolution + permanent pruning of underperformers."""
        super().evolve()

        survivors = []
        for sector in ("COMMODITY", "FINANCE"):
            sector_funds = [f for f in self.funds if f['sector'] == sector]
            if not sector_funds:
                continue
            threshold    = (
                np.mean([f['balance'] for f in sector_funds])
                * self.prune_threshold
            )
            sector_funds.sort(key=lambda f: f['balance'], reverse=True)
            alive = [f for f in sector_funds if f['balance'] >= threshold]
            if len(alive) < self.min_per_sector:
                alive = sector_funds[: self.min_per_sector]
            survivors.extend(alive)

        self.funds = survivors


# ══════════════════════════════════════════════════════════════════════
# Active Template 4 — HybridEngineMarket  (Çift Motorlu Piyasa)
# ══════════════════════════════════════════════════════════════════════

class HybridEngineMarket(BemeMarket):
    """Half the agents use LogisticRegression; half use calibrated LinearSVC.

    COMMODITY sector → LogisticRegression
    FINANCE   sector → CalibratedClassifierCV(LinearSVC)

    This is now equivalent to passing engine_type='hybrid' to BemeMarket.
    The subclass is kept for explicit intent and custom lr/svc kwargs.

    The base_model argument to fit() / initialize_market() is ignored.

    Parameters
    ----------
    lr_kwargs  : dict or None   Extra kwargs for LogisticRegression.
    svc_kwargs : dict or None   Extra kwargs for LinearSVC.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    # Equivalent forms:
    model = HybridEngineMarket(n_funds=50).fit(X_train, y_train)
    model = BemeMarket(n_funds=50, engine_type='hybrid').fit(X_train, y_train)
    """

    def __init__(
        self,
        lr_kwargs:  dict = None,
        svc_kwargs: dict = None,
        **kwargs,
    ):
        kwargs['engine_type'] = 'hybrid'
        super().__init__(**kwargs)
        self.lr_kwargs  = lr_kwargs  or {}
        self.svc_kwargs = svc_kwargs or {}

    def initialize_market(self, base_model, X_train, y_train, engine_type=None):
        """Ignore base_model; use engine_type='hybrid' with custom kwargs."""
        from sklearn.model_selection import train_test_split

        if self._bounty_config == 'auto':
            y_arr  = y_train.toarray() if scipy.sparse.issparse(y_train) else np.asarray(y_train)
            counts = np.maximum(y_arr.sum(axis=0), 1).astype(float)
            self.bounty_map = {
                c: float(1.0 + np.log(counts.max() / counts[c]))
                for c in range(len(counts))
            }

        make_lr  = lambda: self._make_lr(**self.lr_kwargs)
        make_svm = lambda: self._make_svm(**self.svc_kwargs)

        self.funds = []
        for i in range(self.n_funds):
            X_in, _, y_in, _ = train_test_split(
                X_train, y_train, test_size=0.3, random_state=i)
            if scipy.sparse.issparse(y_in):
                y_in = y_in.toarray()

            sector = "COMMODITY" if i < (self.n_funds // 2) else "FINANCE"
            m      = make_lr() if sector == "COMMODITY" else make_svm()

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
        """base_model is optional — HybridEngineMarket provides its own models."""
        if not self.funds:
            self.initialize_market(base_model, X, y)

        n_samples = X.shape[0]
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc="BEME Hybrid", unit="day")
        except ImportError:
            iterator = range(n_samples)
            print(f"BEME: running {n_samples} trading days...")

        for i in iterator:
            x_i = X[i] if scipy.sparse.issparse(X) else X[i:i+1]
            y_i = y[i]
            if hasattr(y_i, "toarray"):
                y_i = y_i.toarray().flatten()
            self.run_trading_day(x_i, y_i)

        return self


# ══════════════════════════════════════════════════════════════════════
# Legacy Templates (now identical to BemeMarket base — kept for compat)
# ══════════════════════════════════════════════════════════════════════

class ConfidenceMarket(BemeMarket):
    """[Legacy] Confidence-weighted P&L.

    This mechanism is now integrated into BemeMarket v1.0 base.
    Kept for API backwards compatibility — behaviour is identical to BemeMarket.
    """


class FutureValueMarket(BemeMarket):
    """[Legacy] Early-stage compound-interest growth booster.

    This mechanism is now integrated into BemeMarket v1.0 base.
    Kept for API backwards compatibility — behaviour is identical to BemeMarket.

    Parameters
    ----------
    discount_factor : float   Accepted but not used (core formula is fixed).
    """
    def __init__(self, discount_factor: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.discount_factor = discount_factor


class AdaptiveEvolutionMarket(BemeMarket):
    """[Legacy] Dynamic rebalancing: fast warm-up then stable phase.

    This mechanism is now integrated into BemeMarket v1.0 base
    (warm-up day < 250 → rebalance every 20d; stable → every rebalance_period).
    Kept for API backwards compatibility — behaviour is identical to BemeMarket.

    Parameters
    ----------
    warmup_days   : int   Accepted but not used (core threshold is 250).
    fast_period   : int   Accepted but not used.
    stable_period : int   Accepted but not used.
    """
    def __init__(
        self,
        warmup_days: int   = 200,
        fast_period: int   = 15,
        stable_period: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.warmup_days   = warmup_days
        self.fast_period   = fast_period
        self.stable_period = stable_period
