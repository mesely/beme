"""
beme/templates.py
=================
Production-ready BemeMarket subclasses built on the Template Method Pattern.

Each template overrides one or more lifecycle hooks from BemeMarket:

    _on_day_start(day_index, x_i)           — before predictions
    _calculate_pnl(agent, market_price,
                   y_true_c, agent_pred,
                   bounty)                  — per-class P&L delta (float)
    _on_day_end()                           — after all balances updated
    _on_rebalance(sector_name,
                  parent, child)            — inside evolution, before clone

All templates inherit .fit(), .predict(), .evolve(), .portfolio_summary()
from BemeMarket — no boilerplate needed.

──────────────────────────────────────────────────────────────────────
Template Builder
──────────────────────────────────────────────────────────────────────
Create your own template on the fly in any notebook:

    from beme import BemeMarket

    # Example: Template 4 — TaxMarket
    class Template4_TaxMarket(BemeMarket):
        \"\"\"A market that taxes the richest agent every day.\"\"\"
        def _on_day_end(self):
            richest = max(self.funds, key=lambda f: f['balance'])
            if richest['balance'] > 500:
                richest['balance'] *= 0.8  # 20% wealth tax

    model = Template4_TaxMarket().fit(X_train, y_train, base_model=base)
──────────────────────────────────────────────────────────────────────
"""

import copy
import numpy as np
import scipy.sparse

from .market import BemeMarket
from .utils  import MultiLabelWrapper


# ══════════════════════════════════════════════════════════════════════
# Template 1 — DecayMarket  (Eskime Payı)
# ══════════════════════════════════════════════════════════════════════

class DecayMarket(BemeMarket):
    """Daily wealth decay — agents that stop earning slowly rust away.

    After each trading day (hook: _on_day_end):

        B_{t+1} = max(1.0,  B_t × (1 − λ))

    A low λ=0.001 ensures poor agents survive long enough to be coached
    by evolution rather than collapsing before the next rebalance cycle.

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

    def _on_day_end(self):
        multiplier = 1.0 - self.decay_rate
        for f in self.funds:
            f['balance'] = max(1.0, f['balance'] * multiplier)


# ══════════════════════════════════════════════════════════════════════
# Template 2 — AlphaMarket  (Sektörel Volatilite / The Champion)
# ══════════════════════════════════════════════════════════════════════

class AlphaMarket(BemeMarket):
    """Sector-specific leverage and per-trade loss caps.

    COMMODITY agents trade with lower leverage (stable, long-term signals).
    FINANCE agents trade with higher leverage (volatile, short-term signals).
    Each sector also has an independent maximum single-trade loss floor.

    Default sector configs:
        COMMODITY — leverage: 35.0,  max_loss: -10.0
        FINANCE   — leverage: 15.0,  max_loss: -5.0

    Parameters
    ----------
    sector_configs : dict or None
        Override per-sector parameters::

            {
                'COMMODITY': {'leverage': float, 'max_loss': float},
                'FINANCE':   {'leverage': float, 'max_loss': float},
            }

        Missing keys fall back to global self.leverage / no cap.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = AlphaMarket().fit(X_train, y_train, base_model=base)
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

        delta = (
            (y_true_c    - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * leverage
            * bounty
        )
        return max(delta, max_loss)


# ══════════════════════════════════════════════════════════════════════
# Template 3 — FutureValueMarket  (Zamanın Değeri / İskonto)
# ══════════════════════════════════════════════════════════════════════

class FutureValueMarket(BemeMarket):
    """Early-stage compound-interest growth booster for poor agents.

    When an agent makes a correct call (delta > 0), a balance-inverse
    multiplier amplifies the reward:

        multiplier = 1 + (100 / (B_i + 1)) × γ

    A fresh agent at B=100 receives multiplier ≈ 1+γ.
    A wealthy agent at B=1000 receives multiplier ≈ 1.09 (negligible).

    Parameters
    ----------
    discount_factor : float
        γ ∈ [0, 1]. Strength of the early-stage boost. Default: 0.9.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = FutureValueMarket(discount_factor=0.95).fit(
        X_train, y_train, base_model=base)
    """

    def __init__(self, discount_factor: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.discount_factor = discount_factor

    def _calculate_pnl(
        self, agent, market_price, y_true_c, agent_pred, bounty
    ) -> float:
        delta = (
            (y_true_c    - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * self.leverage
            * bounty
        )
        if delta > 0:
            multiplier = 1.0 + (100.0 / (agent['balance'] + 1.0)) * self.discount_factor
            delta *= multiplier
        return delta


# ══════════════════════════════════════════════════════════════════════
# Template 4 — ConfidenceMarket  (Hinge-Bounty)
# ══════════════════════════════════════════════════════════════════════

class ConfidenceMarket(BemeMarket):
    """Profit scaled by how confident the agent was.

    An agent that predicted 0.9 (far from the 0.5 decision boundary)
    earns more than one that hedged at 0.6 — reward certainty.

    Formula:
        Confidence_i  = |p_{i,c} − 0.5| × 2        ∈ [0, 1]
        delta_B       = base_pnl × (1 + Confidence_i)

    Parameters
    ----------
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = ConfidenceMarket(leverage=20.0).fit(
        X_train, y_train, base_model=base)
    """

    def _calculate_pnl(
        self, agent, market_price, y_true_c, agent_pred, bounty
    ) -> float:
        base_pnl = (
            (y_true_c    - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * self.leverage
            * bounty
        )
        confidence = abs(agent_pred - 0.5) * 2.0   # ∈ [0, 1]
        return base_pnl * (1.0 + confidence)


# ══════════════════════════════════════════════════════════════════════
# Template 5 — AdaptiveEvolutionMarket  (Isınma Odaklı)
# ══════════════════════════════════════════════════════════════════════

class AdaptiveEvolutionMarket(BemeMarket):
    """Dynamic rebalancing: aggressive warm-up phase, then stabilisation.

    During the first `warmup_days` trading days the rebalance period is set
    to `fast_period` (frequent evolution, rapid selection pressure).
    Afterwards it switches to `stable_period` (slower, steady-state).

    Hook used: _on_day_start — fires every day, cheaply adjusts
    self.rebalance_period before any computation happens.

    Parameters
    ----------
    warmup_days   : int   Threshold day index for phase switch. Default: 200.
    fast_period   : int   Rebalance interval during warm-up. Default: 15.
    stable_period : int   Rebalance interval after warm-up.  Default: 100.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = AdaptiveEvolutionMarket(warmup_days=300).fit(
        X_train, y_train, base_model=base)
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

    def _on_day_start(self, day_index: int, x_i):
        self.rebalance_period = (
            self.fast_period if day_index < self.warmup_days else self.stable_period
        )


# ══════════════════════════════════════════════════════════════════════
# Template 6 — PruningMarket  (Dinamik İnfaz)
# ══════════════════════════════════════════════════════════════════════

class PruningMarket(BemeMarket):
    """Permanently remove agents whose balance falls below a sector threshold.

    After each standard evolution cycle, any agent whose balance is below
    `prune_threshold × mean(sector_balances)` is removed from self.funds.
    A minimum of `min_per_sector` agents is always kept per sector to
    ensure the market can still function.

    Parameters
    ----------
    prune_threshold : float
        Fraction of sector mean balance below which an agent is culled.
        Default: 0.6.
    min_per_sector : int
        Minimum agents to keep alive per sector. Default: 2.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = PruningMarket(prune_threshold=0.5).fit(
        X_train, y_train, base_model=base)
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
        # Step 1: standard clone/replace (includes _on_rebalance hook)
        super().evolve()

        # Step 2: prune per sector
        survivors = []
        for sector in ("COMMODITY", "FINANCE"):
            sector_funds = [f for f in self.funds if f['sector'] == sector]
            if not sector_funds:
                continue

            threshold = (
                np.mean([f['balance'] for f in sector_funds])
                * self.prune_threshold
            )

            # Sort descending so we always keep the best ones
            sector_funds.sort(key=lambda f: f['balance'], reverse=True)
            alive = [f for f in sector_funds if f['balance'] >= threshold]

            # Enforce minimum
            if len(alive) < self.min_per_sector:
                alive = sector_funds[: self.min_per_sector]

            survivors.extend(alive)

        self.funds = survivors


# ══════════════════════════════════════════════════════════════════════
# Template 7 — HybridEngineMarket  (Çift Motorlu Piyasa)
# ══════════════════════════════════════════════════════════════════════

class HybridEngineMarket(BemeMarket):
    """Half the agents use LogisticRegression; half use calibrated LinearSVC.

    COMMODITY sector → LogisticRegression (probabilistic, good for sparse NLP)
    FINANCE sector   → CalibratedClassifierCV(LinearSVC) (margin-based, strong
                       on dense tabular features)

    Both model types are wrapped in MultiOutputClassifier + MultiLabelWrapper
    so that predict_proba(X) always returns (n_samples, n_classes).

    The base_model argument to fit() / initialize_market() is **ignored** —
    HybridEngineMarket builds its own models internally.

    Parameters
    ----------
    lr_kwargs  : dict or None   Extra kwargs for LogisticRegression.
    svc_kwargs : dict or None   Extra kwargs for LinearSVC.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = HybridEngineMarket(n_funds=50).fit(X_train, y_train)
    # base_model not required — HybridEngineMarket provides its own
    """

    def __init__(
        self,
        lr_kwargs:  dict = None,
        svc_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr_kwargs  = lr_kwargs  or {}
        self.svc_kwargs = svc_kwargs or {}

    def initialize_market(self, base_model, X_train, y_train):
        """Override: ignore base_model; build LR + SVC agents internally."""
        from sklearn.linear_model        import LogisticRegression
        from sklearn.svm                 import LinearSVC
        from sklearn.calibration         import CalibratedClassifierCV
        from sklearn.multioutput         import MultiOutputClassifier
        from sklearn.model_selection     import train_test_split

        # ── Auto-Bounty (reuse parent logic) ─────────────────────
        if self._bounty_config == 'auto':
            y_arr  = y_train.toarray() if scipy.sparse.issparse(y_train) else np.asarray(y_train)
            counts = np.maximum(y_arr.sum(axis=0), 1).astype(float)
            self.bounty_map = {
                c: float(1.0 + np.log(counts.max() / counts[c]))
                for c in range(len(counts))
            }

        # ── Model factories ───────────────────────────────────────
        def make_lr():
            defaults = dict(solver='liblinear', max_iter=1000, class_weight='balanced')
            defaults.update(self.lr_kwargs)
            return MultiLabelWrapper(MultiOutputClassifier(LogisticRegression(**defaults)))

        def make_svc():
            defaults = dict(max_iter=2000, class_weight='balanced')
            defaults.update(self.svc_kwargs)
            return MultiLabelWrapper(
                MultiOutputClassifier(CalibratedClassifierCV(LinearSVC(**defaults)))
            )

        # ── Spawn agents ─────────────────────────────────────────
        self.funds = []
        for i in range(self.n_funds):
            X_in, _, y_in, _ = train_test_split(
                X_train, y_train, test_size=0.3, random_state=i)

            if scipy.sparse.issparse(y_in):
                y_in = y_in.toarray()

            # COMMODITY (first half) → LR; FINANCE (second half) → SVC
            sector = "COMMODITY" if i < (self.n_funds // 2) else "FINANCE"
            m      = make_lr() if sector == "COMMODITY" else make_svc()

            try:
                m.fit(X_in, y_in)
            except ValueError:
                y_full = y_train.toarray() if scipy.sparse.issparse(y_train) else y_train
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
            # Pass a sentinel; initialize_market ignores it
            self.initialize_market(base_model, X, y)

        n_samples = X.shape[0]
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc="BEME Hybrid training", unit="day")
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
