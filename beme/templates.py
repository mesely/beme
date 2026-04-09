"""
beme/templates.py
=================
Experimental BemeMarket subclasses built on the Template Method Pattern.

Each template overrides one or more lifecycle hooks from BemeMarket:

    _on_day_start   — before predictions
    _calculate_pnl  — per-class profit/loss delta
    _on_day_end     — after all balances updated
    _on_rebalance   — inside evolution, before model copy

All templates fully inherit .fit(), .predict(), .evolve(), and
.portfolio_summary() from BemeMarket — no boilerplate needed.

──────────────────────────────────────────────────────────────────────
Template Builder (native Python inheritance)
──────────────────────────────────────────────────────────────────────
Users can create their own template on the fly in any notebook or script:

    from beme import BemeMarket

    class Template4_TaxMarket(BemeMarket):
        \"\"\"A market that taxes the richest agent every day.\"\"\"
        def _on_day_end(self):
            richest = max(self.funds, key=lambda f: f['balance'])
            if richest['balance'] > 500:
                richest['balance'] *= 0.8  # 20% wealth tax

No registration or factory required — standard Python subclassing is the
extensibility mechanism.
──────────────────────────────────────────────────────────────────────
"""

from .market import BemeMarket


# ══════════════════════════════════════════════════════════════════════
# Template 1 — DecayMarket (Eskime Payı)
# ══════════════════════════════════════════════════════════════════════

class DecayMarket(BemeMarket):
    """BemeMarket where agents lose a fixed fraction of wealth every day.

    Motivation
    ----------
    In standard BEME, an agent that performed well early can coast on its
    accumulated balance indefinitely. DecayMarket applies a daily "rust tax"
    that forces every agent to keep earning or shrink — preventing stagnation
    and keeping evolutionary pressure alive between rebalancing cycles.

    Mechanism
    ---------
    After each trading day (hook: _on_day_end):

        B_{t+1} = max(1.0,  B_t · (1 − λ))

    where λ is decay_rate.  With λ=0.005 an agent earning nothing loses
    ~39% of its balance over 100 days.

    Parameters
    ----------
    decay_rate : float
        Daily wealth decay fraction λ ∈ (0, 1). Default: 0.005.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = DecayMarket(decay_rate=0.01).fit(X_train, y_train, base_model=base)
    """

    def __init__(self, decay_rate: float = 0.005, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate

    def _on_day_end(self):
        """Apply daily decay to every agent's balance."""
        multiplier = 1.0 - self.decay_rate
        for f in self.funds:
            f['balance'] = max(1.0, f['balance'] * multiplier)


# ══════════════════════════════════════════════════════════════════════
# Template 2 — AlphaMarket (Sectoral Differentiation)
# ══════════════════════════════════════════════════════════════════════

class AlphaMarket(BemeMarket):
    """BemeMarket with sector-specific leverage and single-trade loss caps.

    Motivation
    ----------
    Real financial markets have very different volatility profiles across
    sectors.  AlphaMarket models this by giving COMMODITY and FINANCE sectors
    independent leverage multipliers and per-trade loss floors, so the two
    room types can have fundamentally different risk/reward dynamics.

    Mechanism
    ---------
    During P&L calculation (hook: _calculate_pnl):

        delta = base_pnl_formula × sector_leverage   (not global leverage)
        delta = max(delta, max_loss)                  (clamp downside)

    Default sector configs:
        COMMODITY — leverage: 10.0, max_loss: -5.0
        FINANCE   — leverage: 35.0, max_loss: -20.0

    Parameters
    ----------
    sector_configs : dict or None
        Override sector parameters.  Format::

            {
                'COMMODITY': {'leverage': float, 'max_loss': float},
                'FINANCE':   {'leverage': float, 'max_loss': float},
            }

        Missing keys fall back to global self.leverage / no cap.
    **kwargs
        All BemeMarket constructor parameters.

    Example
    -------
    model = AlphaMarket(
        sector_configs={'COMMODITY': {'leverage': 5.0, 'max_loss': -2.0},
                        'FINANCE':   {'leverage': 50.0, 'max_loss': -30.0}}
    ).fit(X_train, y_train, base_model=base)
    """

    DEFAULT_SECTOR_CONFIGS = {
        'COMMODITY': {'leverage': 10.0, 'max_loss': -5.0},
        'FINANCE':   {'leverage': 35.0, 'max_loss': -20.0},
    }

    def __init__(self, sector_configs: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.sector_configs = sector_configs or dict(self.DEFAULT_SECTOR_CONFIGS)

    def _calculate_pnl(
        self, agent, market_price, agent_pred, y_true_c, bounty
    ) -> float:
        """Use sector-specific leverage and clamp the downside."""
        cfg      = self.sector_configs.get(agent['sector'], {})
        leverage = cfg.get('leverage', self.leverage)
        max_loss = cfg.get('max_loss', -float('inf'))

        delta = (
            (y_true_c   - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * leverage
            * bounty
        )
        return max(delta, max_loss)


# ══════════════════════════════════════════════════════════════════════
# Template 3 — FutureValueMarket (Zamanın Değeri / İskonto)
# ══════════════════════════════════════════════════════════════════════

class FutureValueMarket(BemeMarket):
    """BemeMarket that gives early-stage agents a compounding growth boost.

    Motivation
    ----------
    A new agent cloned during evolution starts with a small balance and faces
    established rivals with hundreds of accumulated points.  Without help, it
    may never catch up before the next rebalancing cycle replaces it.
    FutureValueMarket acts like venture capital: correct calls by poor agents
    are amplified, mimicking early-stage compound interest.

    Mechanism
    ---------
    During P&L calculation (hook: _calculate_pnl):

        delta = base_pnl_formula(...)

        if delta > 0:
            multiplier = 1.0 + (100.0 / agent['balance']) × γ
            delta      = delta × multiplier

    where γ is discount_factor.

    When an agent's balance is 100 (fresh clone) the multiplier is 1 + γ.
    When balance is 1000 the multiplier collapses toward 1.0, so rich agents
    get no special treatment.

    Parameters
    ----------
    discount_factor : float
        γ ∈ [0, 1]. Controls the strength of the early-stage boost.
        Default: 0.9.
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
        self, agent, market_price, agent_pred, y_true_c, bounty
    ) -> float:
        """Apply compound-interest multiplier to profitable trades by poor agents."""
        delta = (
            (y_true_c   - market_price)
            * (agent_pred - market_price)
            * agent['risk']
            * self.leverage
            * bounty
        )
        if delta > 0:
            multiplier = 1.0 + (100.0 / max(agent['balance'], 1.0)) * self.discount_factor
            delta *= multiplier
        return delta
