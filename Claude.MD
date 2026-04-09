# BEME Framework — Agent Build Instructions

You are going to convert a research prototype into a clean, installable Python library
called **beme** and push it to `https://github.com/mesely/beme.git`.

Read every section of this file before writing a single line of code. The math, the code
snippets, and the architecture described here are all **canonical** — do not simplify,
skip, rename, or reinterpret anything. If a formula says divide by the sum of balances,
you divide by the sum of balances. If a default value is 25.0, it stays 25.0.

---

## 1. What Is BEME?

BEME stands for **Bounty-driven Evolutionary Market Ensemble**.

It solves a specific hard problem: **multi-label text classification when the dataset is
heavily imbalanced** (some classes appear rarely). Classic ensembles treat all classes
equally, so they learn to ignore rare ones. BEME fixes this by wrapping classifiers inside
a **prediction market simulation** where:

- Every classifier is a "Hedge Fund" agent that bets on its predictions.
- Agents that correctly predict rare (minority) classes earn **extra money** (bounty).
- Agents that consistently lose money get **replaced** by copies of the best agents.

This creates an automatic economic pressure that forces the ensemble to pay attention to
hard-to-detect classes.

Benchmark result: **Macro F1 = 0.8246** on Reuters-21578, beating all classic ensemble
baselines especially on minority classes.

---

## 2. Repository Layout

Create the repository exactly like this. Every file listed here must exist.

```
beme/
├── CLAUDE.md                  <- this file, keep it in the repo
├── README.md                  <- user-facing docs (see Section 13 for content)
├── setup.py
├── pyproject.toml
├── requirements.txt
├── beme/
│   ├── __init__.py            <- exports BemeMarket
│   ├── market.py              <- BemeMarket class (main entry point)
│   ├── pricing.py             <- market price discovery
│   ├── evolution.py           <- evolutionary cloning
│   ├── bounty.py              <- bounty helper
│   └── utils.py               <- logit/sigmoid wrappers, portfolio summary
└── examples/
    └── reuters_demo.py
```

---

## 3. How the System Works — Pipeline

Data flows like this on every single sample:

```
Raw input (x_i)
      |
      v
Each agent runs predict_proba(x_i)  ->  produces p_{i,c} for every class c
      |
      v
Market Price Discovery:
  For each class c, aggregate all agents in that sector using
  balance-weighted logit average, then apply temperature-scaled sigmoid
  -> produces P_c  (the market's consensus probability for class c)
      |
      v
Final prediction: y_hat[c] = 1 if P_c > 0.5 else 0
      |
      v
If true label y_true is known (training mode):
  Calculate each agent's profit/loss using the P&L formula
  Apply bounty multipliers for minority classes
  Update agent balances
      |
      v
Every H days: run evolutionary rebalancing
  (replace weakest agent in each sector with a clone of the strongest)
```

---

## 4. Agents (Hedge Funds)

Each agent is a Python dict with these exact keys:

```python
{
    'model':          <fitted sklearn estimator>,
    'balance':        100.0,                    # starts at 100, updated each day
    'risk':           <float in [0.5, 1.0]>,    # sampled once at creation
    'sector':         'COMMODITY' or 'FINANCE',
    'active_classes': [list of int],            # which classes this agent trades
}
```

### Initialisation rule

Given `n_funds` agents and a `base_model`:

- Agents `0` to `n_funds // 2 - 1`       -> sector = `'COMMODITY'`, active_classes = `[1, 2, 4, 7, 9]`
- Agents `n_funds // 2` to `n_funds - 1` -> sector = `'FINANCE'`,   active_classes = `[0, 3, 5, 6, 8]`

For each agent `i`:
1. Draw a stratified 70/30 train/test split of (X_train, y_train) with `random_state=i`.
2. Deep-copy `base_model` (so each agent starts from the same architecture but trains independently).
3. Fit the copy on the 70% portion.
4. Sample `risk` from `numpy.random.uniform(0.5, 1.0)`.
5. Set `balance = 100.0`.

```python
# Reference initialisation code -- implement this logic exactly
import copy
import numpy as np
from sklearn.model_selection import train_test_split

def initialize_market(base_model, X_train, y_train, n_funds=50):
    funds = []
    for i in range(n_funds):
        X_in, _, y_in, _ = train_test_split(
            X_train, y_train, test_size=0.3, random_state=i)
        m = copy.deepcopy(base_model)
        m.fit(X_in, y_in)
        sector = "COMMODITY" if i < (n_funds // 2) else "FINANCE"
        active_classes = (
            [1, 2, 4, 7, 9] if sector == "COMMODITY" else [0, 3, 5, 6, 8])
        funds.append({
            'model':          m,
            'balance':        100.0,
            'risk':           np.random.uniform(0.5, 1.0),
            'sector':         sector,
            'active_classes': active_classes,
        })
    return funds
```

---

## 5. Market Price Discovery -- `beme/pricing.py`

This is the heart of the consensus mechanism. Given all agent predictions for a single
sample, compute one probability per class that represents the market's collective opinion.

### Formula

For each class `c`:

```
P_c = sigmoid( L_bar_c / T )
```

Where `L_bar_c` is the balance-weighted average logit:

```
L_bar_c = sum_{i in Sector_c}( B_i * logit(p_{i,c}) )
          -------------------------------------------------
               sum_{i in Sector_c}( B_i )
```

Variables:
- `B_i`      = agent i's current balance
- `p_{i,c}`  = agent i's predicted probability for class c, clipped to [0.01, 0.99]
  (clipping is mandatory -- raw logit blows up at 0 or 1)
- `T`        = temperature (default 1.5)
- `logit(p)` = ln(p / (1-p))    -- use scipy.special.logit
- `sigmoid`  = 1 / (1 + exp(-x)) -- use scipy.special.expit

Only agents whose `active_classes` list contains `c` participate in pricing `c`.

### Reference implementation for `beme/pricing.py`

```python
import numpy as np
from scipy.special import logit, expit


def get_market_prices(funds, preds, T=1.5):
    """
    Compute balance-weighted market consensus probabilities.

    Parameters
    ----------
    funds : list of agent dicts
    preds : np.ndarray shape (n_funds, n_classes), already clipped to [0.01, 0.99]
    T     : temperature scalar

    Returns
    -------
    market_prices : np.ndarray shape (n_classes,)
    """
    n_classes = preds.shape[1]
    market_prices = np.zeros(n_classes)

    for c in range(n_classes):
        idx = [j for j, f in enumerate(funds) if c in f['active_classes']]
        balances = np.array([funds[k]['balance'] for k in idx])
        weights  = balances / np.sum(balances)
        weighted_logit = np.average(logit(preds[idx, c]), weights=weights)
        market_prices[c] = expit(weighted_logit / T)

    return market_prices
```

---

## 6. Profit / Loss and Bounty -- `beme/bounty.py` and `beme/market.py`

After observing the true label `y_true` (binary vector of length n_classes), update
every agent's balance.

### P&L Formula

For each agent `i`:

```
delta_B_i = sum over c in agent's active_classes:
    (y_true[c] - P_c)       # how wrong the market was
  x (p_{i,c}  - P_c)       # how much better than the market the agent was
  x risk_i                  # agent's risk appetite
  x L                       # leverage (default 25.0)
  x beta_c                  # bounty multiplier for class c

agent['balance'] = max(1.0, agent['balance'] + delta_B_i)
```

The `max(1.0, ...)` floor prevents agents from going bankrupt with zero balance, but keeps
them very weakened so evolution will replace them shortly.

### Default bounty map

```python
DEFAULT_BOUNTY_MAP = {1: 1.5, 2: 2.0, 7: 1.5, 8: 2.0}
# Any class not in the map defaults to 1.0
# If bounty_map is None or {}, all classes get beta_c = 1.0 (bounty disabled)
```

### Reference balance-update code (goes inside `run_trading_day` in market.py)

```python
for idx, f in enumerate(funds):
    profit_day = 0
    for c in f['active_classes']:
        beta = bounty_map.get(c, 1.0) if bounty_map else 1.0
        profit_day += (
            (y_true[c] - market_prices[c])
          * (preds[idx, c] - market_prices[c])
          * f['risk']
          * leverage
          * beta
        )
    f['balance'] = max(1.0, f['balance'] + profit_day)
```

### `beme/bounty.py`

```python
DEFAULT_BOUNTY_MAP = {1: 1.5, 2: 2.0, 7: 1.5, 8: 2.0}


def get_bounty(bounty_map, class_index):
    """
    Return the bounty multiplier beta_c for a given class.
    Returns 1.0 if class is not in the map or map is disabled.
    """
    if not bounty_map:
        return 1.0
    return bounty_map.get(class_index, 1.0)
```

---

## 7. Evolutionary Selection -- `beme/evolution.py`

Every `rebalance_period` days (default 100), run this for each sector independently.

### Steps

1. Filter agents belonging to the sector.
2. Sort them by `balance` descending.
3. `parent` = agent with the highest balance (index 0).
4. `child`  = agent with the lowest balance (index -1).
5. Replace `child['model']` with a deep copy of `parent['model']`.
6. Transfer 10% of parent's balance to child:

```
transfer           = parent['balance'] * 0.1
parent['balance'] -= transfer
child['balance']   = max(100.0, transfer)
```

This gives the cloned agent a fresh start (at least 100.0) while slightly penalising the
parent to keep competitive pressure alive.

### Reference implementation for `beme/evolution.py`

```python
import copy


def evolve_sector(funds, sector_name):
    """
    Run one evolutionary step for the given sector.
    Mutates the funds list in-place.
    """
    sector_funds = [f for f in funds if f['sector'] == sector_name]
    sector_funds.sort(key=lambda x: x['balance'], reverse=True)

    parent = sector_funds[0]
    child  = sector_funds[-1]

    child['model']     = copy.deepcopy(parent['model'])
    transfer           = parent['balance'] * 0.1
    parent['balance'] -= transfer
    child['balance']   = max(100.0, transfer)


def evolve(funds):
    """Run one evolutionary step for all sectors."""
    for sector_name in ["COMMODITY", "FINANCE"]:
        evolve_sector(funds, sector_name)
```

---

## 8. BemeMarket Class -- `beme/market.py`

This is the single class users import. Wire everything above into it.

```python
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
```

---

## 9. Utilities -- `beme/utils.py`

```python
import numpy as np
import pandas as pd
from scipy.special import logit as _logit, expit as _expit


def safe_logit(p):
    """Logit with mandatory clipping to avoid -inf/+inf."""
    return _logit(np.clip(p, 0.01, 0.99))


def safe_sigmoid(x):
    """Numerically stable sigmoid."""
    return _expit(x)


def portfolio_summary(funds):
    """
    Build a DataFrame snapshot of all agents.

    Returns
    -------
    pd.DataFrame with columns: agent_id, sector, balance, risk, active_classes
    """
    rows = []
    for i, f in enumerate(funds):
        rows.append({
            'agent_id':       i,
            'sector':         f['sector'],
            'balance':        round(f['balance'], 4),
            'risk':           round(f['risk'], 4),
            'active_classes': f['active_classes'],
        })
    return pd.DataFrame(rows)
```

---

## 10. Package Entrypoint -- `beme/__init__.py`

```python
from .market import BemeMarket

__all__ = ["BemeMarket"]
__version__ = "0.1.0"
```

---

## 11. Package Config Files

### `requirements.txt`

```
numpy>=1.21
scipy>=1.7
scikit-learn>=1.0
pandas>=1.3
```

### `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="beme",
    version="0.1.0",
    description="Bounty-driven Evolutionary Market Ensemble for multi-label classification",
    author="Mehmet Selman Yilmaz",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
    ],
    python_requires=">=3.8",
)
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.backends.legacy:build"
```

---

## 12. Example Script -- `examples/reuters_demo.py`

Write a runnable end-to-end demo. If Reuters-21578 is not available, fall back to a
synthetic dataset via `sklearn.datasets.make_multilabel_classification`.
The script must show: initialize_market -> training loop with run_trading_day -> predict
on a held-out test set -> print Macro F1.

```python
"""
BEME end-to-end demo.
Falls back to synthetic multi-label data if Reuters-21578 is unavailable.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from beme import BemeMarket

# --- Data -----------------------------------------------------------
X, y = make_multilabel_classification(
    n_samples=2000, n_features=100, n_classes=10,
    n_labels=2, allow_unlabeled=False, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Build market ---------------------------------------------------
market = BemeMarket(
    n_funds=50,
    leverage=25.0,
    temperature=1.5,
    bounty_map={1: 1.5, 2: 2.0, 7: 1.5, 8: 2.0},
    rebalance_period=100,
)

base_model = LogisticRegression(max_iter=1000, multi_class='ovr')
market.initialize_market(base_model, X_train, y_train)

# --- Online training loop -------------------------------------------
print("Running trading days...")
for i in range(len(X_train)):
    market.run_trading_day(X_train[i:i+1], y_train[i])

# --- Evaluation -----------------------------------------------------
y_pred = market.predict(X_test)
score  = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"Macro F1: {score:.4f}")

# --- Portfolio snapshot ---------------------------------------------
print("\nTop 5 agents by balance:")
print(market.portfolio_summary().sort_values('balance', ascending=False).head())
```

---

## 13. README.md -- Required Content and Section Order

Write README.md with exactly these sections in this exact order.

---

### Section 1: Title and Badge Line

```
# BEME — Bounty-driven Evolutionary Market Ensemble

Multi-label text classification under class imbalance, powered by prediction markets and evolutionary selection.
```

---

### Section 2: Installation

```markdown
## Installation

git clone https://github.com/mesely/beme.git
cd beme
pip install -e .
```

Or once published to PyPI:

```
pip install beme
```

---

### Section 3: Quick Start (this must come BEFORE any theory)

Show a complete minimal example that a user can copy-paste and run immediately.

```markdown
## Quick Start

from beme import BemeMarket
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 1. Prepare data
X, y = make_multilabel_classification(n_samples=1000, n_features=50,
                                      n_classes=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Create and initialise the market
market = BemeMarket(n_funds=50, leverage=25.0, temperature=1.5)
market.initialize_market(LogisticRegression(max_iter=500), X_train, y_train)

# 3. Train (online loop -- each sample is one "trading day")
for i in range(len(X_train)):
    market.run_trading_day(X_train[i:i+1], y_train[i])

# 4. Predict on held-out test set (no balance updates)
y_pred = market.predict(X_test)
print(f1_score(y_test, y_pred, average='macro'))

# 5. Inspect the portfolio
print(market.portfolio_summary())
```

---

### Section 4: API Reference

Document every public method of BemeMarket with its full signature and all parameters.

---

### Section 5: Hyperparameter Guide

Include this table:

| Parameter          | Symbol | Default                              | Description |
|--------------------|--------|--------------------------------------|-------------|
| `n_funds`          | N      | 50                                   | Number of agent hedge funds |
| `leverage`         | L      | 25.0                                 | Scales P&L. High = fast convergence, higher bankruptcy risk on noisy data |
| `temperature`      | T      | 1.5                                  | Consensus softness. T->0: richest agent dominates. T>>1: all agents vote equally |
| `bounty_map`       | beta_c | {1:1.5, 2:2.0, 7:1.5, 8:2.0}        | Per-class reward multipliers. Pass {} to disable |
| `rebalance_period` | H      | 100                                  | Days between evolutionary rebalancing cycles |

**NLP / sparse text data:** Keep `temperature` high (1.5-2.5). Logistic models on text
produce extreme probabilities; high T dampens outliers and keeps the market calibrated.

**Time-series / noisy signals:** Keep `leverage` low (5.0-10.0). Noise spikes can
bankrupt successful agents before they accumulate enough history.

---

### Section 6: How It Works (Three Pillars)

Write one paragraph each on:
1. Sectoral Isolation -- agents are divided into COMMODITY and FINANCE rooms; each room
   covers a disjoint set of classes, reducing noise and fostering specialisation.
2. Economic Incentive (Bounty) -- rare classes carry higher beta_c multipliers, so agents
   that detect them earn disproportionately more. This pulls ensemble capacity toward
   hard classes automatically.
3. Evolutionary Selection -- every H days, the lowest-balance agent in each sector is
   replaced by a deep copy of the highest-balance agent, transferring learned weights to
   the new generation.

---

### Section 7: Theory (Mathematical Formulas)

Include the three core formulas in readable form (LaTeX or code-block math):

**Market Price Discovery:**
```
P_c = sigmoid( L_bar_c / T )

L_bar_c = sum_i( B_i * logit(p_{i,c}) ) / sum_i( B_i )
          where i ranges over agents in Sector_c
```

**Agent P&L:**
```
delta_B_i = sum_c [ (y_true[c] - P_c) * (p_{i,c} - P_c) * risk_i * L * beta_c ]
B_i = max(1.0, B_i + delta_B_i)
```

**Evolutionary Cloning:**
```
child.model  = deepcopy(parent.model)
transfer     = parent.balance * 0.1
parent.balance -= transfer
child.balance  = max(100.0, transfer)
```

---

### Section 8: Architecture Diagram (ASCII)

```
  Input x_i
      |
      v
 [Agent 1..N]  predict_proba(x_i)
      |
      v
 Market Price Discovery (balance-weighted logit avg)
      |
      v
  P_c  ------>  y_hat = (P_c > 0.5)
      |
      v
  P&L + Bounty Update  <-- y_true (training only)
      |
      v
  Every H days: Evolutionary Rebalancing
```

---

### Section 9: License

MIT

---

## 14. Implementation Checklist

Work through this in order. Do not move to the next item until the current one passes.

- [ ] Create beme/ package directory and all __init__.py files
- [ ] Implement beme/utils.py (safe_logit, safe_sigmoid, portfolio_summary)
- [ ] Implement beme/pricing.py (get_market_prices -- match formula exactly)
- [ ] Implement beme/bounty.py (DEFAULT_BOUNTY_MAP, get_bounty)
- [ ] Implement beme/evolution.py (evolve_sector, evolve)
- [ ] Implement beme/market.py (BemeMarket with all methods)
- [ ] Wire beme/__init__.py
- [ ] Write setup.py and pyproject.toml
- [ ] Write requirements.txt
- [ ] Write examples/reuters_demo.py
- [ ] Write README.md with sections in the order specified in Section 13
- [ ] Verify: python -c "from beme import BemeMarket; print('OK')" exits 0
- [ ] Verify: python examples/reuters_demo.py runs without error
- [ ] All public methods have docstrings
- [ ] Commit and push to https://github.com/mesely/beme.git

---

## 15. Git Instructions

```bash
git clone https://github.com/mesely/beme.git
cd beme
# create all files listed in Section 2
git add .
git commit -m "feat: initial BEME framework v0.1.0"
git push origin main
```

If the remote already has commits, run `git pull --rebase origin main` before pushing.
Do not force-push unless the branch is confirmed empty.