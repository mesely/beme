# BEME — Bounty-driven Evolutionary Market Ensemble

Multi-label text classification under class imbalance, powered by prediction markets and evolutionary selection.

## Installation

```bash
git clone https://github.com/mesely/beme.git
cd beme
pip install -e .
```

Or once published to PyPI:

```bash
pip install beme
```

## Quick Start

**v0.2.0 one-liner:**

```python
from beme import BemeMarket, MultiLabelWrapper
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

model = BemeMarket(bounty_map='auto').fit(
    X_train, y_train,
    base_model=MultiLabelWrapper(MultiOutputClassifier(LogisticRegression(max_iter=1000)))
)
y_pred = model.predict(X_test)
```

**Full example:**

```python
from beme import BemeMarket, MultiLabelWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 1. Prepare data
X, y = make_multilabel_classification(n_samples=1000, n_features=50,
                                      n_classes=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Wrap your sklearn estimator
base_model = MultiLabelWrapper(MultiOutputClassifier(LogisticRegression(max_iter=1000)))

# 3. Fit — initialises market + runs all trading days in one call
market = BemeMarket(
    n_funds=50,
    leverage=25.0,
    temperature=1.5,
    bounty_map='auto',      # auto-compute bounties from class frequencies
    rebalance_period=100,
)
market.fit(X_train, y_train, base_model=base_model)

# 4. Predict on held-out test set (no balance updates)
y_pred = market.predict(X_test)
print(f1_score(y_test, y_pred, average='macro'))

# 5. Inspect the portfolio
print(market.portfolio_summary())
```

## API Reference

### `BemeMarket(n_funds=50, leverage=25.0, temperature=1.5, bounty_map=None, rebalance_period=100, sector_map=None)`

Main entry point. Creates a BEME ensemble.

**Parameters:**
- `n_funds` *(int)*: Number of hedge fund agents. Default: 50.
- `leverage` *(float)*: Scales profit/loss magnitude. Default: 25.0.
- `temperature` *(float)*: Consensus softness. Default: 1.5.
- `bounty_map` *(dict, `'auto'`, or None)*: Per-class reward multipliers.
  - `'auto'` — computed from class frequencies: `β_c = 1.0 + log(max_count / count_c)`
  - `None` — uses default map `{1: 1.5, 2: 2.0, 7: 1.5, 8: 2.0}`
  - `{}` — disables bounty entirely
- `rebalance_period` *(int)*: Days between evolutionary rebalancing cycles. Default: 100.
- `sector_map` *(dict or None)*: Reserved for future custom sector definitions. Currently unused.

---

### `market.fit(X, y, base_model=None) -> self`

High-level scikit-learn compatible training entry point. Initialises the market (if not done yet) and runs the full online training loop. Returns `self` for method chaining.

**Parameters:**
- `X`: array-like or sparse matrix of shape `(n_samples, n_features)`
- `y`: array-like or sparse matrix of shape `(n_samples, n_classes)`
- `base_model`: sklearn estimator. Required on first call; ignored if market already initialised.

---

### `market.initialize_market(base_model, X_train, y_train)`

Low-level initialisation: spawns agents and (if `bounty_map='auto'`) computes class-frequency bounties.

**Parameters:**
- `base_model`: sklearn estimator implementing `fit()` and `predict_proba()`. Deep-copied `n_funds` times.
- `X_train`: array-like or sparse matrix of shape `(n_samples, n_features)`
- `y_train`: array-like or sparse matrix of shape `(n_samples, n_classes)`

---

### `market.run_trading_day(x_i, y_true) -> np.ndarray`

Process one labelled sample. Updates all agent balances. Triggers evolution every `rebalance_period` calls.

**Parameters:**
- `x_i`: array-like of shape `(1, n_features)`, or sparse row
- `y_true`: array-like of shape `(n_classes,)`, binary ground truth. Sparse rows are automatically densified.

**Returns:** `np.ndarray` of shape `(n_classes,)`, binary prediction

---

### `market.predict(X) -> np.ndarray`

Batch inference without updating balances or triggering evolution. Each agent's `predict_proba` is called once on the full `X` matrix for efficiency.

**Parameters:**
- `X`: array-like or sparse matrix of shape `(n_samples, n_features)`

**Returns:** `np.ndarray` of shape `(n_samples, n_classes)`, binary

---

### `market.evolve()`

Manually trigger one evolutionary rebalancing cycle.

---

### `market.portfolio_summary() -> pd.DataFrame`

Return a DataFrame with the current state of every agent.

**Columns:** `agent_id`, `sector`, `balance`, `risk`, `active_classes`

## Hyperparameter Guide

| Parameter          | Symbol | Default                              | Description |
|--------------------|--------|--------------------------------------|-------------|
| `n_funds`          | N      | 50                                   | Number of agent hedge funds |
| `leverage`         | L      | 25.0                                 | Scales P&L. High = fast convergence, higher bankruptcy risk on noisy data |
| `temperature`      | T      | 1.5                                  | Consensus softness. T->0: richest agent dominates. T>>1: all agents vote equally |
| `bounty_map`       | β_c    | `None` → default map                 | `'auto'`, a custom dict, `{}` to disable, or `None` for the default map |
| `rebalance_period` | H      | 100                                  | Days between evolutionary rebalancing cycles |

**NLP / sparse text data:** Keep `temperature` high (1.5-2.5). Logistic models on text produce extreme probabilities; high T dampens outliers and keeps the market calibrated.

**Time-series / noisy signals:** Keep `leverage` low (5.0-10.0). Noise spikes can bankrupt successful agents before they accumulate enough history.

**Unknown class imbalance:** Use `bounty_map='auto'`. The formula `β_c = 1.0 + log(max_count / count_c)` assigns a multiplier of 1.0 to the most frequent class and proportionally larger values to rarer ones.

## How It Works (Three Pillars)

**1. Sectoral Isolation** — Agents are divided into COMMODITY and FINANCE rooms; each room covers a disjoint set of classes (COMMODITY: classes 1,2,4,7,9 — FINANCE: classes 0,3,5,6,8). This reduces cross-class noise and fosters specialisation, allowing each sector to develop deep expertise in its assigned label subset rather than spreading capacity thinly across all classes.

**2. Economic Incentive (Bounty)** — Rare classes carry higher `β_c` multipliers, so agents that correctly detect them earn disproportionately more balance. This creates automatic economic pressure pulling ensemble capacity toward hard-to-detect minority classes — the agents that ignore rare classes fall behind in wealth and eventually get replaced, while those that specialise in detecting them accumulate capital and gain more influence over the final prediction.

**3. Evolutionary Selection** — Every H days, the lowest-balance agent in each sector is replaced by a deep copy of the highest-balance agent, transferring its learned weights to the new generation. A 10% balance transfer gives the clone a fresh start while slightly penalising the parent to keep competitive pressure alive, ensuring the market never stagnates around a single dominant strategy.

## Theory (Mathematical Formulas)

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

**Auto-Bounty Formula:**

```
beta_c = 1.0 + log( max(counts) / counts_c )
```

**Evolutionary Cloning:**

```
child.model  = deepcopy(parent.model)
transfer     = parent.balance * 0.1
parent.balance -= transfer
child.balance  = max(100.0, transfer)
```

## Architecture Diagram

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

## License

MIT
