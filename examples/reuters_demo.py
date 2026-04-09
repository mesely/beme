"""
BEME end-to-end demo.
Falls back to synthetic multi-label data if Reuters-21578 is unavailable.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from beme import BemeMarket, MultiLabelWrapper

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

base_model = MultiLabelWrapper(MultiOutputClassifier(LogisticRegression(max_iter=1000)))
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
