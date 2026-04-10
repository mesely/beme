import numpy as np
import pandas as pd
from scipy.special import logit as _logit, expit as _expit
import copy


def safe_logit(p):
    """Logit with mandatory clipping to avoid -inf/+inf."""
    return _logit(np.clip(p, 0.01, 0.99))


def safe_sigmoid(x):
    """Numerically stable sigmoid."""
    return _expit(x)


class MultiLabelWrapper:
    """
    Wraps a MultiOutputClassifier (or any estimator whose predict_proba
    returns a list of per-class arrays) so that predict_proba(X) returns
    a single array of shape (n_samples, n_classes) as expected by BemeMarket.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        raw = self.estimator.predict_proba(X)
        if isinstance(raw, list):
            # MultiOutputClassifier returns a list of (n_samples, n_classes_c) arrays.
            # Normally n_classes_c == 2 (binary per label).
            # DummyClassifier on a degenerate column returns (n_samples, 1) —
            # in that case there is only one known class (all 0s or all 1s).
            # We return 0.0 probability of the positive class for those columns.
            n_samples = raw[0].shape[0]
            cols = []
            for arr in raw:
                if arr.shape[1] >= 2:
                    cols.append(arr[:, 1])
                else:
                    # Degenerate: only one class seen during training.
                    # Positive-class probability is 0 (safe conservative default).
                    cols.append(np.zeros(n_samples))
            return np.column_stack(cols)
        return raw

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.estimator = copy.deepcopy(self.estimator, memo)
        return result


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
