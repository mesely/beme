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
