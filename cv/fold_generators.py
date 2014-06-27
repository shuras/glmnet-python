import numpy as np
from sklearn.cross_validation import KFold

unweighted_k_fold = KFold

def weighted_k_fold(n, n_folds=2, shuffle=False, weights=None):
    if weights is None:
        raise ValueError("Weights must be supplied to weighted k-fold "
                         "cross validation."
              )
    weights = np.asarray(weights)
    if weights.shape[0] != n:
        raise ValueError("Weights must have length n.")
    samples = np.arange(n)
    if shuffle:
        np.random.shuffle(samples)
        pdf = weights[samples]
    else:
        pdf = weights
    cdf = np.cumsum(pdf)
    wsum = np.sum(weights)
    cutoffs = np.linspace(0, wsum, n_folds + 1)
    for i in range(1, n_folds + 1):
        train_interval = (cdf >= cutoffs[i-1])*(cdf < cutoffs[i])
        test_interval = 1 - train_interval
        train_inds = list(samples[np.nonzero(train_interval)])
        test_inds = list(samples[np.nonzero(test_interval)])
        if i == n_folds: 
            train_inds.append(samples[n-1])
            test_inds.pop()
        yield train_inds, test_inds


