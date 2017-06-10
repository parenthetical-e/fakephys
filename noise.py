import numpy as np
from numpy.random import seed as set_seed
from scipy.ndimage.filters import gaussian_filter


def artifiact_fill(X):
    # Removed random segments, and set to last known good value
    raise NotImplementedError("TODO")
    pass


def random_covariance(X, cov, K, seed=None):
    """Add covariance between K randomly selected electrode"""
    raise NotImplementedError("TODO")
    pass


def normal(X, scale=1.0, seed=None):
    """Add white noise"""
    set_seed(seed)

    return X + np.random.normal(0, scale, size=X.shape)


def lognormal(X, scale=1.0, seed=None):
    """Add lognormal noise"""
    return np.log(normal(X, scale, seed))


def brown(X, scale, seed=None):
    """Add brown noise"""

    set_seed(seed)

    X = np.atleast_2d(X)
    M, N = X.shape
    noi = np.zeros_like(X)

    for j in range(M):
        d = np.random.normal(0, scale)
        rates = [
            d,
        ]

        for _ in range(N - 1):
            d += np.random.normal(0, scale)
            rates.append(d)

        noi[j, :] = rates

    return X + noi


def spatial_guassian(X, scale=1, order=0):
    """Add spatial correlations"""

    X = np.copy(X)

    M = X.shape[1]
    for j in range(M):
        X[:, j] = gaussian_filter(X[:, j], scale, order)

    return X


def temporal_autocorr(X, k=5, rho=0.1):
    """Additive temporal autocorrelation"""

    X = np.copy(X)
    X = np.atleast_2d(X)

    N, M = X.shape
    for j in range(M):
        x = X[:, j]
        for i in range(k, N - k):
            x[i] += np.sum(rho * x[i:i + k])

        X[:, j] = x

    return X
