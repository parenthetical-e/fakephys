from numpy.random import seed as set_seed
from numpy.random import normal as norm
from scipy.ndimage.filters import gaussian_filter


def normal(X, scale=1.0, seed=None):
    """Add white noise"""
    set_seed(seed)

    return X + norm(0, scale, size=X.shape)


def lognormal(X, scale=1.0, seed=None):
    """Add lognormal noise"""
    return np.log(white(X, scale, seed))


def brown(X, scale, seed=None):
    """Add brown noise"""

    set_seed(seed)
    N, M = X.shape

    for j in range(M):
        d = normal(0, scale)
        rates = [
            d,
        ]

        for i in range(N):
            d += normal(0, scale)
            rates.append(d)

        noi[:, j] = np.array(rates)

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

    N, M = X.shape
    for j in range(M):
        x = X[:, j]
        for i in range(k, N - k):
            x[i] += np.sum(rho * x[i:i + k])

        X[:, j] = x

    return X
