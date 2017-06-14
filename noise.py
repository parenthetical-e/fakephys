import numpy as np

from numpy.random import seed as set_seed
from scipy.ndimage.filters import gaussian_filter

from brian2 import *
prefs.codegen.target = 'numpy'


def random_covariance(X, cov=0.1, K=2, seed=None, dt=None):
    """Add covariance between K randomly selected electrode pairs."""
    set_seed(seed)

    # Note sizes
    M, N = X.shape

    # Init (will contain covar electrodes)
    noi = np.copy(X)

    # Pick K random pairs of M rows (electrodes)
    index = range(M)
    np.random.shuffle(index)
    index = index[0:(k * 2)]

    # Add covar
    L = len(index)
    for i in range(0, L, 2):
        e1 = X[index[i], :]
        e2 = X[index[i + 1], :]
        e1 += (e2 * cov)

        noi[i, :] = e1

    return noi


def paired_covariance(X, cov=0.1, pairs=None, seed=None, dt=None):
    """Add covariance between K randomly selected electrode pairs."""
    set_seed(seed)

    # Note sizes
    M, N = X.shape

    # Init (will contain covar electrodes)
    noi = np.copy(X)

    # Add covar
    for pair in pairs:
        e1 = X[pair[0], :]
        e2 = X[pair[1], :]

        e1 += (e2 * cov)
        e2 += (e1 * cov)

        noi[pair[0], :] = e1
        noi[pair[1], :] = e2

    return noi


def normal(X, scale=1.0, seed=None, dt=None):
    """Add white noise"""
    set_seed(seed)

    return X + np.random.normal(0, scale, size=X.shape)


def gamma(X, shape=2, scale=2, seed=None, dt=None):
    set_seed(seed)

    return X + np.random.gamma(shape, scale=scale, size=X.shape)


def brown(X, scale=0.5, seed=None, dt=None):
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


def _lif(time, r_e=10, r_i=10, tau_e=5e-3, tau_i=10e-3, g_l=10e-9, dt=1e-3):

    time_step = dt * second
    defaultclock.dt = time_step

    I = 0  # No bias
    g_l *= siemens

    # Balanced input, numbers from:
    # Destexhe, A., Rudolph, M. & Paré, D., 2003.
    # The high-conductance state of neocortical neurons in vivo.
    # Nature Reviews Neuroscience, 4(9), pp.739–751.
    w_e = 0.73 * g_l
    w_i = 3.67 * g_l

    # -
    # unit = 1
    w_e /= g_l
    w_i /= g_l
    g_l = g_l / g_l

    # Approx 10,000 backgroun neurons without
    # having to sim that many
    Nb = 1000
    z = 10000 / Nb
    r_e = (r_e * z) * Hz
    r_i = (r_i * z) * Hz

    # Fixed
    Et = 1000 * mvolt  # Eff. infinite (shadow mode)
    Er = -65 * mvolt
    Ereset = -60 * mvolt

    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 20 * ms
    tau_ampa = tau_e * second
    tau_gaba = tau_i * second

    # --
    lif = """
    dv/dt = (g_l * (Er - v) + I_syn + I) / tau_m : volt
    I_syn = g_e * (Ee - v) + g_i * (Ei - v) : volt
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1
    I : volt
    """

    P_be = PoissonGroup(Nb, r_e)
    P_bi = PoissonGroup(Nb, r_i)

    # Our one neuron to gain control
    P_e = NeuronGroup(
        1,
        lif,
        threshold='v > Et',
        reset='v = Er',
        refractory=2 * ms,
        method='rk2')

    P_e.v = Ereset
    P_e.I = I * volt

    # Set up the 'network'
    C_be = Synapses(P_be, P_e, on_pre='g_e += w_e')
    C_be.connect('i == j')

    C_bi = Synapses(P_bi, P_e, on_pre='g_i += w_i')
    C_bi.connect('i == j')

    # Data acq
    traces_e = StateMonitor(P_e, ['v'], record=True)

    # Run
    report = None
    run(time * second, report=report)

    return np.asarray(traces_e.v_)


def balanced(X,
             scale=0.1,
             rate_e=10,
             rate_i=10,
             tau_e=5e-3,
             tau_i=10e-3,
             g_l=10e-9,
             dt=1e-3):
    """ Add excatitory-inhibitory background (1/F) noise.

    Params
    ------
    

    References
    ----------
    This way of simulating 1/F noise is justified in ephys based on:

    - Gao, R., Peterson, E. J., & Voytek, B. (2016). Inferring Synaptic 
      Excitation/Inhibition Balance from Field Potentials. bioRxiv, 1–31.
    
    But the idea has a much longer history in intra-cellular recodings:
    
    - Destexhe, A., Rudolph, M. & Paré, D., 2003. 
      The high-conductance state of neocortical neurons in vivo. 
      Nature Reviews Neuroscience, 4(9), pp.739–751.
    """

    X = np.atleast_2d(X)
    M, N = X.shape
    noi = np.zeros_like(X)

    for j in range(M):
        t = float(N * dt)

        noi[j, :] = _lif(
            t,
            r_e=rate_e,
            r_i=rate_i,
            tau_e=tau_e,
            tau_i=tau_i,
            g_l=g_l,
            dt=dt)

    return X + (scale * (noi - np.median(noi)))


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
