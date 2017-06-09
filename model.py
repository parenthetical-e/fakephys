from brian2 import *
from scipy.integrate import odeint
import numpy as np


def create_times(t, dt=1e-3):
    n_steps = int(t * (1.0 / dt))
    times = np.linspace(0, t, n_steps)

    return times


def osc(t, N, a, f, phase=0, dt=1e-3):
    """Sinusoidal oscillation"""
    times = create_times(t, dt)

    rates = []
    for _ in range(N):
        r = a + (a / 2.0) * np.sin(times * f * 2 * np.pi + phase)
        rates.append(r)
    rates = np.vstack(rates)

    return rates


def ie(t,
       N,
       P0,
       PN,
       c1=15.0,
       c2=15.0,
       c3=15.0,
       c4=3.0,
       Q=1,
       tau_e=5e-3,
       tau_i=10e-3,
       dt=1e-3,
       sigma=0.01,
       drivefn=np.linspace):
    # --
    time = t * second
    time_step = dt * second

    # -
    # Fixed parameters.
    re = 1.0
    ri = 0.5

    kn = 1.0
    k = 1.0

    tau_e *= second
    tau_i *= second

    # -
    # Define the drifting drive
    times = create_times(t, dt)
    P = drivefn(P0, PN, len(times))

    # Scale it
    P = P * (2**-0.03)

    # Format for Brian2
    P = TimedArray(P, dt=time_step)

    # -
    eqs = """
            dE/dt = -E/tau_e + ((1 - re * E) * (1 / (1 + exp(-(k * c1 * E - k * c2 * I+ k * P(t) - 2))) - 1/(1 + exp(2*1.0)))) / tau_e  + (sigma / tau_e**.5 * xi_e) : 1
            dI/dt = -I/tau_i + ((1 - ri * I) * (1 / (1 + exp(-2 * (kn * c3 * E - kn * c4 * I + kn * Q - 2.5))) - 1/(1 + exp(2*2.5)))) / tau_i + (sigma / tau_i**.5 * xi_i) : 1
        """

    pops = NeuronGroup(N, model=eqs, namespace={'Q': Q})
    pops.E = 0
    pops.I = 0

    # --
    # Record
    mon = StateMonitor(pops, ('E', 'I'), record=True)

    # --
    # Run
    defaultclock.dt = time_step
    run(time)

    return np.asarray(mon.I_) + np.asarray(mon.E_)


# --
def kuramoto(t, N, a, b, K, dt=1e-3):
    """Simulate a Kuramoto model."""

    times = np.linspace(0, t, int(t / dt))

    omegas = np.random.uniform(a, b, size=N)
    theta0 = np.random.uniform(-np.pi * 2, np.pi * 2, size=N)

    def _f(theta, t, omega, K, N):
        # In classic kuramoto...
        # each oscillator gets the same wieght K
        # normalized by the number of oscillators
        c = K / N

        # and all oscillators are connected to all
        # oscillators
        theta = np.atleast_2d(theta)  # opt for broadcasting
        W = np.sum(np.sin(theta - theta.T), 1)

        return omega + (c * W)

    # Run the Kuramoto
    thetas = odeint(_f, theta0, times, args=(omegas, K, N))
    thetas = np.mod(thetas, 2 * np.pi)
    thetas -= np.pi

    # Move to time domain from the unit circle
    waves = []
    for n in range(N):
        th = thetas[:, n]
        f = omegas[n]

        wave = np.sin(f * 2 * np.pi * times + th)
        waves.append(wave)
    waves = np.vstack(waves)

    return waves
