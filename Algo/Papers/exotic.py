"""
exotic.py
---------

Section 7 of the paper works through two examples that go beyond the
plain vanilla call, to show the kernel method generalizes:

  1. The digital (binary) call: pays 1 if S_T > K, else 0 (Section 7.1).
     Its price is just a single Gaussian tail probability, e^{-rT} * N(d2).

  2. The probability that a drifted Brownian motion stays below a fixed
     barrier B for the whole interval [0, T] (Section 7.2). This one is
     genuinely path-dependent -- it can't be computed from the terminal
     kernel alone -- and the paper derives it via the "method of images":
     you subtract off the contribution of a reflected Gaussian source
     centred on the mirror image of the starting point across the
     barrier, with an exponential re-weighting factor that comes from the
     drift-shift lemma. This reproduces the classical reflection-
     principle result.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .black_scholes import _d1_d2


def digital_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Price of a cash-or-nothing digital call paying 1 if S_T > K (Eq. 7.4):

        C_dig = e^{-rT} * N(d2)
    """
    _, d2 = _d1_d2(S0, K, r, sigma, T)
    return np.exp(-r * T) * norm.cdf(d2)


def prob_stay_below_barrier(S0: float, B: float, r: float, sigma: float, T: float) -> float:
    """Probability that log-price never exceeds the barrier B on [0, T] (Eq. 7.8):

        P = N( (B - x0 - nu*T) / (sigma*sqrt(T)) )
            - exp(2*nu*(B - x0)/sigma^2) * N( (-B + x0 - nu*T) / (sigma*sqrt(T)) )

    where x0 = log(S0) and nu = r - sigma^2/2. Requires S0 < B (the process
    starts strictly below the barrier), otherwise the "never crosses"
    event is either impossible or the formula's log(B/S0) term misbehaves.
    """
    if S0 >= B:
        raise ValueError("S0 must be strictly below the barrier B")

    x0 = np.log(S0)
    b = np.log(B)
    nu = r - 0.5 * sigma ** 2
    sqrt_T = np.sqrt(T)

    term1 = norm.cdf((b - x0 - nu * T) / (sigma * sqrt_T))
    reflection_weight = np.exp(2 * nu * (b - x0) / sigma ** 2)
    term2 = reflection_weight * norm.cdf((-b + x0 - nu * T) / (sigma * sqrt_T))

    return term1 - term2


def reflected_kernel(kernel_density, T: float, x0: float, b: float, y, nu: float, sigma: float):
    """The "image charge" constrained kernel from Eq. 7.7:

        G^B_T(x0, y) = G_T(x0, y) - exp(2*nu*(b - x0)/sigma^2) * G_T(2b - x0, y)

    This is the transition density restricted to paths that never cross
    the barrier b, built by subtracting a reflected Gaussian source. It's
    exposed here mainly so you can plot it and see the density vanish
    exactly at y = b, which is the boundary condition the method of
    images is designed to enforce.

    `kernel_density` should be a callable with the same signature as
    GaussianKernel.density, i.e. kernel_density(T, x, y).
    """
    weight = np.exp(2 * nu * (b - x0) / sigma ** 2)
    return kernel_density(T, x0, y) - weight * kernel_density(T, 2 * b - x0, y)
