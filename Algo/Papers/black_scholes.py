"""
black_scholes.py
-----------------

This is the main result of the paper (Theorem 3): the European call price

    C = e^{-rT} * integral over y of max(e^y - K, 0) * G_T(log S0, y) dy

evaluates in closed form to the familiar

    C = S0 * N(d1) - K * e^{-rT} * N(d2)

We give two independent implementations:

  1. `black_scholes_call` / `black_scholes_put` -- the classic closed-form
     formula, computed directly from d1/d2.
  2. `black_scholes_call_via_kernel` -- the *same* price obtained by
     literally integrating the payoff against the Gaussian kernel, as
     the paper does. Comparing the two is a good sanity check that the
     "closed form" and the "kernel integral" really do agree numerically,
     which is exactly the content of Theorem 3.

We also add the two basic Greeks mentioned in Remark 4 (delta and gamma),
obtained by differentiating the closed-form price -- no stochastic
calculus required, just calculus.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .evolution import EvolutionOperator
from .kernel import GaussianKernel


def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    """Shared helper: compute the standard d1, d2 quantities."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be strictly positive")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Closed-form European call price: C = S0*N(d1) - K*e^{-rT}*N(d2)."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Closed-form European put price via put-call parity.

    Not the headline result of the paper, but it comes for free once you
    have the call price and is handy for cross-checks.
    """
    call = black_scholes_call(S0, K, r, sigma, T)
    return call - S0 + K * np.exp(-r * T)


def call_delta(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """dC/dS0 = N(d1)  (Remark 4)."""
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return norm.cdf(d1) #type: ignore


def call_gamma(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """d^2C/dS0^2 = N'(d1) / (S0 * sigma * sqrt(T))  (Remark 4)."""
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


def black_scholes_call_via_kernel(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Compute the *same* call price directly from the kernel integral (Eq. 1.3),

        C = e^{-rT} * (P_T payoff)(log S0),   payoff(y) = max(e^y - K, 0)

    instead of plugging into the closed formula. This is what the paper
    actually derives from first principles; `black_scholes_call` above is
    just the algebraically simplified end result.
    """
    kernel = GaussianKernel(r=r, sigma=sigma)
    op = EvolutionOperator(kernel)
    x0 = np.log(S0)
    payoff = lambda y: np.maximum(np.exp(y) - K, 0.0)
    undiscounted_expected_payoff = op.apply(T, x0, payoff)
    return np.exp(-r * T) * undiscounted_expected_payoff
