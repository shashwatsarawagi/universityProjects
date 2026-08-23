"""
evolution.py
------------

The evolution (a.k.a. pricing) operator from Definition 2:

    (P_t psi)(x) = integral over y of G_t(x, y) * psi(y) dy

Applied to a payoff function psi(log S_T) and evaluated at x = log S0,
this is exactly the undiscounted expected payoff of a European claim.
Discounting by e^{-rT} turns it into the fair price.

We also implement the rescaled form (Lemma 3), which rewrites the
integral as an expectation over a *standard* normal variable u:

    (P_t psi)(x) = E[ psi(x + nu*t + sigma*sqrt(t)*u) ],  u ~ N(0, 1)

This form is what you'd actually use for a quick Monte-Carlo check, and
it's also what makes the "strong continuity" result (P_t psi -> psi as
t -> 0) visually obvious: as t shrinks, the argument of psi collapses
onto x.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
from scipy.stats import norm

from .kernel import GaussianKernel


class EvolutionOperator:
    """Wraps a GaussianKernel and applies it to arbitrary payoff functions."""

    def __init__(self, kernel: GaussianKernel):
        self.kernel = kernel

    def apply(self, t: float, x: float, payoff: Callable[[np.ndarray], np.ndarray],
              n_std: float = 10.0) -> float:
        """Compute (P_t payoff)(x) by direct numerical integration of the kernel.

        n_std controls how many standard deviations out from the mean we
        integrate -- 10 is already absurd overkill for a Gaussian but it
        keeps the truncation error far below anything else in the pipeline.
        """
        mean = x + self.kernel.nu * t
        std = self.kernel.sigma * np.sqrt(t)
        lo, hi = mean - n_std * std, mean + n_std * std

        integrand = lambda y: self.kernel.density(t, x, y) * payoff(y)
        value, _ = integrate.quad(integrand, lo, hi, limit=200)
        return value

    def apply_rescaled(self, t: float, x: float,
                        payoff: Callable[[np.ndarray], np.ndarray]) -> float:
        """Same thing, but via the standard-Gaussian substitution of Lemma 3:

            (P_t payoff)(x) = integral over u of phi(u) * payoff(x + nu*t + sigma*sqrt(t)*u) du

        where phi is the standard normal density. This is mathematically
        identical to `apply`, just centred/rescaled -- a useful independent
        check that the two integration paths agree.
        """
        nu, sigma = self.kernel.nu, self.kernel.sigma
        shifted_payoff = lambda u: norm.pdf(u) * payoff(x + nu * t + sigma * np.sqrt(t) * u)
        value, _ = integrate.quad(shifted_payoff, -10, 10, limit=200)
        return value


def demonstrate_strong_continuity(kernel: GaussianKernel, x: float,
                                   payoff: Callable[[np.ndarray], np.ndarray],
                                   t_values=None) -> list[tuple[float, float]]:
    """Numerically illustrate Theorem 2: (P_t psi)(x) -> psi(x) as t -> 0+.

    Returns a list of (t, P_t_psi(x)) pairs for decreasing t, which should
    visibly converge towards psi(x).
    """
    if t_values is None:
        t_values = [1.0, 0.1, 0.01, 0.001, 0.0001]

    op = EvolutionOperator(kernel)
    return [(t, op.apply(t, x, payoff)) for t in t_values]
