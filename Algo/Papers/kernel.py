"""
kernel.py
---------

The heart of the whole paper is a single object: the Gaussian transition
density of a drifted Brownian motion,

    G_t(x, y) = (2 * pi * sigma^2 * t)^(-1/2) * exp( -(y - x - nu*t)^2 / (2*sigma^2*t) )

where nu = r - sigma^2 / 2 is the risk-neutral log-drift.

This is just the density of X_t ~ N(x + nu*t, sigma^2*t), i.e. of the
log-price at time t given it started at log-price x. Everything else in
the paper (Black-Scholes, digital options, barrier probabilities,
Chernoff splitting) is built by integrating this kernel against various
payoffs, or by composing it with itself.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate


class GaussianKernel:
    """The drifted Gaussian transition density G_t(x, y).

    Parameters
    ----------
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying (must be > 0).
    """

    def __init__(self, r: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive")
        self.r = r
        self.sigma = sigma
        # nu is the risk-neutral drift of the *log-price*, not of the price
        # itself -- this is the usual Ito correction, -sigma^2/2.
        self.nu = r - 0.5 * sigma ** 2

    def density(self, t: float, x: float, y) -> np.ndarray:
        """Evaluate G_t(x, y). y may be a scalar or a numpy array."""
        if t <= 0:
            raise ValueError("t must be positive")
        y = np.asarray(y, dtype=float)
        variance = self.sigma ** 2 * t
        mean = x + self.nu * t
        norm_const = 1.0 / np.sqrt(2.0 * np.pi * variance)
        return norm_const * np.exp(-((y - mean) ** 2) / (2.0 * variance))

    # Convenience alias so the object itself is "callable" like a function,
    # matching the G_t(x, y) notation in the paper.
    def __call__(self, t: float, x: float, y):
        return self.density(t, x, y)

    def total_mass(self, t: float, x: float) -> float:
        """Numerically check that G_t(x, .) integrates to 1 (Lemma 1).

        We integrate over a wide enough window around the mean that the
        Gaussian tails are negligible, rather than literally over
        (-inf, inf), since quad handles infinite limits but a finite
        window converges just as well and is faster.
        """
        mean = x + self.nu * t
        std = self.sigma * np.sqrt(t)
        lo, hi = mean - 12 * std, mean + 12 * std
        mass, _ = integrate.quad(lambda y: self.density(t, x, y), lo, hi)
        return mass

    def check_semigroup(self, s: float, t: float, x: float, y: float) -> tuple[float, float]:
        """Check the Chapman-Kolmogorov / semigroup law (Theorem 1):

            integral over z of G_s(x, z) * G_t(z, y) dz  ==  G_(s+t)(x, y)

        Returns a (numerical_lhs, exact_rhs) pair so the caller can compare
        them directly.
        """
        # Integrate the "glue" variable z over a wide window centred on
        # where the product of the two Gaussians actually lives.
        mean = x + self.nu * s
        std = self.sigma * np.sqrt(s)
        lo, hi = mean - 12 * std, mean + 12 * std

        integrand = lambda z: self.density(s, x, z) * self.density(t, z, y)
        lhs, _ = integrate.quad(integrand, lo, hi)
        rhs = self.density(s + t, x, y)
        return lhs, rhs #type: ignore
