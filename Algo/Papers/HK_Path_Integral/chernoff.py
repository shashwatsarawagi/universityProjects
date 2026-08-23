"""
chernoff.py
-----------

Section 5 of the paper makes a nice point: for most Chernoff/Trotter
product formulas, the approximation P_t ~= (D_{t/n} S_{t/n})^n only
converges as n -> infinity. But because the drift and diffusion
generators here are *constant coefficient* (they commute), the splitting
is actually exact at every finite n -- gluing n+1 short-time kernels
together via the semigroup law reproduces the single long-time kernel
exactly, with no approximation error at all (Theorem 4):

    G*^n_s(x, y) = G_{(n+1)s}(x, y)

where G*^n_s is the n-fold convolution (over n intermediate positions) of
the one-step kernel G_s.

Here we check this numerically for small n by literally performing the
nested integrals over the intermediate time-slice positions and comparing
against the single kernel evaluated at the combined time. Since each
extra intermediate variable costs another numerical integration, this is
only practical for a handful of slices -- but that's plenty to see that
the "error" is just numerical-quadrature noise, not a systematic
approximation gap.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from .kernel import GaussianKernel


def _n_fold_convolution(kernel: GaussianKernel, n: int, s: float, x: float, y: float) -> float:
    """Compute G*^n_s(x, y): glue n+1 copies of the one-step kernel G_s
    by integrating out n intermediate positions z_1, ..., z_n.

    n = 0 is the base case: no intermediate variables, just G_s(x, y)
    itself (one "step" of size s already connects x to y).
    """
    if n == 0:
        return kernel.density(s, x, y) #type: ignore

    # Integration window for each intermediate variable: centred on the
    # drift-shifted mean, wide enough that the Gaussian tails are
    # negligible relative to the values we're comparing.
    std = kernel.sigma * np.sqrt(s)

    def integrand(z):
        # Recursively glue: one step from x to z, then the (n-1)-fold
        # convolution from z to y.
        return kernel.density(s, x, z) * _n_fold_convolution(kernel, n - 1, s, z, y)

    mean = x + kernel.nu * s
    lo, hi = mean - 10 * std, mean + 10 * std
    value, _ = integrate.quad(integrand, lo, hi, limit=100)
    return value


def chernoff_convolution_check(kernel: GaussianKernel, n: int, s: float,
                                x: float, y: float) -> tuple[float, float]:
    """Compare the n-fold convolution G*^n_s(x, y) against the single
    kernel G_{(n+1)s}(x, y). Returns (numerical_convolution, exact_single_step).

    For n >= 3 this gets slow (each extra intermediate variable is another
    nested numerical integral), so keep n small -- n=0,1,2 is already
    enough to see the pattern clearly.
    """
    convolved = _n_fold_convolution(kernel, n, s, x, y)
    single_step = kernel.density((n + 1) * s, x, y)
    return convolved, single_step #type: ignore
