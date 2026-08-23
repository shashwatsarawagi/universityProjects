"""
demo.py
-------

A little tour through the package. Run this directly:

    python -m hk_pricing.demo

It doesn't prove anything (that's what the Lean file in the paper is
for) -- it just exercises every piece and checks that the different
routes to the same number actually agree, which is a decent numerical
substitute for "does this all hang together".
"""

from __future__ import annotations

from .kernel import GaussianKernel
from .evolution import demonstrate_strong_continuity
from .black_scholes import (
    black_scholes_call,
    black_scholes_call_via_kernel,
    black_scholes_put,
    call_delta,
    call_gamma,
)
from .exotic import digital_call_price, prob_stay_below_barrier
from .chernoff import chernoff_convolution_check


def main():
    # A fairly ordinary set of market parameters, used throughout.
    S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 1.0

    print("=" * 70)
    print("1) Gaussian kernel: total mass and Chapman-Kolmogorov semigroup")
    print("=" * 70)
    kernel = GaussianKernel(r=r, sigma=sigma)
    mass = kernel.total_mass(t=T, x=0.0)
    print(f"  integral of G_T(0, y) dy  = {mass:.10f}   (should be 1)")

    lhs, rhs = kernel.check_semigroup(s=0.4, t=0.6, x=0.0, y=0.1)
    print(f"  (G_0.4 * G_0.6)(0, 0.1)   = {lhs:.10f}")
    print(f"  G_1.0(0, 0.1)             = {rhs:.10f}   (should match the line above)")

    print()
    print("=" * 70)
    print("2) Evolution operator: strong continuity as t -> 0")
    print("=" * 70)
    # A simple smooth bounded payoff so the convergence is easy to see.
    payoff = lambda y: 1.0 / (1.0 + y ** 2)
    x0 = 0.3
    print(f"  target value psi(x) = {payoff(x0):.6f}")
    for t, value in demonstrate_strong_continuity(kernel, x=x0, payoff=payoff):
        print(f"  t = {t:<8g} -> (P_t psi)(x) = {value:.6f}")

    print()
    print("=" * 70)
    print("3) Black-Scholes: closed form vs. direct kernel integration")
    print("=" * 70)
    closed_form = black_scholes_call(S0, K, r, sigma, T)
    via_kernel = black_scholes_call_via_kernel(S0, K, r, sigma, T)
    put = black_scholes_put(S0, K, r, sigma, T)
    print(f"  call price (closed form)   = {closed_form:.6f}")
    print(f"  call price (kernel integral) = {via_kernel:.6f}")
    print(f"  difference                  = {abs(closed_form - via_kernel):.2e}")
    print(f"  put price (parity)          = {put:.6f}")
    print(f"  delta                       = {call_delta(S0, K, r, sigma, T):.6f}")
    print(f"  gamma                       = {call_gamma(S0, K, r, sigma, T):.6f}")

    print()
    print("=" * 70)
    print("4) Digital call and barrier-stay probability")
    print("=" * 70)
    dig = digital_call_price(S0, K, r, sigma, T)
    print(f"  digital call price          = {dig:.6f}")

    B = 130.0  # barrier comfortably above S0
    p_stay = prob_stay_below_barrier(S0, B, r, sigma, T)
    print(f"  P(never crosses {B:.0f} in [0,{T:.0f}]) = {p_stay:.6f}")

    print()
    print("=" * 70)
    print("5) Chernoff/Trotter splitting: exactness at finite n")
    print("=" * 70)
    x, y = 0.0, 0.2
    total_time = 0.9
    for n in (0, 1, 2):
        s = total_time / (n + 1)
        convolved, single = chernoff_convolution_check(kernel, n=n, s=s, x=x, y=y)
        print(f"  n={n}: {n+1} slices of size {s:.4f} -> "
              f"convolution = {convolved:.6f}, single kernel = {single:.6f}")


if __name__ == "__main__":
    main()
