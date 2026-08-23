# hk_pricing

A small numerical Python translation of the constructions in Berdinsky &
Ushakov, *"Henstock–Kurzweil Path Integral in Financial Mathematics: A
Machine-Verified Pricing of European and Barrier Options"* (arXiv:2608.19223).

The paper's real content is a Lean 4 formal proof that a Gaussian
transition kernel reproduces the Black–Scholes formula. This package
doesn't prove anything — it just implements the same mathematical
objects (the kernel, the pricing/evolution operator, the closed-form
price, the Chernoff splitting, and the barrier formula) as ordinary
floating-point Python, so you can run the numbers and see how the pieces
fit together.

## Files

| File               | Corresponds to (paper section)                          |
|---------------------|-----------------------------------------------------------|
| `kernel.py`          | Definition 1, Lemma 1 (total mass), Theorem 1 (semigroup) |
| `evolution.py`       | Definition 2, Lemma 3 (rescaling), Theorem 2 (strong continuity) |
| `black_scholes.py`   | Theorem 3 (closed-form call price), Remark 4 (Greeks)     |
| `exotic.py`          | Section 7 (digital option, barrier-stay probability)      |
| `chernoff.py`        | Section 5, Theorem 4 (exact Chernoff/Trotter splitting)   |
| `demo.py`            | Runs everything and cross-checks the results              |

## Usage

```bash
python -m hk_pricing.demo
```

or, piece by piece:

```python
from hk_pricing import black_scholes_call, GaussianKernel

price = black_scholes_call(S0=100, K=105, r=0.03, sigma=0.25, T=1.0)

kernel = GaussianKernel(r=0.03, sigma=0.25)
kernel.total_mass(t=1.0, x=0.0)   # ~= 1.0
```

## Notes

- The "kernel integral" version of the call price
  (`black_scholes_call_via_kernel`) is deliberately kept separate from
  the closed-form formula so you can see them agree numerically —
  that agreement *is* the content of Theorem 3.
- `chernoff.py` gets slow for more than 2-3 time slices, since each
  extra slice adds another nested numerical integral. That's a
  limitation of doing this with brute-force quadrature, not a
  limitation of the underlying math (which the paper shows is exact
  at *any* number of slices).
- None of this is a substitute for the Lean proof — it's exploratory
  tooling, not verification.
