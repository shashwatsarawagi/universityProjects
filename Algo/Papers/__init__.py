"""
hk_pricing
==========

A small Python re-implementation of the constructions in:

    Berdinsky & Ushakov, "Henstock-Kurzweil Path Integral in Financial
    Mathematics: A Machine-Verified Pricing of European and Barrier
    Options" (arXiv:2608.19223)

The paper's actual content is a Lean 4 formal proof. What we do here is
much more modest: we translate the *mathematical objects* (the Gaussian
transition kernel, the pricing/evolution operator, the Black-Scholes
closed form, the Chernoff/Trotter splitting, and the barrier-probability
formula) into ordinary numerical Python so the results can be explored,
plotted, and sanity-checked against each other.
"""

from .kernel import GaussianKernel
from .evolution import EvolutionOperator
from .black_scholes import black_scholes_call, black_scholes_put, call_delta, call_gamma
from .exotic import digital_call_price, prob_stay_below_barrier
from .chernoff import chernoff_convolution_check

__all__ = [
    "GaussianKernel",
    "EvolutionOperator",
    "black_scholes_call",
    "black_scholes_put",
    "call_delta",
    "call_gamma",
    "digital_call_price",
    "prob_stay_below_barrier",
    "chernoff_convolution_check",
]
