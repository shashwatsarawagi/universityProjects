ACT: Anti-Crosstalk Learning for Cross-Sectional Stock Ranking
*Implementation of the ACT framework proposed by Juntao Li and Liang Zhang.*

Cross-sectional stock ranking predicts the relative return ordering of a universe of stocks on each trading day. Existing graph-based methods encode every stock into a single unified embedding and then propagate that embedding across a relational graph.  This causes two forms of "crosstalk":

1.  Temporal-scale crosstalk – long-run trends, short-horizon fluctuations and event-driven shocks are all bundled into one vector.  When that vector is shared across the graph, stock-specific noise contaminates every neighbour. Logically, only long-run trends should be passed, short-horizon should be locally message passed, while shocks completely individual.

2.  Structural crosstalk – multiple qualitatively different graphs (industry, region, latent co-movement) are naively averaged, so the distinct signal of each graph is lost. This contains information which could significantly boost prediction accuracy

ACT fixes these problems over the following steps:
- TCD  (Temporal Component Decomposition) – splits each price sequence into trend, fluctuation, and shock via recursive causal moving averages then splitting it to the three branches below as required

    a. PSPE is the most complex module. For each static graph (industry, region), it computes both a forward signal (useful relation-aware info to keep) and a backward signal (relation-specific interference to subtract from the residual). This bidirectional design ensures industry signals can't contaminate the region branch. After static purification, a dynamic k-NN graph is built from cosine similarities of the purified residual, and a GAT propagates on it. Static and dynamic results are gate-fused.

    b. FCI uses a gated TCN applied independently per stock: Z = ReLU(P ⊙ σ(Q) + R) where P, Q, R are three independent Conv1d streams. Stock-local convolutions keep fluctuations from ever entering the graph.

    c. SCI creates a smoothed counterfactual shock sequence (what shocks would look like without extreme outliers), then feeds both the raw and smoothed versions through a two-layer MLP. The contrast lets the model isolate genuine event signals.

- ACF stacks the three embeddings and learns per-stock attention weights $`\alpha`$ via a two-layer attention MLP, allowing each stock to emphasise whichever component is currently most predictive.

Now, for the loss function, the paper combines IC and MSE loss justifying this as "learns both correct cross-sectional ranking and reasonable return magnitude calibration". The loss is thus

```math
 \mathcal{L} = \mathcal{L}_{IC} + \lambda\mathcal{L}_{MSE}
```

where $`\lambda\in[0, 1]`$

Trading strategy (backtesting, as in the paper):
At the end of each day, ACT outputs a score vector $`\hat{y}\in\\mathbb{R}^N`$ for all $`N`$ stocks.
The strategy is a long-only TopK-Dropout portfolio ($`K=50`$, drop $`N_{\text{drop}}=5`$ each rebalance).
Hence, my Trader class will:
- Ranks all stocks by their predicted score.
- Buys the top-K stocks not already held (or re-ranks existing holdings).
- Sells positions that have fallen out of the top-K window.
- Returns a list of (stockname, trade_qty, price_limit) tuples where
    - trade_qty > 0 -> BUY  trade_qty shares at a price ≤ price_limit
    - trade_qty < 0 -> SELL |trade_qty| shares at a price ≥ price_limit

Dependencies: torch, torch.nn, numpy, scipy (optional for IC metric)


Simple pipeline: data feeds into trainer, which uses ACT model to inform the decisions of the trader.