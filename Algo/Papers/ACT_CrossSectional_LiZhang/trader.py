import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from actModel import ACTModel

class Trader:
    def __init__(
        self,
        stock_names:   List[str],
        industry_adj:  np.ndarray,     # (N, N) binary industry graph
        region_adj:    np.ndarray,     # (N, N) binary region graph
        n_features:    int   = 6,      # OHLCV + VWAP (simplified; Alpha158 has 158)
        hidden_dim:    int   = 64,
        T:             int   = 40,     # look-back window (days)
        top_k:         int   = 50,     # portfolio size
        n_drop:        int   = 5,      # stocks to rotate out each step
        target_value:  float = 1e6,    # total portfolio target value (currency units)
        device:        str   = "cpu",
    ):
        self.stock_names = stock_names
        self.N = len(stock_names)
        self.T = T
        self.top_k = top_k
        self.n_drop = n_drop
        self.target_value = target_value
        self.device = torch.device(device)

        # Convert numpy adjacency matrices to tensors
        self.industry_adj = torch.tensor(industry_adj, dtype=torch.float32, device=self.device)
        self.region_adj = torch.tensor(region_adj, dtype=torch.float32, device=self.device)

        # Initialise the ACT model
        self.model = ACTModel(n_features=n_features, hidden_dim=hidden_dim, T=T).to(self.device)
        self.model.eval()  # inference mode by default

        # Current portfolio: dict of {stock_name: shares_held}
        self.portfolio: Dict[str, int] = {}

        # Price buffer: stores the most recent T rows of shape (N, F)
        # In production this would be fed from a live data feed.
        self._price_buffer: np.ndarray | None = None   # (T, N, F)


    def update_price_buffer(self, new_row: np.ndarray) -> None:
        if self._price_buffer is None:
            # First call: initialise buffer with repeated first row
            self._price_buffer = np.repeat(new_row[np.newaxis], self.T, axis=0)
        else:
            # Slide the window forward: drop oldest, append new
            self._price_buffer = np.concatenate(
                [self._price_buffer[1:], new_row[np.newaxis]], axis=0
            )

    def load_weights(self, path: str) -> None:
        """Load pre-trained ACT model weights from a .pt file."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _get_scores(self) -> np.ndarray:
        x = torch.tensor(self._price_buffer, dtype=torch.float32, device=self.device)              # (T, N, F)

        with torch.no_grad():
            scores = self.model(x, self.industry_adj, self.region_adj)  # (N,)

        return scores.cpu().numpy()

    def _compute_trades(
        self,
        scores:         np.ndarray,    # (N,) predicted return scores
        current_prices: np.ndarray,    # (N,) last observed price per stock
    ) -> List[Tuple[str, int, float]]:
        N = self.N

        # 1) Rank stocks by score (descending)
        rank = np.argsort(-scores) # indices, best first
        score_rank = np.empty(N, dtype=int)
        score_rank[rank] = np.arange(N) # rank[i] = rank of stock i

        # 2) Identify the target portfolio  (top-K indices)
        target_set = set(rank[:self.top_k].tolist())

        # 3) Determine sells  (stocks to exit)
        # Drop stocks currently held whose rank has fallen beyond top_k + n_drop
        to_sell = []
        for name, shares in list(self.portfolio.items()):
            idx = self.stock_names.index(name)
            if score_rank[idx] >= self.top_k + self.n_drop:
                to_sell.append(idx)

        # 4) Determine buys  (new entries into target portfolio)
        held_indices = {self.stock_names.index(n) for n in self.portfolio}
        to_buy = [i for i in target_set if i not in held_indices]

        # 5) Compute order sizes  (equal notional per position)
        # Expected portfolio size after rebalance
        expected_size = max(1, len(held_indices) - len(to_sell) + len(to_buy))
        notional_per_stock = self.target_value / expected_size

        orders: List[Tuple[str, int, float]] = []

        # SELL orders: sell entire position, limit price = current price
        # (trade < 0  -> sell at min price of p, i.e. do not accept less than p)
        for idx in to_sell:
            name   = self.stock_names[idx]
            shares = self.portfolio.get(name, 0)
            price  = float(current_prices[idx])
            if shares > 0:
                orders.append((name, -shares, price))
                del self.portfolio[name]

        # BUY orders: buy into target notional, limit price = current price
        # (trade > 0  -> buy at max price of p, i.e. do not pay more than p)
        for idx in to_buy:
            name  = self.stock_names[idx]
            price = float(current_prices[idx])
            if price <= 0:
                continue
            shares = int(notional_per_stock / price)
            if shares > 0:
                orders.append((name, shares, price))
                self.portfolio[name] = shares

        return orders

    def run(
        self,
        current_prices: np.ndarray | None = None,
    ) -> List[Tuple[str, int, float]]:

        if current_prices is None:
            if self._price_buffer is None:
                raise RuntimeError("No price data available.")
            # Assuming the 4th feature is the close / VWAP price
            current_prices = self._price_buffer[-1, :, 3]

        # Get ACT model scores for all stocks
        scores = self._get_scores()                          # (N,)

        # Compute and return trade orders
        orders = self._compute_trades(scores, current_prices)

        return orders