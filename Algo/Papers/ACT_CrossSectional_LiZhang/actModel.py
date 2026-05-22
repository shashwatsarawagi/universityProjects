import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# Temporal Component Decomposition (TCD)
class TemporalComponentDecomposition(nn.Module):
    def __init__(self, trend_window: int = 10, fluct_window: int = 3):
        super().__init__()
        self.tau   = trend_window   # wider window for trend
        self.sigma = fluct_window   # narrower window for fluctuation

    def _causal_moving_average(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """
        Compute a causal (i.e. no look-ahead) moving average along dim=0 (time).

        x     : shape (T, N, F)
        window: integer MA window
        returns shape (T, N, F) - same as input, boundary filled with x[0]
        """
        T, N, F = x.shape
        out = torch.zeros_like(x)
        for t in range(T):
            lo  = max(0, t - window + 1)
            # average over [lo, t] inclusive in time dimension
            out[t] = x[lo: t + 1].mean(dim=0)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape (T, N, F)
            Raw multi-variate stock feature tensor over a look-back window.

        Returns
        trend  : (T, N, F) - slow-moving, transferable across related stocks
        fluct  : (T, N, F) - medium-speed, stock-local oscillations
        shock  : (T, N, F) - fast/sparse, idiosyncratic event disturbances
        """
        trend  = self._causal_moving_average(x, self.tau)
        detrend = x - trend                                   # remove trend
        fluct  = self._causal_moving_average(detrend, self.sigma)
        shock  = x - trend - fluct                            # residual
        return trend, fluct, shock


# Progressive Structural Purification Encoder (PSPE)

class GraphConvLayer(nn.Module):
    """
    Simple symmetric graph convolution  H = A_norm @ X @ W
    (GCN-style, as in Kipf & Welling 2017).
    The adjacency matrix is expected to be pre-normalised (row-stochastic or
    symmetric-normalised) by the caller.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (N, in_dim)
        adj : (N, N)  – normalised adjacency
        """
        # Aggregate neighbours then project
        agg = torch.matmul(adj, x)          # (N, in_dim)
        return self.linear(agg)             # (N, out_dim)


class ProgressiveStructuralPurificationEncoder(nn.Module):
    """
    Three-stage pipeline on the trend component's last time-step snapshot:

    Stage 1 - Static Relation Purification
      For each static graph r ∈ {industry, region}:
        h_r = LeakyReLU( GCN_r(x0, A_r) )
        f_r = LeakyReLU( W_f_r @ h_r )       forward signal  (useful)
        b_r =            W_b_r @ h_r          backward signal (interference)
      purified_residual  u  = x0 - Σ_r b_r   subtract structural interference
      z_static = W_s @ [f_ind ; f_reg]        concatenate forward signals

    Stage 2 - Dynamic Structural Refinement
      Build a k-NN graph from cosine similarities of projected u.
      Apply GAT on that sparse dynamic graph → z_dynamic.

    Stage 3 - Gate Fusion
      g = sigma( MLP([z_static ; z_dynamic]) )
      z_trend = LayerNorm( z_static + g ⊙ z_dynamic )
    """

    def __init__(
        self,
        in_dim:       int,
        hidden_dim:   int,
        n_static_graphs: int = 2,   # industry + region
        knn_k:        int = 10,     # neighbours for dynamic graph
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.knn_k       = knn_k
        self.hidden_dim  = hidden_dim
        self.n_static    = n_static_graphs


        ## Step 1
        # Eq 4.2
        self.proj        = nn.Linear(in_dim, hidden_dim) #Projection onto hidden layers
        self.ln0         = nn.LayerNorm(hidden_dim) #LN

        # Eq 4.3
        self.gcn_layers   = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim) for _ in range(n_static_graphs)]) #GCN
        self.W_forward    = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_static_graphs)]) #W_f
        self.W_backward   = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_static_graphs)]) #W_b
        # z_static: project concatenated forward signals
        self.W_static     = nn.Linear(hidden_dim * n_static_graphs, hidden_dim) #W_s


        ## Step 2
        # Eq 4.4
        self.W_dyn_proj   = nn.Linear(hidden_dim, hidden_dim)

        # Eq 4.6
        self.W_gat        = nn.Linear(hidden_dim, hidden_dim, bias=False) #W
        self.att_vec      = nn.Parameter(torch.empty(2 * hidden_dim)) #a
        nn.init.xavier_uniform_(self.att_vec.unsqueeze(0)) #initial guess

        # Eq 4.7
        self.W_out_dyn    = nn.Linear(hidden_dim, hidden_dim) #Wo

        ## Step 3
        # Eq 4.8 (We choose an arbitrary MLP of Linear, ReLU, Linear, Sigmoid)
        self.gate_mlp     = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.ln_trend     = nn.LayerNorm(hidden_dim)
        self.leaky        = nn.LeakyReLU(0.2)
        self.drop         = nn.Dropout(dropout)

    def _normalise_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Row-normalise a binary adjacency matrix (add self-loops first)."""
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adj / row_sum

    def _build_knn_graph(self, u_tilde: torch.Tensor) -> torch.Tensor:
        """
        Construct a sparse k-NN adjacency from cosine similarities  (eq. 4.4-4.5).
        u_tilde : (N, hidden_dim)
        returns   (N, N) binary adjacency
        """
        N = u_tilde.size(0)
        # Cosine similarity matrix
        u_norm = F.normalize(u_tilde, dim=-1)        # (N, d)
        sim    = torch.matmul(u_norm, u_norm.T)      # (N, N)
        # For each row keep top-k neighbours (excluding self)
        sim.fill_diagonal_(-1e9)
        _, top_idx = sim.topk(self.knn_k, dim=-1)    # (N, k)
        adj = torch.zeros(N, N, device=u_tilde.device)
        adj.scatter_(1, top_idx, 1.0)
        return adj                                   # (N, N)

    def _gat_pass(self, u_tilde: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Single-head GAT message passing on the dynamic k-NN graph  (eq. 4.6-4.7).
        u_tilde : (N, hidden_dim)
        adj     : (N, N) binary
        """
        N = u_tilde.size(0)
        
        Wh = self.W_gat(u_tilde)                     # (N, d)
        # Compute pairwise attention logits Eq 4.6.1
        Wh_i = Wh.unsqueeze(1).expand(N, N, -1)     # (N, N, d)
        Wh_j = Wh.unsqueeze(0).expand(N, N, -1)     # (N, N, d)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)    # (N, N, 2d)
        e = self.leaky(concat @ self.att_vec)        # (N, N)

        # Mask out non-edges and softmax as Eq 4.6.2
        mask = (adj == 0)
        e    = e.masked_fill(mask, -1e9)
        alpha = F.softmax(e, dim=-1)                 # (N, N)

        # Aggregate Eq 4.7
        H_d = torch.matmul(alpha, Wh)               # (N, d) #inner part of 4.7 sum
        z_d = self.leaky(self.W_out_dyn(H_d))       # (N, d)
        return z_d

    def forward(
        self,
        x_trend: torch.Tensor,                       # (T, N, F)
        static_adjs: List[torch.Tensor],             # list of (N, N) binary matrices
    ) -> torch.Tensor:
        """
        Returns z_trend : (N, hidden_dim) - purified trend embedding per stock.
        """
        
        ## Static Relation Purification
        
        # Use only the final time-step snapshot of the trend component hence x_last = X_T is just the final element
        x_last = x_trend[-1]                         # (N, F)
        x0 = self.ln0(self.proj(x_last))             # (N, d)

        forward_signals = []
        backward_sum    = torch.zeros_like(x0)

        for gcn, Wf, Wb, adj_r in zip(self.gcn_layers, self.W_forward, self.W_backward, static_adjs):
            A_norm = self._normalise_adj(adj_r)       # (N, N)
            h_r    = self.leaky(gcn(x0, A_norm))      # (N, d)
            f_r    = self.leaky(Wf(h_r))              # forward  – useful signal
            b_r    = Wb(h_r)                          # backward – interference estimate
            forward_signals.append(f_r)
            backward_sum = backward_sum + b_r         # accumulate interference

        # Subtract structural interference from x0
        u = x0 - backward_sum                         # eq 4.3

        # Fuse forward signals → z_static
        z_static = self.W_static(torch.cat(forward_signals, dim=-1))  # (N, d)

        ## Dynamic Structural Refinement
        u_tilde = self.leaky(self.W_dyn_proj(u))      # project purified residual
        adj_dyn = self._build_knn_graph(u_tilde)      # (N, N) sparse k-NN
        z_dyn   = self._gat_pass(u_tilde, adj_dyn)   # (N, d) #H_d in paper

        ## Adaptive Gate Fusion
        gate     = self.gate_mlp(torch.cat([z_static, z_dyn], dim=-1))  # (N, d)
        z_trend  = self.ln_trend(z_static + gate * z_dyn)               # (N, d)
        return z_trend

# Fluctuation Component Isolation (FCI)

class FluctuationComponentIsolation(nn.Module):
    """
    Captures short-horizon oscillatory dynamics through a stock-local Temporal
    Convolutional Network (TCN).  Because convolutions are applied independently
    to each stock, local oscillations stay isolated from cross-stock propagation.

    Pipeline:
      x_fluct → Linear projection → LayerNorm → Reshape to (N, d, T)
              → Gated TCN (three parallel Conv1d heads)
              → ReLU + Dropout
              → take last time-step  → z_fluct  ∈ R^{N x d}

    The gated TCN is:  Z = DropReLU( P ⊙ sigma(Q) + R )
    where P, Q, R are three independent dilated Conv1d streams.
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        T:          int,           # look-back window length
        kernel_size: int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.T = T

        # Initial projection
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.ln   = nn.LayerNorm([hidden_dim, T])   # applied after reshape

        # Three conv streams (same causal padding so T is preserved)
        pad = kernel_size - 1
        self.conv_P = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad, dilation=1)
        self.conv_Q = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad, dilation=1)
        self.conv_R = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad, dilation=1)

        self.drop   = nn.Dropout(dropout)
        self.relu   = nn.ReLU()

    def forward(self, x_fluct: torch.Tensor) -> torch.Tensor:
        """
        x_fluct : (T, N, F)
        returns   z_fluct (N, hidden_dim)
        """
        T, N, F = x_fluct.shape

        # Project each (T, F) slice per stock  → (N, T, d)
        x_proj = self.proj(x_fluct.permute(1, 0, 2))       # (N, T, d)
        # Reshape to (N, d, T) for Conv1d
        F_in = x_proj.permute(0, 2, 1)                     # (N, d, T)
        F_in = self.ln(F_in)

        # Causal conv: trim future-leaking padding on the right
        P = self.conv_P(F_in)[..., :T]                     # (N, d, T)
        Q = self.conv_Q(F_in)[..., :T]
        R = self.conv_R(F_in)[..., :T]

        # Gated activation: element-wise gate Q controls P, additive skip R
        Z = self.drop(self.relu(P * torch.sigmoid(Q) + R)) # (N, d, T)

        # Take the representation at the final time step
        z_fluct = Z[:, :, -1]                              # (N, d)
        return z_fluct


# Shock Component Isolation (SCI)

class ShockComponentIsolation(nn.Module):
    """
    Models idiosyncratic event disturbances via a counterfactual buffer.
    Since shocks are sparse and high-variance, the model is trained to compare
    the *raw* shock at the final time-step against a *smoothed counterfactual*
    (what the shock sequence would look like without extreme outliers).

    Pipeline:
      S_T      = x_shock[-1]                    raw shock embedding
      S_bar_T  = smooth CausalMA over shock     counterfactual
      x        = LN( Proj(S_T) )
      x'       = LN( Proj(S_bar_T) )
      z_shock  = W2 · Dropout( LeakyReLU( W1 · [x ; x'] ) )
    """

    def __init__(
        self,
        in_dim:         int,
        hidden_dim:     int,
        smooth_window:  int = 5,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.ws   = smooth_window
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.ln   = nn.LayerNorm(hidden_dim)

        # Two-layer stock-local MLP on concatenated [raw; counterfactual]
        self.W1   = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W2   = nn.Linear(hidden_dim, hidden_dim)
        self.leaky  = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    def _counterfactual(self, shock: torch.Tensor) -> torch.Tensor:
        """
        Causal moving average over the shock sequence (eq. 4.11).
        shock : (T, N, F) → returns (T, N, F) smoothed
        """
        T = shock.size(0)
        out = torch.zeros_like(shock)
        for t in range(T):
            lo       = max(0, t - self.ws + 1)
            out[t]   = shock[lo: t + 1].mean(dim=0)
        return out

    def forward(self, x_shock: torch.Tensor) -> torch.Tensor:
        """
        x_shock : (T, N, F)
        returns   z_shock (N, hidden_dim)
        """
        # Smoothed counterfactual sequence Eq 4.11
        shock_smooth = self._counterfactual(x_shock)    # (T, N, F)

        # Take final-step snapshots  Eq 4.12
        s_raw    = x_shock[-1]                          # (N, F)
        s_smooth = shock_smooth[-1]                     # (N, F)

        x  = self.ln(self.proj(s_raw))                 # (N, d)
        xp = self.ln(self.proj(s_smooth))              # (N, d)

        # Contrastive two-layer MLP  (eq. 4.13)
        concat  = torch.cat([x, xp], dim=-1)           # (N, 2d)
        z_shock = self.W2(self.drop(self.leaky(self.W1(concat))))  # (N, d)
        return z_shock


# Adaptive Component Fusion (ACF)

class AdaptiveComponentFusion(nn.Module):
    """
    Stacks the three branch embeddings and computes a per-stock soft-attention
    weight over them.  Each stock may weight trend, fluctuation, and shock
    information differently depending on its current regime.

    Attention  s_{i,c} = W2 · tanh( W1 · z_{i,c} + b1 )
    Weight     sigma_{i,c} = softmax over c
    Output     z_i     = Σ_c alpha_{i,c} · z_{i,c}
    """

    def __init__(self, hidden_dim: int, n_components: int = 3):
        super().__init__()
        self.Wa1 = nn.Linear(hidden_dim, hidden_dim)
        self.Wa2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        z_trend: torch.Tensor,   # (N, d)
        z_fluct: torch.Tensor,   # (N, d)
        z_shock: torch.Tensor,   # (N, d)
    ) -> torch.Tensor:
        """
        Returns fused embedding z ∈ R^{N×d}.
        """
        # Stack into (N, 3, d)  Eq 4.14
        C = torch.stack([z_trend, z_fluct, z_shock], dim=1)   # (N, 3, d)

        # Attention scores  (N, 3, 1) → (N, 3)
        s = self.Wa2(torch.tanh(self.Wa1(C))).squeeze(-1)     # (N, 3)
        alpha = F.softmax(s, dim=-1).unsqueeze(-1)             # (N, 3, 1)

        # Weighted sum  (N, d)
        z = (alpha * C).sum(dim=1)                             # (N, d)
        return z


# Full ACT Model

class ACTModel(nn.Module):
    """
    Anti-CrossTalk (ACT) framework as a whole

    Input
    -----
    x           : (T, N, F)   - look-back feature tensor
    industry_adj: (N, N)      - binary industry adjacency
    region_adj  : (N, N)      - binary region adjacency

    Output
    ------
    scores : (N,) - cross-sectional return prediction scores (higher = better)
    """

    def __init__(
        self,
        n_features:    int,
        hidden_dim:    int   = 64,
        T:             int   = 40,     # look-back window
        trend_window:  int   = 10,
        fluct_window:  int   = 3,
        shock_smooth:  int   = 5,
        knn_k:         int   = 10,
        tcn_kernel:    int   = 3,
        dropout:       float = 0.1,
        lambda_mse:    float = 0.5,    # weight on MSE loss Eq. 4.19
    ):
        super().__init__()
        self.lambda_mse = lambda_mse

        # Module 1: Temporal Component Decomposition
        self.tcd = TemporalComponentDecomposition(
            trend_window=trend_window,
            fluct_window=fluct_window,
        )

        # Module 2: PSPE (trend branch, uses graphs)
        self.pspe = ProgressiveStructuralPurificationEncoder(
            in_dim=n_features,
            hidden_dim=hidden_dim,
            n_static_graphs=2,         # industry + region
            knn_k=knn_k,
            dropout=dropout,
        )

        # Module 3: FCI (fluctuation branch, stock-local TCN)
        self.fci = FluctuationComponentIsolation(
            in_dim=n_features,
            hidden_dim=hidden_dim,
            T=T,
            kernel_size=tcn_kernel,
            dropout=dropout,
        )

        # Module 4: SCI (shock branch, counterfactual buffer)
        self.sci = ShockComponentIsolation(
            in_dim=n_features,
            hidden_dim=hidden_dim,
            smooth_window=shock_smooth,
            dropout=dropout,
        )

        # Module 5: Adaptive Component Fusion
        self.acf = AdaptiveComponentFusion(hidden_dim=hidden_dim)

        # Final predictor: Linear -> scalar score per stock
        self.predictor = nn.Linear(hidden_dim, 1) #W_out

    def forward(
        self,
        x:            torch.Tensor,               # (T, N, F)
        industry_adj: torch.Tensor,               # (N, N)
        region_adj:   torch.Tensor,               # (N, N)
    ) -> torch.Tensor:
        """
        Returns predicted return scores y_hat ∈ R^N  Eq 4.16.3
        """
        trend, fluct, shock = self.tcd(x)          # each (T, N, F)


        z_trend = self.pspe(trend, [industry_adj, region_adj])  # (N, d)
        z_fluct = self.fci(fluct)                               # (N, d)
        z_shock = self.sci(shock)                               # (N, d)

        z = self.acf(z_trend, z_fluct, z_shock)                 # (N, d)

        scores = self.predictor(z).squeeze(-1)                  # (N,)
        return scores

    def compute_loss(
        self,
        scores: torch.Tensor,       # (N,) model predictions
        returns: torch.Tensor,      # (N,) realised next-day returns
    ) -> torch.Tensor:
        """
        Combined IC + MSE loss  (Section 4.6, eq. 4.17-4.19).

        IC Loss:  1 - Pearson correlation(ŷ, clip(y, -0.1, 0.1))
          This directly optimises the cross-sectional rank quality.

        MSE Loss: mean squared error on clipped returns
          Provides auxiliary supervision on return magnitude.

        Final:  L = L_IC + lambda * L_MSE
        """
        eps  = 1e-8
        # Clamps returns to [-10%, +10%] to reduce outlier influence precursor to IC loss
        y_c  = returns.clamp(-0.10, 0.10)

        # IC loss = 1 minus Pearson correlation
        yhat_mean = scores.mean()
        y_mean    = y_c.mean()

        num   = ((scores - yhat_mean) * (y_c - y_mean)).sum()
        denom = (((scores - yhat_mean) ** 2).sum() + eps).sqrt() * (((y_c - y_mean) ** 2).sum() + eps).sqrt()
        ic    = num / denom
        loss_ic  = 1.0 - ic                        # minimise → maximise IC

        # MSE loss on clamped returns  Eq 4.18
        loss_mse = F.mse_loss(scores, y_c)

        return loss_ic + self.lambda_mse * loss_mse
    
def train_act(
    model:         ACTModel,
    data_loader,                            # yields (x, y, ind_adj, reg_adj)
    n_epochs:      int   = 20,
    lr:            float = 1e-3,
    device:        str   = "cpu",
) -> List[float]:
    """
    Train the ACT model.

    The data_loader should yield batches of:
      x        : (T, N, F)  - look-back feature tensor
      y        : (N,)       - next-day VWAP return  (eq. 3.2)
      ind_adj  : (N, N)     - industry adjacency
      reg_adj  : (N, N)     - region adjacency

    Returns
    epoch_losses : list of average loss per epoch
    """
    model    = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.5)
    model.train()

    epoch_losses = []
    for epoch in range(n_epochs):
        batch_losses = []
        for x, y, ind_adj, reg_adj in data_loader:
            x       = x.to(device).float()
            y       = y.to(device).float()
            ind_adj = ind_adj.to(device).float()
            reg_adj = reg_adj.to(device).float()

            optimiser.zero_grad()
            scores = model(x, ind_adj, reg_adj)       # (N,)
            loss   = model.compute_loss(scores, y)
            loss.backward()

            # Gradient clipping for stability in financial time series
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        avg_loss = float(np.mean(batch_losses))
        epoch_losses.append(avg_loss)
        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs}  |  Loss: {avg_loss:.6f}")

    return epoch_losses

class StockDataset(Dataset):
    """
    Each sample is one trading day's snapshot.
    X_raw : (total_days, N, F)  — your full feature history
    y_raw : (total_days, N)     — next-day returns
    """
    def __init__(self, X_raw, y_raw, ind_adj, reg_adj, T=40):
        self.X = X_raw
        self.y = y_raw
        self.ind_adj = torch.tensor(ind_adj, dtype=torch.float32)
        self.reg_adj = torch.tensor(reg_adj, dtype=torch.float32)
        self.T = T

    def __len__(self):
        return len(self.X) - self.T

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx : idx + self.T], dtype=torch.float32)  # (T, N, F)
        y = torch.tensor(self.y[idx + self.T],        dtype=torch.float32)  # (N,)
        return x, y, self.ind_adj, self.reg_adj