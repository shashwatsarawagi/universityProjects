from actModel import ACTModel, train_act, StockDataset
from trader import Trader
from torch.utils.data import DataLoader
import numpy as np
import torch

N, F, T, DAYS = 500, 6, 40, 500
X_raw  = np.random.randn(DAYS, N, F).astype(np.float32)   # replace with real OHLCV
y_raw  = np.random.randn(DAYS, N).astype(np.float32)      # replace with real returns
ind_adj = (np.random.rand(N, N) < 0.05).astype(float)
reg_adj = (np.random.rand(N, N) < 0.03).astype(float)

stock_names = [f"STOCK_{i:03d}" for i in range(N)]

dataset = StockDataset(X_raw, y_raw, ind_adj, reg_adj, T=T)
loader  = DataLoader(dataset, batch_size=1, shuffle=True)

model = ACTModel(n_features=F, hidden_dim=64, T=T)
losses = train_act(model, loader, n_epochs=20, lr=1e-3, device="cpu")

# Save weights for the Trader to load later
torch.save(model.state_dict(), "act_weights.pt")


# Instantiate trader
trader = Trader(
    stock_names=stock_names,
    industry_adj=ind_adj,
    region_adj=reg_adj,
    n_features=F,
    hidden_dim=32,
    T=T,
    top_k=5,      # keep small for demo
    n_drop=1,
    target_value=100_000,
    device="cpu",
)

trader.load_weights("act_weights.pt")


print("=" * 60)
print("ACT Framework - Smoke Test")
print("=" * 60)
print(f"Universe : {N} stocks   |   Look-back : {T} days   |   Features : {F}")
print()

# Simulate 3 trading days
for day in range(1, 4):
    # Generate a synthetic (N, F) feature row for today
    feature_row = np.random.randn(N, F).astype(np.float32)
    # Last column (index 3) is the "close price" – must be positive
    feature_row[:, 3] = np.abs(feature_row[:, 3]) * 10 + 10   # prices in [10, 30]

    trader.update_price_buffer(feature_row)

    # current_prices = today's close
    prices = feature_row[:, 3]

    orders = trader.run(current_prices=prices)

    print(f"Day {day}: {len(orders)} order(s)")
    for name, trade, price in orders:
        direction = "BUY " if trade > 0 else "SELL"
        print(f"  {direction}  {abs(trade):>6} shares of {name}  "
                f"limit={price:>8.2f}")
    print()

print(f"Final portfolio: {dict(list(trader.portfolio.items())[:5])} ...")