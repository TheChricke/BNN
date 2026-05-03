# pip install pandas numpy scikit-learn torch

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from math import log, pi

SQLITE_DB_PATH = "binance_data.db"
TABLE_NAME = "alpha_klines"
LOOKBACK = 30       # time dimension (lookback)
MA_LONG = 30        # moving average 30t
MA_SHORT = 2        # moving average 2t
VOL_WINDOW = 30     # volatility window
BATCH_SIZE = 64
EPOCHS = 1
LR = 1e-3
PRIOR_STD = 1.0     # prior std for weights (N(0, prior_std^2))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


#load data from sqlite
def load_kline_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# Feature engineering
def compute_features(df, ma_long=30, ma_short=2, vol_window=30):
    # expect df has 'close_price', 'volume', 'number_of_trades'
    # try to detect timestamp-like column for ordering; otherwise use row index
    timestamp_cols = [c for c in df.columns if c.lower() in ("close_time", "ts", "time", "open_time", "date", "datetime")]
    if len(timestamp_cols) > 0:
        time_col = timestamp_cols[0]
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # compute log returns
    df['log_ret'] = np.log(df['close_price']).diff()
    # moving averages
    df[f'ma_{ma_long}'] = df['close_price'].rolling(ma_long, min_periods=1).mean()
    df[f'ma_{ma_short}'] = df['close_price'].rolling(ma_short, min_periods=1).mean()
    # volatility as rolling std of log returns
    df[f'vol_{vol_window}'] = df['log_ret'].rolling(vol_window, min_periods=1).std().fillna(0.0)
    # drop NaNs (first few rows may have NaN log_ret)
    df = df.dropna(subset=['close_price', 'volume', 'number_of_trades']).reset_index(drop=True)
    return df

# Make sequences (per token)
def make_sequences(df, lookback=30, feature_cols=None, token_col='alpha_token_id'):
    """
    Returns X: (N_sequences, lookback, n_features) and y: (N_sequences,)
    Label y is 1 if close_price at t+1 > close_price at t (i.e., up next period), else 0.
    Sequences are created per token_id to avoid mixing across tokens.
    """
    if feature_cols is None:
        feature_cols = ['close_price', 'volume', 'number_of_trades', f'ma_{MA_LONG}', f'ma_{MA_SHORT}', f'vol_{VOL_WINDOW}']

    sequences = []
    labels = []
    token_ids = []
    grouped = df.groupby(token_col)
    for token, g in grouped:
        g = g.sort_index().reset_index(drop=True)  # already sorted earlier, but safe
        values = g[feature_cols].values
        closes = g['close_price'].values
        n = len(g)
        # create sequences s.t. sequence covers [i, i+lookback-1] and label is whether close at i+lookback > close at i+lookback-1
        for i in range(0, n - lookback - 0):  # label uses next period after sequence end: index i+lookback
            seq = values[i:i+lookback]
            # next close is at i+lookback (one step after sequence end)
            next_close_idx = i + lookback
            if next_close_idx >= n:
                break
            curr_close = closes[i + lookback - 1]
            next_close = closes[next_close_idx]
            label = 1 if next_close > curr_close else 0
            sequences.append(seq)
            labels.append(label)
            token_ids.append(token)
    X = np.array(sequences)   # shape (N, lookback, features)
    y = np.array(labels).astype(np.int64)# shape (N, 1) ?
    tokens = np.array(token_ids)
    return X, y, tokens, feature_cols

# Dataset and scaling
class KlineSequenceDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, T, F)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def standard_scale_sequences(X_train, X_val=None):
    # fit scaler on flattened training sets per feature (we scale each feature independently across time)
    N, T, F = X_train.shape
    flat = X_train.reshape(-1, F)
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    X_train_scaled = flat_scaled.reshape(N, T, F)
    if X_val is not None:
        Nv = X_val.shape[0]
        flatv = X_val.reshape(-1, F)
        flatv_scaled = scaler.transform(flatv)
        X_val_scaled = flatv_scaled.reshape(Nv, T, F)
        return X_train_scaled, X_val_scaled, scaler
    return X_train_scaled, None, scaler


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # variational params for weights
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -3.0)  # rho -> softplus for sigma
        # variational params for bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -3.0)
        self.prior_mean = 0.0
        self.prior_std = prior_std
        # store last sampled weights for kl computation and forward
        self._eps_weight = None
        self._eps_bias = None

    def forward(self, x, sample=True):
        # x: (batch, in_features)
        if self.training or sample:
            # reparameterization trick
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)
            eps_w = torch.randn_like(self.weight_mu)
            eps_b = torch.randn_like(self.bias_mu)
            w = self.weight_mu + weight_sigma * eps_w
            b = self.bias_mu + bias_sigma * eps_b
            # save for KL
            self._last_w = (self.weight_mu, weight_sigma)
            self._last_b = (self.bias_mu, bias_sigma)
            return F.linear(x, w, b)
        else:
            # use mean
            return F.linear(x, self.weight_mu, self.bias_mu)

    def kl_divergence(self):
        # KL(q||p) where q ~ N(mu, sigma^2), p ~ N(0, prior_std^2)
        w_mu, w_sigma = self._last_w
        b_mu, b_sigma = self._last_b
        # analytical KL for Gaussians (elementwise)
        def kl_elem(q_mu, q_sigma, p_sigma):
            # KL = log(p_sigma/q_sigma) + (q_sigma^2 + q_mu^2)/(2 p_sigma^2) - 1/2
            term = torch.log(p_sigma / (q_sigma + 1e-12)) + (q_sigma**2 + q_mu**2) / (2.0 * p_sigma**2) - 0.5
            return term.sum()
        p_sigma = torch.tensor(self.prior_std, device=w_mu.device)
        k_w = kl_elem(w_mu, w_sigma, p_sigma)
        k_b = kl_elem(b_mu, b_sigma, p_sigma)
        return k_w + k_b


class BNNClassifier(nn.Module):
    def __init__(self, lookback, n_features, hidden_units_over_features=2, prior_std=1.0):
        """
        Hidden units = n_features + hidden_units_over_features (per user's spec "2 more neurons than the input features").
        Input is expected as (batch, time, features). We'll flatten to (batch, time * features) then apply Bayesian FC layers.
        """
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features
        self.input_dim = lookback * n_features
        hidden_units = 56

        # Bayesian layers, layers can be added or removed easily
        self.fc1 = BayesianLinear(self.input_dim, hidden_units, prior_std=prior_std)
        self.fc2 = BayesianLinear(hidden_units, hidden_units, prior_std=prior_std)
        self.out = BayesianLinear(hidden_units, 1, prior_std=prior_std)

    def forward(self, x, sample=True):
        # x: (batch, time, features)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # flatten time/features
        x = F.relu(self.fc1(x, sample=sample))
        x = F.relu(self.fc2(x, sample=sample))
        logits = self.out(x, sample=sample)  # (batch, 1)
        probs = torch.sigmoid(logits)
        return probs

    def kl_divergence(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.out.kl_divergence()


def train_bnn(model, train_loader, val_loader=None, epochs=10, lr=1e-3, prior_scale=1.0, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    N_data = len(train_loader.dataset)
    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_kl = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            probs = model(xb, sample=True)
            # likelihood: Bernoulli => binary cross entropy
            bce = F.binary_cross_entropy(probs, yb, reduction='sum')  # sum over batch
            # KL from variational layers
            kl = model.kl_divergence()
            # ELBO negative: (bce + (kl / N_data))
            loss = bce + kl / float(N_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_bce += bce.item()
            total_kl += kl.item()
        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)

        # validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    probs = model(xb, sample=False)  # use mean
                    bce = F.binary_cross_entropy(probs, yb, reduction='sum')
                    kl = model.kl_divergence()
                    loss = bce + kl / float(N_data)
                    val_loss += loss.item()
            history["val_loss"].append(val_loss / len(val_loader.dataset))
            print(f"Epoch {ep+1}/{epochs}  train_loss={avg_loss:.6f}  train_bce_sum={total_bce:.3f}  kl_sum={total_kl:.3f}  val_loss={history['val_loss'][-1]:.6f}")
        else:
            print(f"Epoch {ep+1}/{epochs}  train_loss={avg_loss:.6f}  train_bce_sum={total_bce:.3f}  kl_sum={total_kl:.3f}")
    return history

def main():
    print("Loading data...")
    df = load_kline_table(SQLITE_DB_PATH, TABLE_NAME)
    if df.shape[0] == 0:
        raise RuntimeError("No rows found in table. Check your DB path and table.")

    print("Computing engineered features...")
    df = compute_features(df, ma_long=MA_LONG, ma_short=MA_SHORT, vol_window=VOL_WINDOW)

    print("Creating sequences...")
    X, y, tokens, feature_cols = make_sequences(df, lookback=LOOKBACK, token_col='alpha_token_id')
    print(f"Sequences shape: {X.shape}; labels shape: {y.shape}; features used: {feature_cols}")

    # If too few samples, warn
    if X.shape[0] < 10:
        print("WARNING: very few sequences generated. Check lookback or data continuity per token.")

    # train / val split
    N = X.shape[0]
    print(np)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    print("Scaling features (standard scaler fitted on training set)...")
    X_train_scaled, X_val_scaled, scaler = standard_scale_sequences(X_train, X_val)

    train_ds = KlineSequenceDataset(X_train_scaled, y_train)
    val_ds = KlineSequenceDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_features = X.shape[2]
    print(f"Building BNN: lookback={LOOKBACK}, n_features={n_features}")
    model = BNNClassifier(lookback=LOOKBACK, n_features=n_features, hidden_units_over_features=2, prior_std=PRIOR_STD)
    print("Training...")
    history = train_bnn(model, train_loader, val_loader=val_loader, epochs=EPOCHS, lr=LR, prior_scale=PRIOR_STD, device=DEVICE)
    print("Done training.")

    T = 50  # number of samples (you can tune this)

    with torch.no_grad():
        xb, yb = next(iter(val_loader)) #run test only on first batch
        xb = xb.to(DEVICE)

        # Collect multiple stochastic forward passes
        samples = torch.stack([model(xb, sample=True) for _ in range(T)])  
        # shape: (T, batch_size, 1) or (T, batch_size)

        # Compute statistics
        mean_probs = samples.mean(dim=0).cpu().numpy().squeeze()
        std_probs  = samples.std(dim=0).cpu().numpy().squeeze()

        preds = (mean_probs >= 0.5).astype(int)

        #show forst 10 samples in the batch
        print("Mean probs :", mean_probs[:10])
        print("Uncertainty:", std_probs[:10])
        print("Preds      :", preds[:10])
        print("True       :", yb.numpy()[:10].squeeze().astype(int))

    torch.save(model.state_dict(), "bnn_model_state.pt")
if __name__ == "__main__":
    main()
