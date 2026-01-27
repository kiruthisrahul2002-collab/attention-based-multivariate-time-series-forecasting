# =============================================================================
# ATTENTION-BASED MULTIVARIATE TIME SERIES FORECASTING
# Complete Runnable Implementation
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------------------------------------------
# 1. REPRODUCIBILITY
# -----------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 2. DATASET GENERATION (5+ FEATURES, 1000+ SAMPLES)
# -----------------------------------------------------------------------------
def generate_time_series(n_steps=3000):
    t = np.arange(n_steps)

    trend = 0.003 * t
    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / 168)
    monthly = np.sin(2 * np.pi * t / 720)
    nonlinear = np.sin(0.01 * t) * np.cos(0.03 * t)
    noise = np.random.normal(0, 0.3, n_steps)

    target = trend + daily + weekly + monthly + nonlinear + noise

    exog_1 = np.cos(2 * np.pi * t / 12)
    exog_2 = np.random.normal(0, 1, n_steps)
    exog_3 = np.sin(0.005 * t)
    exog_4 = np.random.normal(0, 0.5, n_steps)

    return np.column_stack([target, exog_1, exog_2, exog_3, exog_4])

data = generate_time_series()

# -----------------------------------------------------------------------------
# 3. SEQUENCE CREATION
# -----------------------------------------------------------------------------
LOOKBACK = 48
HORIZON = 12

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon, 0])
    return np.array(X), np.array(y)

# -----------------------------------------------------------------------------
# 4. WALK-FORWARD VALIDATION
# -----------------------------------------------------------------------------
def walk_forward_split(data, train_size, step):
    for i in range(train_size, len(data) - step, step):
        yield data[:i], data[i:i+step]

# -----------------------------------------------------------------------------
# 5. MODELS
# -----------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(d_model, HORIZON)

    def forward(self, x):
        x = self.proj(x)
        attn_out = self.encoder(x)
        return self.fc(attn_out[:, -1, :]), attn_out

# -----------------------------------------------------------------------------
# 6. TRAINING FUNCTION
# -----------------------------------------------------------------------------
def train_model(model, X, y, epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)[0] if isinstance(model, TransformerModel) else model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    return model

# -----------------------------------------------------------------------------
# 7. MASE METRIC
# -----------------------------------------------------------------------------
def mase(y_true, y_pred, training_series):
    naive = np.abs(training_series[1:] - training_series[:-1]).mean()
    return np.mean(np.abs(y_true - y_pred)) / naive

# -----------------------------------------------------------------------------
# 8. WALK-FORWARD TRAINING & EVALUATION
# -----------------------------------------------------------------------------
scaler = StandardScaler()
lstm_errors, transformer_errors = [], []

for train_data, test_data in walk_forward_split(data, train_size=2000, step=HORIZON):

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled, LOOKBACK, HORIZON)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK, HORIZON)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    lstm = train_model(LSTMModel(5), X_train, y_train)
    transformer = train_model(TransformerModel(5), X_train, y_train)

    with torch.no_grad():
        lstm_pred = lstm(X_test).numpy().flatten()
        transformer_pred, attn = transformer(X_test)
        transformer_pred = transformer_pred.numpy().flatten()

    y_true = y_test.numpy().flatten()

    lstm_errors.append([
        mean_absolute_error(y_true, lstm_pred),
        np.sqrt(mean_squared_error(y_true, lstm_pred)),
        mase(y_true, lstm_pred, train_data[:, 0])
    ])

    transformer_errors.append([
        mean_absolute_error(y_true, transformer_pred),
        np.sqrt(mean_squared_error(y_true, transformer_pred)),
        mase(y_true, transformer_pred, train_data[:, 0])
    ])

# -----------------------------------------------------------------------------
# 9. RESULTS
# -----------------------------------------------------------------------------
lstm_results = np.mean(lstm_errors, axis=0)
transformer_results = np.mean(transformer_errors, axis=0)

print("\nFINAL RESULTS (AVERAGED WALK-FORWARD)\n")
print("LSTM -> MAE:", lstm_results[0], " RMSE:", lstm_results[1], " MASE:", lstm_results[2])
print("Transformer -> MAE:", transformer_results[0], " RMSE:", transformer_results[1], " MASE:", transformer_results[2])

# -----------------------------------------------------------------------------
# 10. ATTENTION VISUALIZATION
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.imshow(attn[0].detach().numpy(), aspect="auto", cmap="viridis")
plt.colorbar(label="Attention Weight")
plt.xlabel("Feature Dimension")
plt.ylabel("Time Steps")
plt.title("Attention Heatmap (Transformer Encoder)")
plt.show()
