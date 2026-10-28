"""
train_models.py — Train 4 models (LR, RF, XGBoost, LSTM), evaluate, serialize, and produce comparison artefacts.

Part of: "Enhancing Solar Energy Forecasting and Grid Management Using Explainable AI (XAI)"
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Metric helpers ──────────────────────────────────────────────────────────
def safe_mape(y_true, y_pred, eps=1.0):
    """MAPE with a floor on the denominator to avoid division-by-zero."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(y_true, y_pred, model_name="model"):
    """Return dict of MAE, RMSE, MAPE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "R2": r2}


# ── LSTM model definition ──────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, seq, hidden)
        out = self.fc(out[:, -1, :])   # last time-step
        return out.squeeze(-1)


# ── LSTM data helpers ───────────────────────────────────────────────────────
SEQ_LEN = 6


def make_sequences(X, y, seq_len=SEQ_LEN):
    """Create sliding-window sequences for the LSTM."""
    Xs, ys = [], []
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train_sc, y_train, X_val_sc, y_val, input_size, device,
               hidden_size=64, num_layers=2, dropout=0.2,
               lr=1e-3, epochs=150, patience=20, batch_size=128):
    """Train LSTM with early stopping on validation loss."""
    X_tr_seq, y_tr_seq = make_sequences(X_train_sc, y_train.values)
    X_va_seq, y_va_seq = make_sequences(X_val_sc, y_val.values)

    train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    val_ds = TensorDataset(torch.tensor(X_va_seq), torch.tensor(y_va_seq))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(input_size, hidden_size, num_layers, dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [LSTM] epoch {epoch:>3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [LSTM] Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_lstm(model, X_sc, y_series, device):
    """Generate LSTM predictions aligned to the last (n − SEQ_LEN) points."""
    X_seq, _ = make_sequences(X_sc, y_series.values)
    with torch.no_grad():
        preds = model(torch.tensor(X_seq).to(device)).cpu().numpy()
    return preds


# ── Model serialization ────────────────────────────────────────────────────
def serialize_models(lr_model, rf_model, xgb_model, lstm_model,
                     models_dir="models"):
    """Save all trained models to disk for later loading without retraining."""
    os.makedirs(models_dir, exist_ok=True)

    # Scikit-learn & XGBoost models via joblib
    lr_path = os.path.join(models_dir, "linear_regression_final.pkl")
    joblib.dump(lr_model, lr_path)
    print(f"[serialize] Linear Regression saved -> {lr_path}")

    rf_path = os.path.join(models_dir, "random_forest_final.pkl")
    joblib.dump(rf_model, rf_path)
    print(f"[serialize] Random Forest saved -> {rf_path}")

    xgb_path = os.path.join(models_dir, "xgboost_final.pkl")
    joblib.dump(xgb_model, xgb_path)
    print(f"[serialize] XGBoost saved -> {xgb_path}")

    # LSTM weights via torch.save
    lstm_path = os.path.join(models_dir, "lstm_final.pth")
    torch.save(lstm_model.state_dict(), lstm_path)
    print(f"[serialize] LSTM weights saved -> {lstm_path}")


# ── Main training entry point ──────────────────────────────────────────────
def train_and_evaluate(prep: dict, results_dir: str = "results",
                       models_dir: str = "models"):
    """Train all 4 models, compute metrics, serialize, save comparison table and plot."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train = prep["X_train"]
    y_train = prep["y_train"]
    X_val = prep["X_val"]
    y_val = prep["y_val"]
    X_test = prep["X_test"]
    y_test = prep["y_test"]
    X_train_sc = prep["X_train_sc"]
    X_val_sc = prep["X_val_sc"]
    X_test_sc = prep["X_test_sc"]
    feature_cols = prep["feature_cols"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] PyTorch device: {device}")

    metrics_rows = []
    predictions = {}

    # ── 1. Linear Regression ────────────────────────────────────────────────
    print("\n-- Training Linear Regression --")
    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)
    lr_pred = lr_model.predict(X_test_sc)
    metrics_rows.append(compute_metrics(y_test, lr_pred, "Linear Regression"))
    predictions["Linear Regression"] = lr_pred
    print(f"   R2 = {metrics_rows[-1]['R2']:.4f}")

    # ── 2. Random Forest ────────────────────────────────────────────────────
    print("\n-- Training Random Forest (300 trees) --")
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    metrics_rows.append(compute_metrics(y_test, rf_pred, "Random Forest"))
    predictions["Random Forest"] = rf_pred
    print(f"   R2 = {metrics_rows[-1]['R2']:.4f}")

    # ── 3. XGBoost ──────────────────────────────────────────────────────────
    print("\n-- Training XGBoost --")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_pred = xgb_model.predict(X_test)
    metrics_rows.append(compute_metrics(y_test, xgb_pred, "XGBoost"))
    predictions["XGBoost"] = xgb_pred
    print(f"   R2 = {metrics_rows[-1]['R2']:.4f}  (best_iteration={xgb_model.best_iteration})")

    # ── 4. LSTM ─────────────────────────────────────────────────────────────
    print("\n-- Training LSTM --")
    input_size = X_train_sc.shape[1]
    lstm_model = train_lstm(X_train_sc, y_train, X_val_sc, y_val,
                            input_size=input_size, device=device)
    lstm_pred_full = predict_lstm(lstm_model, X_test_sc, y_test, device)
    # The LSTM predictions are shorter by SEQ_LEN; align the ground truth.
    y_test_lstm = y_test.iloc[SEQ_LEN:].reset_index(drop=True)
    metrics_rows.append(compute_metrics(y_test_lstm, lstm_pred_full, "LSTM"))
    predictions["LSTM"] = lstm_pred_full
    print(f"   R2 = {metrics_rows[-1]['R2']:.4f}")

    # ── Save metrics table ──────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(results_dir, "final_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[train] Metrics saved -> {metrics_path}")
    print(metrics_df.to_string(index=False))

    # Also save model_comparison_metrics.csv for backwards compatibility
    metrics_df.to_csv(os.path.join(results_dir, "model_comparison_metrics.csv"), index=False)

    # ── Serialize all trained models ────────────────────────────────────────
    serialize_models(lr_model, rf_model, xgb_model, lstm_model,
                     models_dir=models_dir)

    # Also save the scaler so users can load and scale new data
    joblib.dump(prep["scaler"], os.path.join(models_dir, "scaler.pkl"))

    # ── Predictions comparison plot ─────────────────────────────────────────
    _plot_predictions(y_test, predictions, results_dir)

    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "lstm_model": lstm_model,
        "metrics_df": metrics_df,
        "predictions": predictions,
        "device": device,
    }


# ── Plot helper ─────────────────────────────────────────────────────────────
def _plot_predictions(y_test, predictions, results_dir, window=150):
    """Overlay actual vs predicted for 4 models on a 150-point test window."""
    fig, ax = plt.subplots(figsize=(14, 5))
    start = 0
    end = min(window, len(y_test))
    actual_slice = y_test.iloc[start:end].values

    ax.plot(actual_slice, linewidth=2, label="Actual", color="black")

    colours = {"Linear Regression": "#e74c3c", "Random Forest": "#2ecc71",
               "XGBoost": "#3498db", "LSTM": "#9b59b6"}
    for name, pred in predictions.items():
        pred_slice = pred[start:end]
        ax.plot(pred_slice, linewidth=1.2, linestyle="--", label=name, color=colours.get(name, "grey"))

    ax.set_xlabel("Test Time Step")
    ax.set_ylabel("Power Output (W)")
    ax.set_title("Actual vs Predicted Power Output — 150-Point Test Window")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(results_dir, "model_predictions_comparison.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[train] Prediction plot saved -> {path}")
