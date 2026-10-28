"""
preprocess.py — Data loading, cleaning, feature engineering, chronological split, and scaling.

Part of: "Enhancing Solar Energy Forecasting and Grid Management Using Explainable AI (XAI)"
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
def load_data(filepath: str, sheet_name: str = "final") -> pd.DataFrame:
    """Load the merged solar Excel file and parse DateTime."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[preprocess] Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# 2. Sensor calibration fixes
# ---------------------------------------------------------------------------
def fix_sensor_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Clip Relative Humidity to [0, 100] and wrap Wind Direction to [0, 360)."""
    df["Relative Humidity (%)"] = df["Relative Humidity (%)"].clip(upper=100.0)
    df["Wind Direction (Degree)"] = df["Wind Direction (Degree)"] % 360.0
    print("[preprocess] Sensor calibration drift fixed.")
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, cyclical, domain, lagged, and soiling-baseline features."""
    # Decimal hour
    df["Hour"] = df["DateTime"].dt.hour + df["DateTime"].dt.minute / 60.0

    # Cyclical encodings
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["Month_sin"] = np.sin(2 * np.pi * df["DateTime"].dt.month / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["DateTime"].dt.month / 12.0)

    # Panel thermal differential
    df["Temp_Diff"] = df["Temp1 (degC)"] - df["Air Temp (degC)"]

    # Lagged features (1-step and 2-step)
    for lag in [1, 2]:
        df[f"Pyranometer_lag{lag}"] = df["Pyranometer (Watt/m2)"].shift(lag)
        df[f"Power_Output_lag{lag}"] = df["Power_Output (W)"].shift(lag)

    # Drop the first 2 rows with NaN lags
    df = df.iloc[2:].reset_index(drop=True)

    # ── Soiling Baseline ────────────────────────────────────────────────────
    df["Rolling_Soiling_Factor"] = (
        df["Power_Output_lag1"] / (df["Pyranometer_lag1"] + 1)
    ).rolling(16, min_periods=1).median()

    df["Expected_Baseline_Power"] = (
        df["Pyranometer (Watt/m2)"] * df["Rolling_Soiling_Factor"]
    )

    print(f"[preprocess] Feature engineering done. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. Prepare feature matrix & target
# ---------------------------------------------------------------------------
FEATURES_TO_DROP = ["DateTime", "Power_Output (W)", "Voltage (V)", "Current (A)"]


def get_feature_target(df: pd.DataFrame):
    """Return X (features) and y (target), dropping leaky & non-feature columns."""
    drop_cols = FEATURES_TO_DROP.copy()
    for col in drop_cols:
        if col not in df.columns:
            drop_cols.remove(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["Power_Output (W)"].copy()
    print(f"[preprocess] Feature matrix X: {X.shape}, target y: {y.shape}")
    print(f"[preprocess] Feature columns: {list(X.columns)}")
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# 5. Monthly-Block Chronological Split (70 / 15 / 15)
# ---------------------------------------------------------------------------
def monthly_block_split(X, y, df, train_frac=0.70, val_frac=0.15):
    """
    Monthly-block chronological split.
    Group by Year-Month. For each month, take the first 70% of rows for Train,
    next 15% for Validation, and final 15% for Test.
    """
    # Create a year-month key from the original DataFrame's DateTime
    ym_key = df["DateTime"].dt.to_period("M")

    train_idx, val_idx, test_idx = [], [], []

    for period, group_indices in sorted(ym_key.groupby(ym_key).groups.items()):
        group_indices = sorted(group_indices)
        n = len(group_indices)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_idx.extend(group_indices[:train_end])
        val_idx.extend(group_indices[train_end:val_end])
        test_idx.extend(group_indices[val_end:])

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print(f"[preprocess] Monthly-block split sizes -- "
          f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# 6. Standard scaling (fitted on training only)
# ---------------------------------------------------------------------------
def scale_data(X_train, X_val, X_test):
    """Fit StandardScaler on train, transform train/val/test. Returns arrays."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    print("[preprocess] StandardScaler fitted on training set and applied.")
    return X_train_sc, X_val_sc, X_test_sc, scaler


# ---------------------------------------------------------------------------
# 7. Master preprocessing function
# ---------------------------------------------------------------------------
def run_preprocessing(data_path: str, results_dir: str = "results"):
    """Execute the full preprocessing pipeline and return all artefacts."""
    os.makedirs(results_dir, exist_ok=True)

    df = load_data(data_path)
    df = fix_sensor_drift(df)
    df = engineer_features(df)
    X, y, feature_cols = get_feature_target(df)
    X_train, y_train, X_val, y_val, X_test, y_test = monthly_block_split(X, y, df)
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_data(X_train, X_val, X_test)

    # ── Save split datasets to CSV ──────────────────────────────────────────
    X_train.to_csv(os.path.join(results_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(results_dir, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(results_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(results_dir, "y_val.csv"), index=False)
    X_test.to_csv(os.path.join(results_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(results_dir, "y_test.csv"), index=False)
    print(f"[preprocess] Split datasets saved to {results_dir}/")

    return {
        "df": df,
        "feature_cols": feature_cols,
        # Unscaled splits (for tree models)
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        # Scaled splits (for LR and LSTM)
        "X_train_sc": X_train_sc,
        "X_val_sc": X_val_sc,
        "X_test_sc": X_test_sc,
        "scaler": scaler,
    }
