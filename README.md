# Enhancing Solar Energy Forecasting Using Explainable AI (XAI)

A complete machine learning research pipeline for short-term solar power output forecasting, with SHAP-based explainability and PV ramp event detection for grid management.

## Project Architecture

```
solar_xai_research/
├── merged_solar_data_2.xlsx   # 24-month raw dataset (sheet: "final")
├── run_pipeline.py            # Master script — runs the full pipeline end-to-end
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Data loading, cleaning, feature engineering, split
│   ├── train_models.py        # Model training, evaluation, serialization
│   └── xai_and_analysis.py    # SHAP explanations, ramp detection, summary report
├── models/                    # Serialized trained models
│   ├── linear_regression_final.pkl
│   ├── random_forest_final.pkl
│   ├── xgboost_final.pkl
│   ├── lstm_final.pth
│   └── scaler.pkl
└── results/                   # Output CSVs, plots, and reports
    ├── X_train.csv / y_train.csv
    ├── X_val.csv   / y_val.csv
    ├── X_test.csv  / y_test.csv
    ├── final_model_metrics.csv
    ├── model_predictions_comparison.png
    ├── shap_beeswarm_summary.png
    ├── shap_global_bar.png
    ├── shap_local_ramp_waterfall.png
    ├── pv_ramp_events_report.csv
    └── RESEARCH_RESULTS_SUMMARY.md
```

## How to Run

```bash
# 1. Using the project virtual environment (Windows):
cmd /c .\.venv\Scripts\python.exe run_pipeline.py
```

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| **B** | `src/preprocess.py` | Loads the 24-month Excel dataset, applies sensor calibration (humidity ≤ 100, wind direction % 360), engineers cyclic time features, lag features, soiling baseline, and performs a Monthly-Block Chronological Split (70/15/15). |
| **C** | `src/train_models.py` | Trains 4 models (Linear Regression, Random Forest 300 trees, XGBoost with early stopping, LSTM with early stopping), evaluates on the test set (MAE, RMSE, MAPE, R²), and serializes all trained models to `models/`. |
| **D** | `src/xai_and_analysis.py` | Loads the saved XGBoost model, computes SHAP values (beeswarm and global bar plots), detects PV ramp events, produces a local waterfall plot for the worst negative ramp, and writes `RESEARCH_RESULTS_SUMMARY.md`. |

## Key Features & Preprocessing Rules

- **Voltage/Current exclusion:** `Voltage (V)` and `Current (A)` are strictly dropped from features to prevent data leakage (Power = V × I).
- **Soiling Baseline:** A rolling median of the power-to-irradiance ratio approximates panel soiling degradation:
  $$\text{Rolling\_Soiling\_Factor} = \text{rolling}_{16}\left(\frac{\text{Power\_Output}_{\text{lag1}}}{\text{Pyranometer}_{\text{lag1}} + 1}\right)$$
  $$\text{Expected\_Baseline\_Power} = \text{Pyranometer} \times \text{Rolling\_Soiling\_Factor}$$
- **Monthly-Block Chronological Split:** Within each calendar month, the first 70% of rows are allocated to Train, next 15% to Validation, and final 15% to Test — preserving chronological order while ensuring every season is represented across all sets.

## Loading Saved Models

### Scikit-learn / XGBoost (`.pkl` files)

```python
import joblib

# Load any of the sklearn/XGBoost models
lr_model = joblib.load("models/linear_regression_final.pkl")
rf_model = joblib.load("models/random_forest_final.pkl")
xgb_model = joblib.load("models/xgboost_final.pkl")

# Predict on new data (features must match X_train columns)
predictions = xgb_model.predict(X_new)
```

### PyTorch LSTM (`.pth` weights)

```python
import torch
from src.train_models import LSTMForecaster

# Instantiate the LSTM model architecture (input_size=19 features)
model = LSTMForecaster(input_size=19, hidden_size=64, num_layers=2, dropout=0.2)

# Load saved weights
model.load_state_dict(torch.load("models/lstm_final.pth", map_location="cpu"))
model.eval()

# Predict with input tensor of shape (batch_size, seq_len=6, 19)
```

## Models Trained & Evaluated

| Model | Implementation | Details |
|-------|----------------|---------|
| Linear Regression | `sklearn.linear_model.LinearRegression` | Scaled features baseline |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` | 300 estimators, parallel execution |
| XGBoost | `xgboost.XGBRegressor` | 500 estimators, learning rate 0.05, early stopping on val RMSE |
| LSTM | `torch.nn.LSTM` | 2-layer LSTM, hidden size 64, dropout 0.2, sequence length 6 |

## License

Research project — all rights reserved.
