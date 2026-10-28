"""
run_pipeline.py — Master script that runs the entire research pipeline end-to-end.

Usage:
    python run_pipeline.py
"""

import os
import sys
import time
import joblib

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import run_preprocessing
from src.train_models import train_and_evaluate
from src.xai_and_analysis import global_shap_analysis, ramp_analysis, write_summary


def main():
    t0 = time.time()
    DATA_PATH = os.path.join(PROJECT_ROOT, "merged_solar_data_2.xlsx")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Step B: Preprocessing & Feature Engineering ─────────────────────────
    print("=" * 70)
    print("STEP B — Data Preprocessing & Feature Engineering")
    print("=" * 70)
    prep = run_preprocessing(DATA_PATH, results_dir=RESULTS_DIR)

    # ── Step C: Model Training, Evaluation & Serialization ──────────────────
    print("\n" + "=" * 70)
    print("STEP C — Model Training, Evaluation & Serialization")
    print("=" * 70)
    models = train_and_evaluate(prep, results_dir=RESULTS_DIR, models_dir=MODELS_DIR)

    # ── Step D: Explainable AI & Grid Ramp Analysis ─────────────────────────
    print("\n" + "=" * 70)
    print("STEP D — Explainable AI (SHAP) & Grid Ramp Analysis")
    print("=" * 70)
    # Load the serialized XGBoost model to ensure it is verified end-to-end
    xgb_saved_path = os.path.join(MODELS_DIR, "xgboost_final.pkl")
    print(f"[XAI] Loading saved XGBoost model from {xgb_saved_path} …")
    saved_xgb_model = joblib.load(xgb_saved_path)

    explainer, shap_values, importance_df = global_shap_analysis(
        saved_xgb_model,
        prep["X_test"],
        prep["feature_cols"],
        results_dir=RESULTS_DIR,
    )

    ramp_stats, worst_ramp_idx = ramp_analysis(
        saved_xgb_model,
        prep["X_test"],
        prep["y_test"],
        prep["feature_cols"],
        explainer,
        results_dir=RESULTS_DIR,
    )

    write_summary(models["metrics_df"], importance_df, ramp_stats, results_dir=RESULTS_DIR)

    # ── Final verification ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION — Checking output artefacts")
    print("=" * 70)
    expected_files = [
        # Results
        "results/final_model_metrics.csv",
        "results/model_predictions_comparison.png",
        "results/shap_beeswarm_summary.png",
        "results/shap_global_bar.png",
        "results/pv_ramp_events_report.csv",
        "results/shap_local_ramp_waterfall.png",
        "results/RESEARCH_RESULTS_SUMMARY.md",
        "results/X_train.csv",
        "results/y_train.csv",
        "results/X_val.csv",
        "results/y_val.csv",
        "results/X_test.csv",
        "results/y_test.csv",
        # Serialized models
        "models/linear_regression_final.pkl",
        "models/random_forest_final.pkl",
        "models/xgboost_final.pkl",
        "models/lstm_final.pth",
    ]
    all_ok = True
    for rel_path in expected_files:
        fpath = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  [OK] {rel_path:50s} ({size_kb:>8.1f} KB)")
        else:
            print(f"  [MISS] {rel_path:50s} -- MISSING or EMPTY")
            all_ok = False

    elapsed = time.time() - t0
    print(f"\nTotal pipeline time: {elapsed:.1f}s")
    if all_ok:
        print("\n***  All artefacts verified successfully. Pipeline complete!")
    else:
        print("\n[WARNING]  Some artefacts are missing -- review logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
