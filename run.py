# -*- coding: utf-8 -*-
"""Execution script to reproduce best predictions for Project 1.

EPFL Machine Learning Course (CS-433)
Fall 2026

High-Performance Competitive Pipeline:
1. Staged data ingestion with low-memory footprint (< 600 MB peak RAM).
2. Advanced Domain Preprocessing & Feature Engineering:
   - BRFSS CDC code cleaning (88/888 -> 0 for count variables, 7/9/77/99/777/999/9999 -> NaN)
   - Retention of high-yield screening features up to 98% missingness (CVDASPRN, BPMEDS, etc.)
   - Ternary {-1, 0, +1} mapping for binary survey questions (Yes -> +1, No -> -1, Unasked -> 0)
   - One-hot categorical / dummy expansion for discrete ordinal/nominal features (cardinality 3-16)
   - Clinically grounded comorbidity interactions (Age * BP, Age * Diabetes, Sex * Age, Fitness, etc.)
   - Explaining away mobility terms (DIFFWALK * HAVARTH3 to disentangle arthritis from heart disease)
   - Polynomial quadratic expansions and continuous feature standardization
   - Direct output generation into preallocated contiguous float32 matrices (zero heap fragmentation)
3. 5-Fold Stratified Ridge Regression Ensemble with Cost-Sensitive Weighting:
   - Out-of-fold continuous score calibration and global decision threshold optimization
   - Variance-reducing ensemble averaging across all 5 fold models for test set scoring
4. Generates predictions on x_test.csv and exports submission.csv via create_csv_submission.

Usage:
    python run.py
"""

import os
import sys
import time
import gc
import numpy as np

# Ensure local imports work cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from helpers import create_csv_submission
from src.preprocessing import (
    PreprocessingPipeline,
    load_train_dataset,
    load_test_dataset,
)
from src.models import KFoldRidgeEnsemble
from src.metrics import evaluate_predictions


def find_data_directory():
    """Locate the dataset directory containing x_train.csv, y_train.csv, and x_test.csv."""
    candidates = [
        os.path.join(CURRENT_DIR, "data", "dataset"),
        os.path.join(CURRENT_DIR, "data"),
        CURRENT_DIR,
    ]
    for path in candidates:
        if (
            os.path.isfile(os.path.join(path, "x_train.csv"))
            and os.path.isfile(os.path.join(path, "y_train.csv"))
            and os.path.isfile(os.path.join(path, "x_test.csv"))
        ):
            return path
    raise FileNotFoundError(
        "Could not locate dataset files (x_train.csv, y_train.csv, x_test.csv). "
        "Please extract data/dataset_to_release.zip into projects/project1/data/dataset."
    )


def main():
    print("=" * 75)
    print(" EPFL CS-433: Machine Learning — Project 1 High-Performance Pipeline")
    print("=" * 75)

    data_dir = find_data_directory()

    # 1. Locate and load training dataset
    print(f"\n[1/4] Loading training dataset from: {data_dir}")
    t0 = time.time()
    with open(os.path.join(data_dir, "x_train.csv"), "r") as f:
        header = f.readline().strip().split(",")
    feature_names = header[1:]

    X_train_raw, y_train, train_ids = load_train_dataset(data_dir)
    print(
        f"      Loaded {X_train_raw.shape[0]} training samples across {X_train_raw.shape[1]} features "
        f"in {time.time() - t0:.2f}s."
    )
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == -1)
    print(
        f"      Target distribution: {n_pos} positive ({n_pos / len(y_train):.2%}) | "
        f"{n_neg} negative ({n_neg / len(y_train):.2%})"
    )

    # 2. Fit and transform training features
    print("\n[2/4] Fitting advanced preprocessing & feature engineering pipeline...")
    t0 = time.time()
    pipeline = PreprocessingPipeline(
        max_missing_ratio=0.98,
        clean_cdc_codes=True,
        add_missing_indicators=True,
        ternary_binary=True,
        one_hot_categorical=True,
        add_clinical_interactions=True,
        standardize=True,
        add_bias=True,
        feature_names=feature_names,
    )
    tx_train = pipeline.fit_transform(X_train_raw)
    print(
        f"      Transformed feature space: {X_train_raw.shape[1]} raw -> {tx_train.shape[1]} engineered features "
        f"({tx_train.nbytes / 1e6:.1f} MB) in {time.time() - t0:.2f}s."
    )

    # Free raw training arrays from memory
    del X_train_raw
    gc.collect()

    # 3. Stratified 5-Fold Ridge Ensemble Training & Out-Of-Fold Validation
    print(
        "\n[3/4] Training 5-Fold Stratified Ridge Regression Ensemble (Cost-Sensitive)..."
    )
    t0 = time.time()
    ensemble = KFoldRidgeEnsemble(
        lambda_=1e-2,
        pos_weight=3.5,
        k_fold=5,
        seed=42,
    )
    ensemble.fit(tx_train, y_train)

    oof_preds = np.where(ensemble.oof_scores >= ensemble.threshold, 1, -1)
    oof_metrics = evaluate_predictions(y_train, oof_preds)

    print(f"      Ensemble training completed in {time.time() - t0:.2f}s.")
    print(
        f"      Out-Of-Fold Cross-Validation Performance (Optimal tau = {ensemble.threshold:+.4f}):"
    )
    print(
        f"      - Accuracy:  {oof_metrics['accuracy']:.4f} ({oof_metrics['accuracy']:.2%})"
    )
    print(f"      - Precision: {oof_metrics['precision']:.4f}")
    print(f"      - Recall:    {oof_metrics['recall']:.4f}")
    print(f"      - F1 Score:  {oof_metrics['f1']:.4f}")

    # Free training transformed matrix from memory
    del tx_train
    gc.collect()

    # 4. Load test dataset, transform, predict, and export CSV submission
    print("\n[4/4] Loading test dataset and generating fold-averaged predictions...")
    t0 = time.time()
    X_test_raw, test_ids = load_test_dataset(data_dir)
    tx_test = pipeline.transform(X_test_raw)
    del X_test_raw
    gc.collect()

    raw_test_scores = ensemble.predict_raw(tx_test)
    y_pred_test = np.where(raw_test_scores >= ensemble.threshold, 1, -1)
    del tx_test
    gc.collect()

    output_csv = os.path.join(CURRENT_DIR, "submission.csv")
    create_csv_submission(test_ids, y_pred_test, output_csv)

    n_test_pos = np.sum(y_pred_test == 1)
    n_test_neg = np.sum(y_pred_test == -1)
    print(
        f"      Predictions completed and saved to: {output_csv} in {time.time() - t0:.2f}s."
    )
    print(
        f"      Test set prediction breakdown: {n_test_pos} positive ({n_test_pos / len(y_pred_test):.2%}) | "
        f"{n_test_neg} negative ({n_test_neg / len(y_pred_test):.2%})"
    )
    print("=" * 75)
    print(" Pipeline execution finished successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
