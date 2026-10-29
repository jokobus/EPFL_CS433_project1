# -*- coding: utf-8 -*-
"""Stratified Cross-Validation and Hyperparameter Optimization for Project 1.

EPFL Machine Learning Course (CS-433)
Fall 2026

Implements Stratified K-Fold cross validation and grid search routines without external ML libraries.
"""

import numpy as np
from src.metrics import evaluate_predictions, find_optimal_threshold


def build_k_fold_indices(y, k_fold=5, seed=42, stratified=True):
    """Generate train/validation split indices for K-Fold cross-validation.

    Args:
        y: numpy array of shape (N,) containing binary target labels.
        k_fold: integer, number of folds (default: 5).
        seed: integer, random seed for reproducible fold shuffling.
        stratified: bool, if True preserves class balance (ratio of +1 to -1) in each fold.

    Returns:
        folds: list of (train_indices, val_indices) tuples.
    """
    np.random.seed(seed)
    n_samples = len(y)

    if not stratified:
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // k_fold
        folds = []
        for k in range(k_fold):
            val_idx = indices[
                k * fold_size : (k + 1) * fold_size if k != k_fold - 1 else n_samples
            ]
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
            folds.append((train_idx, val_idx))
        return folds

    # Stratified K-Fold: balance positive and negative indices
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y != 1)[0]

    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    pos_splits = np.array_split(pos_indices, k_fold)
    neg_splits = np.array_split(neg_indices, k_fold)

    folds = []
    for k in range(k_fold):
        val_idx = np.concatenate([pos_splits[k], neg_splits[k]])
        np.random.shuffle(val_idx)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]
        np.random.shuffle(train_idx)

        folds.append((train_idx, val_idx))

    return folds


def cross_validate_pipeline(
    model_fn,
    tx,
    y,
    k_fold=5,
    tune_threshold=True,
    metric="f1",
    seed=42,
):
    """Execute K-Fold cross-validation for a model on preprocessed features tx.

    Args:
        model_fn: callable, factory returning a freshly instantiated model (with .fit and .predict_raw).
        tx: numpy array of shape (N, D) containing processed features.
        y: numpy array of shape (N,) containing targets in {-1, 1}.
        k_fold: integer, number of folds.
        tune_threshold: bool, whether to optimize threshold on train fold and evaluate on val fold.
        metric: string, metric to optimize when tuning threshold ('f1' or 'accuracy').
        seed: integer, random seed.

    Returns:
        results: dict containing fold metrics and aggregated mean/std statistics.
    """
    folds = build_k_fold_indices(y, k_fold=k_fold, seed=seed, stratified=True)
    val_metrics = []
    thresholds = []

    for fold_k, (tr_idx, va_idx) in enumerate(folds):
        tx_tr, y_tr = tx[tr_idx], y[tr_idx]
        tx_va, y_va = tx[va_idx], y[va_idx]

        model = model_fn()
        model.fit(tx_tr, y_tr)

        if tune_threshold:
            raw_tr = model.predict_raw(tx_tr)
            best_tau, _, _, _ = find_optimal_threshold(y_tr, raw_tr, metric=metric)
            thresholds.append(best_tau)
            preds_va = model.predict(tx_va, threshold=best_tau)
        else:
            preds_va = model.predict(tx_va)

        m = evaluate_predictions(y_va, preds_va)
        val_metrics.append(m)

    # Compute aggregate statistics
    accs = [m["accuracy"] for m in val_metrics]
    precs = [m["precision"] for m in val_metrics]
    recs = [m["recall"] for m in val_metrics]
    f1s = [m["f1"] for m in val_metrics]

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_precision": float(np.mean(precs)),
        "std_precision": float(np.std(precs)),
        "mean_recall": float(np.mean(recs)),
        "std_recall": float(np.std(recs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "mean_threshold": float(np.mean(thresholds)) if thresholds else 0.0,
        "fold_metrics": val_metrics,
    }


def grid_search_ridge_lambda(
    tx,
    y,
    lambdas=None,
    k_fold=5,
    tune_threshold=True,
    metric="f1",
):
    """Grid search across regularization values lambda for Ridge Regression.

    Args:
        tx: numpy array of shape (N, D).
        y: numpy array of shape (N,) with labels in {-1, 1}.
        lambdas: list/array of regularization values.
        k_fold: integer, number of cross-validation folds.
        tune_threshold: bool, whether to tune decision threshold per fold.
        metric: string, metric to maximize.

    Returns:
        best_lambda: float, optimal lambda value.
        best_result: dict, cross-validation metrics at optimal lambda.
        all_results: list of (lambda, result_dict) tuples.
    """
    from src.models import RidgeRegressionModel

    if lambdas is None:
        lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    all_results = []
    best_lambda = None
    best_score = -1.0
    best_result = None

    for lam in lambdas:
        model_fn = lambda l=lam: RidgeRegressionModel(lambda_=l)
        res = cross_validate_pipeline(
            model_fn,
            tx,
            y,
            k_fold=k_fold,
            tune_threshold=tune_threshold,
            metric=metric,
        )
        all_results.append((lam, res))
        score = res[f"mean_{metric}"]
        if score > best_score:
            best_score = score
            best_lambda = lam
            best_result = res

    return best_lambda, best_result, all_results
