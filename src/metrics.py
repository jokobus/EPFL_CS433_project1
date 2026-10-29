# -*- coding: utf-8 -*-
"""Evaluation metrics and decision threshold optimization for Project 1.

EPFL Machine Learning Course (CS-433)
Fall 2026

Implements accuracy, precision, recall, F1 score, confusion matrix,
and automatic threshold tuning for binary classification under class imbalance.
All implementations use NumPy and Python standard library only.
"""

import numpy as np


def compute_confusion_matrix(y_true, y_pred):
    """Compute binary confusion matrix elements.

    Supports ground-truth and prediction labels in {-1, 1} or {0, 1}.

    Args:
        y_true: numpy array of shape (N,) containing true binary labels.
        y_pred: numpy array of shape (N,) containing predicted binary labels.

    Returns:
        dict: containing counts for 'tp', 'fp', 'tn', 'fn'.
    """
    # Normalize labels to {0, 1} for computation
    y_t = np.where(y_true == 1, 1, 0)
    y_p = np.where(y_pred == 1, 1, 0)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_accuracy(y_true, y_pred):
    """Compute overall classification accuracy.

    Args:
        y_true: numpy array of shape (N,) containing true binary labels.
        y_pred: numpy array of shape (N,) containing predicted binary labels.

    Returns:
        accuracy: float, fraction of correctly classified samples.
    """
    return float(np.mean(y_true == y_pred))


def compute_precision(y_true, y_pred):
    """Compute precision for the positive class (+1).

    Args:
        y_true: numpy array of shape (N,) containing true binary labels.
        y_pred: numpy array of shape (N,) containing predicted binary labels.

    Returns:
        precision: float in [0.0, 1.0].
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["tp"] + cm["fp"]
    if denominator == 0:
        return 0.0
    return float(cm["tp"] / denominator)


def compute_recall(y_true, y_pred):
    """Compute recall (sensitivity) for the positive class (+1).

    Args:
        y_true: numpy array of shape (N,) containing true binary labels.
        y_pred: numpy array of shape (N,) containing predicted binary labels.

    Returns:
        recall: float in [0.0, 1.0].
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    denominator = cm["tp"] + cm["fn"]
    if denominator == 0:
        return 0.0
    return float(cm["tp"] / denominator)


def compute_f1_score(y_true, y_pred):
    """Compute harmonic mean of precision and recall (F1 score) for positive class (+1).

    Args:
        y_true: numpy array of shape (N,) containing true binary labels.
        y_pred: numpy array of shape (N,) containing predicted binary labels.

    Returns:
        f1: float in [0.0, 1.0].
    """
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    if prec + rec == 0.0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def evaluate_predictions(y_true, y_pred):
    """Compute comprehensive dictionary of evaluation metrics.

    Args:
        y_true: numpy array of shape (N,) containing true binary labels in {-1, 1}.
        y_pred: numpy array of shape (N,) containing predicted binary labels in {-1, 1}.

    Returns:
        metrics: dict with accuracy, precision, recall, f1, and confusion matrix.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    f1 = compute_f1_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def find_optimal_threshold(
    y_true,
    raw_scores,
    n_thresholds=200,
    metric="f1",
):
    """Find the optimal decision threshold that maximizes a chosen metric on validation data.

    Given continuous predictions or probabilities (raw_scores), classifies sample as +1
    if raw_score >= threshold, else -1.

    Args:
        y_true: numpy array of shape (N,) with true labels in {-1, 1}.
        raw_scores: numpy array of shape (N,) with continuous model predictions.
        n_thresholds: integer, number of threshold points to evaluate across range.
        metric: string, metric to maximize ('f1' or 'accuracy').

    Returns:
        best_threshold: float, threshold yielding highest metric score.
        best_score: float, highest metric score achieved.
        all_thresholds: numpy array of evaluated thresholds.
        all_scores: numpy array of corresponding metric values.
    """
    min_score = np.percentile(raw_scores, 1.0)
    max_score = np.percentile(raw_scores, 99.0)
    thresholds = np.linspace(min_score, max_score, n_thresholds)

    best_score = -1.0
    best_threshold = 0.0
    scores = []

    for tau in thresholds:
        y_pred = np.where(raw_scores >= tau, 1, -1)
        if metric == "f1":
            score = compute_f1_score(y_true, y_pred)
        elif metric == "accuracy":
            score = compute_accuracy(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        scores.append(score)
        if score > best_score:
            best_score = score
            best_threshold = tau

    return best_threshold, best_score, thresholds, np.array(scores)
