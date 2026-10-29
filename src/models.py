# -*- coding: utf-8 -*-
"""Model wrappers integrating implementations.py with training, scoring, and prediction.

EPFL Machine Learning Course (CS-433)
Fall 2026

Wraps the core functions of implementations.py into a clean, unified object-oriented interface.
Supports cost-sensitive / sample-weighted Ridge Regression and K-Fold Out-Of-Fold Ensembling.
All implementations rely solely on implementations.py and NumPy.
"""

import os
import sys
import numpy as np

# Ensure implementations.py can be imported from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import implementations as imp
from src.metrics import find_optimal_threshold


class BaseModel:
    """Base interface for linear classification models."""

    def __init__(self):
        self.w = None
        self.loss = None
        self.threshold = 0.0

    def fit(self, tx, y):
        raise NotImplementedError

    def predict_raw(self, tx):
        raise NotImplementedError

    def predict(self, tx, threshold=None):
        """Predict binary class labels in {-1, 1}."""
        if threshold is None:
            threshold = self.threshold
        raw = self.predict_raw(tx)
        return np.where(raw >= threshold, 1, -1)

    def optimize_threshold(self, tx_val, y_val, metric="f1"):
        """Tune decision threshold on validation set."""
        raw_val = self.predict_raw(tx_val)
        best_tau, best_score, _, _ = find_optimal_threshold(
            y_val, raw_val, metric=metric
        )
        self.threshold = best_tau
        return best_tau, best_score


class LeastSquaresModel(BaseModel):
    """Least Squares Linear Regression for classification."""

    def fit(self, tx, y):
        """Fit model weights using normal equations."""
        self.w, self.loss = imp.least_squares(y, tx)
        return self

    def predict_raw(self, tx):
        """Compute linear projection tx @ w."""
        return tx.dot(self.w)


class RidgeRegressionModel(BaseModel):
    """Ridge Regression (L2 Regularized Least Squares) with optional class reweighting."""

    def __init__(self, lambda_=1e-4, pos_weight=1.0):
        super().__init__()
        self.lambda_ = lambda_
        self.pos_weight = pos_weight

    def fit(self, tx, y):
        """Fit regularized weights using normal equations.

        Args:
            tx: numpy array of shape (N, D).
            y: numpy array of shape (N,) with labels in {-1, 1}.
        """
        n_samples, n_features = tx.shape

        if self.pos_weight == 1.0:
            self.w, self.loss = imp.ridge_regression(y, tx, self.lambda_)
        else:
            # Weighted ridge regression normal equations:
            # (X^T W X + 2 N lambda I) w = X^T W y
            weights = np.where(y == 1, self.pos_weight, 1.0).astype(np.float32)
            A = tx.T.dot(tx * weights[:, None])
            b = tx.T.dot(weights * y)
            try:
                self.w = np.linalg.solve(A + reg, b)
            except np.linalg.LinAlgError:
                self.w = np.linalg.pinv(A + reg).dot(b)
            self.loss = imp.compute_mse_loss(y, tx, self.w)

        return self

    def predict_raw(self, tx):
        """Compute linear projection tx @ w."""
        return tx.dot(self.w)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression using Gradient Descent."""

    def __init__(self, max_iters=500, gamma=0.01):
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.threshold = 0.5

    def fit(self, tx, y, initial_w=None):
        y_bin = np.where(y == 1, 1.0, 0.0)
        n_features = tx.shape[1]
        if initial_w is None:
            initial_w = np.zeros(n_features, dtype=np.float32)

        self.w, self.loss = imp.logistic_regression(
            y_bin, tx, initial_w, self.max_iters, self.gamma
        )
        return self

    def predict_raw(self, tx):
        return imp.sigmoid(tx.dot(self.w))


class RegularizedLogisticRegressionModel(BaseModel):
    """Regularized Logistic Regression with L2 penalty using Gradient Descent."""

    def __init__(self, lambda_=0.01, max_iters=500, gamma=0.01):
        super().__init__()
        self.lambda_ = lambda_
        self.max_iters = max_iters
        self.gamma = gamma
        self.threshold = 0.5

    def fit(self, tx, y, initial_w=None):
        y_bin = np.where(y == 1, 1.0, 0.0)
        n_features = tx.shape[1]
        if initial_w is None:
            initial_w = np.zeros(n_features, dtype=np.float32)

        self.w, self.loss = imp.reg_logistic_regression(
            y_bin, tx, self.lambda_, initial_w, self.max_iters, self.gamma
        )
        return self

    def predict_raw(self, tx):
        return imp.sigmoid(tx.dot(self.w))


class KFoldRidgeEnsemble(BaseModel):
    """5-Fold Ridge Regression Ensemble with Out-Of-Fold Threshold Optimization."""

    def __init__(self, lambda_=1e-4, pos_weight=3.5, k_fold=5, seed=42):
        super().__init__()
        self.lambda_ = lambda_
        self.pos_weight = pos_weight
        self.k_fold = k_fold
        self.seed = seed
        self.models = []
        self.oof_scores = None

    def fit(self, tx, y):
        """Train K fold models using fast zero-copy matrix subtraction."""
        import gc
        from src.cross_validation import build_k_fold_indices

        n_samples, n_features = tx.shape
        folds = build_k_fold_indices(
            y, k_fold=self.k_fold, seed=self.seed, stratified=True
        )
        self.oof_scores = np.zeros(n_samples, dtype=np.float32)
        self.models = []

        # Compute full covariance and projection once without full weights broadcast:
        # A_full = X^T X + (w+ - 1) X_pos^T X_pos
        pos_mask = y == 1
        X_pos = tx[pos_mask]
        diff_w = float(self.pos_weight - 1.0)

        A_full = tx.T.dot(tx)
        if diff_w != 0.0 and len(X_pos) > 0:
            A_full += diff_w * X_pos.T.dot(X_pos)

        b_full = tx.T.dot(y.astype(np.float32))
        if diff_w != 0.0 and len(X_pos) > 0:
            b_full += diff_w * X_pos.T.dot(np.ones(len(X_pos), dtype=np.float32))

        del X_pos
        gc.collect()

        for tr_idx, va_idx in folds:
            X_va = tx[va_idx]
            y_va = y[va_idx]
            n_tr = len(tr_idx)

            # Compute validation fold covariance
            va_pos_mask = y_va == 1
            X_va_pos = X_va[va_pos_mask]
            A_va = X_va.T.dot(X_va)
            if diff_w != 0.0 and len(X_va_pos) > 0:
                A_va += diff_w * X_va_pos.T.dot(X_va_pos)

            b_va = X_va.T.dot(y_va.astype(np.float32))
            if diff_w != 0.0 and len(X_va_pos) > 0:
                b_va += diff_w * X_va_pos.T.dot(
                    np.ones(len(X_va_pos), dtype=np.float32)
                )

            # Training normal equations via exact subtraction
            A_tr = A_full - A_va
            b_tr = b_full - b_va
            reg = 2.0 * n_tr * self.lambda_ * np.eye(n_features, dtype=np.float32)

            m = RidgeRegressionModel(lambda_=self.lambda_, pos_weight=self.pos_weight)
            try:
                m.w = np.linalg.solve(A_tr + reg, b_tr)
            except np.linalg.LinAlgError:
                m.w = np.linalg.pinv(A_tr + reg).dot(b_tr)

            self.models.append(m)
            self.oof_scores[va_idx] = X_va.dot(m.w)

        # Optimize global decision threshold on out-of-fold predictions
        best_tau, best_score, _, _ = find_optimal_threshold(
            y, self.oof_scores, metric="f1"
        )
        self.threshold = best_tau
        return self

    def predict_raw(self, tx):
        """Ensemble continuous score: average predictions of all K fold models."""
        scores = np.zeros(len(tx), dtype=np.float32)
        for m in self.models:
            scores += m.predict_raw(tx)
        return scores / float(len(self.models))
