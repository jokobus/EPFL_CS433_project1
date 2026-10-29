# -*- coding: utf-8 -*-
"""Implementations of the six core machine learning methods for Project 1.

EPFL Machine Learning Course (CS-433)
Fall 2026

All functions adhere strictly to the project specification:
- Return type: (w, loss) where w has shape (D,) and loss is a scalar float.
- Iterative methods use initial_w, max_iters, and step size gamma.
- Regularized methods (ridge_regression, reg_logistic_regression) return unpenalized loss.
- Vectors are represented as 1D numpy arrays of shape (N,) or (D,).
"""

import numpy as np


def compute_mse_loss(y, tx, w):
    """Compute the Mean Squared Error (MSE) loss with factor 1/(2N).

    Args:
        y: numpy array of shape (N,) containing target labels/values.
        tx: numpy array of shape (N, D) containing input feature matrix.
        w: numpy array of shape (D,) containing model weight vector.

    Returns:
        loss: scalar float representing the MSE loss.
    """
    n_samples = len(y)
    e = y - tx.dot(w)
    return np.float64(np.sum(e**2) / (2.0 * n_samples))


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss function.

    Args:
        y: numpy array of shape (N,) containing target labels/values.
        tx: numpy array of shape (N, D) containing input feature matrix.
        w: numpy array of shape (D,) containing model weight vector.

    Returns:
        grad: numpy array of shape (D,) containing gradient vector.
    """
    n_samples = len(y)
    e = y - tx.dot(w)
    return -tx.T.dot(e) / float(n_samples)


def sigmoid(t):
    """Apply numerically stable sigmoid function element-wise.

    Args:
        t: scalar or numpy array of values.

    Returns:
        sig: numpy array with values mapped to (0, 1).
    """
    clipped_t = np.clip(t, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped_t))


def compute_logistic_loss(y, tx, w):
    """Compute negative log-likelihood (cross-entropy loss) for logistic regression.

    Args:
        y: numpy array of shape (N,) with binary labels in {0, 1}.
        tx: numpy array of shape (N, D) containing input feature matrix.
        w: numpy array of shape (D,) containing model weight vector.

    Returns:
        loss: scalar float representing the negative log-likelihood loss.
    """
    n_samples = len(y)
    pred = sigmoid(tx.dot(w))
    eps = 1e-15
    clipped_pred = np.clip(pred, eps, 1.0 - eps)
    loss = -np.mean(y * np.log(clipped_pred) + (1.0 - y) * np.log(1.0 - clipped_pred))
    return np.float64(loss)


def compute_logistic_gradient(y, tx, w):
    """Compute gradient of the negative log-likelihood for logistic regression.

    Args:
        y: numpy array of shape (N,) with binary labels in {0, 1}.
        tx: numpy array of shape (N, D) containing input feature matrix.
        w: numpy array of shape (D,) containing model weight vector.

    Returns:
        grad: numpy array of shape (D,) containing the gradient vector.
    """
    n_samples = len(y)
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y) / float(n_samples)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch Gradient Descent.

    Args:
        y: numpy array of shape (N,) containing target values.
        tx: numpy array of shape (N, D) containing input feature matrix.
        initial_w: numpy array of shape (D,) containing initial weight vector.
        max_iters: integer, total number of gradient descent iterations.
        gamma: float, learning rate / step size.

    Returns:
        w: numpy array of shape (D,), final weight vector after optimization.
        loss: scalar float, MSE loss computed with the final weights.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w -= gamma * grad

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Stochastic Gradient Descent (mini-batch size 1).

    Args:
        y: numpy array of shape (N,) containing target values.
        tx: numpy array of shape (N, D) containing input feature matrix.
        initial_w: numpy array of shape (D,) containing initial weight vector.
        max_iters: integer, total number of SGD steps.
        gamma: float, learning rate / step size.

    Returns:
        w: numpy array of shape (D,), final weight vector after optimization.
        loss: scalar float, full-dataset MSE loss computed with final weights.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = len(y)

    for _ in range(max_iters):
        if n_samples == 1:
            idx = 0
        else:
            idx = np.random.randint(0, n_samples)

        y_sample = y[idx : idx + 1]
        tx_sample = tx[idx : idx + 1]

        grad = compute_mse_gradient(y_sample, tx_sample, w)
        w -= gamma * grad

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Args:
        y: numpy array of shape (N,) containing target values.
        tx: numpy array of shape (N, D) containing input feature matrix.

    Returns:
        w: numpy array of shape (D,), optimal closed-form weight vector.
        loss: scalar float, MSE loss computed with optimal weights.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a).dot(b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the regularized normal equations.

    Solves: (X^T X + 2 * N * lambda * I) w = X^T y.

    Args:
        y: numpy array of shape (N,) containing target values.
        tx: numpy array of shape (N, D) containing input feature matrix.
        lambda_: float, regularization penalty coefficient.

    Returns:
        w: numpy array of shape (D,), optimal regularized weight vector.
        loss: scalar float, unpenalized MSE loss computed with optimal weights.
    """
    n_samples, n_features = tx.shape
    reg_matrix = 2.0 * n_samples * lambda_ * np.eye(n_features)
    a = tx.T.dot(tx) + reg_matrix
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using full-batch Gradient Descent for binary labels in {0, 1}.

    Args:
        y: numpy array of shape (N,) containing binary targets in {0, 1}.
        tx: numpy array of shape (N, D) containing input feature matrix.
        initial_w: numpy array of shape (D,) containing initial weight vector.
        max_iters: integer, total number of gradient descent iterations.
        gamma: float, learning rate / step size.

    Returns:
        w: numpy array of shape (D,), final weight vector after optimization.
        loss: scalar float, negative log-likelihood computed with final weights.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w -= gamma * grad

    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression with L2 penalty using Gradient Descent.

    Args:
        y: numpy array of shape (N,) containing binary targets in {0, 1}.
        tx: numpy array of shape (N, D) containing input feature matrix.
        lambda_: float, L2 regularization coefficient.
        initial_w: numpy array of shape (D,) containing initial weight vector.
        max_iters: integer, total number of gradient descent iterations.
        gamma: float, learning rate / step size.

    Returns:
        w: numpy array of shape (D,), final weight vector after optimization.
        loss: scalar float, unpenalized negative log-likelihood with final weights.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w -= gamma * grad

    loss = compute_logistic_loss(y, tx, w)
    return w, loss
