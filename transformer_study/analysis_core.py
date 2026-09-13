from __future__ import annotations

import numpy as np


def fit_affine_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a matrix")
    if y.ndim == 1:
        y = y[:, None]
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows")
    xa = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    reg = np.eye(xa.shape[1], dtype=x.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    gram = xa.T @ xa + reg
    rhs = xa.T @ y
    try:
        theta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(gram) @ rhs
    w, b = theta[:-1], theta[-1]
    if b.size == 1:
        b = b.reshape(1)
    return w, b


def apply_affine(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(x) @ np.asarray(w) + np.asarray(b)


def compose_affine(
    w_ab: np.ndarray, b_ab: np.ndarray, w_bc: np.ndarray, b_bc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return w_ab @ w_bc, b_ab @ w_bc + b_bc


def _one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {int(c): i for i, c in enumerate(classes)}
    out = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, value in enumerate(labels):
        out[row, lookup[int(value)]] = 1.0
    return out


def ridge_classifier(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ridge: float = 1e-3,
) -> float:
    y_fit = np.asarray(y_fit)
    y_test = np.asarray(y_test)
    classes = np.unique(np.concatenate([y_fit, y_test]))
    target = _one_hot(y_fit, classes)
    w, b = fit_affine_ridge(x_fit, target, ridge=ridge)
    scores = apply_affine(x_test, w, b)
    pred = classes[np.argmax(scores, axis=1)]
    return float(np.mean(pred == y_test))


def orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def known_task_basis(task_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(task_means, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError("task_means must be a matrix")
    center = means.mean(axis=0)
    centered = means - center
    if not np.any(centered):
        return center, np.zeros((means.shape[1], 0), dtype=np.float64)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if len(s) == 0 or s[0] == 0:
        return center, np.zeros((means.shape[1], 0), dtype=np.float64)
    tol = max(centered.shape) * np.finfo(centered.dtype).eps * s[0]
    basis = vt[s > tol].T
    return center, basis


def rho_perp(h: np.ndarray, center: np.ndarray, basis: np.ndarray, eps: float = 1e-12):
    z = np.asarray(h, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    if basis.size:
        projection = z @ basis @ basis.T
    else:
        projection = np.zeros_like(z)
    residual = z - projection
    numerator = np.sum(residual * residual, axis=-1)
    denominator = np.sum(z * z, axis=-1) + eps
    value = numerator / denominator
    if np.ndim(value) == 0:
        return float(value)
    return value


def normalized_mse(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mse = np.mean((pred - true) ** 2)
    variance = np.mean((true - true.mean(axis=0, keepdims=True)) ** 2)
    return float(mse / (variance + eps))


def mean_cosine(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    dot = np.sum(pred * true, axis=1)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + eps
    return float(np.mean(dot / denom))
