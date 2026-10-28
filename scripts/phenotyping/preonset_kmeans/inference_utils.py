from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def modified_poisson(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Log-Poisson regression with HC0 sandwich covariance for adjusted risk ratios."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)

    def objective(beta: np.ndarray) -> float:
        eta = np.clip(x @ beta, -20, 20)
        return float(np.sum(np.exp(eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = np.clip(x @ beta, -20, 20)
        return x.T @ (np.exp(eta) - y)

    initial = np.zeros(x.shape[1])
    initial[0] = np.log(np.clip(y.mean(), 1e-5, None))
    result = minimize(
        objective, initial, jac=gradient, method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    beta = result.x
    mu = np.exp(np.clip(x @ beta, -20, 20))
    bread_inv = np.linalg.pinv(x.T @ (mu[:, None] * x))
    scores = x * (y - mu)[:, None]
    meat = scores.T @ scores
    n, p = x.shape
    covariance = bread_inv @ meat @ bread_inv * (n / max(n - p, 1))
    scaled_gradient = float(np.max(np.abs(gradient(beta))) / len(y))
    return beta, covariance, bool(result.success or scaled_gradient < 1e-6)
