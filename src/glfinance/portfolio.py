from __future__ import annotations

import numpy as np


def unconstrained_gmv_weights(covariance: np.ndarray) -> np.ndarray:
    """Closed-form fully-invested GMV weights; short positions are allowed."""
    matrix = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    n_assets = matrix.shape[0]
    ones = np.ones(n_assets)
    inverse = np.linalg.pinv(matrix, rcond=1.0e-10)
    numerator = inverse @ ones
    denominator = float(ones @ numerator)
    if not np.isfinite(denominator) or abs(denominator) < 1.0e-14:
        raise RuntimeError("GMV normalization denominator is numerically zero.")
    weights = numerator / denominator
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("GMV calculation produced non-finite weights.")
    return weights


def equal_weights(n_assets: int) -> np.ndarray:
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")
    return np.ones(n_assets) / float(n_assets)


def long_only_gmv_weights(
    covariance: np.ndarray,
    tolerance: float = 1.0e-10,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Solve the fully-invested long-only GMV problem by an active-set method."""
    matrix = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    n_assets = matrix.shape[0]
    if matrix.shape != (n_assets, n_assets):
        raise ValueError("Covariance must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Covariance contains non-finite values.")

    free = set(range(n_assets))
    weights = equal_weights(n_assets)
    iteration_limit = max_iterations or 10 * n_assets

    for _ in range(iteration_limit):
        free_indices = np.asarray(sorted(free), dtype=int)
        submatrix = matrix[np.ix_(free_indices, free_indices)]
        subweights = unconstrained_gmv_weights(submatrix)
        candidate = np.zeros(n_assets)
        candidate[free_indices] = subweights

        negative = free_indices[candidate[free_indices] < -tolerance]
        if negative.size:
            direction = candidate - weights
            eligible = np.where(direction < -tolerance)[0]
            ratios = -weights[eligible] / direction[eligible]
            positive_ratios = ratios[ratios >= 0.0]
            if positive_ratios.size == 0:
                raise RuntimeError("Long-only active-set step has no feasible boundary.")
            alpha = min(1.0, float(np.min(positive_ratios)))
            weights = weights + alpha * direction
            weights[np.abs(weights) <= tolerance] = 0.0
            hit = [index for index in free if weights[index] <= tolerance]
            for index in hit:
                free.remove(index)
            if not free:
                raise RuntimeError("Long-only active set became empty.")
            continue

        weights = candidate
        gradient = matrix @ weights
        free_level = float(np.mean(gradient[free_indices]))
        active_indices = np.asarray(
            sorted(set(range(n_assets)).difference(free)), dtype=int
        )
        if active_indices.size == 0:
            weights = np.maximum(weights, 0.0)
            return weights / weights.sum()
        violations = active_indices[
            gradient[active_indices] < free_level - tolerance
        ]
        if violations.size == 0:
            weights = np.maximum(weights, 0.0)
            return weights / weights.sum()
        add_index = int(violations[np.argmin(gradient[violations])])
        free.add(add_index)

    # Degenerate low-rank graph covariances can make an active set cycle within
    # numerical tolerance. Fall back to a general constrained optimizer rather
    # than returning an unchecked approximation.
    from scipy.optimize import minimize

    reference = minimize(
        lambda candidate: float(candidate @ matrix @ candidate),
        x0=np.maximum(weights, 0.0) / np.maximum(np.sum(np.maximum(weights, 0.0)), 1.0e-15),
        jac=lambda candidate: 2.0 * matrix @ candidate,
        bounds=[(0.0, 1.0)] * n_assets,
        constraints={
            "type": "eq",
            "fun": lambda candidate: float(candidate.sum() - 1.0),
            "jac": lambda candidate: np.ones_like(candidate),
        },
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 2000, "disp": False},
    )
    if not reference.success:
        raise RuntimeError(
            f"Long-only GMV solvers did not converge: {reference.message}"
        )
    result = np.maximum(np.asarray(reference.x, dtype=float), 0.0)
    result /= result.sum()
    if np.min(result) < -tolerance or abs(result.sum() - 1.0) > 1.0e-9:
        raise RuntimeError("Long-only GMV fallback returned infeasible weights.")
    return result


def turnover(weights: np.ndarray, previous_weights: np.ndarray | None) -> float:
    if previous_weights is None:
        return 0.0
    return float(np.sum(np.abs(np.asarray(weights) - np.asarray(previous_weights))))
