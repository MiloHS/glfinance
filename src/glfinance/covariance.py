from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return 0.5 * (values + values.T)


def graph_basis(laplacian: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal eigenvectors in increasing graph-frequency order."""
    matrix = _symmetrize(laplacian)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    retained = min(matrix.shape[0], max(1, int(k)))
    return eigenvectors[:, np.argsort(eigenvalues)[:retained]]


def projection_energy(values: np.ndarray, basis: np.ndarray) -> tuple[float, float]:
    """Return retained and total squared energy for already-centered observations."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    basis = np.asarray(basis, dtype=float)
    if values.shape[1] != basis.shape[0]:
        raise ValueError("Observations and projection basis have different asset dimensions.")
    return float(np.sum((values @ basis) ** 2)), float(np.sum(values ** 2))


def graph_covariance(
    laplacian: np.ndarray,
    demeaned_returns: np.ndarray,
    k: int,
    rho: float,
    delta: float,
    subspace_estimator: str,
    restore_residual: bool = False,
    sample_ddof: int = 1,
) -> np.ndarray:
    """Construct a graph-filtered covariance using Sample or Ledoit-Wolf."""
    returns = np.asarray(demeaned_returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] != laplacian.shape[0]:
        raise ValueError("Returns must have shape (observations, assets).")
    n_assets = laplacian.shape[0]
    basis = graph_basis(laplacian, k)
    projected = returns @ basis

    estimator = subspace_estimator.lower()
    if estimator == "ledoitwolf":
        subspace_covariance = LedoitWolf().fit(projected).covariance_
    elif estimator == "sample":
        subspace_covariance = np.atleast_2d(
            np.cov(projected, rowvar=False, ddof=sample_ddof)
        )
    else:
        raise ValueError(f"Unknown subspace estimator: {subspace_estimator}")

    low_rank = basis @ subspace_covariance @ basis.T
    result = (1.0 - rho) * low_rank + rho * np.diag(np.diag(low_rank))
    if restore_residual:
        residual = returns - projected @ basis.T
        result += np.diag(np.var(residual, axis=0, ddof=1))
    result = _symmetrize(result) + float(delta) * np.eye(n_assets)
    return result


def pca_factor_covariance(
    demeaned_returns: np.ndarray,
    k: int,
    delta: float,
    subspace_estimator: str = "sample",
    rho: float = 0.0,
) -> np.ndarray:
    """Leading-PC factor covariance plus diagonal sample residual variances.

    Sample/zero-rho is the conventional diagonal-residual factor benchmark.
    LedoitWolf/nonzero-rho matches the graph reconstruction's regularization.
    This implements diagonal residuals, not the full POET estimator.
    """
    returns = np.asarray(demeaned_returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 2:
        raise ValueError("Returns must have shape (observations>=2, assets).")
    _, singular_values, right_vectors = np.linalg.svd(returns, full_matrices=False)
    tolerance = np.finfo(float).eps * max(returns.shape) * singular_values[0]
    rank = min(int(np.count_nonzero(singular_values > tolerance)), returns.shape[0] - 1)
    retained = min(max(1, int(k)), rank)
    if retained == 0:
        return float(delta) * np.eye(returns.shape[1])
    basis = right_vectors[:retained].T
    projected = returns @ basis
    if subspace_estimator.lower() == "sample":
        factor_covariance = np.atleast_2d(np.cov(projected, rowvar=False, ddof=1))
    elif subspace_estimator.lower() == "ledoitwolf":
        factor_covariance = LedoitWolf().fit(projected).covariance_
    else:
        raise ValueError(f"Unknown subspace estimator: {subspace_estimator}")
    low_rank = basis @ factor_covariance @ basis.T
    residual = returns - projected @ basis.T
    result = (1.0 - rho) * low_rank + rho * np.diag(np.diag(low_rank))
    result += np.diag(np.var(residual, axis=0, ddof=1))
    return _symmetrize(result) + float(delta) * np.eye(returns.shape[1])


def sample_covariance(demeaned_returns: np.ndarray, delta: float = 0.0) -> np.ndarray:
    values = np.asarray(demeaned_returns, dtype=float)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    return _symmetrize(covariance) + float(delta) * np.eye(values.shape[1])


def ledoit_wolf_covariance(demeaned_returns: np.ndarray) -> np.ndarray:
    return _symmetrize(LedoitWolf().fit(demeaned_returns).covariance_)

