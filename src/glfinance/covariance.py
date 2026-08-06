from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return 0.5 * (values + values.T)


def graph_covariance(
    laplacian: np.ndarray,
    demeaned_returns: np.ndarray,
    k: int,
    rho: float,
    delta: float,
    subspace_estimator: str,
) -> np.ndarray:
    """Construct a graph-filtered covariance using Sample or Ledoit-Wolf."""
    laplacian = _symmetrize(laplacian)
    returns = np.asarray(demeaned_returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] != laplacian.shape[0]:
        raise ValueError("Returns must have shape (observations, assets).")
    n_assets = laplacian.shape[0]
    retained = max(1, min(int(k), n_assets))
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    basis = eigenvectors[:, np.argsort(eigenvalues)[:retained]]
    projected = returns @ basis

    estimator = subspace_estimator.lower()
    if estimator == "ledoitwolf":
        subspace_covariance = LedoitWolf().fit(projected).covariance_
    elif estimator == "sample":
        subspace_covariance = np.atleast_2d(
            np.cov(projected, rowvar=False, ddof=1)
        )
    else:
        raise ValueError(f"Unknown subspace estimator: {subspace_estimator}")

    low_rank = basis @ subspace_covariance @ basis.T
    result = (1.0 - rho) * low_rank + rho * np.diag(np.diag(low_rank))
    result = _symmetrize(result) + float(delta) * np.eye(n_assets)
    return result


def sample_covariance(demeaned_returns: np.ndarray, delta: float = 0.0) -> np.ndarray:
    values = np.asarray(demeaned_returns, dtype=float)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    return _symmetrize(covariance) + float(delta) * np.eye(values.shape[1])


def ledoit_wolf_covariance(demeaned_returns: np.ndarray) -> np.ndarray:
    return _symmetrize(LedoitWolf().fit(demeaned_returns).covariance_)

