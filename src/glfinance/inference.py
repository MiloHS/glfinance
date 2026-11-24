"""Paired circular-bootstrap inference for portfolio variances.

Studentization follows Ledoit and Wolf (2011), using their 2008 Section 3.2.2:
log variance ratios, prewhitened Parzen HAC, and complete-block bootstrap SEs.
Reference equations and author code: https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html
"""
from __future__ import annotations

import numpy as np


def circular_bootstrap_indices(
    observations: int, block_length: int, replications: int, seed: int,
    sample_size: int | None = None,
) -> np.ndarray:
    """Draw contiguous wrapping blocks, keeping paired strategies together."""
    size = observations if sample_size is None else sample_size
    if observations < 2 or not 1 <= block_length <= observations or replications < 1 or size < 1:
        raise ValueError("Invalid circular-bootstrap sample size, block length, or replication count.")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, observations, size=(replications, int(np.ceil(size / block_length))))
    blocks = (starts[:, :, None] + np.arange(block_length)) % observations
    return blocks.reshape(replications, -1)[:, :size]


def _parzen_long_run_covariance(values: np.ndarray, projection: np.ndarray) -> tuple[float, float]:
    """Andrews AR(1) plug-in bandwidth with the reference finite-sample adjustment."""
    n = len(values)
    numerator = denominator = 0.0
    for column in values.T:
        design = np.column_stack([np.ones(n - 1), column[:-1]])
        if np.linalg.cond(design.T @ design, p=1) > 1e10:
            continue
        intercept, rho = np.linalg.lstsq(design, column[1:], rcond=None)[0]
        if abs(1.0 - rho) < 1e-6:
            continue
        residual_variance = np.mean((column[1:] - intercept - rho * column[:-1]) ** 2)
        numerator += 4 * rho ** 2 * residual_variance ** 2 / (1 - rho) ** 8
        denominator += residual_variance ** 2 / (1 - rho) ** 4
    alpha = numerator / denominator if denominator > 0 else 0.0
    bandwidth = min(2.6614 * (alpha * n) ** 0.2, n - 1)
    # Project first to avoid cancellation when two strategy paths are very similar.
    scores = values @ projection
    covariance = scores @ scores / n
    for lag in range(1, int(np.ceil(bandwidth))):
        x = lag / bandwidth
        weight = 1 - 6 * x ** 2 + 6 * x ** 3 if x < 0.5 else 2 * (1 - x) ** 3
        covariance += 2 * weight * (scores[lag:] @ scores[:-lag]) / n
    return float(covariance * n / (n - 4)), float(bandwidth)


def _log_variance_standard_error(returns: np.ndarray) -> tuple[float, float, bool]:
    """Delta-method SE from four raw moments, with bounded VAR(1) prewhitening."""
    n = len(returns)
    mean = returns.mean(axis=0)
    variance = returns.var(axis=0, ddof=0)
    gradient = np.array([-2 * mean[0] / variance[0], 2 * mean[1] / variance[1],
                         1 / variance[0], -1 / variance[1]])
    moments = np.column_stack([returns - mean, returns ** 2 - (returns ** 2).mean(axis=0)])
    design = moments[:-1]
    prewhitened = np.linalg.cond(design.T @ design, p=1) <= 1e10
    if prewhitened:
        coefficients = np.linalg.lstsq(design, moments[1:], rcond=None)[0].T
        left, singular, right = np.linalg.svd(coefficients)
        coefficients = (left * np.minimum(singular, 0.97)) @ right
        residuals = moments[1:] - design @ coefficients.T
        recolor = np.linalg.solve(np.eye(4) - coefficients, np.eye(4))
        covariance, bandwidth = _parzen_long_run_covariance(residuals, recolor.T @ gradient)
    else:
        covariance, bandwidth = _parzen_long_run_covariance(moments, gradient)
    return float(np.sqrt(max(0.0, covariance / n))), bandwidth, bool(prewhitened)


def studentized_variance_test(
    primary_returns: np.ndarray, benchmark_returns: np.ndarray,
    replications: int = 5000, block_length: int = 4, seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Ledoit-Wolf symmetric studentized interval and two-sided variance-ratio test.

    The observed statistic uses every observation. Each resample has floor(n/b)
    complete blocks, drawn from all n original dates (including the last dates
    when n is not divisible by b). No automatic block-length calibration is used.
    """
    primary = np.asarray(primary_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if primary.ndim != 1 or primary.shape != benchmark.shape:
        raise ValueError("Paired returns must be equal-length vectors.")
    returns = np.column_stack([primary, benchmark])
    n = len(returns)
    if n < 8 or not np.isfinite(returns).all() or not 0 < confidence < 1:
        raise ValueError("Inference requires at least eight finite paired observations and valid confidence.")
    if not 1 <= block_length <= n // 2 or replications < 2:
        raise ValueError("Inference requires at least two complete blocks and two resamples.")
    variance = returns.var(axis=0, ddof=0)
    if np.any(variance <= 0):
        raise ValueError("Both return series must have positive variance.")
    log_ratio = float(np.log(variance[0]) - np.log(variance[1]))
    influence = ((returns[:, 0] - primary.mean()) ** 2 / variance[0]
                 - (returns[:, 1] - benchmark.mean()) ** 2 / variance[1])
    sample_size = n // block_length * block_length
    result = {
        "weeks": n, "variance_ratio": float(variance[0] / variance[1]),
        "log_variance_ratio": log_ratio, "replications": replications,
        "block_length": block_length, "bootstrap_weeks": sample_size,
        "confidence": confidence, "hac_kernel": "Parzen",
    }
    if np.max(np.abs(influence)) < 1e-12:
        return {**result, "standard_error": 0.0, "hac_bandwidth": 0.0,
                "prewhitened": False, "critical_value": 0.0,
                "ratio_ci_lower": result["variance_ratio"], "ratio_ci_upper": result["variance_ratio"],
                "p_value": 1.0 if abs(log_ratio) < 1e-12 else 0.0, "degenerate_draws": replications}
    standard_error, bandwidth, prewhitened = _log_variance_standard_error(returns)
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise RuntimeError("The observed log variance ratio has an invalid HAC standard error.")
    indices = circular_bootstrap_indices(n, block_length, replications, seed, sample_size)
    samples = returns[indices]
    centered = samples - samples.mean(axis=1, keepdims=True)
    variances = np.mean(centered ** 2, axis=1)
    if np.any(variances <= 0):
        raise RuntimeError("A bootstrap sample has zero variance; studentization is undefined.")
    bootstrap_log_ratios = np.log(variances[:, 0]) - np.log(variances[:, 1])
    # Equivalent to the four-moment gradient in the paper's bootstrap SE.
    scores = centered[:, :, 0] ** 2 / variances[:, 0, None] - centered[:, :, 1] ** 2 / variances[:, 1, None]
    block_scores = scores.reshape(replications, -1, block_length).sum(axis=2) / np.sqrt(block_length)
    bootstrap_se = np.sqrt(np.mean(block_scores ** 2, axis=1) / (sample_size - 4))
    numerator = np.abs(bootstrap_log_ratios - log_ratio)
    roots = np.divide(numerator, bootstrap_se, out=np.full(replications, np.inf), where=bootstrap_se > 0)
    roots[(bootstrap_se == 0) & (numerator < 1e-12)] = 0.0
    rank = min(replications - 1, int(np.ceil(confidence * (replications + 1))) - 1)
    critical = float(np.sort(roots)[rank])
    half_width = critical * standard_error
    return {
        **result, "standard_error": standard_error, "hac_bandwidth": bandwidth,
        "prewhitened": prewhitened, "critical_value": critical,
        "ratio_ci_lower": float(np.exp(log_ratio - half_width)),
        "ratio_ci_upper": float(np.exp(log_ratio + half_width)),
        "p_value": float((1 + np.count_nonzero(roots >= abs(log_ratio) / standard_error)) / (replications + 1)),
        "degenerate_draws": int(np.count_nonzero(bootstrap_se == 0)),
    }
