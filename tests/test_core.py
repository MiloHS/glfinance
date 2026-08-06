import numpy as np

from glfinance.covariance import graph_covariance
from glfinance.graph import correlation_distance, solve_causal_laplacian
from glfinance.portfolio import long_only_gmv_weights, unconstrained_gmv_weights


def complete_graph_laplacian(n_assets: int) -> np.ndarray:
    adjacency = np.ones((n_assets, n_assets)) - np.eye(n_assets)
    return np.diag(adjacency.sum(axis=1)) - adjacency


def test_correlation_distance_is_symmetric_and_zero_diagonal() -> None:
    signals = np.array(
        [
            [0.1, 0.2, -0.1, 0.0],
            [0.2, 0.4, -0.2, 0.0],
            [-0.1, 0.0, 0.1, 0.2],
        ]
    )
    distance = correlation_distance(signals)
    np.testing.assert_allclose(distance, distance.T)
    np.testing.assert_allclose(np.diag(distance), 0.0)
    assert np.all(distance >= 0.0)


def test_causal_solver_returns_valid_laplacian() -> None:
    signals = np.array(
        [
            [0.01, 0.02, -0.01, 0.03, 0.01],
            [0.02, 0.01, -0.02, 0.02, 0.00],
            [-0.01, 0.00, 0.02, -0.01, 0.03],
            [0.00, -0.01, 0.01, 0.00, 0.02],
        ]
    )
    distance = correlation_distance(signals)
    config = {
        "gamma_1": 0.1,
        "gamma_2": 0.1,
        "gamma_3": 0.1,
        "diagonal_floor": 1.0e-6,
        "prune_threshold": 1.0e-7,
        "solver": "SCS",
        "solver_options": {"eps": 1.0e-5, "max_iters": 10000, "verbose": False},
    }
    laplacian, status, _, _ = solve_causal_laplacian(
        signals[:, -2],
        signals[:, -1],
        distance,
        previous_laplacian=None,
        graph_config=config,
    )
    assert status in {"optimal", "optimal_inaccurate"}
    np.testing.assert_allclose(laplacian, laplacian.T, atol=1.0e-8)
    np.testing.assert_allclose(laplacian.sum(axis=1), 0.0, atol=1.0e-8)
    off_diagonal = laplacian.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.all(off_diagonal <= 1.0e-10)
    assert np.all(np.linalg.eigvalsh(laplacian) >= -1.0e-8)


def test_covariance_and_gmv_invariants() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(size=(30, 5))
    returns -= returns.mean(axis=0)
    laplacian = complete_graph_laplacian(5)
    for estimator in ("Sample", "LedoitWolf"):
        covariance = graph_covariance(
            laplacian,
            returns,
            k=3,
            rho=0.15,
            delta=1.0e-6,
            subspace_estimator=estimator,
        )
        np.testing.assert_allclose(covariance, covariance.T)
        assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
        weights = unconstrained_gmv_weights(covariance)
        np.testing.assert_allclose(weights.sum(), 1.0)


def test_long_only_gmv_matches_scipy_reference() -> None:
    from scipy.optimize import minimize

    rng = np.random.default_rng(23)
    loadings = rng.normal(size=(7, 7))
    covariance = loadings @ loadings.T + 0.1 * np.eye(7)
    actual = long_only_gmv_weights(covariance)
    reference = minimize(
        lambda weights: float(weights @ covariance @ weights),
        x0=np.ones(7) / 7.0,
        jac=lambda weights: 2.0 * covariance @ weights,
        bounds=[(0.0, 1.0)] * 7,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 2000},
    )
    assert reference.success
    assert np.all(actual >= -1.0e-12)
    np.testing.assert_allclose(actual.sum(), 1.0)
    np.testing.assert_allclose(
        actual @ covariance @ actual,
        reference.fun,
        rtol=1.0e-7,
        atol=1.0e-10,
    )
