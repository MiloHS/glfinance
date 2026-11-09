import inspect

import numpy as np
import pyomo.environ as pyo
try:
    import pytest
except ImportError:
    import contextlib
    import re

    class _PytestFallback:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as e:
                if match and not re.search(match, str(e)):
                    raise AssertionError(f"Pattern {match!r} not found in {str(e)!r}")
            else:
                raise AssertionError(f"Expected exception {exc} but none was raised")

    pytest = _PytestFallback()

from glfinance.experiments.crisis_top100 import (
    run_crisis_experiment,
    solve_crisis_model,
)
from glfinance.graph import k_lag_graph_learning_model, solve_instance


def small_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([[1.0, 3.0, 2.0], [2.0, -1.0, 4.0]])
    regressors = np.zeros((3, 2, 2))
    regressors[:, :, 0] = 1.0
    regressors[:, :, 1] = np.array([[0.0, 1.0], [2.0, -1.0], [1.0, 3.0]])
    distances = np.zeros((3, 2, 2))
    return x, regressors, distances


def build_small_model(k: int = 1):
    x, regressors, distances = small_inputs()
    return k_lag_graph_learning_model(
        x,
        regressors,
        distances,
        gamma1=40.0,
        gamma2=1.5e-5,
        gamma3=1.5e-2,
        k=k,
    )


def test_crisis_defaults_match_manuscript() -> None:
    expected_gammas = (40.0, 1.5e-5, 1.5e-2)
    for function in (solve_crisis_model, run_crisis_experiment):
        parameters = inspect.signature(function).parameters
        assert parameters["k"].default == 1
        assert parameters["gammas"].default == expected_gammas


def test_one_lag_model_has_only_first_order_coefficients() -> None:
    model = build_small_model()

    assert model.j_map == {(1, 1): 0}
    assert list(model.Cidx) == [0]
    assert set(model.TPpow) == {(0, 1), (1, 1)}
    assert all(power >= 1 for _, power in model.TPpow)
    assert model.L[0, 0, 0].lb == 1e-6


def test_one_lag_prediction_matches_manuscript_equation() -> None:
    model = build_small_model()
    x, regressors, _ = small_inputs()
    laplacian = np.array([[2.0, -2.0], [-2.0, 2.0]])
    adjacency = np.array([[0.0, 2.0], [2.0, 0.0]])
    degree = np.diag([2.0, 2.0])

    for t in model.AllT:
        for i in model.Nidx:
            for j in model.Nidx:
                model.L[t, i, j].value = laplacian[i, j]
    model.c1[0].value = 0.3
    model.c2[0].value = -0.2

    for t in model.PredT:
        beta_t = np.array([0.1 * t, -0.25])
        for j in model.Jidx:
            model.beta[j, t].value = beta_t[j]
        expected = (
            0.3 * adjacency @ x[:, t - 1]
            - 0.2 * degree @ x[:, t - 1]
            + regressors[t] @ beta_t
        )
        actual = np.array([pyo.value(model.s_hat[i, t]) for i in model.Nidx])
        np.testing.assert_allclose(actual, expected)


def test_model_rejects_zero_lags() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_small_model(k=0)


def test_solver_outputs_exclude_initial_time_point(monkeypatch=None) -> None:
    class FakeSolver:
        def __init__(self) -> None:
            self.options = {}

        def solve(self, model, tee=False):
            return object()

    orig_factory = pyo.SolverFactory
    if monkeypatch is not None:
        monkeypatch.setattr(pyo, "SolverFactory", lambda name: FakeSolver())
    else:
        pyo.SolverFactory = lambda name: FakeSolver()

    try:
        x, regressors, distances = small_inputs()
        _, _, _, laplacians, c1, c2, beta, predictions = solve_instance(
            x,
            regressors,
            distances,
            k=1,
            gammas=(40.0, 1.5e-5, 1.5e-2),
        )

        assert laplacians.shape == (3, 2, 2)
        assert c1.shape == (1,)
        assert c2.shape == (1,)
        assert beta.shape == (2, 2)
        assert predictions.shape == (2, 2)
    finally:
        if monkeypatch is None:
            pyo.SolverFactory = orig_factory
