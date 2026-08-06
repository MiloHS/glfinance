import numpy as np
import pandas as pd

from glfinance.backtest import run_nested_walk_forward
from glfinance.data import synthetic_inputs
from glfinance.graph import learn_causal_graphs


def complete_graph_laplacian(n_assets: int) -> np.ndarray:
    adjacency = np.ones((n_assets, n_assets)) - np.eye(n_assets)
    return np.diag(adjacency.sum(axis=1)) - adjacency


def small_configs() -> tuple[dict, dict]:
    experiment = {
        "validation_weeks": 8,
        "recalibration_weeks": 4,
    }
    portfolio = {
        "training_windows": [5, 7],
        "k_fractions": [0.5, 0.75],
        "rho_values": [0.1, 0.2],
        "delta": 1.0e-6,
        "methods": [
            "GraphLW",
            "GraphSample",
            "LedoitWolf",
            "Sample",
            "EqualWeight",
        ],
    }
    return experiment, portfolio


def test_selections_are_fixed_inside_each_unseen_block() -> None:
    tickers = ["A", "B", "C", "D"]
    returns, risk_free, _ = synthetic_inputs(tickers, observations=36, seed=3)
    graph = complete_graph_laplacian(len(tickers))
    graphs = {pd.Timestamp(date): graph for date in returns.index[:-1]}
    experiment, portfolio = small_configs()
    artifacts = run_nested_walk_forward(
        returns,
        risk_free,
        graphs,
        outer_start_position=20,
        experiment_config=experiment,
        portfolio_config=portfolio,
    )
    assert len(artifacts.weekly_performance["date"].unique()) == 16
    for (_, method), group in artifacts.weekly_performance.groupby(
        ["block_id", "method"]
    ):
        assert group["training_window"].nunique(dropna=False) == 1
        assert group["k"].nunique(dropna=False) == 1
        assert group["rho"].nunique(dropna=False) == 1
    assert artifacts.weekly_performance.groupby("block_id")["date"].nunique().max() == 4


def test_future_changes_do_not_change_earlier_graphs() -> None:
    tickers = ["A", "B", "C", "D"]
    returns, _, _ = synthetic_inputs(tickers, observations=40, seed=11)
    changed = returns.copy()
    changed.iloc[30:] = changed.iloc[30:] * 100.0
    config = {
        "gamma_1": 0.1,
        "gamma_2": 0.1,
        "gamma_3": 0.1,
        "diagonal_floor": 1.0e-6,
        "prune_threshold": 1.0e-7,
        "solver": "SCS",
        "solver_options": {"eps": 1.0e-5, "max_iters": 10000, "verbose": False},
    }
    original_graphs, _ = learn_causal_graphs(
        returns,
        first_graph_position=20,
        last_graph_position=29,
        lookback_weeks=12,
        graph_config=config,
    )
    changed_graphs, _ = learn_causal_graphs(
        changed,
        first_graph_position=20,
        last_graph_position=29,
        lookback_weeks=12,
        graph_config=config,
    )
    for date in original_graphs:
        np.testing.assert_allclose(
            original_graphs[date], changed_graphs[date], atol=1.0e-7
        )

