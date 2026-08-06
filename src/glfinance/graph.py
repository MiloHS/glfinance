from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphSolveRecord:
    date: pd.Timestamp
    status: str
    objective: float
    solver: str
    iterations: int | None
    solve_time: float | None


def correlation_distance(signals: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Return the absolute-correlation dissimilarity used by the manuscript."""
    values = np.asarray(signals, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("signals must have shape (assets, observations>=2).")
    correlation = np.corrcoef(values)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    distance = 1.0 / (np.abs(correlation) + epsilon) - 1.0 / (1.0 + epsilon)
    np.fill_diagonal(distance, 0.0)
    return 0.5 * (distance + distance.T)


def adjacency_and_degree(laplacian: cp.Expression) -> tuple[cp.Expression, cp.Expression]:
    diagonal = cp.diag(cp.diag(laplacian))
    adjacency = -laplacian + diagonal
    degree = cp.diag(cp.sum(adjacency, axis=1))
    return adjacency, degree


def _prune_valid_laplacian(laplacian: np.ndarray, threshold: float) -> np.ndarray:
    """Prune adjacency weights, then rebuild the diagonal to retain row sums."""
    symmetric = 0.5 * (laplacian + laplacian.T)
    adjacency = -symmetric
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency < threshold] = 0.0
    adjacency = 0.5 * (adjacency + adjacency.T)
    result = np.diag(adjacency.sum(axis=1)) - adjacency
    return 0.5 * (result + result.T)


def solve_causal_laplacian(
    x_previous: np.ndarray,
    x_current: np.ndarray,
    distance: np.ndarray,
    previous_laplacian: np.ndarray | None,
    graph_config: dict[str, Any],
) -> tuple[np.ndarray, str, float, cp.SolverStats]:
    """Learn one graph using current/past observations and at most L(t-1)."""
    x_previous = np.asarray(x_previous, dtype=float).reshape(-1)
    x_current = np.asarray(x_current, dtype=float).reshape(-1)
    distance = np.asarray(distance, dtype=float)
    n_assets = x_current.size
    if x_previous.shape != x_current.shape:
        raise ValueError("Previous and current signals must have the same shape.")
    if distance.shape != (n_assets, n_assets):
        raise ValueError("Distance matrix has the wrong shape.")
    if previous_laplacian is not None and np.shape(previous_laplacian) != distance.shape:
        raise ValueError("Previous Laplacian has the wrong shape.")

    laplacian = cp.Variable((n_assets, n_assets), symmetric=True)
    adjacency, degree = adjacency_and_degree(laplacian)
    prediction = (adjacency + degree) @ x_previous

    n_squared = float(n_assets * n_assets)
    dirichlet = -cp.trace(laplacian @ distance) / n_squared
    log_barrier = -float(graph_config["gamma_1"]) * cp.sum(
        cp.log(cp.diag(laplacian))
    ) / float(n_assets)
    frobenius = float(graph_config["gamma_2"]) * cp.sum_squares(
        laplacian
    ) / n_squared
    prediction_error = cp.sum_squares(x_current - prediction) / float(n_assets)

    objective_terms = [dirichlet, log_barrier, frobenius, prediction_error]
    if previous_laplacian is not None:
        temporal = float(graph_config["gamma_3"]) * cp.sum_squares(
            laplacian - np.asarray(previous_laplacian)
        ) / n_squared
        objective_terms.append(temporal)

    off_diagonal_mask = np.ones((n_assets, n_assets)) - np.eye(n_assets)
    constraints = [
        cp.diag(laplacian) >= float(graph_config["diagonal_floor"]),
        laplacian @ np.ones(n_assets) == 0,
        cp.multiply(off_diagonal_mask, laplacian) <= 0,
    ]
    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
    solver_name = str(graph_config.get("solver", "SCS")).upper()
    solver = getattr(cp, solver_name, solver_name)
    options = dict(graph_config.get("solver_options", {}))
    problem.solve(solver=solver, **options)

    accepted = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if problem.status not in accepted or laplacian.value is None:
        raise RuntimeError(
            f"Graph solve failed with status={problem.status!r}, solver={solver_name}."
        )
    cleaned = _prune_valid_laplacian(
        np.asarray(laplacian.value), float(graph_config["prune_threshold"])
    )
    return cleaned, str(problem.status), float(problem.value), problem.solver_stats


def learn_causal_graphs(
    returns: pd.DataFrame,
    first_graph_position: int,
    last_graph_position: int,
    lookback_weeks: int,
    graph_config: dict[str, Any],
) -> tuple[dict[pd.Timestamp, np.ndarray], list[GraphSolveRecord]]:
    """Learn chronological graphs for an inclusive positional date range."""
    if first_graph_position < lookback_weeks - 1:
        raise ValueError("First graph date lacks the configured distance lookback.")
    if last_graph_position >= len(returns):
        raise ValueError("Last graph position is outside the returns data.")
    if first_graph_position > last_graph_position:
        raise ValueError("First graph position must not exceed last graph position.")

    graphs: dict[pd.Timestamp, np.ndarray] = {}
    records: list[GraphSolveRecord] = []
    previous: np.ndarray | None = None
    values = returns.to_numpy(dtype=float)

    for position in range(first_graph_position, last_graph_position + 1):
        window = values[
            position - lookback_weeks + 1 : position + 1
        ].T
        distance = correlation_distance(window)
        graph, status, objective, stats = solve_causal_laplacian(
            values[position - 1],
            values[position],
            distance,
            previous,
            graph_config,
        )
        date = pd.Timestamp(returns.index[position])
        graphs[date] = graph
        records.append(
            GraphSolveRecord(
                date=date,
                status=status,
                objective=objective,
                solver=str(graph_config.get("solver", "SCS")),
                iterations=getattr(stats, "num_iters", None),
                solve_time=getattr(stats, "solve_time", None),
            )
        )
        previous = graph
    return graphs, records

