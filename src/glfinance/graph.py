from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import os
from pathlib import Path

import scipy

# Ensure IDAES solver directory and gfortran libraries are in PATH / DYLD_LIBRARY_PATH
idaes_bin = str(Path.home() / ".idaes" / "bin")
if idaes_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + idaes_bin

scipy_dylibs = str(Path(scipy.__file__).parent / ".dylibs")
current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
if scipy_dylibs not in current_dyld:
    os.environ["DYLD_LIBRARY_PATH"] = scipy_dylibs + (os.pathsep + current_dyld if current_dyld else "")




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


def k_lag_graph_learning_model(
    X_data: np.ndarray,
    P_data: np.ndarray,
    Z_data: np.ndarray,
    gamma1: float,
    gamma2: float,
    gamma3: float,
    k: int,
):
    """Pyomo model for the paper's finite-lag polynomial graph process."""
    import pyomo.environ as pyo

    X_data = np.asarray(X_data, dtype=float)
    P_data = np.asarray(P_data, dtype=float)
    Z_data = np.asarray(Z_data, dtype=float)

    N, M = X_data.shape
    assert P_data.shape[0] == M and P_data.shape[1] == N, "P_data must be (M, N, J)"
    J = P_data.shape[2]
    assert Z_data.shape == (M, N, N), "Z_data must be (M, N, N)"
    assert M >= 2, "Need at least two time points"
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    effective_k = min(int(k), M - 1)

    model = pyo.ConcreteModel()

    model.Nidx = pyo.RangeSet(0, N - 1)
    model.AllT = pyo.RangeSet(0, M - 1)
    model.PredT = pyo.RangeSet(1, M - 1)
    model.PowT = pyo.RangeSet(0, M - 2)
    model.Jidx = pyo.RangeSet(0, J - 1)

    j_map = {}
    idx = 0
    for lag in range(1, effective_k + 1):
        for p in range(1, lag + 1):
            j_map[(lag, p)] = idx
            idx += 1
    model.j_map = j_map
    J_total = idx
    model.Cidx = pyo.RangeSet(0, J_total - 1)

    def max_pow_init(m, s):
        return min(effective_k, (M - 1) - s)

    model.max_pow = pyo.Param(model.PowT, initialize=max_pow_init, within=pyo.NonNegativeIntegers, mutable=False)

    def TPpow_init(m):
        return (
            (s, p)
            for s in m.PowT
            for p in range(1, int(m.max_pow[s]) + 1)
        )

    model.TPpow = pyo.Set(dimen=2, initialize=TPpow_init)

    model.L = pyo.Var(model.AllT, model.Nidx, model.Nidx, domain=pyo.Reals)

    for t in model.AllT:
        for i in model.Nidx:
            model.L[t, i, i].setlb(1e-6)

    model.c1 = pyo.Var(model.Cidx, domain=pyo.Reals)
    model.c2 = pyo.Var(model.Cidx, domain=pyo.Reals)
    model.beta = pyo.Var(model.Jidx, model.PredT, domain=pyo.Reals)

    model.X = pyo.Param(model.Nidx, model.AllT, initialize=lambda m, i, t: float(X_data[i, t]), mutable=False)
    model.P = pyo.Param(model.AllT, model.Nidx, model.Jidx, initialize=lambda m, t, i, j: float(P_data[t, i, j]), mutable=False)
    model.Z = pyo.Param(model.AllT, model.Nidx, model.Nidx, initialize=lambda m, t, i, j: float(Z_data[t, i, j]), mutable=False)

    def A_rule(m, t, i, j):
        if i == j:
            return 0.0
        return -m.L[t, i, j]

    model.A = pyo.Expression(model.AllT, model.Nidx, model.Nidx, rule=A_rule)

    def D_rule(m, t, i, j):
        if i != j:
            return 0.0
        return pyo.quicksum(m.A[t, i, r] for r in m.Nidx)

    model.D = pyo.Expression(model.AllT, model.Nidx, model.Nidx, rule=D_rule)

    def Apow_rule(m, s, p, i, j):
        if p == 1:
            return m.A[s, i, j]
        return pyo.quicksum(m.A[s, i, r] * m.A_pow[s, p - 1, r, j] for r in m.Nidx)

    def Dpow_rule(m, s, p, i, j):
        if p == 1:
            return m.D[s, i, j]
        return pyo.quicksum(m.D[s, i, r] * m.D_pow[s, p - 1, r, j] for r in m.Nidx)

    model.A_pow = pyo.Expression(model.TPpow, model.Nidx, model.Nidx, rule=Apow_rule)
    model.D_pow = pyo.Expression(model.TPpow, model.Nidx, model.Nidx, rule=Dpow_rule)

    model.s_true = pyo.Expression(model.Nidx, model.PredT, rule=lambda m, i, t: m.X[i, t])

    def s_hat_rule(m, i, t):
        expr = pyo.quicksum(m.P[t, i, j] * m.beta[j, t] for j in m.Jidx)
        max_lag = min(effective_k, t)
        for lag in range(1, max_lag + 1):
            s = t - lag
            for p in range(1, lag + 1):
                idx = m.j_map[(lag, p)]
                expr += m.c1[idx] * pyo.quicksum(m.A_pow[s, p, i, r] * m.X[r, s] for r in m.Nidx)
                expr += m.c2[idx] * pyo.quicksum(m.D_pow[s, p, i, r] * m.X[r, s] for r in m.Nidx)
        return expr

    model.s_hat = pyo.Expression(model.Nidx, model.PredT, rule=s_hat_rule)

    def obj_rule(m):
        trace_term = -pyo.quicksum(m.L[t, i, j] * m.Z[t, j, i] for t in m.AllT for i in m.Nidx for j in m.Nidx)
        log_term = -gamma1 * pyo.quicksum(pyo.log(m.L[t, i, i]) for t in m.AllT for i in m.Nidx)
        frob_L = gamma2 * pyo.quicksum(m.L[t, i, j] ** 2 for t in m.AllT for i in m.Nidx for j in m.Nidx)
        pred = pyo.quicksum((m.s_true[i, t] - m.s_hat[i, t]) ** 2 for t in m.PredT for i in m.Nidx)
        smooth = gamma3 * pyo.quicksum((m.L[t, i, j] - m.L[t - 1, i, j]) ** 2 for t in m.AllT if t > 0 for i in m.Nidx for j in m.Nidx)
        return trace_term + log_term + frob_L + pred + smooth

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def zero_row_rule(m, t, i):
        return pyo.quicksum(m.L[t, i, j] for j in m.Nidx) == 0.0

    model.zero_row = pyo.Constraint(model.AllT, model.Nidx, rule=zero_row_rule)

    def symmetry_rule(m, t, i, j):
        if i < j:
            return m.L[t, i, j] == m.L[t, j, i]
        return pyo.Constraint.Skip

    model.symmetry = pyo.Constraint(model.AllT, model.Nidx, model.Nidx, rule=symmetry_rule)

    def offdiag_rule(m, t, i, j):
        if i != j:
            return m.L[t, i, j] <= 0.0
        return pyo.Constraint.Skip

    model.offdiag = pyo.Constraint(model.AllT, model.Nidx, model.Nidx, rule=offdiag_rule)

    return model


def solve_instance(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    k: int = 2,
    gammas: tuple[float, float, float] = (1e-3, 1e-1, 1e-3),
    ipopt_opts: dict[str, Any] | None = None,
):
    """Solve the joint graph learning model with IPOPT."""
    import pyomo.environ as pyo

    gamma1, gamma2, gamma3 = gammas
    m = k_lag_graph_learning_model(X, P, Z, gamma1, gamma2, gamma3, k)

    N, M = X.shape
    for t in range(M):
        for i in range(N):
            for j in range(N):
                if i == j:
                    m.L[t, i, i].value = 1.0
                elif i < j:
                    m.L[t, i, j].value = -0.05
                    m.L[t, j, i].value = -0.05
        for i in range(N):
            row_off = sum(m.L[t, i, j].value for j in range(N) if j != i)
            m.L[t, i, i].value = max(-row_off, 1e-6)

    for j in m.Cidx:
        m.c1[j].value = 0.0
        m.c2[j].value = 0.0
    for t in m.PredT:
        for j in m.Jidx:
            m.beta[j, t].value = 0.0

    solver = pyo.SolverFactory("ipopt")
    if ipopt_opts:
        for k_, v_ in ipopt_opts.items():
            solver.options[k_] = v_
    results = solver.solve(m, tee=True)

    obj_val = pyo.value(m.obj)

    L = np.zeros((M, N, N))
    for t in range(M):
        for i in range(N):
            for j in range(N):
                L[t, i, j] = pyo.value(m.L[t, i, j])

    J_total = len(m.Cidx)
    c1 = np.array([pyo.value(m.c1[j]) for j in range(J_total)])
    c2 = np.array([pyo.value(m.c2[j]) for j in range(J_total)])

    J = len(m.Jidx)
    beta = np.zeros((J, M - 1))
    for t in m.PredT:
        for j in m.Jidx:
            beta[j, t - 1] = pyo.value(m.beta[j, t])

    s_hat = np.zeros((N, M - 1))
    for t in m.PredT:
        for i in m.Nidx:
            s_hat[i, t - 1] = pyo.value(m.s_hat[i, t])

    return m, results, obj_val, L, c1, c2, beta, s_hat

