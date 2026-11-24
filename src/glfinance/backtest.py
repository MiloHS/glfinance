from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from glfinance.covariance import (
    graph_covariance,
    ledoit_wolf_covariance,
    pca_factor_covariance,
    sample_covariance,
)
from glfinance.portfolio import (
    equal_weights,
    long_only_gmv_weights,
    portfolio_step,
    turnover,
    unconstrained_gmv_weights,
)


GRAPH_METHODS = {"GraphLW", "GraphSample", "GraphLWResidual"}
PCA_METHODS = {"PCAFactor", "PCALWResidual"}


@dataclass(frozen=True)
class Hyperparameters:
    training_window: int | None = None
    k_fraction: float | None = None
    rho: float | None = None

    def retained_modes(self, n_assets: int, *, pca: bool = False) -> int | None:
        if self.k_fraction is None:
            return None
        retained = min(n_assets, max(1, int(np.ceil(self.k_fraction * n_assets))))
        return min(retained, self.training_window - 1) if pca else retained


@dataclass
class BacktestArtifacts:
    weekly_performance: pd.DataFrame
    weekly_weights: pd.DataFrame
    selection_history: pd.DataFrame


@dataclass(frozen=True)
class CandidatePath:
    first_target_position: int
    realized_returns: np.ndarray
    weights: np.ndarray
    turnovers: np.ndarray

    def offsets(self, target_positions: range) -> np.ndarray:
        return np.arange(
            target_positions.start,
            target_positions.stop,
            target_positions.step,
            dtype=int,
        ) - self.first_target_position


def candidate_grid(
    method: str, portfolio_config: dict[str, Any], n_assets: int | None = None
) -> list[Hyperparameters]:
    training_windows = [int(value) for value in portfolio_config["training_windows"]]
    if method in GRAPH_METHODS | PCA_METHODS:
        candidates = [
            Hyperparameters(int(training), float(k_fraction), float(rho))
            for training, k_fraction, rho in product(
                training_windows,
                portfolio_config["k_fractions"],
                [0.0] if method == "PCAFactor" else portfolio_config["rho_values"],
            )
        ]
        if method in PCA_METHODS and n_assets is not None:
            unique = {}
            for parameters in candidates:
                key = (parameters.training_window, parameters.retained_modes(n_assets, pca=True), parameters.rho)
                unique.setdefault(key, parameters)
            return list(unique.values())
        return candidates
    if method == "LedoitWolfLong":
        return [Hyperparameters(value) for value in sorted(set(training_windows + [100]))]
    if method == "LedoitWolf100":
        return [Hyperparameters(100)]
    if method in {"LedoitWolf", "Sample"}:
        return [Hyperparameters(training_window=value) for value in training_windows]
    if method == "EqualWeight":
        return [Hyperparameters()]
    raise ValueError(f"Unknown portfolio method: {method}")


def _training_matrix(
    returns_values: np.ndarray, target_position: int, training_window: int
) -> np.ndarray:
    start = target_position - int(training_window)
    if start < 0:
        raise ValueError("Target does not have enough covariance training history.")
    matrix = returns_values[start:target_position]
    return matrix - matrix.mean(axis=0, keepdims=True)


def weights_for_target(
    method: str,
    parameters: Hyperparameters,
    target_position: int,
    returns: pd.DataFrame,
    graphs: dict[pd.Timestamp, np.ndarray],
    delta: float,
    long_only: bool = False,
    sample_ddof: int = 1,
) -> np.ndarray:
    """Calculate weights using data strictly before target_position."""
    n_assets = returns.shape[1]
    if method == "EqualWeight":
        return equal_weights(n_assets)
    if parameters.training_window is None:
        raise ValueError(f"{method} requires a covariance training window.")

    values = returns.to_numpy(dtype=float)
    training = _training_matrix(
        values, target_position, parameters.training_window
    )
    if method in GRAPH_METHODS:
        graph_date = pd.Timestamp(returns.index[target_position - 1])
        if graph_date not in graphs:
            raise KeyError(f"No causal graph is available for {graph_date.date()}.")
        retained = parameters.retained_modes(n_assets)
        if retained is None or parameters.rho is None:
            raise ValueError(f"{method} requires k_fraction and rho.")
        covariance = graph_covariance(
            graphs[graph_date],
            training,
            k=retained,
            rho=parameters.rho,
            delta=delta,
            subspace_estimator=(
                "Sample" if method == "GraphSample" else "LedoitWolf"
            ),
            restore_residual=method == "GraphLWResidual",
            sample_ddof=sample_ddof,
        )
    elif method in PCA_METHODS:
        retained = parameters.retained_modes(n_assets, pca=True)
        if retained is None or parameters.rho is None:
            raise ValueError(f"{method} requires k_fraction and rho.")
        covariance = pca_factor_covariance(
            training,
            k=retained,
            delta=delta,
            subspace_estimator="sample" if method == "PCAFactor" else "LedoitWolf",
            rho=parameters.rho,
        )
    elif method in {"LedoitWolf", "LedoitWolfLong", "LedoitWolf100"}:
        covariance = ledoit_wolf_covariance(training)
    elif method == "Sample":
        covariance = sample_covariance(training)
    else:
        raise ValueError(f"Unknown portfolio method: {method}")
    if long_only:
        return long_only_gmv_weights(covariance)
    return unconstrained_gmv_weights(covariance)


def evaluate_fixed_schedule(
    method: str,
    schedule: pd.DataFrame,
    returns: pd.DataFrame,
    risk_free: pd.Series,
    graphs: dict[pd.Timestamp, np.ndarray],
    delta: float,
    long_only: bool = False,
    sample_ddof: int = 1,
) -> BacktestArtifacts:
    """Replay a saved block-selection schedule without selecting any new parameters."""
    performance_rows, weight_rows = [], []
    previous_drifted = None
    previous_position = None
    values = returns.to_numpy(dtype=float)
    for row in schedule.sort_values("unseen_start").itertuples():
        if pd.Timestamp(row.validation_end) >= pd.Timestamp(row.unseen_start):
            raise ValueError("Saved validation must end before its unseen block.")
        parameters = Hyperparameters(int(row.training_window), float(row.k_fraction), float(row.rho))
        start = returns.index.get_loc(pd.Timestamp(row.unseen_start))
        stop = returns.index.get_loc(pd.Timestamp(row.unseen_end)) + 1
        for position in range(start, stop):
            if previous_position is not None and position != previous_position + 1:
                raise ValueError("Saved blocks must cover consecutive, nonoverlapping weeks.")
            weights = weights_for_target(
                method, parameters, position, returns, graphs, delta,
                long_only=long_only, sample_ddof=sample_ddof,
            )
            realized, drifted = portfolio_step(weights, values[position])
            date = returns.index[position]
            risk_free_return = float(risk_free.loc[date])
            performance_rows.append({
                "date": date, "block_id": row.block_id, "method": method,
                "portfolio_return": realized, "risk_free_return": risk_free_return,
                "excess_return": realized - risk_free_return,
                "turnover": turnover(weights, previous_drifted),
                "gross_exposure": float(np.abs(weights).sum()),
                "training_window": parameters.training_window,
                "k_fraction": parameters.k_fraction, "k": parameters.retained_modes(returns.shape[1]),
                "rho": parameters.rho,
            })
            weight_rows.extend({"date": date, "block_id": row.block_id, "method": method,
                                "symbol": symbol, "weight": float(weight)}
                               for symbol, weight in zip(returns.columns, weights))
            previous_drifted, previous_position = drifted, position
    return BacktestArtifacts(pd.DataFrame(performance_rows), pd.DataFrame(weight_rows), schedule.copy())


def evaluate_parameters(
    method: str,
    parameters: Hyperparameters,
    target_positions: Iterable[int],
    returns: pd.DataFrame,
    graphs: dict[pd.Timestamp, np.ndarray],
    delta: float,
    long_only: bool = False,
) -> tuple[float, float]:
    """Return validation realized variance and mean turnover."""
    portfolio_returns: list[float] = []
    turnovers: list[float] = []
    previous_weights: np.ndarray | None = None
    values = returns.to_numpy(dtype=float)
    for target_position in target_positions:
        weights = weights_for_target(
            method,
            parameters,
            target_position,
            returns,
            graphs,
            delta,
            long_only=long_only,
        )
        realized, drifted = portfolio_step(weights, values[target_position])
        portfolio_returns.append(realized)
        turnovers.append(turnover(weights, previous_weights))
        previous_weights = drifted
    if len(portfolio_returns) < 2:
        raise ValueError("Validation requires at least two target weeks.")
    return (
        float(np.var(portfolio_returns, ddof=1)),
        float(np.nanmean(turnovers)),
    )


def select_parameters(
    method: str,
    validation_positions: range,
    returns: pd.DataFrame,
    graphs: dict[pd.Timestamp, np.ndarray],
    portfolio_config: dict[str, Any],
) -> tuple[Hyperparameters, float, float]:
    """Grid search with deterministic variance/turnover/model tie-breaking."""
    evaluated: list[tuple[float, float, int, float, float, Hyperparameters]] = []
    delta = float(portfolio_config["delta"])
    long_only = bool(portfolio_config.get("long_only", False))
    for parameters in candidate_grid(method, portfolio_config, returns.shape[1]):
        realized_variance, mean_turnover = evaluate_parameters(
            method,
            parameters,
            validation_positions,
            returns,
            graphs,
            delta,
            long_only=long_only,
        )
        # Exact ties: lower turnover, longer T, smaller k, larger rho.
        evaluated.append(
            (
                realized_variance,
                mean_turnover,
                -(parameters.training_window or 0),
                parameters.k_fraction or 0.0,
                -(parameters.rho or 0.0),
                parameters,
            )
        )
    best = min(evaluated, key=lambda item: item[:-1])
    return best[-1], best[0], best[1]


def _precompute_candidate_paths(
    methods: list[str],
    first_target_position: int,
    returns: pd.DataFrame,
    graphs: dict[pd.Timestamp, np.ndarray],
    portfolio_config: dict[str, Any],
) -> dict[str, dict[Hyperparameters, CandidatePath]]:
    """Calculate each candidate's weekly weights once for all needed dates."""
    values = returns.to_numpy(dtype=float)
    delta = float(portfolio_config["delta"])
    long_only = bool(portfolio_config.get("long_only", False))
    target_positions = range(first_target_position, len(returns))
    paths: dict[str, dict[Hyperparameters, CandidatePath]] = {}

    for method in methods:
        method_paths: dict[Hyperparameters, CandidatePath] = {}
        for parameters in candidate_grid(method, portfolio_config, returns.shape[1]):
            weights_rows: list[np.ndarray] = []
            realized_returns: list[float] = []
            turnovers: list[float] = []
            previous_drifted: np.ndarray | None = None
            for target_position in target_positions:
                weights = weights_for_target(
                    method,
                    parameters,
                    target_position,
                    returns,
                    graphs,
                    delta,
                    long_only=long_only,
                )
                weights_rows.append(weights)
                realized, drifted = portfolio_step(weights, values[target_position])
                realized_returns.append(realized)
                turnovers.append(turnover(weights, previous_drifted))
                previous_drifted = drifted
            method_paths[parameters] = CandidatePath(
                first_target_position=first_target_position,
                realized_returns=np.asarray(realized_returns, dtype=float),
                weights=np.vstack(weights_rows),
                turnovers=np.asarray(turnovers, dtype=float),
            )
        paths[method] = method_paths
    return paths


def _select_from_candidate_paths(
    method_paths: dict[Hyperparameters, CandidatePath],
    validation_positions: range,
) -> tuple[Hyperparameters, float, float]:
    evaluated: list[tuple[float, float, int, float, float, Hyperparameters]] = []
    for parameters, path in method_paths.items():
        offsets = path.offsets(validation_positions)
        realized = path.realized_returns[offsets]
        weekly_turnovers = path.turnovers[offsets].copy()
        weekly_turnovers[0] = np.nan  # Exclude establishment of this validation path.
        evaluated.append(
            (
                float(np.var(realized, ddof=1)),
                float(np.nanmean(weekly_turnovers)),
                -(parameters.training_window or 0),
                parameters.k_fraction or 0.0,
                -(parameters.rho or 0.0),
                parameters,
            )
        )
    best = min(evaluated, key=lambda item: item[:-1])
    return best[-1], best[0], best[1]


def run_nested_walk_forward(
    returns: pd.DataFrame,
    risk_free: pd.Series,
    graphs: dict[pd.Timestamp, np.ndarray],
    outer_start_position: int,
    experiment_config: dict[str, Any],
    portfolio_config: dict[str, Any],
) -> BacktestArtifacts:
    """Run quarterly recalibration and weekly unseen portfolio evaluation."""
    methods = list(portfolio_config["methods"])
    validation_weeks = int(experiment_config["validation_weeks"])
    recalibration_weeks = int(experiment_config["recalibration_weeks"])
    n_observations, n_assets = returns.shape
    if outer_start_position - validation_weeks < 0:
        raise ValueError("Outer period lacks the requested validation history.")

    performance_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    previous_weights: dict[str, np.ndarray | None] = {
        method: None for method in methods
    }
    returns_values = returns.to_numpy(dtype=float)
    first_target_position = outer_start_position - validation_weeks
    candidate_paths = _precompute_candidate_paths(
        methods,
        first_target_position,
        returns,
        graphs,
        portfolio_config,
    )

    block_id = 0
    for block_start in range(
        outer_start_position, n_observations, recalibration_weeks
    ):
        block_stop = min(block_start + recalibration_weeks, n_observations)
        validation = range(
            block_start - validation_weeks,
            block_start,
        )
        selected: dict[str, Hyperparameters] = {}
        for method in methods:
            parameters, validation_variance, validation_turnover = (
                _select_from_candidate_paths(
                    candidate_paths[method],
                    validation,
                )
            )
            selected[method] = parameters
            selection_rows.append(
                {
                    "block_id": block_id,
                    "method": method,
                    "validation_start": returns.index[validation.start],
                    "validation_end": returns.index[validation.stop - 1],
                    "unseen_start": returns.index[block_start],
                    "unseen_end": returns.index[block_stop - 1],
                    "training_window": parameters.training_window,
                    "k_fraction": parameters.k_fraction,
                    "k": parameters.retained_modes(n_assets, pca=method in PCA_METHODS),
                    "rho": parameters.rho,
                    "validation_variance": validation_variance,
                    "validation_turnover": validation_turnover,
                }
            )

        for target_position in range(block_start, block_stop):
            date = pd.Timestamp(returns.index[target_position])
            for method in methods:
                path = candidate_paths[method][selected[method]]
                offset = target_position - path.first_target_position
                weights = path.weights[offset]
                portfolio_return, drifted = portfolio_step(
                    weights, returns_values[target_position]
                )
                risk_free_return = float(risk_free.loc[date])
                current_turnover = turnover(weights, previous_weights[method])
                previous_weights[method] = drifted
                performance_rows.append(
                    {
                        "date": date,
                        "block_id": block_id,
                        "method": method,
                        "portfolio_return": portfolio_return,
                        "risk_free_return": risk_free_return,
                        "excess_return": portfolio_return - risk_free_return,
                        "turnover": current_turnover,
                        "gross_exposure": float(np.sum(np.abs(weights))),
                        "training_window": selected[method].training_window,
                        "k_fraction": selected[method].k_fraction,
                        "k": selected[method].retained_modes(n_assets, pca=method in PCA_METHODS),
                        "rho": selected[method].rho,
                    }
                )
                for symbol, weight in zip(returns.columns, weights):
                    weight_rows.append(
                        {
                            "date": date,
                            "block_id": block_id,
                            "method": method,
                            "symbol": symbol,
                            "weight": float(weight),
                        }
                    )
        block_id += 1

    return BacktestArtifacts(
        weekly_performance=pd.DataFrame(performance_rows),
        weekly_weights=pd.DataFrame(weight_rows),
        selection_history=pd.DataFrame(selection_rows),
    )
