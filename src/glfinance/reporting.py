from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_performance(
    weekly_performance: pd.DataFrame, group_columns: Iterable[str]
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    columns = list(group_columns)
    grouper = columns[0] if len(columns) == 1 else columns
    for keys, group in weekly_performance.groupby(grouper, sort=False):
        key_values = (keys,) if len(columns) == 1 else tuple(keys)
        returns = group["portfolio_return"].to_numpy(dtype=float)
        excess = group["excess_return"].to_numpy(dtype=float)
        realized_variance = float(np.var(returns, ddof=1))
        realized_volatility = float(np.sqrt(realized_variance))
        excess_std = float(np.std(excess, ddof=1))
        row: dict[str, float | str | int] = dict(zip(columns, key_values))
        row.update(
            {
                "weeks": int(len(group)),
                "realized_variance": realized_variance,
                "weekly_volatility": realized_volatility,
                "annualized_volatility": realized_volatility * np.sqrt(52.0),
                "annualized_mean_return": float(np.mean(returns) * 52.0),
                "annualized_sharpe": (
                    float(np.mean(excess) / excess_std * np.sqrt(52.0))
                    if excess_std > 0
                    else np.nan
                ),
                "mean_turnover": float(group["turnover"].mean()),
                "mean_gross_exposure": float(group["gross_exposure"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def moving_block_bootstrap_variance_difference(
    primary_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    replications: int,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired moving-block bootstrap of Var(primary)-Var(benchmark)."""
    primary = np.asarray(primary_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if primary.shape != benchmark.shape or primary.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be equal-length vectors.")
    if len(primary) < 2 or block_length < 1:
        raise ValueError("Insufficient data or invalid block length.")
    rng = np.random.default_rng(seed)
    n_observations = len(primary)
    differences = np.empty(replications)
    blocks_needed = int(np.ceil(n_observations / block_length))
    for replication in range(replications):
        starts = rng.integers(0, n_observations, size=blocks_needed)
        indices = np.concatenate(
            [
                (np.arange(start, start + block_length) % n_observations)
                for start in starts
            ]
        )[:n_observations]
        differences[replication] = np.var(
            primary[indices], ddof=1
        ) - np.var(benchmark[indices], ddof=1)
    observed_primary = float(np.var(primary, ddof=1))
    observed_benchmark = float(np.var(benchmark, ddof=1))
    return {
        "weeks": n_observations,
        "primary_variance": observed_primary,
        "benchmark_variance": observed_benchmark,
        "variance_difference": observed_primary - observed_benchmark,
        "variance_reduction_percent": (
            100.0 * (observed_benchmark - observed_primary) / observed_benchmark
        ),
        "ci_lower": float(np.percentile(differences, 2.5)),
        "ci_upper": float(np.percentile(differences, 97.5)),
        "probability_primary_lower": float(np.mean(differences < 0.0)),
        "replications": int(replications),
        "block_length": int(block_length),
    }


def bootstrap_comparisons(
    weekly_performance: pd.DataFrame,
    primary_methods: Iterable[str],
    benchmark_method: str,
    replications: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    benchmark = (
        weekly_performance.query("method == @benchmark_method")
        .set_index("date")["portfolio_return"]
        .sort_index()
    )
    for method in primary_methods:
        primary = (
            weekly_performance.query("method == @method")
            .set_index("date")["portfolio_return"]
            .sort_index()
        )
        joined = pd.concat(
            [primary.rename("primary"), benchmark.rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        result = moving_block_bootstrap_variance_difference(
            joined["primary"].to_numpy(),
            joined["benchmark"].to_numpy(),
            replications,
            block_length,
            seed,
        )
        rows.append(
            {
                "primary_method": method,
                "benchmark_method": benchmark_method,
                **result,
            }
        )
    return pd.DataFrame(rows)


def add_block_win_rates(
    summary: pd.DataFrame, block_summary: pd.DataFrame, benchmark: str
) -> pd.DataFrame:
    result = summary.copy()
    benchmark_variances = (
        block_summary.query("method == @benchmark")
        .set_index("block_id")["realized_variance"]
    )
    win_rates: dict[str, float] = {}
    for method, group in block_summary.groupby("method"):
        if method == benchmark:
            win_rates[method] = np.nan
            continue
        aligned = group.set_index("block_id")["realized_variance"].to_frame("method")
        aligned["benchmark"] = benchmark_variances
        aligned = aligned.dropna()
        win_rates[method] = float((aligned["method"] < aligned["benchmark"]).mean())
    result["block_win_rate_vs_ledoit_wolf"] = result["method"].map(win_rates)
    return result


def create_figures(
    weekly_performance: pd.DataFrame,
    selection_history: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pivot = weekly_performance.pivot(
        index="date", columns="method", values="portfolio_return"
    ).sort_index()
    cumulative = np.exp(pivot.cumsum())
    axis = cumulative.plot(figsize=(9, 5), linewidth=1.4)
    axis.set_title("Cumulative out-of-sample portfolio value")
    axis.set_ylabel("Value of one unit invested")
    axis.set_xlabel("")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "cumulative_performance.png", dpi=220)
    plt.close(axis.figure)

    graph_selections = selection_history[
        selection_history["method"].isin(["GraphLW", "GraphSample"])
    ].copy()
    if not graph_selections.empty:
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for method, group in graph_selections.groupby("method"):
            ordered = group.sort_values("unseen_start")
            axes[0].step(
                ordered["unseen_start"],
                ordered["training_window"],
                where="post",
                label=method,
            )
            axes[1].step(
                ordered["unseen_start"],
                ordered["k"],
                where="post",
                label=method,
            )
            axes[2].step(
                ordered["unseen_start"],
                ordered["rho"],
                where="post",
                label=method,
            )
        axes[0].set_ylabel("T")
        axes[1].set_ylabel("k")
        axes[2].set_ylabel("rho")
        axes[2].set_xlabel("")
        axes[0].legend()
        fig.suptitle("Walk-forward hyperparameter selections")
        fig.tight_layout()
        fig.savefig(output / "selected_hyperparameters.png", dpi=220)
        plt.close(fig)

    block_variances = (
        weekly_performance[
            weekly_performance["method"].isin(["GraphLW", "LedoitWolf"])
        ]
        .groupby(["block_id", "method"])["portfolio_return"]
        .var(ddof=1)
        .unstack("method")
    )
    if not block_variances.empty:
        axis = block_variances.plot(
            kind="bar", figsize=(10, 4.5), width=0.8
        )
        axis.set_title("Realized variance by unseen recalibration block")
        axis.set_xlabel("Unseen block")
        axis.set_ylabel("Weekly realized variance")
        axis.legend(["Graph-LW", "Ledoit-Wolf"])
        axis.figure.tight_layout()
        axis.figure.savefig(output / "block_realized_variance.png", dpi=220)
        plt.close(axis.figure)
