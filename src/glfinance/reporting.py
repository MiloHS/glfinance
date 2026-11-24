from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glfinance.inference import circular_bootstrap_indices

def cumulative_wealth(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound weekly simple returns from one unit of initial wealth."""
    values = simple_returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("Cumulative wealth requires finite returns greater than -1.")
    wealth = (1.0 + simple_returns).cumprod()
    if not np.all(np.isfinite(wealth.to_numpy())):
        raise RuntimeError("Cumulative wealth became non-finite.")
    return wealth


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


def circular_block_bootstrap_variance_difference(
    primary_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    replications: int,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired circular-block bootstrap of Var(primary)-Var(benchmark)."""
    primary = np.asarray(primary_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if primary.shape != benchmark.shape or primary.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be equal-length vectors.")
    if len(primary) < 2 or block_length < 1:
        raise ValueError("Insufficient data or invalid block length.")
    n_observations = len(primary)
    indices = circular_bootstrap_indices(n_observations, block_length, replications, seed)
    differences = np.var(primary[indices], axis=1, ddof=1) - np.var(benchmark[indices], axis=1, ddof=1)
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
        result = circular_block_bootstrap_variance_difference(
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


def create_energy_figure(energy_summary: pd.DataFrame, path: str | Path) -> None:
    """Compare both graph bases using each saved parameter-selection schedule."""
    selected = energy_summary.query("kind == 'selected'")
    universes = list(selected.universe.unique())
    labels = ["Abs / LW", "Abs / Sample", "Pos / LW", "Pos / Sample"]
    schedules = pd.MultiIndex.from_product(
        [["absolute", "positive"], ["GraphLW", "GraphSample"]],
        names=["selection_graph", "selection_method"],
    )
    fig, axes = plt.subplots(2, len(universes), figsize=(12, 6), squeeze=False)
    for column, universe in enumerate(universes):
        group = selected.query("universe == @universe")
        for row, metric in enumerate(("training_energy_fraction", "unseen_energy_fraction")):
            pivot = group.pivot(index=["selection_graph", "selection_method"],
                                columns="basis_graph", values=metric).reindex(schedules)
            axis = axes[row, column]
            for offset, (family, label) in enumerate([
                ("absolute", "Absolute basis"), ("positive", "Positive basis")
            ]):
                axis.bar(np.arange(4) + (offset - 0.5) * 0.36,
                         100 * pivot[family], width=0.36, label=label)
            axis.set_xticks(np.arange(4), labels, rotation=25, ha="right", fontsize=8)
            axis.set_ylim(0, 100)
            axis.set_title(f"{universe}: {'training' if row == 0 else 'unseen'}")
            if column == 0:
                axis.set_ylabel("Retained energy (%)")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Retained return energy at saved selected settings")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
    cumulative = cumulative_wealth(pivot)
    axis = cumulative.plot(figsize=(9, 5), linewidth=1.4)
    axis.set_title("Cumulative out-of-sample portfolio value")
    axis.set_ylabel("Value of one unit invested")
    axis.set_xlabel("")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "cumulative_performance.png", dpi=220)
    plt.close(axis.figure)

    graph_selections = selection_history[
        selection_history["method"].str.startswith(("Graph", "PCA"))
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

    plotted_methods = (
        ["GraphLW", "LedoitWolf"]
        if "GraphLW" in pivot.columns else list(pivot.columns)
    )
    block_variances = (
        weekly_performance[
            weekly_performance["method"].isin(plotted_methods)
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
        axis.figure.tight_layout()
        axis.figure.savefig(output / "block_realized_variance.png", dpi=220)
        plt.close(axis.figure)
