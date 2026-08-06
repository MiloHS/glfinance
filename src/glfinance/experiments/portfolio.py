from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glfinance.backtest import run_nested_walk_forward
from glfinance.config import dump_config, load_config, resolved_config
from glfinance.data import load_or_download_inputs, synthetic_inputs
from glfinance.graph import learn_causal_graphs
from glfinance.reporting import (
    add_block_win_rates,
    bootstrap_comparisons,
    create_figures,
    summarize_performance,
)


def _repository_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return Path.cwd().resolve()


def _save_graphs(
    graphs: dict[pd.Timestamp, np.ndarray], path: Path
) -> None:
    dates = np.asarray([str(date.date()) for date in graphs], dtype="U10")
    laplacians = np.stack(list(graphs.values()), axis=0)
    np.savez_compressed(path, dates=dates, laplacians=laplacians)


def _write_json(value: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def run_experiment(
    config: dict[str, Any],
    synthetic: bool = False,
    refresh_data: bool = False,
    data_only: bool = False,
    output_override: str | None = None,
) -> Path:
    data_config = config["data"]
    experiment_config = config["experiment"]
    symbols = [item["symbol"] for item in data_config["tickers"]]
    if synthetic:
        minimum_observations = (
            int(experiment_config["outer_weeks"])
            + int(experiment_config["validation_weeks"])
            + int(experiment_config["distance_lookback_weeks"])
            + 2
        )
        returns, risk_free, manifest = synthetic_inputs(
            symbols,
            observations=max(420, minimum_observations),
            seed=int(experiment_config["seed"]),
        )
    else:
        returns, risk_free, manifest = load_or_download_inputs(
            config, refresh=refresh_data
        )

    output_root = Path(
        output_override or config["output"]["root_dir"]
    ) / ("synthetic" if synthetic else "real")
    output_root.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_root / "resolved_config.yaml")
    _write_json(manifest, output_root / "data_manifest.json")
    if data_only:
        return output_root

    all_performance: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    all_solver_records: list[pd.DataFrame] = []

    outer_weeks = int(experiment_config["outer_weeks"])
    validation_weeks = int(experiment_config["validation_weeks"])
    lookback = int(experiment_config["distance_lookback_weeks"])
    max_training = max(int(value) for value in config["portfolio"]["training_windows"])
    outer_start = len(returns) - outer_weeks
    first_validation_target = outer_start - validation_weeks
    first_graph_position = first_validation_target - 1
    last_graph_position = len(returns) - 2
    minimum_first_graph = max(lookback - 1, max_training - 1)
    if first_graph_position < minimum_first_graph:
        raise RuntimeError(
            "Not enough history for the requested lookback, validation, and outer periods."
        )

    for universe_size in data_config["universe_sizes"]:
        universe_size = int(universe_size)
        universe_returns = returns.iloc[:, :universe_size].copy()
        universe_name = f"N{universe_size}"
        graphs, solve_records = learn_causal_graphs(
            universe_returns,
            first_graph_position,
            last_graph_position,
            lookback,
            config["graph"],
        )
        _save_graphs(graphs, output_root / f"laplacians_{universe_name}.npz")
        solver_frame = pd.DataFrame([asdict(record) for record in solve_records])
        solver_frame.insert(0, "universe", universe_name)
        all_solver_records.append(solver_frame)

        artifacts = run_nested_walk_forward(
            universe_returns,
            risk_free,
            graphs,
            outer_start,
            experiment_config,
            config["portfolio"],
        )
        for frame in (
            artifacts.weekly_performance,
            artifacts.weekly_weights,
            artifacts.selection_history,
        ):
            frame.insert(0, "universe", universe_name)
            frame.insert(1, "n_assets", universe_size)
        all_performance.append(artifacts.weekly_performance)
        all_weights.append(artifacts.weekly_weights)
        all_selections.append(artifacts.selection_history)

    weekly_performance = pd.concat(all_performance, ignore_index=True)
    weekly_weights = pd.concat(all_weights, ignore_index=True)
    selection_history = pd.concat(all_selections, ignore_index=True)
    solver_status = pd.concat(all_solver_records, ignore_index=True)
    weekly_performance.to_csv(output_root / "weekly_performance.csv", index=False)
    weekly_weights.to_csv(output_root / "weekly_weights.csv", index=False)
    selection_history.to_csv(output_root / "selection_history.csv", index=False)
    solver_status.to_csv(output_root / "solver_status.csv", index=False)

    summary_parts: list[pd.DataFrame] = []
    block_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    figures_root = output_root / "figures"
    for universe, group in weekly_performance.groupby("universe", sort=False):
        summary = summarize_performance(group, ["method"])
        summary.insert(0, "universe", universe)
        block = summarize_performance(group, ["block_id", "method"])
        block.insert(0, "universe", universe)
        summary = add_block_win_rates(summary, block, "LedoitWolf")
        bootstrap = bootstrap_comparisons(
            group,
            primary_methods=["GraphLW", "GraphSample"],
            benchmark_method="LedoitWolf",
            replications=int(experiment_config["bootstrap_replications"]),
            block_length=int(experiment_config["bootstrap_block_weeks"]),
            seed=int(experiment_config["seed"]),
        )
        bootstrap.insert(0, "universe", universe)
        summary_parts.append(summary)
        block_parts.append(block)
        bootstrap_parts.append(bootstrap)
        create_figures(
            group,
            selection_history.query("universe == @universe"),
            figures_root / universe,
        )

    pd.concat(summary_parts, ignore_index=True).to_csv(
        output_root / "summary.csv", index=False
    )
    pd.concat(block_parts, ignore_index=True).to_csv(
        output_root / "block_summary.csv", index=False
    )
    pd.concat(bootstrap_parts, ignore_index=True).to_csv(
        output_root / "bootstrap.csv", index=False
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the causal walk-forward GLFinance portfolio experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/portfolio_walk_forward.yaml",
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic data and avoid network access.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Redownload and replace cached processed inputs.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Prepare inputs and manifest, then stop before graph learning.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the configured output root directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = resolved_config(
        load_config(config_path), _repository_root(config_path)
    )
    output = run_experiment(
        config,
        synthetic=args.synthetic,
        refresh_data=args.refresh_data,
        data_only=args.data_only,
        output_override=args.output_dir,
    )
    print(f"Experiment artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

