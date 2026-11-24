from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glfinance.backtest import candidate_grid, run_nested_walk_forward
from glfinance.config import dump_config, load_config, resolved_config
from glfinance.data import _sha256, load_or_download_inputs, synthetic_inputs
from glfinance.graph import PORTFOLIO_GRAPH_OBJECTIVE, learn_causal_graphs
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


def _load_graphs(
    path: Path, expected_dates: pd.DatetimeIndex, n_assets: int
) -> dict[pd.Timestamp, np.ndarray]:
    """Reuse graphs only when their dates and asset dimensions match exactly."""
    with np.load(path, allow_pickle=False) as saved:
        dates = pd.DatetimeIndex(pd.to_datetime(saved["dates"]))
        laplacians = saved["laplacians"]
    if not dates.equals(expected_dates):
        raise ValueError(f"Saved graph dates do not match the requested run: {path}")
    if laplacians.shape != (len(dates), n_assets, n_assets):
        raise ValueError(f"Saved graphs have the wrong shape: {path}")
    if not np.all(np.isfinite(laplacians)):
        raise ValueError(f"Saved graphs contain non-finite values: {path}")
    return dict(zip(dates, laplacians))


def run_experiment(
    config: dict[str, Any],
    synthetic: bool = False,
    refresh_data: bool = False,
    data_only: bool = False,
    output_override: str | None = None,
    reuse_graphs_from: str | None = None,
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
    graph_source = Path(reuse_graphs_from).resolve() if reuse_graphs_from else None
    if graph_source is not None and (synthetic or graph_source == output_root.resolve()):
        raise ValueError("Reuse real graphs from a separate results directory.")
    output_root.mkdir(parents=True, exist_ok=True)
    run_config = {**config, "output": {**config["output"], "root_dir": str(output_root.parent.resolve())}}
    dump_config(run_config, output_root / "resolved_config.yaml")
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
    max_training = max(
        parameters.training_window or 0
        for method in config["portfolio"]["methods"]
        for parameters in candidate_grid(method, config["portfolio"])
    )
    outer_start = len(returns) - outer_weeks
    first_validation_target = outer_start - validation_weeks
    first_graph_position = first_validation_target - 1
    last_graph_position = len(returns) - 2
    minimum_first_graph = max(lookback - 1, max_training - 1)
    if first_graph_position < minimum_first_graph:
        raise RuntimeError(
            "Not enough history for the requested lookback, validation, and outer periods."
        )

    if graph_source is not None:
        source_config = load_config(graph_source / "resolved_config.yaml")
        source_manifest = json.loads((graph_source / "data_manifest.json").read_text())
        if source_manifest.get("graph_objective") != PORTFOLIO_GRAPH_OBJECTIVE:
            raise ValueError("Saved graphs predate the Frobenius-penalty correction; recompute them without --reuse-graphs-from.")
        same_settings = (
            source_config["graph"] == config["graph"]
            and source_config["experiment"]["distance_lookback_weeks"] == lookback
            and source_config["data"]["tickers"] == data_config["tickers"]
        )
        price_file = "weekly_log_returns.csv"
        same_prices = source_manifest["files"][price_file] == manifest["files"][price_file]
        if not same_settings or not same_prices:
            raise ValueError("Saved graphs require identical graph settings, asset order, and prices.")
        source_solver_status = pd.read_csv(graph_source / "solver_status.csv", parse_dates=["date"])
        manifest["reused_graphs"] = {"source": str(graph_source), "files": {}}

    for universe_size in data_config["universe_sizes"]:
        universe_size = int(universe_size)
        universe_returns = returns.iloc[:, :universe_size].copy()
        universe_name = f"N{universe_size}"
        print(f"{universe_name}: preparing graphs and evaluating portfolio candidates", flush=True)
        if graph_source is None:
            graphs, solve_records = learn_causal_graphs(
                universe_returns,
                first_graph_position,
                last_graph_position,
                lookback,
                config["graph"],
            )
            solver_frame = pd.DataFrame([asdict(record) for record in solve_records])
            solver_frame.insert(0, "universe", universe_name)
        else:
            graph_file = graph_source / f"laplacians_{universe_name}.npz"
            graphs = _load_graphs(
                graph_file,
                returns.index[first_graph_position : last_graph_position + 1],
                universe_size,
            )
            solver_frame = source_solver_status.query("universe == @universe_name").copy()
            if not pd.DatetimeIndex(solver_frame["date"]).equals(pd.DatetimeIndex(graphs)):
                raise ValueError(f"Saved solver records do not match {universe_name} graphs.")
            manifest["reused_graphs"]["files"][graph_file.name] = _sha256(graph_file)
        _save_graphs(graphs, output_root / f"laplacians_{universe_name}.npz")
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
        print(f"{universe_name}: completed", flush=True)

    _write_json(manifest, output_root / "data_manifest.json")
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
            primary_methods=[
                method for method in config["portfolio"]["methods"]
                if method not in {"LedoitWolf", "EqualWeight"}
            ],
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
    # Mark the corrected objective only after the complete run succeeds.
    manifest["graph_objective"] = PORTFOLIO_GRAPH_OBJECTIVE
    _write_json(manifest, output_root / "data_manifest.json")
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
    parser.add_argument(
        "--reuse-graphs-from",
        help="Reuse graphs from a results directory with identical prices and graph settings.",
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
        reuse_graphs_from=args.reuse_graphs_from,
    )
    print(f"Experiment artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

