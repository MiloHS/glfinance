"""One-at-a-time graph sensitivity, holding saved portfolio selections fixed."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glfinance.backtest import evaluate_fixed_schedule
from glfinance.data import _sha256
from glfinance.experiments.portfolio import _load_graphs, _save_graphs, _write_json
from glfinance.graph import PORTFOLIO_GRAPH_OBJECTIVE, learn_causal_graphs
from glfinance.reporting import summarize_performance


def sensitivity_cases(settings: dict, graph_config: dict) -> list[tuple[str, str, float, dict]]:
    """Validate a small one-at-a-time grid; never mutate the baseline configuration."""
    allowed = {"gamma_1", "gamma_2", "gamma_3", "epsilon"}
    if not settings or set(settings) - allowed:
        raise ValueError("Sensitivity settings must contain gamma_1/2/3 or epsilon only.")
    cases = []
    for parameter, values in settings.items():
        baseline = float(graph_config.get(parameter, 0.01))
        if not isinstance(values, list) or not values:
            raise ValueError(f"Specify a nonempty sensitivity list for {parameter}.")
        seen = {baseline}
        for value in values:
            value = float(value)
            if not np.isfinite(value) or value <= 0 or value in seen:
                raise ValueError(f"Sensitivity values must be unique, positive, and different from baseline: {parameter}")
            seen.add(value)
            config = deepcopy(graph_config)
            config[parameter] = value
            cases.append((f"{parameter}_{value:g}", parameter, value, config))
    return cases


def graph_changes(graphs: dict, reference: dict) -> dict:
    """Paired graph changes over the identical chronological graph sequence."""
    if list(graphs) != list(reference):
        raise ValueError("Sensitivity and baseline graph dates differ.")
    changes = [np.linalg.norm(graphs[date] - graph, "fro") / max(np.linalg.norm(graph, "fro"), 1e-15)
               for date, graph in reference.items()]
    return {"mean_relative_graph_change": float(np.mean(changes)),
            "max_relative_graph_change": float(np.max(changes))}


def _figure(summary: pd.DataFrame, path: Path) -> None:
    universes = sorted(summary.universe.unique(), key=lambda value: int(value[1:]))
    variants = summary.loc[summary.variant != "baseline", "variant"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(universes), figsize=(14, 5), squeeze=False)
    for column, universe in enumerate(universes):
        axis = axes[0, column]
        for method in ["GraphLW", "GraphSample"]:
            data = summary.query("universe == @universe and method == @method").set_index("variant")
            axis.plot(data.loc[variants, "variance_change_vs_baseline_percent"], range(len(variants)), "o-", label=method)
        axis.axvline(0, color="black", linewidth=.7)
        axis.set_yticks(range(len(variants)), variants if column == 0 else [])
        axis.invert_yaxis()
        axis.set_title(universe)
        axis.set_xlabel("Variance change from own baseline (%)")
        axis.grid(alpha=.2)
    axes[0, -1].legend()
    figure.suptitle("One-at-a-time sensitivity; baseline T, k and diagonal shrinkage held fixed")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_sensitivity(spec: dict, root: Path, base: dict, baseline_dir: Path) -> Path:
    cases = sensitivity_cases(spec["sensitivity"], base["graph"])
    output = (root / spec["output_dir"]).parent / "sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads((baseline_dir / "data_manifest.json").read_text())
    archived_objective = baseline_manifest.get("graph_objective") != PORTFOLIO_GRAPH_OBJECTIVE
    cache = Path(base["data"]["cache_dir"])
    symbols = [item["symbol"] for item in base["data"]["tickers"]]
    returns = pd.read_csv(cache / "weekly_log_returns.csv", index_col="date", parse_dates=True)[symbols]
    risk_free = pd.read_csv(cache / "weekly_risk_free_dtb3_simple.csv", index_col="date", parse_dates=True).iloc[:, 0]
    selection_path = baseline_dir / "selection_history.csv"
    selections = pd.read_csv(selection_path, parse_dates=["validation_end", "unseen_start", "unseen_end"])
    performance_path = baseline_dir / "weekly_performance.csv"
    weight_path = baseline_dir / "weekly_weights.csv"
    original = pd.read_csv(performance_path, parse_dates=["date"])
    methods = ["GraphLW", "GraphSample"]
    performances, weights, solver_records, diagnostics = [], [], [], []
    first_graph = len(returns) - int(base["experiment"]["outer_weeks"]) - int(base["experiment"]["validation_weeks"]) - 1
    last_graph = len(returns) - 2
    graph_dates = returns.index[first_graph:last_graph + 1]
    source_files = {str(path.relative_to(root)): _sha256(path) for path in
                    [selection_path, performance_path, weight_path, baseline_dir / "resolved_config.yaml"]}
    for size in base["data"]["universe_sizes"]:
        universe = f"N{size}"
        data = returns.iloc[:, :size]
        baseline_graph_path = baseline_dir / f"laplacians_{universe}.npz"
        source_files[str(baseline_graph_path.relative_to(root))] = _sha256(baseline_graph_path)
        archived_graphs = _load_graphs(baseline_graph_path, graph_dates, size)
        reference_weights = {}
        # Recompute the reference as well, permitting an explicitly labelled
        # sensitivity replay of archived selections without reusing old graphs.
        for variant, parameter, value, graph_config in [("baseline", "none", np.nan, base["graph"])] + cases:
            print(f"Sensitivity {universe}/{variant}: learning {len(graph_dates)} chronological graphs", flush=True)
            graphs, records = learn_causal_graphs(
                data, first_graph, last_graph, int(base["experiment"]["distance_lookback_weeks"]), graph_config,
            )
            directory = output / "graphs" / variant
            directory.mkdir(parents=True, exist_ok=True)
            _save_graphs(graphs, directory / f"laplacians_{universe}.npz")
            statuses = pd.DataFrame([asdict(record) for record in records])
            statuses["variant"], statuses["universe"] = variant, universe
            solver_records.append(statuses)
            if variant == "baseline":
                reference_graphs = graphs
            changes = graph_changes(graphs, reference_graphs)
            archived_changes = graph_changes(graphs, archived_graphs)
            for method in methods:
                schedule = selections.query("universe == @universe and method == @method").copy()
                artifact = evaluate_fixed_schedule(
                    method, schedule, data, risk_free, graphs, base["portfolio"]["delta"],
                    long_only=base["portfolio"].get("long_only", False),
                )
                for frame in [artifact.weekly_performance, artifact.weekly_weights]:
                    frame["variant"], frame["universe"] = variant, universe
                    frame["changed_parameter"], frame["parameter_value"] = parameter, value
                performances.append(artifact.weekly_performance)
                weights.append(artifact.weekly_weights)
                current = artifact.weekly_performance.sort_values("date")
                saved = original.query("universe == @universe and method == @method").sort_values("date")
                if not current.date.reset_index(drop=True).equals(saved.date.reset_index(drop=True)):
                    raise ValueError("Sensitivity changed the unseen evaluation dates.")
                for column in ["block_id", "training_window", "k", "rho"]:
                    np.testing.assert_allclose(current[column], saved[column], rtol=0, atol=0)
                allocation = artifact.weekly_weights.pivot(index="date", columns="symbol", values="weight")
                if variant == "baseline":
                    reference_weights[method] = allocation
                reference = reference_weights[method]
                if not allocation.index.equals(reference.index) or not allocation.columns.equals(reference.columns):
                    raise ValueError("Sensitivity and baseline allocation dates/assets differ.")
                diagnostics.append({"variant": variant, "universe": universe, "method": method, **changes,
                                    "mean_relative_graph_change_vs_archived": archived_changes["mean_relative_graph_change"],
                                    "mean_half_l1_weight_change": float(.5 * (allocation - reference).abs().sum(axis=1).mean())})
            print(f"Sensitivity {universe}/{variant}: complete", flush=True)
    performance = pd.concat(performances, ignore_index=True)
    summary = summarize_performance(performance, ["variant", "universe", "method"])
    metadata = {name: (parameter, value) for name, parameter, value, _ in cases}
    summary["changed_parameter"] = summary.variant.map(lambda name: metadata.get(name, ("none", np.nan))[0])
    summary["parameter_value"] = summary.variant.map(lambda name: metadata.get(name, ("none", np.nan))[1])
    own_baseline = summary.query("variant == 'baseline'").set_index(["universe", "method"]).realized_variance
    lw = original.query("method == 'LedoitWolf'").groupby("universe").portfolio_return.var()
    summary["variance_change_vs_baseline_percent"] = [
        100 * (row.realized_variance / own_baseline.loc[(row.universe, row.method)] - 1)
        for row in summary.itertuples()]
    summary["variance_reduction_vs_lw_percent"] = 100 * (1 - summary.realized_variance / summary.universe.map(lw))
    summary = summary.merge(pd.DataFrame(diagnostics), on=["variant", "universe", "method"], how="left")
    summary.to_csv(output / "summary.csv", index=False)
    performance.to_csv(output / "weekly_performance.csv", index=False)
    pd.concat(weights, ignore_index=True).to_csv(output / "weekly_weights.csv", index=False)
    statuses = pd.concat(solver_records, ignore_index=True)
    statuses.to_csv(output / "solver_status.csv", index=False)
    _figure(summary, output / "sensitivity.png")
    _write_json({"base_config": base, "sensitivity_grid": spec["sensitivity"], "source_files": source_files,
                 "graph_objective": PORTFOLIO_GRAPH_OBJECTIVE,
                 "baseline_uses_corrected_objective": not archived_objective,
                 "cached_inputs": {name: _sha256(cache / name) for name in
                                   ["weekly_log_returns.csv", "weekly_risk_free_dtb3_simple.csv"]},
                 "design": "one parameter at a time; replay each method's saved baseline T/k/rho schedule; no retuning or best-variant selection",
                 "graph_initialization": "same first graph date as baseline, with no temporal term at that date; thereafter previous graph from the same sensitivity variant",
                 "baseline_epsilon": float(base["graph"].get("epsilon", .01)),
                 "reference": "baseline graphs recomputed with vectorized Frobenius penalty; saved baseline portfolio selections replayed without retuning",
                 "archived_result_warning": ("Input selections predate the Frobenius-penalty correction; rerun nested experiments before manuscript use"
                                             if archived_objective else None),
                 "relative_graph_change": "Frobenius norm of alternative-minus-baseline divided by baseline norm; mean/max across all graph dates",
                 "scope": "portfolio only; fixed-setting sensitivity, not separately optimized procedures"},
                output / "analysis_manifest.json")
    return output
