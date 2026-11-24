"""Run the portfolio revision comparisons with the maintained portfolio pipeline."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from glfinance.backtest import evaluate_fixed_schedule
from glfinance.config import load_config, resolved_config
from glfinance.covariance import graph_basis, projection_energy
from glfinance.data import _sha256
from glfinance.graph import PORTFOLIO_GRAPH_OBJECTIVE
from glfinance.experiments.portfolio import _load_graphs, _repository_root, _write_json, run_experiment
from glfinance.reporting import create_energy_figure, summarize_performance


def _revision_inputs(config_path: Path, *, allow_archived_objective: bool = False) -> tuple[dict, Path, dict, Path]:
    spec = yaml.safe_load(config_path.read_text())
    root = _repository_root(config_path)
    base = resolved_config(load_config(config_path.parent / spec["base_config"]), root)
    baseline_dir = root / spec["baseline_dir"]
    baseline_config = load_config(baseline_dir / "resolved_config.yaml")
    baseline_manifest = json.loads((baseline_dir / "data_manifest.json").read_text())
    if not allow_archived_objective and baseline_manifest.get("graph_objective") != PORTFOLIO_GRAPH_OBJECTIVE:
        raise ValueError("The saved baseline predates the Frobenius-penalty correction. Rerun the baseline before comparisons/components/inference.")
    if baseline_manifest.get("portfolio_return_convention") != "weighted_asset_simple_returns":
        raise ValueError("Run the corrected simple-return baseline before these comparisons.")
    if any(base[key] != baseline_config[key] for key in ("experiment", "data", "graph", "portfolio")):
        raise ValueError("The corrected baseline configuration does not match the comparison base.")
    for filename, checksum in baseline_manifest["files"].items():
        if _sha256(Path(base["data"]["cache_dir"]) / filename) != checksum:
            raise ValueError(f"Baseline input changed: {filename}")
    return spec, root, base, baseline_dir


def run_comparisons(config_path: Path) -> Path:
    spec, root, base, baseline_dir = _revision_inputs(config_path)
    output = root / spec["output_dir"]
    baseline = pd.read_csv(baseline_dir / "summary.csv")
    baseline.insert(0, "comparison", "baseline")
    summaries = [baseline]
    intervals = []
    baseline_intervals = pd.read_csv(baseline_dir / "bootstrap.csv")
    baseline_intervals.insert(0, "comparison", "baseline")
    intervals.append(baseline_intervals)
    for name, comparison in spec["comparisons"].items():
        print(f"Running comparison: {name}", flush=True)
        config = deepcopy(base)
        config["graph"].update(comparison.get("graph", {}))
        config["portfolio"]["methods"] = comparison["methods"]
        config["portfolio"]["primary_method"] = comparison["methods"][0]
        config["output"]["root_dir"] = str(output / name)
        graph_source = str(baseline_dir) if config["graph"] == base["graph"] else None
        directory = run_experiment(config, reuse_graphs_from=graph_source)
        summary = pd.read_csv(directory / "summary.csv")
        summary.insert(0, "comparison", name)
        # The identical direct-LW reference is already present in baseline.
        summaries.append(summary.query("method != 'LedoitWolf'").copy())
        interval = pd.read_csv(directory / "bootstrap.csv")
        interval.insert(0, "comparison", name)
        intervals.append(interval)

    summary = pd.concat(summaries, ignore_index=True)
    reference = baseline.query("method == 'LedoitWolf'").set_index("universe")["realized_variance"]
    summary["variance_reduction_vs_lw_percent"] = 100 * (
        1 - summary["realized_variance"] / summary["universe"].map(reference)
    )
    summary.to_csv(output / "comparison_summary.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(output / "comparison_bootstrap.csv", index=False)
    return output


def _energy_diagnostics(returns, graphs, selections, portfolio_config, outer_start):
    """Selected and matched-rank energies; every mean and basis precedes the target."""
    values = returns.to_numpy(dtype=float)
    n_assets = returns.shape[1]
    bases = {
        name: {date: graph_basis(graph, n_assets) for date, graph in family.items()}
        for name, family in graphs.items() if name in {"absolute", "positive"}
    }
    schedules = {}
    for family in ("absolute", "positive"):
        for method in ("GraphLW", "GraphSample"):
            rows = selections[family].query("method == @method")
            for row in rows.itertuples():
                for date in returns.loc[row.unseen_start:row.unseen_end].index:
                    schedules[family, method, date] = row

    rows = []
    for position in range(outer_start, len(returns)):
        date, graph_date = returns.index[position], returns.index[position - 1]
        contexts = []
        for family in ("absolute", "positive"):
            for method in ("GraphLW", "GraphSample"):
                chosen = schedules[family, method, date]
                contexts.append(("selected", family, method, int(chosen.training_window), int(chosen.k)))
        for training in portfolio_config["training_windows"]:
            for fraction in portfolio_config["k_fractions"]:
                contexts.append(("matched_grid", "grid", "grid", int(training), int(np.ceil(fraction * n_assets))))
        for kind, selection_graph, selection_method, training, retained in contexts:
            history = values[position - training:position]
            training_mean = history.mean(axis=0)
            centered = history - training_mean
            unseen = values[position] - training_mean
            for family in ("absolute", "positive"):
                basis = bases[family][graph_date][:, :retained]
                train_retained, train_total = projection_energy(centered, basis)
                unseen_retained, unseen_total = projection_energy(unseen, basis)
                rows.append({
                    "date": date, "graph_date": graph_date, "kind": kind,
                    "selection_graph": selection_graph, "selection_method": selection_method,
                    "basis_graph": family, "training_window": training, "k": retained,
                    "training_retained": train_retained, "training_total": train_total,
                    "unseen_retained": unseen_retained, "unseen_total": unseen_total,
                })
    weekly = pd.DataFrame(rows)
    summaries = []
    for kind, group in weekly.groupby("kind"):
        keys = ["selection_graph", "selection_method", "basis_graph"]
        if kind == "matched_grid":
            keys += ["training_window", "k"]
        summary = group.groupby(keys).agg(
            weeks=("date", "size"),
            training_retained=("training_retained", "sum"), training_total=("training_total", "sum"),
            unseen_retained=("unseen_retained", "sum"), unseen_total=("unseen_total", "sum"),
        ).reset_index()
        summary["training_energy_fraction"] = summary.training_retained / summary.training_total.replace(0, np.nan)
        summary["unseen_energy_fraction"] = summary.unseen_retained / summary.unseen_total.replace(0, np.nan)
        summary.insert(0, "kind", kind)
        summaries.append(summary)
    return weekly, pd.concat(summaries, ignore_index=True)


def run_components(config_path: Path) -> Path:
    """Part 4: use existing graphs and selections, without new grid searches."""
    spec, root, base, baseline_dir = _revision_inputs(config_path)
    comparisons = root / spec["output_dir"]
    output = comparisons.parent / "components"
    cache = Path(base["data"]["cache_dir"])
    symbols = [item["symbol"] for item in base["data"]["tickers"]]
    returns = pd.read_csv(cache / "weekly_log_returns.csv", index_col="date", parse_dates=True)[symbols]
    risk_free = pd.read_csv(cache / "weekly_risk_free_dtb3_simple.csv", index_col="date", parse_dates=True).iloc[:, 0]
    directories = {
        "absolute": baseline_dir, "positive": comparisons / "positive_correlation" / "real",
        "no_prediction": comparisons / "no_prediction" / "real",
    }
    date_columns = ["validation_end", "unseen_start", "unseen_end"]
    selections = {}
    source_files = {}
    baseline_manifest = json.loads((baseline_dir / "data_manifest.json").read_text())
    for family, directory in directories.items():
        manifest = json.loads((directory / "data_manifest.json").read_text())
        if manifest.get("graph_objective") != PORTFOLIO_GRAPH_OBJECTIVE:
            raise ValueError(f"Rerun Part 3 with the corrected Frobenius penalty: {family}")
        if manifest["files"] != baseline_manifest["files"]:
            raise ValueError(f"Part 3 inputs differ from baseline: {family}")
        saved_config = load_config(directory / "resolved_config.yaml")
        expected_graph = deepcopy(base["graph"])
        if family != "absolute":
            name = "positive_correlation" if family == "positive" else "no_prediction"
            expected_graph.update(spec["comparisons"][name]["graph"])
        if saved_config["graph"] != expected_graph:
            raise ValueError(f"Part 3 graph settings differ from the specification: {family}")
        path = directory / "selection_history.csv"
        selections[family] = pd.read_csv(path, parse_dates=date_columns)
        source_files[str(path.relative_to(root))] = _sha256(path)

    # Common population normalization isolates LW shrinkage from T vs T-1 scaling.
    cases = [
        ("within_subspace_lw", "GraphLW", "absolute", "GraphSample", 0),
        ("within_subspace_lw", "GraphSample", "absolute", "GraphSample", 0),
        ("residual_variance", "GraphLW", "absolute", "GraphLWResidual", 1),
        ("prediction_term", "GraphLW", "no_prediction", "GraphLW", 1),
        ("correlation_sign", "GraphLW", "positive", "GraphLW", 1),
    ]
    all_performance, all_weights, all_energy, all_energy_summary, differences = [], [], [], [], []
    outer_start = len(returns) - int(base["experiment"]["outer_weeks"])
    first_graph = outer_start - int(base["experiment"]["validation_weeks"]) - 1
    graph_dates = returns.index[first_graph:-1]
    for n_assets in base["data"]["universe_sizes"]:
        universe = f"N{n_assets}"
        print(f"{universe}: fixed-setting comparisons and retained energy", flush=True)
        data = returns.iloc[:, :n_assets]
        graphs = {}
        for family, directory in directories.items():
            path = directory / f"laplacians_{universe}.npz"
            graphs[family] = _load_graphs(path, graph_dates, n_assets)
            source_files[str(path.relative_to(root))] = _sha256(path)
        schedules = {name: frame.query("universe == @universe").copy() for name, frame in selections.items()}
        for case, selector, alternative_graph, alternative_method, sample_ddof in cases:
            schedule = schedules["absolute"].query("method == @selector").copy()
            sides = {}
            for side, family, method in [
                ("reference", "absolute", "GraphLW"),
                ("alternative", alternative_graph, alternative_method),
            ]:
                artifact = evaluate_fixed_schedule(
                    method, schedule, data, risk_free, graphs[family], base["portfolio"]["delta"],
                    long_only=base["portfolio"].get("long_only", False), sample_ddof=sample_ddof,
                )
                for frame in (artifact.weekly_performance, artifact.weekly_weights):
                    frame.insert(0, "universe", universe)
                    frame.insert(0, "selection_method", selector)
                    frame.insert(0, "comparison", case)
                    frame["side"], frame["graph_variant"], frame["sample_ddof"] = side, family, sample_ddof
                all_performance.append(artifact.weekly_performance)
                all_weights.append(artifact.weekly_weights)
                sides[side] = artifact.weekly_weights.pivot(index="date", columns="symbol", values="weight")
            change = 0.5 * (sides["alternative"] - sides["reference"]).abs().sum(axis=1)
            differences.append({"comparison": case, "selection_method": selector, "universe": universe,
                                "mean_half_l1_weight_change": float(change.mean())})
        energy, energy_summary = _energy_diagnostics(data, graphs, schedules, base["portfolio"], outer_start)
        for frame in (energy, energy_summary):
            frame.insert(0, "universe", universe)
        all_energy.append(energy)
        all_energy_summary.append(energy_summary)

    output.mkdir(parents=True, exist_ok=True)
    performance = pd.concat(all_performance, ignore_index=True)
    keys = ["comparison", "selection_method", "universe", "side", "method", "graph_variant", "sample_ddof"]
    summary = summarize_performance(performance, keys)
    pair_keys = ["comparison", "selection_method", "universe"]
    reference = summary.query("side == 'reference'").drop(columns="side")
    alternative = summary.query("side == 'alternative'").drop(columns="side")
    paired = reference.merge(alternative, on=pair_keys, suffixes=("_reference", "_alternative"))
    paired["variance_change_percent"] = 100 * (paired.realized_variance_alternative / paired.realized_variance_reference - 1)
    paired = paired.merge(pd.DataFrame(differences), on=pair_keys)
    summary.to_csv(output / "fixed_setting_summary.csv", index=False)
    paired.to_csv(output / "fixed_setting_comparison.csv", index=False)
    performance.to_csv(output / "fixed_setting_weekly.csv", index=False)
    pd.concat(all_weights, ignore_index=True).to_csv(output / "fixed_setting_weights.csv", index=False)
    pd.concat(all_energy, ignore_index=True).to_csv(output / "energy_weekly.csv", index=False)
    energy_summary = pd.concat(all_energy_summary, ignore_index=True)
    energy_summary.to_csv(output / "energy_summary.csv", index=False)
    create_energy_figure(energy_summary, output / "retained_energy.png")
    _write_json({
        "base_config": base, "revision_config": spec,
        "graph_objective": PORTFOLIO_GRAPH_OBJECTIVE,
        "input_files": baseline_manifest["files"], "source_files": source_files,
        "within_subspace_covariance_ddof": 0,
        "residual_variance_ddof": 1,
        "independently_tuned_sample_covariance_ddof": 1,
        "energy_centering": "trailing training mean, also used for unseen return",
        "energy_aggregation": "sum of retained squared energy / sum of total squared energy",
        "variance_change_sign": "positive means alternative has higher variance",
    }, output / "analysis_manifest.json")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/portfolio_revision.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--components", action="store_true", help="Part 4: analyze saved graphs and selections only.")
    mode.add_argument("--inference", action="store_true", help="Bootstrap robustness and saved block diagnostics; no refitting.")
    mode.add_argument("--sensitivity", action="store_true", help="One-at-a-time graph sensitivity at saved baseline portfolio settings.")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    if args.sensitivity:
        from glfinance.experiments.portfolio_sensitivity import run_sensitivity
        output = run_sensitivity(*_revision_inputs(config_path, allow_archived_objective=True))
    elif args.inference:
        from glfinance.experiments.portfolio_inference import run_inference
        output = run_inference(*_revision_inputs(config_path))
    else:
        run = run_components if args.components else run_comparisons
        output = run(config_path)
    print(f"Comparison results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
