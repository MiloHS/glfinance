"""Inference and time diagnostics from saved portfolio paths; no model refitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glfinance.data import _sha256
from glfinance.graph import PORTFOLIO_GRAPH_OBJECTIVE
from glfinance.inference import studentized_variance_test
from glfinance.reporting import circular_block_bootstrap_variance_difference


def paired_paths(primary: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Require an exact date/block match rather than silently dropping observations."""
    columns = ["date", "block_id", "portfolio_return"]
    a = primary[columns].sort_values("date").set_index("date")
    b = reference[columns].sort_values("date").set_index("date")
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise ValueError("Strategy dates must be unique and identical.")
    if not a.block_id.equals(b.block_id):
        raise ValueError("Strategy block assignments differ.")
    result = a.rename(columns={"portfolio_return": "primary"})
    result["reference"] = b.portfolio_return
    if not np.isfinite(result[["primary", "reference"]]).all().all():
        raise ValueError("Return paths contain missing or nonfinite values.")
    return result


def block_diagnostics(pair: pd.DataFrame) -> pd.DataFrame:
    """Block variances plus additive contributions to the full-period difference."""
    n = len(pair)
    squared_difference = ((pair.primary - pair.primary.mean()) ** 2
                          - (pair.reference - pair.reference.mean()) ** 2)
    rows = []
    for block, group in pair.groupby("block_id", sort=True):
        difference = group.primary.var() - group.reference.var()
        contribution = squared_difference.loc[group.index].sum() / (n - 1)
        rows.append({"block_id": block, "start": group.index.min(), "end": group.index.max(),
                     "weeks": len(group), "variance_difference": difference,
                     "variance_reduction_percent": -100 * difference / group.reference.var(),
                     "full_period_variance_contribution": contribution,
                     "full_period_reduction_contribution_pp": -100 * contribution / pair.reference.var()})
    return pd.DataFrame(rows)


def _figures(blocks: pd.DataFrame, selections: pd.DataFrame, output: Path) -> None:
    baseline = blocks.query("comparison == 'baseline' and method in ['GraphLW', 'GraphSample']")
    universes = sorted(baseline.universe.unique(), key=lambda x: int(x[1:]))
    figure, axes = plt.subplots(2, len(universes), figsize=(14, 7), squeeze=False)
    for column, universe in enumerate(universes):
        for method, group in baseline.query("universe == @universe").groupby("method"):
            group = group.sort_values("block_id")
            dates = pd.to_datetime(group.end)
            axes[0, column].plot(dates, group.variance_difference, marker="o", label=method)
            axes[1, column].plot(dates, group.full_period_reduction_contribution_pp, marker="o", label=method)
        axes[0, column].set_title(universe)
        for axis in axes[:, column]:
            axis.axhline(0, color="black", linewidth=.7)
            axis.tick_params(axis="x", rotation=45)
            axis.grid(alpha=.2)
    axes[0, 0].set_ylabel("Block variance: method − LW")
    axes[1, 0].set_ylabel("Contribution to full-period\nvariance reduction (percentage points)")
    axes[0, -1].legend()
    figure.suptitle("Unseen blocks: 13 weeks each; final block 5 weeks")
    figure.tight_layout()
    figure.savefig(output / "block_variance_differences.png", dpi=180)
    plt.close(figure)
    for universe in universes:
        chosen = selections.query("universe == @universe").copy()
        chosen["procedure"] = chosen.comparison + "/" + chosen.method
        figure, axes = plt.subplots(1, 3, figsize=(16, 8))
        for axis, parameter in zip(axes, ["training_window", "k", "eta"]):
            matrix = chosen.pivot(index="procedure", columns="block_id", values=parameter)
            values = matrix.to_numpy(dtype=float)
            artist = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis")
            axis.set_yticks(range(len(matrix)), matrix.index if parameter == "training_window" else [])
            axis.set_xticks(range(len(matrix.columns)), matrix.columns)
            axis.set_title({"training_window": "Training weeks T", "k": "Retained modes k", "eta": "Diagonal shrinkage η"}[parameter])
            axis.set_xlabel("Unseen block (zero-based)")
            for row, col in zip(*np.where(np.isfinite(values))):
                axis.text(col, row, f"{values[row, col]:g}", ha="center", va="center", fontsize=6,
                          color="white" if artist.norm(values[row, col]) < .5 else "black")
            figure.colorbar(artist, ax=axis, fraction=.035)
        figure.suptitle(f"{universe}: selected parameters (blank = not applicable)")
        figure.tight_layout()
        figure.savefig(output / f"selected_parameters_{universe}.png", dpi=180)
        plt.close(figure)


def run_inference(spec: dict, root: Path, base: dict, baseline_dir: Path) -> Path:
    comparisons = root / spec["output_dir"]
    output = comparisons.parent / "inference"
    settings = spec.get("inference", {})
    lengths = settings.get("block_lengths", [2, 4, 8, 13])
    replications = int(settings.get("replications", 5000))
    seed = int(settings.get("seed", 42))
    baseline = pd.read_csv(baseline_dir / "weekly_performance.csv", parse_dates=["date"])
    paths = {"baseline": baseline}
    sources, schedules = {}, []
    baseline_manifest = json.loads((baseline_dir / "data_manifest.json").read_text())
    for comparison, directory in [("baseline", baseline_dir)] + [
        (name, comparisons / name / "real") for name in spec["comparisons"]
    ]:
        manifest = json.loads((directory / "data_manifest.json").read_text())
        if manifest.get("graph_objective") != PORTFOLIO_GRAPH_OBJECTIVE:
            raise ValueError(f"Rerun saved portfolio paths with the corrected Frobenius penalty: {comparison}")
        if manifest["files"] != baseline_manifest["files"]:
            raise ValueError(f"Input data differ from baseline: {comparison}")
        for filename in ["weekly_performance.csv", "selection_history.csv", "data_manifest.json", "resolved_config.yaml"]:
            path = directory / filename
            sources[str(path.relative_to(root))] = _sha256(path)
        performance = pd.read_csv(directory / "weekly_performance.csv", parse_dates=["date"])
        if comparison != "baseline":
            for universe, group in performance.query("method == 'LedoitWolf'").groupby("universe"):
                pair = paired_paths(group, baseline.query("universe == @universe and method == 'LedoitWolf'"))
                if not np.array_equal(pair.primary, pair.reference):
                    raise ValueError(f"LW reference differs: {comparison}/{universe}")
            performance = performance.query("method != 'LedoitWolf'")
        paths[comparison] = performance
        selection = pd.read_csv(directory / "selection_history.csv")
        if comparison != "baseline":
            selection = selection.query("method != 'LedoitWolf'").copy()
        selection.insert(0, "comparison", comparison)
        schedules.append(selection.rename(columns={"rho": "eta"}))

    percentile, studentized, blocks, concentration = [], [], [], []

    def analyze(metadata, pair, time_diagnostics=False):
        if len(pair) != int(base["experiment"]["outer_weeks"]):
            raise ValueError("Incomplete unseen path.")
        print("Inference: " + "/".join(str(value) for value in metadata.values()), flush=True)
        for length in lengths:
            percentile.append({**metadata, **circular_block_bootstrap_variance_difference(
                pair.primary.to_numpy(), pair.reference.to_numpy(), replications, length, seed)})
            studentized.append({**metadata, **studentized_variance_test(
                pair.primary.to_numpy(), pair.reference.to_numpy(), replications, length, seed)})
        if time_diagnostics:
            diagnostics = block_diagnostics(pair)
            for key, value in metadata.items():
                diagnostics[key] = value
            blocks.append(diagnostics)
            contributions = diagnostics.full_period_reduction_contribution_pp
            positive = contributions.clip(lower=0)
            concentration.append({**metadata, "blocks": len(diagnostics),
                                  "blocks_lower_variance": int((diagnostics.variance_difference < 0).sum()),
                                  "full_period_reduction_percent": contributions.sum(),
                                  "largest_three_positive_contributions_pp": positive.nlargest(3).sum(),
                                  "largest_three_share_of_positive_contributions": positive.nlargest(3).sum() / positive.sum()
                                  if positive.sum() > 0 else np.nan})

    for comparison, frame in paths.items():
        for (universe, method), group in frame.groupby(["universe", "method"]):
            if method == "LedoitWolf":
                continue
            reference = baseline.query("universe == @universe and method == 'LedoitWolf'")
            analyze({"comparison": comparison, "universe": universe, "method": method,
                     "reference": "LedoitWolf", "selection_method": "independently_tuned"},
                    paired_paths(group, reference), True)
    component_path = comparisons.parent / "components" / "fixed_setting_weekly.csv"
    if component_path.exists():
        component_manifest = json.loads((component_path.parent / "analysis_manifest.json").read_text())
        if component_manifest.get("graph_objective") != PORTFOLIO_GRAPH_OBJECTIVE:
            raise ValueError("Rerun components with the corrected Frobenius penalty before inference.")
        sources[str(component_path.relative_to(root))] = _sha256(component_path)
        components = pd.read_csv(component_path, parse_dates=["date"])
        for (case, selector, universe), group in components.groupby(["comparison", "selection_method", "universe"]):
            analyze({"comparison": "fixed_" + case, "universe": universe, "method": "alternative",
                     "reference": "fixed_reference", "selection_method": selector},
                    paired_paths(group.query("side == 'alternative'"), group.query("side == 'reference'")))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(percentile).to_csv(output / "bootstrap_block_sensitivity.csv", index=False)
    pd.DataFrame(studentized).to_csv(output / "studentized_variance_tests.csv", index=False)
    block_frame = pd.concat(blocks, ignore_index=True)
    selection_frame = pd.concat(schedules, ignore_index=True)
    block_frame.to_csv(output / "block_performance.csv", index=False)
    pd.DataFrame(concentration).to_csv(output / "concentration_summary.csv", index=False)
    selection_frame.to_csv(output / "selected_parameters.csv", index=False)
    _figures(block_frame, selection_frame, output)
    manifest = {"settings": {"block_lengths": lengths, "replications": replications, "seed": seed},
                "source_files": sources, "inference_scope": "conditional on saved selected portfolio paths; no retuning",
                "percentile": "paired circular blocks; truncate final drawn block to n observations",
                "studentized": "log variance ratio; prewhitened Parzen HAC; complete bootstrap blocks drawn from all original dates",
                "block_length_selection": "prespecified sensitivity grid; no automatic calibration or significance-based selection",
                "contribution": "sum of differences of globally centered squared returns divided by n-1",
                "reference": "https://www.econ.uzh.ch/dam/jcr:520edf26-2322-4708-8dde-51d61141914a/Ledoit_et_al-2011-Wilmott_Robust_Performance.pdf"}
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
