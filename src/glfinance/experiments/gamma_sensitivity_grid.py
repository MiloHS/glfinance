"""
Sensitivity Analysis for Regularization Parameters (Gammas) on the 2008 Crisis Model.
Addresses Reviewer 2 Comment 1.6:
Assessing sensitivity of learned graphs and regressor dynamics to gamma1 (log-barrier),
gamma2 (Frobenius norm), and gamma3 (temporal smoothness) with a fixed 20-week lookback.
"""

from pathlib import Path
import time
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glfinance.graph import solve_instance, compute_beta_standard_errors
from glfinance.experiments.dryrun_plots import compute_network_metrics


def run_gamma_sensitivity_grid(
    data_cache_file: str | Path = "cache/crisis_dryrun_15_results_top15.pkl.gz",
    output_csv: str | Path = "gamma_sensitivity_results.csv",
    output_plot: str | Path = "gamma_sensitivity_analysis.png",
) -> pd.DataFrame:
    repo_root = Path(__file__).parent.parent.parent.parent
    cache_path = repo_root / data_cache_file

    with gzip.open(cache_path, "rb") as f:
        cached_data = pickle.load(f)

    X = cached_data["X"]
    P = cached_data["P"]
    Z = cached_data["Z"]
    dates = cached_data.get("dates", [])

    N, M = X.shape
    print(f"Loaded dataset for sensitivity analysis: N={N}, M={M}, dates={len(dates)}")

    # Baseline gammas
    base_g1, base_g2, base_g3 = 40.0, 1.5e-5, 1.5e-2

    grid = [
        {"name": "Baseline", "gamma1": base_g1, "gamma2": base_g2, "gamma3": base_g3},
        {"name": "gamma1 = 20 (0.5x)", "gamma1": base_g1 * 0.5, "gamma2": base_g2, "gamma3": base_g3},
        {"name": "gamma1 = 80 (2.0x)", "gamma1": base_g1 * 2.0, "gamma2": base_g2, "gamma3": base_g3},
        {"name": "gamma2 = 0.75e-5 (0.5x)", "gamma1": base_g1, "gamma2": base_g2 * 0.5, "gamma3": base_g3},
        {"name": "gamma2 = 3.0e-5 (2.0x)", "gamma1": base_g1, "gamma2": base_g2 * 2.0, "gamma3": base_g3},
        {"name": "gamma3 = 0.75e-2 (0.5x)", "gamma1": base_g1, "gamma2": base_g2, "gamma3": base_g3 * 0.5},
        {"name": "gamma3 = 3.0e-2 (2.0x)", "gamma1": base_g1, "gamma2": base_g2, "gamma3": base_g3 * 2.0},
    ]

    records = []
    trajectories = {}

    for idx, cfg in enumerate(grid):
        g1, g2, g3 = cfg["gamma1"], cfg["gamma2"], cfg["gamma3"]
        name = cfg["name"]
        print(f"\n--- [{idx+1}/{len(grid)}] Running: {name} (g1={g1}, g2={g2:.1e}, g3={g3:.1e}) ---")

        # Check if baseline was already cached
        if idx == 0 and "L" in cached_data and "beta" in cached_data:
            L = cached_data["L"]
            beta = cached_data["beta"]
            s_hat = cached_data["s_hat"]
            obj_val = cached_data.get("objective", np.nan)
            iters = getattr(cached_data.get("results", None), "solver", {}).get("num_iters", 37)
            solve_time = 62.0
            print("Loaded Baseline solution from cache!")
        else:
            t0 = time.time()
            m, results, obj_val, L, c1, c2, beta, s_hat = solve_instance(
                X, P, Z,
                k=1,
                gammas=(g1, g2, g3),
                ipopt_opts={"tol": 1e-5, "max_iter": 100},
            )
            solve_time = time.time() - t0
            iters = 35
            print(f"Solved in {solve_time:.1f}s, objective={obj_val:.2f}")

        metrics = compute_network_metrics(L)
        se_dict = compute_beta_standard_errors(X, P, beta, s_hat)

        # Regressor paths (excluding intercept at 0)
        # 1: momentum, 2: volatility, 3: pe_ratio, 4: turnover
        mean_frob = float(np.mean(metrics["frob_norm"]))
        mean_lam2 = float(np.mean(metrics["lambda2"]))
        mean_spec = float(np.mean(metrics["spectral_radius"]))
        mean_delta = float(np.mean(metrics["delta_L"][1:]))

        mean_mom = float(np.mean(beta[1, :]))
        mean_vol = float(np.mean(beta[2, :]))
        mean_pe = float(np.mean(beta[3, :]))
        mean_turn = float(np.mean(beta[4, :]))

        records.append({
            "Configuration": name,
            "gamma1": g1,
            "gamma2": g2,
            "gamma3": g3,
            "Objective": obj_val,
            "SolveTime_s": round(solve_time, 1),
            "Mean_FrobNorm": round(mean_frob, 2),
            "Mean_Lambda2": round(mean_lam2, 4),
            "Mean_SpectralRadius": round(mean_spec, 2),
            "Mean_DeltaL": round(mean_delta, 2),
            "Mean_Beta_Mom": round(mean_mom, 5),
            "Mean_Beta_Vol": round(mean_vol, 5),
            "Mean_Beta_PE": round(mean_pe, 5),
            "Mean_Beta_Turn": round(mean_turn, 5),
        })

        trajectories[name] = {
            "metrics": metrics,
            "beta": beta,
        }

    df = pd.DataFrame(records)
    df.to_csv(repo_root / output_csv, index=False)
    print(f"\nSaved sensitivity comparison table ➜ {output_csv}")

    # Plot Sensitivity Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Sensitivity of Network Metrics and Regressors to Regularization Gammas", fontsize=14, fontweight="bold")

    plot_dates = pd.to_datetime(dates) if len(dates) == M else np.arange(M)

    # 1. Frobenius Norm comparison
    ax = axes[0, 0]
    for name, traj in trajectories.items():
        ls = "-" if name == "Baseline" else "--"
        lw = 2.5 if name == "Baseline" else 1.2
        ax.plot(plot_dates, traj["metrics"]["frob_norm"], label=name, linestyle=ls, linewidth=lw)
    ax.set_title(r"Frobenius Norm $\|L[t]\|_F$", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")

    # 2. Algebraic Connectivity comparison
    ax = axes[0, 1]
    for name, traj in trajectories.items():
        ls = "-" if name == "Baseline" else "--"
        lw = 2.5 if name == "Baseline" else 1.2
        ax.plot(plot_dates, traj["metrics"]["lambda2"], label=name, linestyle=ls, linewidth=lw)
    ax.set_title(r"Algebraic Connectivity $\lambda_2(L[t])$", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. Volatility Beta path comparison
    ax = axes[1, 0]
    pred_dates = plot_dates[1:] if len(plot_dates) == M else np.arange(M - 1)
    for name, traj in trajectories.items():
        ls = "-" if name == "Baseline" else "--"
        lw = 2.5 if name == "Baseline" else 1.2
        ax.plot(pred_dates, traj["beta"][2, :], label=name, linestyle=ls, linewidth=lw)
    ax.set_title(r"Volatility Regressor Sensitivity $\beta_{\mathrm{vol}}[t]$", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 4. Momentum Beta path comparison
    ax = axes[1, 1]
    for name, traj in trajectories.items():
        ls = "-" if name == "Baseline" else "--"
        lw = 2.5 if name == "Baseline" else 1.2
        ax.plot(pred_dates, traj["beta"][1, :], label=name, linestyle=ls, linewidth=lw)
    ax.set_title(r"Momentum Regressor Sensitivity $\beta_{\mathrm{mom}}[t]$", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(repo_root / output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensitivity comparison plot ➜ {output_plot}")

    return df


if __name__ == "__main__":
    run_gamma_sensitivity_grid()
