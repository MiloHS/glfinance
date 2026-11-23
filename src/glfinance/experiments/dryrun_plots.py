"""
Plotting utilities for generating the 4 network metrics and 4 regressors with
confidence intervals (Comment 1.5) for both the 2008 Crisis and 2020 COVID periods.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_network_metrics(L: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the 4 network metrics from a sequence of Laplacians L[t].

    1. Algebraic Connectivity lambda_2(L[t])
    2. Frobenius Norm ||L[t]||_F
    3. Spectral Radius rho(L[t])
    4. Structural Change Delta L[t] = ||L[t] - L[t-1]||_F
    """
    M, N, _ = L.shape
    lambda2 = np.zeros(M)
    frob_norm = np.zeros(M)
    spectral_radius = np.zeros(M)
    delta_L = np.zeros(M)

    for t in range(M):
        Lt = L[t]
        frob_norm[t] = np.linalg.norm(Lt, "fro")
        # Eigenvalues of symmetric Laplacian
        evals = np.sort(np.linalg.eigvalsh(Lt))
        # Algebraic connectivity is 2nd smallest eigenvalue
        lambda2[t] = evals[1] if len(evals) > 1 else 0.0
        # Spectral radius is largest eigenvalue
        spectral_radius[t] = evals[-1]

        if t > 0:
            delta_L[t] = np.linalg.norm(Lt - L[t - 1], "fro")
        else:
            delta_L[0] = 0.0

    return {
        "lambda2": lambda2,
        "frob_norm": frob_norm,
        "spectral_radius": spectral_radius,
        "delta_L": delta_L,
    }


def plot_network_metrics(
    dates: list[pd.Timestamp],
    metrics: dict[str, np.ndarray],
    title: str,
    output_path: Path | str,
    crisis_shade: tuple[str, str] | None = None,
) -> None:
    """Plot the 4 network metrics over time in a publication-quality 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    date_series = pd.to_datetime(dates)

    # 1. Algebraic Connectivity
    ax = axes[0, 0]
    ax.plot(date_series, metrics["lambda2"], color="#1f77b4", linewidth=2.0)
    ax.set_title(r"Algebraic Connectivity $\lambda_2(L[t])$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\lambda_2$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2. Frobenius Norm
    ax = axes[0, 1]
    ax.plot(date_series, metrics["frob_norm"], color="#d62728", linewidth=2.0)
    ax.set_title(r"Frobenius Norm $\|L[t]\|_F$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\|L\|_F$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. Spectral Radius
    ax = axes[1, 0]
    ax.plot(date_series, metrics["spectral_radius"], color="#2ca02c", linewidth=2.0)
    ax.set_title(r"Spectral Radius $\rho(L[t])$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\rho$", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 4. Change in Connectivity
    ax = axes[1, 1]
    # Plot from index 1 forward
    ax.plot(date_series[1:], metrics["delta_L"][1:], color="#9467bd", linewidth=2.0)
    ax.set_title(r"Change in Connectivity $\Delta L[t]$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\|L[t] - L[t-1]\|_F$", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Optional crisis shading
    if crisis_shade:
        s_start, s_end = pd.to_datetime(crisis_shade[0]), pd.to_datetime(crisis_shade[1])
        for row in axes:
            for sub_ax in row:
                sub_ax.axvspan(s_start, s_end, color="gray", alpha=0.2, label="Crisis Window")

    for row in axes:
        for sub_ax in row:
            sub_ax.tick_params(axis="x", labelrotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved network metrics plot ➜ {output_path}")


def plot_regressors_with_confidence_bands(
    dates: list[pd.Timestamp],
    beta: np.ndarray,
    beta_se: np.ndarray,
    regressor_names: list[str],
    title: str,
    output_path: Path | str,
    crisis_shade: tuple[str, str] | None = None,
) -> None:
    """Plot the 4 regressors over time with 95% confidence bands (Comment 1.5).

    Creates a 4-panel figure (one for each regressor) showing:
      - Point estimate beta_j[t]
      - Shaded 95% confidence interval [beta - 1.96*SE, beta + 1.96*SE]
      - Zero reference line
    """
    # Exclude intercept at index 0 if beta has J = 1 + len(regressor_names)
    J, M_pred = beta.shape
    if J == len(regressor_names) + 1:
        plot_beta = beta[1:, :]
        plot_se = beta_se[1:, :]
    else:
        plot_beta = beta[:len(regressor_names), :]
        plot_se = beta_se[:len(regressor_names), :]

    # Pred dates correspond to t=1..M-1
    pred_dates = pd.to_datetime(dates[-M_pred:])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for idx, (name, ax, color) in enumerate(zip(regressor_names, axes.flatten(), colors)):
        b_series = plot_beta[idx]
        se_series = plot_se[idx]
        ci_low = b_series - 1.96 * se_series
        ci_high = b_series + 1.96 * se_series

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.plot(pred_dates, b_series, color=color, linewidth=2.0, label=f"$\\beta_{{{name}}}[t]$")
        ax.fill_between(
            pred_dates,
            ci_low,
            ci_high,
            color=color,
            alpha=0.25,
            label="Pointwise 95% Confidence Interval (Robust SE)",
        )

        display_name = {
            "momentum": "Momentum",
            "volatility": "Volatility",
            "pe_ratio": "P/E Ratio",
            "turnover": "Turnover",
        }.get(name, name.capitalize())

        ax.set_title(f"{display_name} Sensitivity ($\\beta_{{{name}}}[t]$)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Coefficient", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=9)

        if crisis_shade:
            s_start, s_end = pd.to_datetime(crisis_shade[0]), pd.to_datetime(crisis_shade[1])
            ax.axvspan(s_start, s_end, color="gray", alpha=0.15)

    for row in axes:
        for sub_ax in row:
            sub_ax.tick_params(axis="x", labelrotation=30)

    axes[1, 0].set_xlabel("Date", fontsize=10)
    axes[1, 1].set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved regressor paths with confidence bands ➜ {output_path}")
