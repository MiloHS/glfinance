"""
Sector-by-Sector Crisis Dynamics Experiment (Section 4.1.2, Figures 2 & 8)
Retooled for US-Only Stocks & 4 Company-Specific Regressors using Existing Crisis Data.

This module analyzes how financial stress spread asynchronously across three key
economic sectors during the 2008 Global Financial Crisis:
  1. Financials: C, BAC, JPM, WFC (excludes non-US: MUFG, BCS)
  2. Energy: XOM, CVX, COP (excludes non-US: TTE, SHEL, BP)
  3. Consumer Staples: PG, PEP, KO, WMT (excludes non-US: UL, BTI)

Scope & Data Specifications:
  - Tickers: Restricts the universe strictly to US equities present in CRSP and Compustat.
  - Regressors: Uses strictly the 4 firm-specific characteristics:
      1. Momentum (6-month return)
      2. Volatility (21-day return std)
      3. P/E Ratio (price / Compustat epsf12)
      4. Turnover (volume / CRSP shares outstanding)
    (Macroeconomic and common factors Mkt-RF, SMB, HML, and Interest Rate are excluded).
  - Existing Crisis Data: Slices directly from existing preprocessed crisis cache
    (e.g., 'cache/XPZ_90da40d598c26518.pkl.gz' or any cached crisis tensor) or builds
    from 'compustat_crisis.csv' and 'crsp_crisis2.csv' in the project.
  - Solvers: Supports both:
      1. Nonlinear polynomial-lag estimation via Pyomo + IPOPT (Algorithm 2, k >= 1)
      2. Convex one-lag estimation via CVXPY (Algorithm 1, k = 1) with auto-fallback
         if the IPOPT binary is not installed locally.

Reproductions:
  - Figure 8: Multi-sector spectral radius lambda_max(L[t]) trajectory over time
  - Figure 2: Network graph visualizations for Financials at milestone crisis dates
  - 4-panel network metric summaries per sector (Algebraic connectivity, Frobenius norm,
    Spectral radius, and Structural change / time consistency).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from itertools import cycle
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Sector Definitions (US Equities Only)
# -----------------------------------------------------------------------------
# Excluded non-US foreign stocks from the original 17/18-stock experiment:
#   Finance: MUFG (Japan), BCS (UK)
#   Energy: TTE (France), SHEL (UK), BP (UK)
#   Consumer Staples: UL (UK/EU), BTI (UK)
SECTOR_TICKERS: dict[str, list[str]] = {
    "Finance": [
        "C",     # Citigroup
        "BAC",   # Bank of America
        "JPM",   # JPMorgan Chase
        "WFC",   # Wells Fargo
    ],
    "Energy": [
        "XOM",   # ExxonMobil
        "CVX",   # Chevron
        "COP",   # ConocoPhillips
    ],
    "Consumer Staples": [
        "PG",    # Procter & Gamble
        "PEP",   # PepsiCo
        "KO",    # Coca-Cola
        "WMT",   # Walmart
    ],
}

# The four firm-specific characteristics (plus intercept)
DEFAULT_REGRESSORS: list[str] = [
    "momentum",
    "volatility",
    "pe_ratio",
    "turnover",
]


# -----------------------------------------------------------------------------
# Data Loading from Existing Crisis Datasets
# -----------------------------------------------------------------------------
def find_crisis_cache_file(cache_dir: Path | str, required_tickers: list[str]) -> Path | None:
    """Locate a cached crisis tensor file in cache_dir containing the required tickers."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    # Check known candidate files first
    priority_candidates = [
        cache_path / "XPZ_90da40d598c26518.pkl.gz",
        cache_path / "XPZ_504c18fcbe29cbc7.pkl.gz",
        cache_path / "XPZ_6383e7ad7190e73a.pkl.gz",
        cache_path / "XPZ_e09a1f23e1fb4cd9.pkl.gz",
    ]
    all_files = priority_candidates + list(cache_path.glob("XPZ_*.pkl.gz"))

    import pickle
    for p in all_files:
        if not p.exists():
            continue
        try:
            with gzip.open(p, "rb") as f:
                data = pickle.load(f)
            if len(data) >= 4:
                cached_tickers = data[3]
                if all(t in cached_tickers for t in required_tickers):
                    return p
        except Exception:
            continue
    return None


def slice_sector_from_full_dataset(
    X_full: np.ndarray,
    P_full: np.ndarray,
    Z_full: np.ndarray,
    tickers_full: list[str],
    sector_tickers: list[str],
    dates_full: list[pd.Timestamp] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[pd.Timestamp]]:
    """Extract sub-tensors for sector_tickers from full preprocessed crisis tensors."""
    idx = [tickers_full.index(t) for t in sector_tickers]

    # Slice X: shape (N_sec, M)
    X_sec = X_full[idx, :]

    # Slice P: shape (M, N_sec, J)
    # Ensure J has intercept + 4 characteristics (5 columns total)
    P_sec = P_full[:, idx, :]
    if P_sec.shape[2] > 5:
        P_sec = P_sec[:, :, :5]

    # Slice Z: shape (M, N_sec, N_sec)
    Z_sec = Z_full[:, idx, :][:, :, idx]

    dates = dates_full if dates_full is not None else []
    return X_sec, P_sec, Z_sec, sector_tickers, dates


def load_or_prepare_sector_data(
    sector_name: str,
    tickers: list[str],
    cache_dir: str | Path = "cache",
    crisis_cache_file: str | Path | None = None,
    compustat_path: str | Path = "compustat_crisis.csv",
    crsp_path: str | Path = "crsp_crisis2.csv",
    start_date: str = "2006-03-01",
    end_date: str = "2010-12-31",
    lookback_weeks: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[pd.Timestamp]]:
    """Load or prepare (X, P, Z) for the given sector from existing crisis datasets.

    Strategy:
      1. Check for sector-specific cached tensor in cache_dir.
      2. If not found, slice from a master crisis cache tensor containing all US tickers.
      3. If neither cache is available, build fresh from compustat_crisis.csv and crsp_crisis2.csv.
    """
    cache_p = Path(cache_dir)
    cache_p.mkdir(parents=True, exist_ok=True)

    sec_slug = sector_name.lower().replace(" ", "_")
    sec_cache_file = cache_p / f"XPZ_sector_{sec_slug}_us_only.pkl.gz"

    import pickle
    if sec_cache_file.exists():
        with gzip.open(sec_cache_file, "rb") as f:
            X, P, Z, sec_tickers, sec_dates = pickle.load(f)
        print(f"[{sector_name}] Loaded sector dataset from cache ➜ {sec_cache_file.name}")
        return X, P, Z, sec_tickers, sec_dates

    # Try slicing from master crisis cache
    full_cache = Path(crisis_cache_file) if crisis_cache_file else find_crisis_cache_file(cache_p, tickers)
    if full_cache and full_cache.exists():
        print(f"[{sector_name}] Slicing US sector tickers from crisis dataset ➜ {full_cache.name}")
        with gzip.open(full_cache, "rb") as f:
            data = pickle.load(f)
        X_full, P_full, Z_full = data[0], data[1], data[2]
        tickers_full = data[3]
        dates_full = data[4] if len(data) >= 5 else []

        X, P, Z, sec_tickers, sec_dates = slice_sector_from_full_dataset(
            X_full, P_full, Z_full, tickers_full, tickers, dates_full
        )

        with gzip.open(sec_cache_file, "wb") as f:
            pickle.dump((X, P, Z, sec_tickers, sec_dates), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{sector_name}] Saved sector slice to cache ➜ {sec_cache_file.name}")
        return X, P, Z, sec_tickers, sec_dates

    # Fallback: build via glfinance.data.prepare_XPZ using existing CRSP and Compustat CSVs
    print(f"[{sector_name}] Preparing sector dataset from CRSP & Compustat...")
    from glfinance.data import prepare_XPZ

    X, P, Z, aligned_tickers, dates = prepare_XPZ(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lookback_weeks=lookback_weeks,
        regressor_names=DEFAULT_REGRESSORS,
        compustat_df=compustat_path,
        crsp_df=crsp_path,
        cache_dir=cache_p,
        return_tickers=True,
        return_dates=True,
    )

    with gzip.open(sec_cache_file, "wb") as f:
        pickle.dump((X, P, Z, aligned_tickers, dates), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{sector_name}] Built and cached sector dataset ➜ {sec_cache_file.name}")
    return X, P, Z, aligned_tickers, dates


# -----------------------------------------------------------------------------
# Solver Implementations: IPOPT (Nonlinear) & CVXPY (Convex Fallback)
# -----------------------------------------------------------------------------
def solve_sector_ipopt(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    k: int = 2,
    gammas: tuple[float, float, float] = (40.0, 1.5e-5, 1.5e-2),
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve polynomial-lag graph learning model using Pyomo and IPOPT (Algorithm 2)."""
    from glfinance.graph import solve_instance

    m, results, obj_val, L, c1, c2, beta, s_hat = solve_instance(
        X, P, Z, k=k, gammas=gammas, ipopt_opts={"tol": 1e-6, "max_iter": max_iter}
    )
    return L, c1, c2, beta, s_hat, obj_val


def solve_sector_cvxpy(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    gammas: tuple[float, float, float] = (40.0, 1.5e-5, 1.5e-2),
    solver: str = "SCS",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the one-lag convex graph learning model via CVXPY (Algorithm 1).

    Convex formulation of Problem (16):
      min sum_t -tr(L[t] Z[t]) - gamma1 sum_i log(L[t]_ii) + gamma2 ||L[t]||_F^2
          + sum_t ||x[t] - ((W[t-1] + D[t-1])x[t-1] + P[t] beta[t])||_2^2
          + gamma3 sum_t ||L[t] - L[t-1]||_F^2
      s.t. L[t] 1 = 0, L[t]_ij <= 0 (i != j), diag(L[t]) >= 1e-6.
    """
    N, M = X.shape
    J = P.shape[2]
    gamma1, gamma2, gamma3 = gammas

    L_vars = [cp.Variable((N, N), symmetric=True) for _ in range(M)]
    Beta_vars = [cp.Variable(J) for _ in range(M - 1)]

    obj_terms = []
    constraints = []
    off_diag_mask = np.ones((N, N)) - np.eye(N)

    for t in range(M):
        Lt = L_vars[t]
        Zt = Z[t]

        # Dirichlet smoothness & regularizers
        obj_terms.append(-cp.trace(Lt @ Zt))
        obj_terms.append(-gamma1 * cp.sum(cp.log(cp.diag(Lt))))
        obj_terms.append(gamma2 * cp.sum_squares(Lt))

        # Constraints
        constraints.append(Lt @ np.ones(N) == 0)
        constraints.append(cp.diag(Lt) >= 1e-6)
        constraints.append(cp.multiply(off_diag_mask, Lt) <= 0)

        # Temporal smoothness
        if t > 0:
            obj_terms.append(gamma3 * cp.sum_squares(Lt - L_vars[t - 1]))

    # Signal prediction error: x[t] - (W[t-1] + D[t-1])x[t-1] - P[t] beta[t]
    # For Laplacian L = D - W, W + D = 2*diag(L) - L (affine in L)
    for t in range(1, M):
        L_prev = L_vars[t - 1]
        W_plus_D = 2.0 * cp.diag(cp.diag(L_prev)) - L_prev
        x_prev = X[:, t - 1]
        x_curr = X[:, t]
        beta_t = Beta_vars[t - 1]
        P_t = P[t]

        x_hat = W_plus_D @ x_prev + P_t @ beta_t
        obj_terms.append(cp.sum_squares(x_curr - x_hat))

    problem = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)
    slv = getattr(cp, solver.upper(), cp.SCS)
    problem.solve(solver=slv, verbose=False)

    L_opt = np.zeros((M, N, N))
    for t in range(M):
        L_opt[t] = L_vars[t].value

    beta_opt = np.zeros((J, M))
    for t in range(1, M):
        beta_opt[:, t] = Beta_vars[t - 1].value

    s_hat_opt = np.zeros((N, M))
    for t in range(1, M):
        L_prev_val = L_opt[t - 1]
        w_plus_d_val = 2.0 * np.diag(np.diag(L_prev_val)) - L_prev_val
        s_hat_opt[:, t] = w_plus_d_val @ X[:, t - 1] + P[t] @ beta_opt[:, t]

    return L_opt, beta_opt, s_hat_opt, float(problem.value)


# -----------------------------------------------------------------------------
# Distance & Metric Helpers
# -----------------------------------------------------------------------------
def distance_matrix_unweighted_corr(X: np.ndarray) -> np.ndarray:
    """Compute distance matrix Z from sample correlation: Z_ij = 1/(|rho_ij| + 0.01) - 1/1.01."""
    corr = np.corrcoef(X)
    Z = 1.0 / (np.abs(corr) + 0.01) - 1.0 / 1.01
    np.fill_diagonal(Z, 0.0)
    return Z


def algebraic_connectivity(L: np.ndarray) -> np.ndarray:
    """Second smallest eigenvalue lambda_2(L[t])."""
    T = L.shape[0]
    lam2 = np.empty(T)
    for t in range(T):
        evals = np.sort(np.linalg.eigvals(L[t]).real)
        lam2[t] = evals[1] if len(evals) > 1 else 0.0
    return lam2


def frobenius_norm(L: np.ndarray) -> np.ndarray:
    """Frobenius norm ||L[t]||_F."""
    return np.linalg.norm(L, axis=(1, 2))


def spectral_radius(L: np.ndarray) -> np.ndarray:
    """Largest absolute eigenvalue lambda_max(L[t])."""
    T = L.shape[0]
    lam_max = np.empty(T)
    for t in range(T):
        lam_max[t] = np.max(np.linalg.eigvals(L[t]).real)
    return lam_max


def time_consistency(L: np.ndarray) -> np.ndarray:
    """Change in connectivity ||L[t] - L[t-1]||_F."""
    diffs = np.linalg.norm(np.diff(L, axis=0), axis=(1, 2))
    return np.insert(diffs, 0, np.nan)


def compute_scale_free_network_metrics(L: np.ndarray) -> dict[str, np.ndarray]:
    """Compute scale-free network metrics normalized for varying node counts N.

    Metrics:
      1. Average Edge Weight (Coupling Density): tr(L[t]) / (N * (N - 1))
         - Represents mean pairwise interaction strength invariant to graph size.
      2. Scale-Free Frobenius Norm: ||L[t]||_F / N
         - Normalizes graph magnitude by number of nodes.
      3. Scale-Free Spectral Radius: lambda_max(L[t]) / N
         - Normalizes dominant eigenvalue by theoretical maximum scaling with N.
      4. Scale-Free Algebraic Connectivity: lambda_2(L[t]) / N
         - Normalizes Fiedler value for comparison across different universe sizes.
      5. Normalized Laplacian Connectivity: lambda_2(D^{-1/2} L[t] D^{-1/2})
         - Standard spectral graph theory measure (Chung 1997), bounded in [0, N/(N-1)].
      6. Relative Structural Change: ||L[t] - L[t-1]||_F / ||L[t-1]||_F
         - Dimensionless rate of topological realignment.
    """
    M, N, _ = L.shape
    lam_max_scaled = np.zeros(M)
    frob_scaled = np.zeros(M)
    lambda2_scaled = np.zeros(M)
    lambda2_norm_lap = np.zeros(M)
    avg_edge_weight = np.zeros(M)
    rel_change = np.zeros(M)
    rel_change[0] = np.nan

    for t in range(M):
        Lt = L[t]
        frob_scaled[t] = np.linalg.norm(Lt) / float(N)
        evals = np.sort(np.linalg.eigvals(Lt).real)
        lam_max_scaled[t] = evals[-1] / float(N)
        lambda2_scaled[t] = evals[1] / float(N) if N > 1 else 0.0

        # Mean coupling per potential edge: tr(L) / (N*(N-1))
        avg_edge_weight[t] = np.trace(Lt) / float(N * (N - 1)) if N > 1 else 0.0

        # Normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}
        d = np.diag(Lt)
        d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
        D_inv = np.diag(d_inv_sqrt)
        L_norm = D_inv @ Lt @ D_inv
        evals_norm = np.sort(np.linalg.eigvals(L_norm).real)
        lambda2_norm_lap[t] = evals_norm[1] if N > 1 else 0.0

        if t > 0:
            norm_prev = np.linalg.norm(L[t - 1])
            rel_change[t] = np.linalg.norm(Lt - L[t - 1]) / norm_prev if norm_prev > 1e-12 else 0.0

    return {
        "spectral_radius_per_node": lam_max_scaled,
        "frobenius_per_node": frob_scaled,
        "algebraic_connectivity_per_node": lambda2_scaled,
        "normalized_algebraic_connectivity": lambda2_norm_lap,
        "average_edge_weight": avg_edge_weight,
        "relative_structural_change": rel_change,
    }


# -----------------------------------------------------------------------------
# Publication-Quality Plotting (Matching Figures 2 & 8 + Scale-Free)
# -----------------------------------------------------------------------------
def plot_scale_free_comparison(
    laps_dict: dict[str, np.ndarray],
    save_path: str | Path = "figure_scale_free_4panel_comparison.png",
    dates: list[pd.Timestamp] | None = None,
    title: str = "2008 Financial Crisis: Cross-Sector Scale-Free Network Metrics",
    crisis_shade: tuple[str, str] | None = ("2007-12-01", "2009-06-30"),
    start_year: int = 2006,
    max_t: int = 60,
    dpi: int = 300,
) -> Figure:
    """Plot 4-panel scale-free metric comparison across sectors of different sizes.

    Matches the 2x2 layout of classical network metrics (dryrun_plots.py / Section 4.1):
      (0, 0): Scale-Free Algebraic Connectivity: lambda_2(L[t]) / N
      (0, 1): Scale-Free Frobenius Magnitude: ||L[t]||_F / N
      (1, 0): Scale-Free Spectral Radius: rho(L[t]) / N
      (1, 1): Relative Structural Change: ||L[t] - L[t-1]||_F / ||L[t-1]||_F
    Uses a single unified legend for the whole figure and matching journal styling.
    """
    lengths = [L.shape[0] for L in laps_dict.values()]
    T = min(min(lengths), max_t)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    if dates is not None and len(dates) >= T:
        date_series = pd.to_datetime(dates[:T])
    else:
        date_series = pd.date_range(start=f"{start_year}-07-31", periods=T, freq="ME")

    sector_styles = {
        "Finance": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.0},
        "Energy": {"color": "#ff7f0e", "marker": "s", "linestyle": "--", "linewidth": 2.0},
        "Consumer Staples": {"color": "#2ca02c", "marker": "^", "linestyle": "-.", "linewidth": 2.0},
    }
    default_styles = cycle([
        {"color": "#9467bd", "marker": "D", "linestyle": ":", "linewidth": 2.0},
        {"color": "#8c564b", "marker": "x", "linestyle": "-", "linewidth": 2.0},
    ])

    metrics_per_sector = {
        label: compute_scale_free_network_metrics(L[:T])
        for label, L in laps_dict.items()
    }

    # (0, 0): Scale-Free Algebraic Connectivity
    ax = axes[0, 0]
    for label, metrics in metrics_per_sector.items():
        st = sector_styles.get(label, next(default_styles))
        ax.plot(
            date_series,
            metrics["algebraic_connectivity_per_node"],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=st.get("linewidth", 2.0),
            markersize=3.5,
            label=label,
        )
    ax.set_title(r"Scale-Free Algebraic Connectivity $\lambda_2(L[t]) / N$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\lambda_2 / N$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # (0, 1): Scale-Free Frobenius Magnitude
    ax = axes[0, 1]
    for label, metrics in metrics_per_sector.items():
        st = sector_styles.get(label, next(default_styles))
        ax.plot(
            date_series,
            metrics["frobenius_per_node"],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=st.get("linewidth", 2.0),
            markersize=3.5,
            label=label,
        )
    ax.set_title(r"Scale-Free Frobenius Magnitude $\|L[t]\|_F / N$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\|L\|_F / N$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # (1, 0): Scale-Free Spectral Radius
    ax = axes[1, 0]
    for label, metrics in metrics_per_sector.items():
        st = sector_styles.get(label, next(default_styles))
        ax.plot(
            date_series,
            metrics["spectral_radius_per_node"],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=st.get("linewidth", 2.0),
            markersize=3.5,
            label=label,
        )
    ax.set_title(r"Scale-Free Spectral Radius $\rho(L[t]) / N$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\rho / N$", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # (1, 1): Relative Structural Change (Plot from index 1 forward)
    ax = axes[1, 1]
    for label, metrics in metrics_per_sector.items():
        st = sector_styles.get(label, next(default_styles))
        ax.plot(
            date_series[1:],
            metrics["relative_structural_change"][1:],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=st.get("linewidth", 2.0),
            markersize=3.5,
            label=label,
        )
    ax.set_title(r"Relative Structural Change $\frac{\|L[t] - L[t-1]\|_F}{\|L[t-1]\|_F}$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\Delta L_{\mathrm{rel}}[t]$", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Optional crisis shading
    if crisis_shade:
        s_start, s_end = pd.to_datetime(crisis_shade[0]), pd.to_datetime(crisis_shade[1])
        if s_start <= date_series[-1] and s_end >= date_series[0]:
            for row in axes:
                for sub_ax in row:
                    sub_ax.axvspan(s_start, s_end, color="gray", alpha=0.2, label="Crisis Window")

    for row in axes:
        for sub_ax in row:
            sub_ax.tick_params(axis="x", labelrotation=30)

    # Single unified legend placed at top of figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(by_label),
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scale-free 4-panel comparison ➜ {out_p}")
    return fig


def plot_spectral_radius_multi(
    laps_dict: dict[str, np.ndarray],
    save_path: str | Path = "sector_spectral_radius.png",
    dates: list[pd.Timestamp] | None = None,
    start_year: int = 2006,
    max_t: int = 60,
    dpi: int = 300,
) -> Figure:
    """Reproduce Figure 8: Multi-sector spectral radius comparison over time.

    Shows Financials, Energy, and Consumer Staples spectral radii over the crisis period.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
    })

    lengths = [L.shape[0] for L in laps_dict.values()]
    T = min(min(lengths), max_t)

    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    # Consistent sector styling matching Figure 8
    sector_styles = {
        "Finance": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "Energy": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "Consumer Staples": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    }
    default_styles = cycle([
        {"color": "#9467bd", "marker": "D", "linestyle": ":"},
        {"color": "#8c564b", "marker": "x", "linestyle": "-"},
    ])
    timesteps = np.arange(T)

    for label, L in laps_dict.items():
        st = sector_styles.get(label, next(default_styles))
        y = spectral_radius(L[:T])
        ax.plot(
            timesteps,
            y,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            label=label,
            markersize=3.5,
            linewidth=1.2,
        )

    tick_pos = np.arange(0, T, 12)
    if dates and len(dates) >= T:
        tick_lbls = [pd.to_datetime(dates[i]).strftime("%Y") for i in tick_pos]
    else:
        tick_lbls = [str(start_year + i) for i in range(len(tick_pos))]

    ax.set_xlabel("Year")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls)
    ax.set_ylabel(r"$\lambda_{\max}(L_t)$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=8)
    fig.tight_layout()

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    print(f"Saved multi-sector spectral radius comparison ➜ {out_p}")
    return fig


def plot_scale_free_spectral_radius_multi(
    laps_dict: dict[str, np.ndarray],
    save_path: str | Path = "figure8_scale_free_spectral_radius.png",
    dates: list[pd.Timestamp] | None = None,
    start_year: int = 2006,
    max_t: int = 60,
    dpi: int = 300,
) -> Figure:
    """Multi-sector comparison of scale-free spectral radius lambda_max(L_t) / N over time."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
    })

    lengths = [L.shape[0] for L in laps_dict.values()]
    T = min(min(lengths), max_t)

    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    sector_styles = {
        "Finance": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "Energy": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "Consumer Staples": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    }
    default_styles = cycle([
        {"color": "#9467bd", "marker": "D", "linestyle": ":"},
        {"color": "#8c564b", "marker": "x", "linestyle": "-"},
    ])
    timesteps = np.arange(T)

    for label, L in laps_dict.items():
        st = sector_styles.get(label, next(default_styles))
        N = L.shape[1]
        y = spectral_radius(L[:T]) / float(N)
        ax.plot(
            timesteps,
            y,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            label=label,
            markersize=3.5,
            linewidth=1.2,
        )

    tick_pos = np.arange(0, T, 12)
    if dates and len(dates) >= T:
        tick_lbls = [pd.to_datetime(dates[i]).strftime("%Y") for i in tick_pos]
    else:
        tick_lbls = [str(start_year + i) for i in range(len(tick_pos))]

    ax.set_xlabel("Year")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls)
    ax.set_ylabel(r"$\lambda_{\max}(L_t) / N$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=8)
    fig.tight_layout()

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    print(f"Saved scale-free spectral radius comparison ➜ {out_p}")
    return fig


def plot_sector_metrics_grid(
    L: np.ndarray,
    sector_name: str,
    save_path: str | Path,
    dates: list[pd.Timestamp] | None = None,
    start_year: int = 2006,
    max_t: int = 60,
    dpi: int = 300,
) -> None:
    """Plot the 4 network metrics for a sector in a 2x2 publication grid."""
    L_sub = L[:max_t]
    T = len(L_sub)
    timesteps = np.arange(T)

    y1 = algebraic_connectivity(L_sub)
    y2 = frobenius_norm(L_sub)
    y3 = spectral_radius(L_sub)
    y4 = time_consistency(L_sub)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.2))
    tick_pos = np.arange(0, T, 12)
    if dates and len(dates) >= T:
        tick_lbls = [pd.to_datetime(dates[i]).strftime("%Y") for i in tick_pos]
    else:
        tick_lbls = [str(start_year + i) for i in range(len(tick_pos))]

    # Colors matching classical network metrics figures
    axs[0, 0].plot(timesteps, y1, color="#1f77b4", marker="o", markersize=3, linewidth=1.2)
    axs[0, 0].set_title(rf"{sector_name} Algebraic Connectivity $\lambda_2(L[t])$")
    axs[0, 0].set_ylabel(r"$\lambda_2$")

    axs[0, 1].plot(timesteps, y2, color="#d62728", marker="s", markersize=3, linewidth=1.2)
    axs[0, 1].set_title(rf"{sector_name} Frobenius Norm $\|L[t]\|_F$")
    axs[0, 1].set_ylabel(r"$\|L\|_F$")

    axs[1, 0].plot(timesteps, y3, color="#2ca02c", marker="^", markersize=3, linewidth=1.2)
    axs[1, 0].set_title(rf"{sector_name} Spectral Radius $\rho(L[t])$")
    axs[1, 0].set_ylabel(r"$\rho$")

    axs[1, 1].plot(timesteps, y4, color="#9467bd", marker="D", markersize=3, linewidth=1.2)
    axs[1, 1].set_title(rf"{sector_name} Change in Connectivity $\Delta L[t]$")
    axs[1, 1].set_ylabel(r"$\Delta L[t]$")

    for ax in axs.flat:
        ax.set_xlabel("Year")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbls)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {sector_name} 4-panel metrics grid ➜ {out_p}")


def plot_network_graphs_at_dates(
    L: np.ndarray,
    tickers: list[str],
    save_dir: str | Path,
    dates: list[pd.Timestamp] | None = None,
    factor: float = 0.08,
    threshold: float = 1e-5,
) -> None:
    """Reproduce Figure 2: Network graphs at selected milestone dates."""
    try:
        import networkx as nx
    except ImportError:
        print("networkx is not installed. Skipping network graph topology plots.")
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = L.shape[1]
    M = L.shape[0]

    # Select representative time steps (early, crisis peak, recovery)
    sample_steps = [min(7, M - 1), min(25, M - 1), min(M - 1, 53)]

    for t in sample_steps:
        if t >= M:
            continue
        date_str = pd.to_datetime(dates[t]).strftime("%Y-%m-%d") if dates and t < len(dates) else f"Step {t}"
        mat = L[t].copy()
        W = -mat
        np.fill_diagonal(W, 0.0)

        G = nx.Graph()
        G.add_nodes_from(tickers)
        for i in range(N):
            for j in range(i + 1, N):
                w = float(W[i, j])
                if w > threshold:
                    G.add_edge(tickers[i], tickers[j], weight=w)

        fig = plt.figure(figsize=(5, 5))
        pos = nx.circular_layout(G, scale=1.5)
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            width=[max(w * factor, 0.5) for w in weights] if weights else 1.0,
            node_size=900,
            font_size=10,
        )
        plt.title(f"Financial Network Topology ({date_str})")
        plt.axis("off")
        fig_path = out_dir / f"network_graph_t{t}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved network graph for t={t} ({date_str}) ➜ {fig_path}")


# -----------------------------------------------------------------------------
# Main Experiment Workflow
# -----------------------------------------------------------------------------
def run_sector_experiment(
    sectors: Sequence[str] | None = None,
    solver_mode: str = "auto",
    k: int = 2,
    gammas: tuple[float, float, float] = (40.0, 1.5e-5, 1.5e-2),
    output_dir: str | Path = "outputs/sectors",
    cache_dir: str | Path = "cache",
    crisis_cache_file: str | Path | None = None,
    compustat_path: str | Path = "compustat_crisis.csv",
    crsp_path: str | Path = "crsp_crisis2.csv",
    start_date: str = "2006-03-01",
    end_date: str = "2010-12-31",
    lookback_weeks: int = 20,
    max_iter: int = 500,
    force_solve: bool = False,
) -> dict[str, Any]:
    """Execute the sector-by-sector experiment across US sector universes.

    Uses the 4 company-specific regressors and slices from existing crisis datasets.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_p = Path(cache_dir)
    cache_p.mkdir(parents=True, exist_ok=True)

    if sectors is None:
        sectors = list(SECTOR_TICKERS.keys())

    # Detect solver availability
    idaes_bin = Path.home() / ".idaes" / "bin"
    has_ipopt = shutil.which("ipopt") is not None or (idaes_bin / "ipopt").exists()

    if solver_mode == "auto":
        use_ipopt = has_ipopt and k >= 1
    elif solver_mode == "ipopt":
        use_ipopt = True
    elif solver_mode == "cvxpy":
        use_ipopt = False
    else:
        raise ValueError(f"Unknown solver mode '{solver_mode}'. Choose 'auto', 'ipopt', or 'cvxpy'.")

    solver_label = "IPOPT (Polynomial-Lag)" if use_ipopt else "CVXPY (One-Lag Convex)"
    print(f"Executing Sector Experiment with Solver: {solver_label}")
    print(f"Regressors: {DEFAULT_REGRESSORS} (4 firm-specific characteristics)")

    learned_laps: dict[str, np.ndarray] = {}
    sector_results: dict[str, Any] = {}
    all_dates: list[pd.Timestamp] = []

    for sec in sectors:
        if sec not in SECTOR_TICKERS:
            print(f"Unknown sector '{sec}'. Available: {list(SECTOR_TICKERS.keys())}")
            continue

        tickers = SECTOR_TICKERS[sec]
        print(f"\n{'='*60}")
        print(f"Processing Sector: {sec} ({len(tickers)} US stocks: {tickers})")
        print(f"{'='*60}")

        X, P, Z, sec_tickers, sec_dates = load_or_prepare_sector_data(
            sector_name=sec,
            tickers=tickers,
            cache_dir=cache_p,
            crisis_cache_file=crisis_cache_file,
            compustat_path=compustat_path,
            crsp_path=crsp_path,
            start_date=start_date,
            end_date=end_date,
            lookback_weeks=lookback_weeks,
        )
        if sec_dates and not all_dates:
            all_dates = sec_dates

        sec_slug = sec.lower().replace(" ", "_")
        res_cache = cache_p / f"sector_result_{sec_slug}_{'ipopt_k' + str(k) if use_ipopt else 'cvxpy'}.pkl.gz"

        import pickle
        if res_cache.exists() and not force_solve:
            with gzip.open(res_cache, "rb") as f:
                res = pickle.load(f)
            print(f"[{sec}] Loaded solved result from cache ➜ {res_cache.name}")
            L = res["L"]
            obj_val = res.get("objective", float("nan"))
            beta = res.get("beta", np.zeros((5, L.shape[0])))
        else:
            if use_ipopt:
                print(f"[{sec}] Solving via Pyomo + IPOPT (k={k}, gammas={gammas})...")
                L, c1, c2, beta, s_hat, obj_val = solve_sector_ipopt(
                    X, P, Z, k=k, gammas=gammas, max_iter=max_iter
                )
            else:
                print(f"[{sec}] Solving via CVXPY Convex SCS solver (gammas={gammas})...")
                L, beta, s_hat, obj_val = solve_sector_cvxpy(
                    X, P, Z, gammas=gammas, solver="SCS"
                )
                c1, c2 = np.array([1.0]), np.array([1.0])

            res = {
                "sector": sec,
                "tickers": sec_tickers,
                "dates": sec_dates,
                "L": L,
                "c1": c1,
                "c2": c2,
                "beta": beta,
                "s_hat": s_hat,
                "objective": obj_val,
                "regressor_names": DEFAULT_REGRESSORS,
            }

            with gzip.open(res_cache, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[{sec}] Saved solved results to cache ➜ {res_cache.name}")

        # Summary statistics
        sr_vals = spectral_radius(L)
        fn_vals = frobenius_norm(L)
        ac_vals = algebraic_connectivity(L)
        sf_metrics = compute_scale_free_network_metrics(L)

        print(f"[{sec}] Solved {L.shape[0]} months for {len(sec_tickers)} stocks (N={len(sec_tickers)}):")
        print(f"       Objective Value: {obj_val:.4f}")
        print(f"       Raw Spectral Radius (lambda_max): mean={sr_vals.mean():.4f}, max={sr_vals.max():.4f}")
        print(f"       [Scale-Free] Spectral Radius / N: mean={sf_metrics['spectral_radius_per_node'].mean():.4f}, max={sf_metrics['spectral_radius_per_node'].max():.4f}")
        print(f"       [Scale-Free] Average Edge Weight (Density): mean={sf_metrics['average_edge_weight'].mean():.4f}, max={sf_metrics['average_edge_weight'].max():.4f}")
        print(f"       [Scale-Free] Frobenius Norm / N: mean={sf_metrics['frobenius_per_node'].mean():.4f}, max={sf_metrics['frobenius_per_node'].max():.4f}")
        print(f"       [Scale-Free] Algebraic Conn / N: mean={sf_metrics['algebraic_connectivity_per_node'].mean():.4f}, max={sf_metrics['algebraic_connectivity_per_node'].max():.4f}")
        print(f"       [Scale-Free] Relative Structural Change: mean={np.nanmean(sf_metrics['relative_structural_change']):.4f}, max={np.nanmax(sf_metrics['relative_structural_change']):.4f}")
        if beta is not None and beta.shape[0] >= 5:
            print("       Fitted Characteristic Betas (mean across time):")
            for j_idx, reg_n in enumerate(DEFAULT_REGRESSORS, start=1):
                print(f"         beta_{reg_n}: {beta[j_idx, 1:].mean():.5f} (std={beta[j_idx, 1:].std():.5f})")

        learned_laps[sec] = L
        sector_results[sec] = res

        # Generate individual sector 4-panel metrics grid
        plot_sector_metrics_grid(
            L,
            sec,
            out_dir / f"{sec_slug}_metrics.png",
            dates=sec_dates,
        )

        # For Finance, generate NetworkX topologies (Figure 2)
        if sec == "Finance":
            plot_network_graphs_at_dates(
                L,
                sec_tickers,
                out_dir / "network_topologies",
                dates=sec_dates,
            )

    # Multi-sector comparisons
    if len(learned_laps) > 1:
        # Scale-Free 4-Panel Comparison (Single unified legend, matching journal styling)
        scale_free_png = out_dir / "figure_scale_free_4panel_comparison.png"
        scale_free_pdf = out_dir / "figure_scale_free_4panel_comparison.pdf"
        plot_scale_free_comparison(
            learned_laps,
            save_path=scale_free_png,
            dates=all_dates,
            max_t=60,
        )
        plot_scale_free_comparison(
            learned_laps,
            save_path=scale_free_pdf,
            dates=all_dates,
            max_t=60,
        )
        # Also save at project root if output_dir is not root
        if out_dir.resolve() != Path(".").resolve():
            plot_scale_free_comparison(
                learned_laps,
                save_path="figure_scale_free_4panel_comparison.png",
                dates=all_dates,
                max_t=60,
            )

    print("\nSector experiment workflow completed successfully!")
    return sector_results


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Sector-by-Sector Crisis Dynamics Experiment (Section 4.1.2) with US Stocks"
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        default=["Finance", "Energy", "Consumer Staples"],
        help="Sectors to evaluate (Finance, Energy, Consumer Staples, or all)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        choices=["auto", "ipopt", "cvxpy"],
        help="Optimization solver mode ('auto', 'ipopt', or 'cvxpy')",
    )
    parser.add_argument("--k", type=int, default=2, help="Polynomial lag order for IPOPT (default: 2)")
    parser.add_argument(
        "--gammas",
        nargs=3,
        type=float,
        default=[40.0, 1.5e-5, 1.5e-2],
        help="Hyperparameters gamma1 gamma2 gamma3 (default: 40.0 1.5e-5 1.5e-2)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/sectors", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory")
    parser.add_argument("--crisis-cache", type=str, default=None, help="Path to master crisis XPZ cache file")
    parser.add_argument("--compustat", type=str, default="compustat_crisis.csv", help="Compustat CSV path")
    parser.add_argument("--crsp", type=str, default="crsp_crisis2.csv", help="CRSP CSV path")
    parser.add_argument("--max-iter", type=int, default=500, help="IPOPT max iterations")
    parser.add_argument("--force-solve", action="store_true", help="Force fresh solving bypassing cache")
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    sectors = args.sectors
    if len(sectors) == 1 and sectors[0].lower() == "all":
        sectors = list(SECTOR_TICKERS.keys())

    run_sector_experiment(
        sectors=sectors,
        solver_mode=args.solver,
        k=args.k,
        gammas=tuple(args.gammas),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        crisis_cache_file=args.crisis_cache,
        compustat_path=args.compustat,
        crsp_path=args.crsp,
        max_iter=args.max_iter,
        force_solve=args.force_solve,
    )


if __name__ == "__main__":
    main()
