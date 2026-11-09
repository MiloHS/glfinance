"""
Experiment: Testing the Impact of a Temporal Penalty on Regressor Coefficients (beta[t])

Question under investigation:
In the joint graph learning model, we apply a temporal smoothness penalty gamma3 * ||L[t] - L[t-1]||_F^2
to prevent the learned Laplacians from changing too drastically between consecutive time steps.
What happens if we also add a similar temporal penalty on the regressor coefficients:
    gamma4 * sum_{t=2}^{M-1} ||beta[:, t] - beta[:, t-1]||_2^2 ?

Does adding this penalty significantly alter the learned Laplacians L[t], or does it primarily
smooth out the regressor trajectories beta[t]?
"""

import gzip
import pickle
import time
import os
from pathlib import Path
import scipy

# Ensure IDAES solver directory and gfortran libraries are in PATH / DYLD_LIBRARY_PATH
idaes_bin = str(Path.home() / ".idaes" / "bin")
if idaes_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + idaes_bin

scipy_dylibs = str(Path(scipy.__file__).parent / ".dylibs")
current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
if scipy_dylibs not in current_dyld:
    os.environ["DYLD_LIBRARY_PATH"] = scipy_dylibs + (os.pathsep + current_dyld if current_dyld else "")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo


def build_model_with_beta_penalty(
    X_data: np.ndarray,
    P_data: np.ndarray,
    Z_data: np.ndarray,
    gamma1: float = 40.0,
    gamma2: float = 1.5e-5,
    gamma3: float = 1.5e-2,
    gamma4: float = 0.0,  # Temporal penalty weight for beta[t]
    k: int = 0,
):
    """Pyomo model for joint graph learning with an additional temporal penalty on beta[t]."""
    X_data = np.asarray(X_data, dtype=float)
    P_data = np.asarray(P_data, dtype=float)
    Z_data = np.asarray(Z_data, dtype=float)

    N, M = X_data.shape
    J = P_data.shape[2]

    model = pyo.ConcreteModel()

    model.Nidx = pyo.RangeSet(0, N - 1)
    model.AllT = pyo.RangeSet(0, M - 1)
    model.PredT = pyo.RangeSet(1, M - 1)
    model.Jidx = pyo.RangeSet(0, J - 1)

    model.L = pyo.Var(model.AllT, model.Nidx, model.Nidx, domain=pyo.Reals)

    for t in model.AllT:
        for i in model.Nidx:
            model.L[t, i, i].setlb(1e-8)

    model.beta = pyo.Var(model.Jidx, model.PredT, domain=pyo.Reals)

    model.X = pyo.Param(model.Nidx, model.AllT, initialize=lambda m, i, t: float(X_data[i, t]), mutable=False)
    model.P = pyo.Param(model.AllT, model.Nidx, model.Jidx, initialize=lambda m, t, i, j: float(P_data[t, i, j]), mutable=False)
    model.Z = pyo.Param(model.AllT, model.Nidx, model.Nidx, initialize=lambda m, t, i, j: float(Z_data[t, i, j]), mutable=False)

    model.s_true = pyo.Expression(model.Nidx, model.PredT, rule=lambda m, i, t: m.X[i, t])

    def s_hat_rule(m, i, t):
        return pyo.quicksum(m.P[t, i, j] * m.beta[j, t] for j in m.Jidx)

    model.s_hat = pyo.Expression(model.Nidx, model.PredT, rule=s_hat_rule)

    def obj_rule(m):
        trace_term = -pyo.quicksum(m.L[t, i, j] * m.Z[t, j, i] for t in m.AllT for i in m.Nidx for j in m.Nidx)
        log_term = -gamma1 * pyo.quicksum(pyo.log(m.L[t, i, i]) for t in m.AllT for i in m.Nidx)
        frob_L = gamma2 * pyo.quicksum(m.L[t, i, j] ** 2 for t in m.AllT for i in m.Nidx for j in m.Nidx)
        pred = pyo.quicksum((m.s_true[i, t] - m.s_hat[i, t]) ** 2 for t in m.PredT for i in m.Nidx)
        smooth_L = gamma3 * pyo.quicksum((m.L[t, i, j] - m.L[t - 1, i, j]) ** 2 for t in m.AllT if t > 0 for i in m.Nidx for j in m.Nidx)
        
        # New temporal smoothness penalty on beta[t]
        smooth_beta = gamma4 * pyo.quicksum((m.beta[j, t] - m.beta[j, t - 1]) ** 2 for t in m.PredT if t > 1 for j in m.Jidx)
        
        return trace_term + log_term + frob_L + pred + smooth_L + smooth_beta

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def zero_row_rule(m, t, i):
        return pyo.quicksum(m.L[t, i, j] for j in m.Nidx) == 0.0

    model.zero_row = pyo.Constraint(model.AllT, model.Nidx, rule=zero_row_rule)

    def symmetry_rule(m, t, i, j):
        if i < j:
            return m.L[t, i, j] == m.L[t, j, i]
        return pyo.Constraint.Skip

    model.symmetry = pyo.Constraint(model.AllT, model.Nidx, model.Nidx, rule=symmetry_rule)

    def offdiag_rule(m, t, i, j):
        if i != j:
            return m.L[t, i, j] <= 0.0
        return pyo.Constraint.Skip

    model.offdiag = pyo.Constraint(model.AllT, model.Nidx, model.Nidx, rule=offdiag_rule)

    return model


def solve_model_instance(model, ipopt_opts=None):
    solver = pyo.SolverFactory("ipopt")
    opts = {"tol": 1e-6, "max_iter": 500}
    if ipopt_opts:
        opts.update(ipopt_opts)
    for k, v in opts.items():
        solver.options[k] = v
    results = solver.solve(model, tee=False)

    M = len(model.AllT)
    N = len(model.Nidx)
    J = len(model.Jidx)

    L_arr = np.zeros((M, N, N))
    for t in model.AllT:
        for i in model.Nidx:
            for j in model.Nidx:
                L_arr[t, i, j] = pyo.value(model.L[t, i, j])

    beta_arr = np.zeros((J, M))
    for t in model.PredT:
        for j in model.Jidx:
            beta_arr[j, t] = pyo.value(model.beta[j, t])

    return L_arr, beta_arr, pyo.value(model.obj)


def run_beta_penalty_experiment():
    print("=== LOADING STANDARDIZED 2008 CRISIS DATASET ===")
    with gzip.open("cache/XPZ_1dc837dbc9b26350.pkl.gz", "rb") as f:
        X, P, Z = pickle.load(f)

    # Use N=30 subset for rapid iteration testing
    N_sub = 30
    X_sub = X[:N_sub, :]
    P_sub = P[:, :N_sub, :]
    Z_sub = Z[:, :N_sub, :N_sub]

    print(f"Data shapes for experiment: X={X_sub.shape}, P={P_sub.shape}, Z={Z_sub.shape}")

    gamma4_values = [0.0, 1.0, 10.0, 100.0]
    results_store = {}

    for g4 in gamma4_values:
        print(f"\n---> Solving model with gamma4 (beta temporal penalty) = {g4}...")
        t0 = time.time()
        m = build_model_with_beta_penalty(X_sub, P_sub, Z_sub, gamma4=g4)
        L_sol, beta_sol, obj_val = solve_model_instance(m)
        t1 = time.time()
        print(f"Done in {t1 - t0:.2f}s | Obj: {obj_val:.2f}")

        results_store[g4] = {"L": L_sol, "beta": beta_sol, "obj": obj_val}

    # Compare learned Laplacians against baseline (gamma4 = 0.0)
    L_base = results_store[0.0]["L"]
    beta_base = results_store[0.0]["beta"]

    print("\n=== EXPERIMENTAL ANALYSIS RESULTS ===")
    print(f"{'gamma4 (Beta Penalty)':<22} | {'||L - L_base||_F (Laplacian Change)':<35} | {'||beta - beta_base||_2 (Beta Change)':<35}")
    print("-" * 98)

    for g4 in gamma4_values:
        L_diff = np.linalg.norm(results_store[g4]["L"] - L_base)
        beta_diff = np.linalg.norm(results_store[g4]["beta"] - beta_base)
        print(f"{g4:<22.1f} | {L_diff:<35.6f} | {beta_diff:<35.6f}")

    # Plot comparisons
    # True evaluation sample begins July 2006 (after 20-week lookback burn-in) through Dec 2010
    dates = pd.date_range(start="2006-07-31", periods=X.shape[1], freq="ME")

    # Plot 1: Frobenius Norm of Laplacians over time across different gamma4
    for g4 in gamma4_values:
        frob = [np.linalg.norm(results_store[g4]["L"][t], ord="fro") for t in range(X.shape[1])]
        axes[0].plot(dates, frob, label=f"gamma4 = {g4}", linewidth=1.2)
    axes[0].set_title(r"Learned Laplacian Frobenius Norm $\|L[t]\|_F$ Across $\gamma_4$", fontsize=9, fontweight="bold")
    axes[0].set_ylabel(r"$\|L[t]\|_F$", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    # Plot 2: Volatility beta[t] trajectory across different gamma4
    vol_idx = 2  # Volatility factor index
    for g4 in gamma4_values:
        axes[1].plot(dates[1:], results_store[g4]["beta"][vol_idx, 1:], label=f"gamma4 = {g4}", linewidth=1.2)
    axes[1].set_title(r"Volatility Regressor Trajectory $\beta_{vol}[t]$ Across $\gamma_4$", fontsize=9, fontweight="bold")
    axes[1].set_ylabel(r"$\beta_{vol,t}$", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = Path("beta_temporal_penalty_experiment.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nExperiment comparison plot saved to: {out_path.absolute()}")

if __name__ == "__main__":
    run_beta_penalty_experiment()
