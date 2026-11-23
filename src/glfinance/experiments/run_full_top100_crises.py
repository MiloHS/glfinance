"""
Master execution script for solving the full N=100 optimization for both
the 2008 Financial Crisis and the 2020 COVID-19 Liquidity Shock,
and generating the 4 publication graphs (network metrics and regressor sensitivities with CIs).
"""

from pathlib import Path
import time
import gzip
import pickle
import numpy as np
import pandas as pd

from glfinance.experiments.crisis_top100 import build_crisis_dataset, solve_crisis_model
from glfinance.experiments.dryrun_plots import compute_network_metrics, plot_network_metrics, plot_regressors_with_confidence_bands


def run_full_pipeline():
    repo_root = Path(__file__).parent.parent.parent.parent
    cache_dir = repo_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART 1: 2020 COVID-19 CRISIS (N=100 stocks, M=28 months)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: BUILDING 2020 COVID-19 DATASET (N=100)")
    print("="*70)

    X_cov, P_cov, Z_cov, tickers_cov, regs_cov, dates_cov = build_crisis_dataset(
        crsp_path="crsp_covid2.csv",
        compustat_path="compustat_covid.csv",
        top_n=100,
        start_date="2019-05-01",
        end_date="2021-12-31",
        lookback_weeks=20,
        experiment_name="covid_top100_final",
    )
    print(f"COVID Dataset ready: X={X_cov.shape}, P={P_cov.shape}, Z={Z_cov.shape}, N={len(tickers_cov)}, M={len(dates_cov)}")

    print("\n" + "="*70)
    print("STEP 2: SOLVING 2020 COVID-19 MODEL WITH IPOPT (N=100)")
    print("="*70)
    t0 = time.time()
    covid_res = solve_crisis_model(
        X_cov, P_cov, Z_cov, tickers_cov,
        dates=dates_cov,
        regressor_names=regs_cov,
        k=0,
        gammas=(40.0, 1.5e-5, 1.5e-2),
        experiment_name="covid_top100_final",
    )
    covid_time = time.time() - t0
    print(f"COVID Optimization completed in {covid_time/60:.2f} minutes (objective={covid_res['objective']:.2f})")

    print("\n" + "="*70)
    print("STEP 3: GENERATING 2020 COVID-19 PLOTS (N=100)")
    print("="*70)
    metrics_cov = compute_network_metrics(covid_res["L"])
    plot_network_metrics(
        dates=covid_res["dates"],
        metrics=metrics_cov,
        title="2020 COVID-19 Crisis: Network Metrics Over Time (N=100)",
        output_path="covid_network_metrics_journal.png",
        crisis_shade=("2020-02-01", "2020-04-30"),
    )
    plot_regressors_with_confidence_bands(
        dates=covid_res["dates"],
        beta=covid_res["beta"],
        beta_se=covid_res["beta_se_robust"],
        regressor_names=covid_res["regressor_names"],
        title="2020 COVID-19 Crisis: Regressor Coefficients with 95% Confidence Bands (N=100)",
        output_path="covid_regressors_with_ci.png",
        crisis_shade=("2020-02-01", "2020-04-30"),
    )

    # =========================================================================
    # PART 2: 2008 GLOBAL FINANCIAL CRISIS (N=100 stocks, M=54 months)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: BUILDING 2008 FINANCIAL CRISIS DATASET (N=100)")
    print("="*70)

    X_cri, P_cri, Z_cri, tickers_cri, regs_cri, dates_cri = build_crisis_dataset(
        crsp_path="crsp_crisis2.csv",
        compustat_path="compustat_crisis.csv",
        top_n=100,
        start_date="2006-03-01",
        end_date="2010-12-31",
        lookback_weeks=20,
        experiment_name="crisis_top100_final",
    )
    print(f"Crisis Dataset ready: X={X_cri.shape}, P={P_cri.shape}, Z={Z_cri.shape}, N={len(tickers_cri)}, M={len(dates_cri)}")

    print("\n" + "="*70)
    print("STEP 5: SOLVING 2008 FINANCIAL CRISIS MODEL WITH IPOPT (N=100)")
    print("="*70)
    t0 = time.time()
    crisis_res = solve_crisis_model(
        X_cri, P_cri, Z_cri, tickers_cri,
        dates=dates_cri,
        regressor_names=regs_cri,
        k=0,
        gammas=(40.0, 1.5e-5, 1.5e-2),
        experiment_name="crisis_top100_final",
    )
    crisis_time = time.time() - t0
    print(f"Crisis Optimization completed in {crisis_time/60:.2f} minutes (objective={crisis_res['objective']:.2f})")

    print("\n" + "="*70)
    print("STEP 6: GENERATING 2008 FINANCIAL CRISIS PLOTS (N=100)")
    print("="*70)
    metrics_cri = compute_network_metrics(crisis_res["L"])
    plot_network_metrics(
        dates=crisis_res["dates"],
        metrics=metrics_cri,
        title="2008 Financial Crisis: Network Metrics Over Time (N=100)",
        output_path="crisis_network_metrics_journal.png",
        crisis_shade=("2007-12-01", "2009-06-30"),
    )
    plot_regressors_with_confidence_bands(
        dates=crisis_res["dates"],
        beta=crisis_res["beta"],
        beta_se=crisis_res["beta_se_robust"],
        regressor_names=crisis_res["regressor_names"],
        title="2008 Financial Crisis: Regressor Coefficients with 95% Confidence Bands (N=100)",
        output_path="crisis_regressors_with_ci.png",
        crisis_shade=("2007-12-01", "2009-06-30"),
    )

    print("\n" + "="*70)
    print("ALL OPTIMIZATIONS COMPLETED AND ALL 4 GRAPHS SAVED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    run_full_pipeline()
