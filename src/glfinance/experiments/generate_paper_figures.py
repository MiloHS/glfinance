import gzip
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from glfinance.experiments.dryrun_plots import (
    compute_network_metrics,
    plot_network_metrics,
    plot_regressors_with_confidence_bands,
)


def main():
    repo_root = Path(__file__).resolve().parents[3]

    # 1. Load newly solved N=100 models from cache
    cov_path = repo_root / "cache" / "covid_top100_sanity_k0_results_top100.pkl.gz"
    cri_path = repo_root / "cache" / "crisis_top100_final_results_top100.pkl.gz"

    with gzip.open(cov_path, "rb") as f:
        cov = pickle.load(f)

    with gzip.open(cri_path, "rb") as f:
        cri = pickle.load(f)

    print(f"Loaded COVID N={len(cov['tickers'])}, L={cov['L'].shape}")
    print(f"Loaded Crisis N={len(cri['tickers'])}, L={cri['L'].shape}")

    # 2. Overwrite all COVID network metric files
    metrics_cov = compute_network_metrics(cov["L"])
    for name in ["covid_network_metrics.png", "covid_network_metrics_journal.png"]:
        out_p = repo_root / name
        plot_network_metrics(
            dates=cov["dates"],
            metrics=metrics_cov,
            title="2020 COVID-19 Crisis: Network Metrics Over Time (N=100)",
            output_path=out_p,
            crisis_shade=("2020-02-01", "2020-04-30"),
        )
        print(f"Updated ➜ {out_p.name}")

    # 3. Overwrite all COVID regressor files
    for name in [
        "covid_regressors_overtime.png",
        "covid_regressors_with_ci.png",
        "covid_std_betas_journal.png",
        "covid_true_learned_betas_standardized.png",
    ]:
        out_p = repo_root / name
        plot_regressors_with_confidence_bands(
            dates=cov["dates"],
            beta=cov["beta"],
            beta_se=cov["beta_se_robust"],
            regressor_names=cov["regressor_names"],
            title="2020 COVID-19 Crisis: Standardized Factor Coefficients with 95% CIs (N=100)",
            output_path=out_p,
            crisis_shade=("2020-02-01", "2020-04-30"),
        )
        print(f"Updated ➜ {out_p.name}")

    # 4. Overwrite all Crisis network metric files
    metrics_cri = compute_network_metrics(cri["L"])
    for name in ["crisis_network_metrics.png", "crisis_network_metrics_journal.png"]:
        out_p = repo_root / name
        plot_network_metrics(
            dates=cri["dates"],
            metrics=metrics_cri,
            title="2008 Financial Crisis: Network Metrics Over Time (N=100)",
            output_path=out_p,
            crisis_shade=("2007-12-01", "2009-06-30"),
        )
        print(f"Updated ➜ {out_p.name}")

    # 5. Overwrite all Crisis regressor files
    for name in [
        "crisis_regressors_overtime.png",
        "crisis_regressors_with_ci.png",
        "crisis_std_betas_journal.png",
        "crisis_true_learned_betas_standardized.png",
    ]:
        out_p = repo_root / name
        plot_regressors_with_confidence_bands(
            dates=cri["dates"],
            beta=cri["beta"],
            beta_se=cri["beta_se_robust"],
            regressor_names=cri["regressor_names"],
            title="2008 Financial Crisis: Standardized Factor Coefficients with 95% CIs (N=100)",
            output_path=out_p,
            crisis_shade=("2007-12-01", "2009-06-30"),
        )
        print(f"Updated ➜ {out_p.name}")

    # 6. Generate regressor_case_studies_4panel.png using brand new N=100 data
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    fig.suptitle(
        "Historical Case Studies of Selected Standardized Factor Coefficients (N=100)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Note: beta matrix has 5 rows: row 0 is Intercept, row 1 is Momentum, row 2 is Volatility, row 3 is PE_Ratio, row 4 is Turnover
    # Panel A: Volatility during 2008 Financial Crisis (row 2)
    dates_cri = pd.to_datetime(cri["dates"])[1:]
    b_vol_cri = cri["beta"][2, :]
    se_vol_cri = cri["beta_se_robust"][2, :]
    ax = axes[0, 0]
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.plot(dates_cri, b_vol_cri, color="#d62728", linewidth=2.0, label=r"$\beta_{\mathrm{vol}}[t]$ (2008 Crisis)")
    ax.fill_between(
        dates_cri,
        b_vol_cri - 1.96 * se_vol_cri,
        b_vol_cri + 1.96 * se_vol_cri,
        color="#d62728",
        alpha=0.25,
        label="Pointwise 95% CI",
    )
    ax.axvspan(pd.to_datetime("2007-12-01"), pd.to_datetime("2009-06-30"), color="gray", alpha=0.15)
    ax.set_title("(A) Volatility Sensitivity around 2008 Lehman Collapse (N=100)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Coefficient (per $1\sigma$ change)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelrotation=30)

    # Panel B: Momentum during 2020 COVID Market Decline (row 1)
    dates_cov = pd.to_datetime(cov["dates"])[1:]
    b_mom_cov = cov["beta"][1, :]
    se_mom_cov = cov["beta_se_robust"][1, :]
    ax = axes[0, 1]
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.plot(dates_cov, b_mom_cov, color="#1f77b4", linewidth=2.0, label=r"$\beta_{\mathrm{mom}}[t]$ (COVID Shock)")
    ax.fill_between(
        dates_cov,
        b_mom_cov - 1.96 * se_mom_cov,
        b_mom_cov + 1.96 * se_mom_cov,
        color="#1f77b4",
        alpha=0.25,
        label="Pointwise 95% CI",
    )
    ax.axvspan(pd.to_datetime("2020-02-01"), pd.to_datetime("2020-04-30"), color="gray", alpha=0.15)
    ax.set_title("(B) Momentum Sensitivity during 2020 COVID-19 Shock (N=100)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Coefficient (per $1\sigma$ change)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelrotation=30)

    # Panel C: P/E Ratio during 2008 Credit Disruption (row 3)
    b_pe_cri = cri["beta"][3, :]
    se_pe_cri = cri["beta_se_robust"][3, :]
    ax = axes[1, 0]
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.plot(dates_cri, b_pe_cri, color="#2ca02c", linewidth=2.0, label=r"$\beta_{\mathrm{pe}}[t]$ (2008 Crisis)")
    ax.fill_between(
        dates_cri,
        b_pe_cri - 1.96 * se_pe_cri,
        b_pe_cri + 1.96 * se_pe_cri,
        color="#2ca02c",
        alpha=0.25,
        label="Pointwise 95% CI",
    )
    ax.axvspan(pd.to_datetime("2007-12-01"), pd.to_datetime("2009-06-30"), color="gray", alpha=0.15)
    ax.set_title("(C) P/E Ratio Valuation Sensitivity during 2008 Crisis (N=100)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(r"Coefficient (per $1\sigma$ change)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelrotation=30)

    # Panel D: Turnover during 2020 COVID Liquidity Shock (row 4)
    b_turn_cov = cov["beta"][4, :]
    se_turn_cov = cov["beta_se_robust"][4, :]
    ax = axes[1, 1]
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.plot(dates_cov, b_turn_cov, color="#ff7f0e", linewidth=2.0, label=r"$\beta_{\mathrm{turn}}[t]$ (COVID Shock)")
    ax.fill_between(
        dates_cov,
        b_turn_cov - 1.96 * se_turn_cov,
        b_turn_cov + 1.96 * se_turn_cov,
        color="#ff7f0e",
        alpha=0.25,
        label="Pointwise 95% CI",
    )
    ax.axvspan(pd.to_datetime("2020-02-01"), pd.to_datetime("2020-04-30"), color="gray", alpha=0.15)
    ax.set_title("(D) Turnover Liquidity Sensitivity during 2020 COVID Shock (N=100)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(r"Coefficient (per $1\sigma$ change)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelrotation=30)

    plt.tight_layout()
    case_path = repo_root / "regressor_case_studies_4panel.png"
    plt.savefig(case_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Updated ➜ {case_path.name}")


if __name__ == "__main__":
    main()
