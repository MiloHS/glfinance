from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from glfinance.data import prepare_XPZ, fred, load_ff_csv
from glfinance.graph import solve_instance


def run_regressor_experiment(
    tickers: list[str],
    start_date: str = "2006-01-01",
    end_date: str = "2012-06-30",
    lookback_weeks: int = 20,
    regressor_names: list[str] | None = None,
    ff_base_path: str = "/content/drive/MyDrive/FF_Factors",
    cache_dir: str = "/content/drive/MyDrive/XPZ_cache",
    k: int = 2,
    gammas: tuple[float, float, float] = (40.0, 1e-5, 1.5e-2),
):
    if regressor_names is None:
        # Focus strictly on firm-specific characteristics (Reviewer 2 comment on identification of common factors)
        regressor_names = [
            "momentum", "volatility", "pe_ratio", "turnover"
        ]

    rates_df = fred() if "interest_rate" in regressor_names else None

    base_path = Path(ff_base_path)
    if any(r in regressor_names for r in ["mkt-rf", "smb", "hml"]) and base_path.exists():
        ff_map = {
            "US": load_ff_csv(base_path / "F-F_Factors_US.csv"),
            "EU": load_ff_csv(base_path / "F-F_Factors_EU.csv"),
            "JAPAN": load_ff_csv(base_path / "F-F_Factors_Jap.csv"),
            "ASIAPAC_EXJAP": load_ff_csv(base_path / "F-F_Factors_AsiaPac_exJap.csv"),
            "NA": load_ff_csv(base_path / "F-F_Factors_NA.csv"),
        }
    else:
        ff_map = {}

    X, P, Z = prepare_XPZ(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lookback_weeks=lookback_weeks,
        regressor_names=regressor_names,
        rates_df=rates_df,
        ff_map=ff_map,
        cache_dir=cache_dir,
    )

    print(f"Dataset shapes ➜ X: {X.shape}, P: {P.shape}, Z: {Z.shape}")

    m, results, obj_val, L, c1, c2, beta, s_hat = solve_instance(
        X, P, Z,
        k=k,
        gammas=gammas,
        ipopt_opts={"tol": 1e-6, "max_iter": 500},
    )

    return {
        "model": m,
        "results": results,
        "objective": obj_val,
        "L": L,
        "c1": c1,
        "c2": c2,
        "beta": beta,
        "s_hat": s_hat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run joint graph learning experiment with regressors.")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of stock tickers")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2012-06-30")
    parser.add_argument("--lookback-weeks", type=int, default=20)
    args = parser.parse_args()

    run_regressor_experiment(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_weeks=args.lookback_weeks,
    )


if __name__ == "__main__":
    main()
