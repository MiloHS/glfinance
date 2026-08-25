from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from glfinance.data import prepare_XPZ, fred, load_ff_csv
from glfinance.graph import solve_instance


def select_top_market_cap_tickers(
    crsp_path: str | Path,
    compustat_path: str | Path,
    top_n: int = 100,
    start_date: str = "2006-03-01",
    end_date: str = "2010-12-31",
) -> list[str]:
    """Identify top N tickers by average market cap (MthPrc * ShrOut) present in both CRSP and Compustat."""
    crsp = pd.read_csv(crsp_path)
    comp = pd.read_csv(compustat_path)

    comp_tickers = set(comp["tic"].dropna().unique())
    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"])

    mask = (crsp["MthCalDt"] >= start_date) & (crsp["MthCalDt"] <= end_date)
    crsp_filtered = crsp[mask].copy()

    crsp_filtered["MthPrc"] = crsp_filtered["MthPrc"].abs()
    crsp_filtered["MktCap"] = crsp_filtered["MthPrc"] * crsp_filtered["ShrOut"]

    valid_crsp = crsp_filtered[crsp_filtered["Ticker"].isin(comp_tickers)]

    comp["datadate"] = pd.to_datetime(comp["datadate"])
    dt_start = pd.to_datetime(start_date)
    comp_sub = comp[(comp["datadate"] >= (dt_start - pd.Timedelta(days=400))) & (comp["datadate"] <= end_date)].copy()
    invalid_eps_tickers = set()
    for tic, grp in comp_sub.groupby("tic"):
        eps_vals = grp["epsf12"].values
        if len(eps_vals) == 0 or pd.isna(eps_vals).any() or (eps_vals <= 0).any():
            invalid_eps_tickers.add(tic)

    valid_crsp = valid_crsp[~valid_crsp["Ticker"].isin(invalid_eps_tickers)]

    ranked_tickers = (
        valid_crsp.groupby("Ticker")["MktCap"].mean().sort_values(ascending=False).index.tolist()
    )

    # Bulk verify active price histories on yfinance
    import yfinance as yf

    bulk_start = (dt_start - pd.Timedelta(days=220)).strftime("%Y-%m-%d")
    candidates = ranked_tickers[: top_n + 150]
    yf_df = yf.download(
        candidates,
        start=bulk_start,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )["Adj Close"]

    top_tickers = []
    min_trading_days = int(len(pd.date_range(bulk_start, end_date, freq="B")) * 0.85)
    for col in yf_df.columns:
        series = yf_df[col].loc[bulk_start:end_date].dropna()
        if len(series) >= min_trading_days:
            top_tickers.append(col)
        if len(top_tickers) == top_n:
            break

    return top_tickers



def get_eps_from_compustat(
    ticker: str,
    day: pd.Timestamp | str,
    compustat_path: str | Path,
) -> float:
    """Retrieve EPS from Compustat without lookahead (only datadate <= target day)."""
    comp = pd.read_csv(compustat_path)
    dt = pd.to_datetime(day)
    comp["datadate"] = pd.to_datetime(comp["datadate"])

    sub = comp[(comp["tic"] == ticker) & (comp["datadate"] <= dt)].sort_values("datadate")
    eps = sub.iloc[-1]["epsf12"]
    return float(eps) if pd.notna(eps) and eps != 0 else np.nan


def build_crisis_dataset(
    crsp_path: str | Path = "crsp_crisis.csv",
    compustat_path: str | Path = "crisis_compustat.csv",
    top_n: int = 100,
    start_date: str = "2006-03-01",
    end_date: str = "2010-12-31",
    lookback_weeks: int = 20,
    regressor_names: list[str] | None = None,
    ff_base_path: str | Path = "Fama-French Factors",
    cache_dir: str | Path = "cache",
    experiment_name: str = "crisis",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Step 1: Extract tickers and build/load the raw (X, P, Z) dataset with diagnostic info."""
    repo_root = Path(__file__).parent.parent.parent.parent
    cache_path = repo_root / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)

    compustat_df = pd.read_csv(compustat_path)
    crsp_df = pd.read_csv(crsp_path)

    # Check if cached tickers file exists to bypass select_top_market_cap_tickers download
    tickers_cache_file = cache_path / f"top_{top_n}_tickers_{experiment_name}.json"
    if tickers_cache_file.exists():
        import json
        with open(tickers_cache_file, "r") as f:
            top_tickers = json.load(f)
        print(f"Loaded {len(top_tickers)} tickers from cache ➜ {tickers_cache_file.name}")
    else:
        print(f"Extracting top market cap tickers for '{experiment_name}' from CRSP & Compustat...")
        top_tickers = select_top_market_cap_tickers(
            crsp_path=crsp_path,
            compustat_path=compustat_path,
            top_n=top_n,
            start_date=start_date,
            end_date=end_date,
        )
        import json
        with open(tickers_cache_file, "w") as f:
            json.dump(top_tickers, f)

    if regressor_names is None:
        regressor_names = [
            "momentum",
            "volatility",
            "pe_ratio",
            "turnover",
            "interest_rate",
            "mkt-rf",
            "smb",
            "hml",
        ]

    rates_df = fred()

    base_path = Path(ff_base_path)
    if not base_path.exists():
        ff_map = {}
    else:
        ff_map = {
            "US": load_ff_csv(base_path / "F-F_Factors_US.csv"),
            "EU": load_ff_csv(base_path / "F-F_Factors_EU.csv"),
            "JAPAN": load_ff_csv(base_path / "F-F_Factors_Jap.csv"),
            "ASIAPAC_EXJAP": load_ff_csv(base_path / "F-F_Factors_AsiaPac_exJap.csv"),
            "NA": load_ff_csv(base_path / "F-F_Factors_NA.csv"),
        }

    X, P, Z = prepare_XPZ(
        tickers=top_tickers,
        start_date=start_date,
        end_date=end_date,
        lookback_weeks=lookback_weeks,
        regressor_names=regressor_names,
        rates_df=rates_df,
        ff_map=ff_map,
        compustat_df=compustat_df,
        crsp_df=crsp_df,
        cache_dir=cache_path,
    )

    print(f"Dataset shapes ➜ X: {X.shape}, P: {P.shape}, Z: {Z.shape}")
    return X, P, Z, top_tickers, regressor_names


def solve_crisis_model(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    top_tickers: list[str],
    k: int = 0,
    gammas: tuple[float, float, float] = (40.0, 1e-5, 1.5e-2),
    experiment_name: str = "crisis",
) -> dict[str, Any]:
    """Step 2: Solve the joint graph learning model given prepared X, P, Z tensors."""
    print(f"Running IPOPT optimization solver for '{experiment_name}'...")
    m, results, obj_val, L, c1, c2, beta, s_hat = solve_instance(
        X,
        P,
        Z,
        k=k,
        gammas=gammas,
        ipopt_opts={"tol": 1e-6, "max_iter": 500},
    )

    out_dict = {
        "tickers": top_tickers,
        "results": results,
        "objective": obj_val,
        "L": L,
        "c1": c1,
        "c2": c2,
        "beta": beta,
        "s_hat": s_hat,
        "X": X,
        "P": P,
        "Z": Z,
    }

    # Automatically save numerical results dictionary to cache
    import gzip, pickle
    repo_root = Path(__file__).parent.parent.parent.parent
    cache_path = repo_root / "cache" / f"{experiment_name}_results_top{len(top_tickers)}.pkl.gz"
    with gzip.open(cache_path, "wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved complete experiment results dictionary ➜ {cache_path.name}")

    out_dict["model"] = m
    return out_dict


def run_crisis_experiment(
    crsp_path: str | Path = "crsp_crisis.csv",
    compustat_path: str | Path = "crisis_compustat.csv",
    top_n: int = 100,
    start_date: str = "2006-03-01",
    end_date: str = "2010-12-31",
    lookback_weeks: int = 20,
    regressor_names: list[str] | None = None,
    ff_base_path: str | Path = "Fama-French Factors",
    cache_dir: str | Path = "cache",
    k: int = 0,
    gammas: tuple[float, float, float] = (40.0, 1e-5, 1.5e-2),
    experiment_name: str = "crisis",
) -> dict[str, Any]:
    """Execute complete workflow: Step 1 (Build dataset) + Step 2 (Solve model)."""
    X, P, Z, top_tickers, reg_names = build_crisis_dataset(
        crsp_path=crsp_path,
        compustat_path=compustat_path,
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
        lookback_weeks=lookback_weeks,
        regressor_names=regressor_names,
        ff_base_path=ff_base_path,
        cache_dir=cache_dir,
        experiment_name=experiment_name,
    )
    return solve_crisis_model(X, P, Z, top_tickers, k=k, gammas=gammas, experiment_name=experiment_name)
