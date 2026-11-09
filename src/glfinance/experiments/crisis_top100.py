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
    price_df: pd.DataFrame | None = None,
) -> list[str]:
    """Identify top N tickers by pre-crisis baseline market cap (MthPrc * ShrOut)

    present in both CRSP and Compustat, without conditioning on post-crisis
    market capitalization, EPS survival rules, or post-crisis survival.
    """
    crsp = pd.read_csv(crsp_path)
    comp = pd.read_csv(compustat_path)

    comp_tickers = set(comp["tic"].dropna().unique())
    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"])
    dt_start = pd.to_datetime(start_date)

    # 1. Pre-crisis baseline ranking: evaluate market cap strictly prior to or at start_date
    pre_mask = crsp["MthCalDt"] <= dt_start
    if not pre_mask.any():
        pre_mask = crsp["MthCalDt"] == crsp["MthCalDt"].min()

    crsp_pre = crsp[pre_mask].copy()
    crsp_pre["MthPrc"] = crsp_pre["MthPrc"].abs()
    crsp_pre["MktCap"] = crsp_pre["MthPrc"] * crsp_pre["ShrOut"]

    valid_crsp = crsp_pre[crsp_pre["Ticker"].isin(comp_tickers)]

    # 2. Ranked strictly by pre-crisis market cap; NO post-crisis conditioning and NO EPS filter
    ranked_tickers = (
        valid_crsp.groupby("Ticker")["MktCap"].mean().sort_values(ascending=False).index.tolist()
    )

    candidates = ranked_tickers[: top_n + 150]
    bulk_start = (dt_start - pd.Timedelta(days=220)).strftime("%Y-%m-%d")
    min_trading_days = int(len(pd.date_range(bulk_start, end_date, freq="B")) * 0.85)

    top_tickers: list[str] = []

    # 3. Fix candidate iteration: iterate strictly in descending market-cap rank order
    if price_df is not None:
        for tic in candidates:
            if tic in price_df.columns:
                series = price_df[tic].loc[bulk_start:end_date].dropna()
                if len(series) >= min_trading_days:
                    top_tickers.append(tic)
            if len(top_tickers) == top_n:
                break
    else:
        try:
            import yfinance as yf
        except Exception:
            yf = None

        if yf is not None:
            try:
                yf_df = yf.download(
                    candidates,
                    start=bulk_start,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                )["Adj Close"]
                for tic in candidates:
                    if tic in yf_df.columns:
                        series = yf_df[tic].loc[bulk_start:end_date].dropna()
                        if len(series) >= min_trading_days:
                            top_tickers.append(tic)
                    if len(top_tickers) == top_n:
                        break
            except Exception as e:
                print(f"Warning: yfinance verification failed ({e}), using top candidate ranking directly.")
                top_tickers = candidates[:top_n]
        else:
            top_tickers = candidates[:top_n]

    # Ensure top_n tickers are returned
    if len(top_tickers) < top_n:
        for tic in candidates:
            if tic not in top_tickers:
                top_tickers.append(tic)
            if len(top_tickers) == top_n:
                break

    return top_tickers



def get_eps_from_compustat(
    ticker: str,
    day: pd.Timestamp | str,
    compustat_path: str | Path,
    lag_days: int = 90,
) -> float:
    """Retrieve EPS from Compustat imposing a 3-month (90-day) reporting lag

    to ensure accounting disclosures were publicly filed and eliminate lookahead bias.
    """
    comp = pd.read_csv(compustat_path)
    dt = pd.to_datetime(day) - pd.Timedelta(days=lag_days)
    comp["datadate"] = pd.to_datetime(comp["datadate"])

    sub = comp[(comp["tic"] == ticker) & (comp["datadate"] <= dt)].sort_values("datadate")
    if sub.empty:
        return np.nan
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
    price_df: pd.DataFrame | None = None,
    volume_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Step 1: Extract tickers and build/load the raw (X, P, Z) dataset with diagnostic info."""
    repo_root = Path(__file__).parent.parent.parent.parent
    cache_path = repo_root / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)

    compustat_df = pd.read_csv(compustat_path)
    crsp_df = pd.read_csv(crsp_path)

    # Check if cached tickers file exists to bypass select_top_market_cap_tickers download
    tickers_cache_file = cache_path / f"top_{top_n}_tickers_{experiment_name}_v2.json"
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
            price_df=price_df,
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

    X, P, Z, aligned_tickers, dates = prepare_XPZ(
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
        price_df=price_df,
        volume_df=volume_df,
        return_tickers=True,
        return_dates=True,
    )

    print(f"Dataset shapes ➜ X: {X.shape}, P: {P.shape}, Z: {Z.shape}, Dates: {len(dates)}")
    return X, P, Z, aligned_tickers, regressor_names, dates


def solve_crisis_model(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    top_tickers: list[str],
    dates: list[pd.Timestamp] | None = None,
    k: int = 1,
    gammas: tuple[float, float, float] = (40.0, 1.5e-5, 1.5e-2),
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
        "dates": [pd.Timestamp(d) for d in dates] if dates is not None else [],
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
    k: int = 1,
    gammas: tuple[float, float, float] = (40.0, 1.5e-5, 1.5e-2),
    experiment_name: str = "crisis",
) -> dict[str, Any]:
    """Execute complete workflow: Step 1 (Build dataset) + Step 2 (Solve model)."""
    X, P, Z, top_tickers, reg_names, dates = build_crisis_dataset(
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
    return solve_crisis_model(X, P, Z, top_tickers, dates=dates, k=k, gammas=gammas, experiment_name=experiment_name)

