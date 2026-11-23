from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import ssl
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

from glfinance import __version__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_downloaded_prices(
    downloaded: pd.DataFrame, tickers: list[str], field: str
) -> pd.DataFrame:
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        level_zero = downloaded.columns.get_level_values(0)
        level_one = downloaded.columns.get_level_values(1)
        if field in level_zero:
            prices = downloaded[field]
        elif field in level_one:
            prices = downloaded.xs(field, axis=1, level=1)
        else:
            raise KeyError(f"Price field {field!r} was not returned by Yahoo Finance.")
    elif len(tickers) == 1 and field in downloaded:
        prices = downloaded[[field]].rename(columns={field: tickers[0]})
    else:
        raise KeyError(f"Could not locate {field!r} in downloaded data.")

    missing = sorted(set(tickers).difference(prices.columns))
    if missing:
        raise RuntimeError(f"Missing downloaded price columns: {missing}")
    return prices.loc[:, tickers].sort_index()


def download_weekly_log_returns(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    field: str = "Adj Close",
    weekly_rule: str = "W-FRI",
) -> pd.DataFrame:
    """Download prices and return chronological Friday-to-Friday log returns."""
    symbols = list(tickers)
    downloaded = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    prices = _normalize_downloaded_prices(downloaded, symbols, field)
    weekly = prices.resample(weekly_rule).last().ffill(limit=1)
    returns = np.log(weekly / weekly.shift(1)).dropna(how="any")
    if returns.empty:
        raise RuntimeError("No complete weekly return observations were produced.")
    returns.index.name = "date"
    return returns


def download_weekly_risk_free(
    series: str, start_date: str, end_date: str, weekly_rule: str = "W-FRI"
) -> pd.Series:
    """Download a FRED annual-percent series and convert it to weekly returns."""
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&cosd={start_date}&coed={end_date}"
    )
    frame = pd.read_csv(url, na_values=".")
    if frame.shape[1] < 2:
        raise RuntimeError("FRED returned an unexpected CSV schema.")
    date_column = "DATE" if "DATE" in frame.columns else frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    if series not in frame:
        raise RuntimeError(f"FRED response did not contain {series}.")
    annual_percent = frame.set_index(date_column)[series].astype(float)
    weekly_annual = annual_percent.resample(weekly_rule).last().ffill()
    weekly_return = (1.0 + weekly_annual / 100.0) ** (1.0 / 52.0) - 1.0
    weekly_return.name = "risk_free"
    weekly_return.index.name = "date"
    return weekly_return.dropna()


def load_or_download_inputs(config: dict[str, Any], refresh: bool = False) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load cached processed inputs, or download and cache them."""
    data_config = config["data"]
    experiment = config["experiment"]
    cache_dir = Path(data_config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    returns_path = cache_dir / "weekly_log_returns.csv"
    risk_free_path = cache_dir / "weekly_risk_free.csv"
    manifest_path = cache_dir / "data_manifest.json"

    symbols = [item["symbol"] for item in data_config["tickers"]]
    if refresh or not returns_path.exists():
        returns = download_weekly_log_returns(
            symbols,
            experiment["start_date"],
            experiment["end_date"],
            field=data_config["price_field"],
            weekly_rule=data_config["weekly_rule"],
        )
        returns.to_csv(returns_path)
    else:
        returns = pd.read_csv(returns_path, index_col="date", parse_dates=True)
        missing = sorted(set(symbols).difference(returns.columns))
        if missing:
            raise RuntimeError(
                f"Cached returns do not contain configured symbols: {missing}. "
                "Run with --refresh-data."
            )
        returns = returns.loc[:, symbols]

    if refresh or not risk_free_path.exists():
        risk_free = download_weekly_risk_free(
            data_config["risk_free_series"],
            experiment["start_date"],
            experiment["end_date"],
            weekly_rule=data_config["weekly_rule"],
        )
        risk_free.to_csv(risk_free_path)
    else:
        risk_free_frame = pd.read_csv(
            risk_free_path, index_col="date", parse_dates=True
        )
        risk_free = risk_free_frame.iloc[:, 0].rename("risk_free")

    common = returns.index.intersection(risk_free.index)
    returns = returns.loc[common].sort_index()
    risk_free = risk_free.loc[common].sort_index()
    if returns.isna().any().any() or risk_free.isna().any():
        raise RuntimeError("Processed inputs contain missing values.")

    manifest = {
        "price_source": data_config["price_source"],
        "price_field": data_config["price_field"],
        "risk_free_source": data_config["risk_free_source"],
        "risk_free_series": data_config["risk_free_series"],
        "start_date": experiment["start_date"],
        "end_date_exclusive": experiment["end_date"],
        "first_week": str(returns.index.min().date()),
        "last_week": str(returns.index.max().date()),
        "observations": int(len(returns)),
        "tickers": symbols,
        "ticker_groups": data_config["tickers"],
        "files": {
            returns_path.name: _sha256(returns_path),
            risk_free_path.name: _sha256(risk_free_path),
        },
        "versions": {
            "glfinance": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "yfinance": yf.__version__,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return returns, risk_free, manifest


def synthetic_inputs(
    tickers: Iterable[str], observations: int = 420, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Create deterministic correlated weekly returns for offline smoke tests."""
    symbols = list(tickers)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-06", periods=observations, freq="W-FRI")
    market = rng.normal(0.001, 0.018, size=(observations, 1))
    sector = rng.normal(0.0, 0.009, size=(observations, 6))
    noise = rng.normal(0.0, 0.012, size=(observations, len(symbols)))
    returns_array = np.empty_like(noise)
    for column in range(len(symbols)):
        returns_array[:, column] = (
            0.65 * market[:, 0] + 0.45 * sector[:, column % 6] + noise[:, column]
        )
    returns = pd.DataFrame(returns_array, index=dates, columns=symbols)
    returns.index.name = "date"
    risk_free = pd.Series(0.00035, index=dates, name="risk_free")
    risk_free.index.name = "date"
    manifest = {
        "price_source": "synthetic",
        "risk_free_source": "synthetic",
        "observations": observations,
        "tickers": symbols,
        "seed": seed,
    }
    return returns, risk_free, manifest


def fred(api_key: str | None = None, start_date: str = "1990-01-01", end_date: str = "2025-07-23") -> pd.DataFrame:
    """Fetch monthly interest rate series across countries via FRED API."""
    import os
    import datetime
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED API key is required. Pass api_key directly or set the FRED_API_KEY environment variable."
        )

    client = Fred(api_key=key)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    series = {
        "US": "IRSTCI01USM156N",
        "EU": "IRSTCI01EZM156N",
        "JAPAN": "IRSTCI01JPM156N",
        "China": "IRSTCI01CNM156N",
        "UK": "IRSTCI01GBM156N",
        "Korea": "IRSTCI01KRM156N",
        "Australia": "IRSTCI01AUM156N",
        "Brazil": "IRSTCI01BRM156N",
    }

    df = pd.concat(
        {econ: client.get_series(sid, start, end) for econ, sid in series.items()},
        axis=1,
    )

    df_monthly = df.resample("ME").last()
    df_monthly.index.name = "Date"
    df_monthly.columns.name = "Economy"
    return df_monthly



_ff_region_map = {
    "US": "US",
    "EU": "EU",
    "JAPAN": "JAPAN",
    "China": "AsiaPac_exJap",
    "UK": "EU",
    "Korea": "AsiaPac_exJap",
    "Australia": "AsiaPac_exJap",
    "Brazil": "NA",
}

_ticker_meta_df: pd.DataFrame | None = None


def _load_ticker_meta(meta_path: str | Path | None = None) -> pd.DataFrame:
    global _ticker_meta_df
    if _ticker_meta_df is None:
        if meta_path is None:
            meta_path = Path("/content/drive/MyDrive/FF_Factors/ticker_meta.csv")
        if Path(meta_path).exists():
            _ticker_meta_df = pd.read_csv(meta_path, index_col=0)
        else:
            _ticker_meta_df = pd.DataFrame()
    return _ticker_meta_df


def get_econ(ticker: str) -> str:
    meta = _load_ticker_meta()
    if not meta.empty and ticker in meta.index and "Economy" in meta.columns:
        return str(meta.loc[ticker, "Economy"])
    return "US"


def _to_float(val: Any) -> float:
    try:
        f = float(val)
        return f if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def get_price_near(date: pd.Timestamp | str, ticker: str) -> float:
    dt = pd.to_datetime(date)
    start = (dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            return np.nan
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            data = data["Adj Close"].dropna()
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0].dropna()
        sub = data[:dt]
        if len(sub) == 0:
            return np.nan
        val = sub.iloc[-1]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)
    except Exception:
        return np.nan




def get_momentum(ticker: str, day: pd.Timestamp | str) -> float:
    dt = pd.to_datetime(day)
    past = dt - pd.Timedelta(days=180)
    p0 = get_price_near(past, ticker)
    p1 = get_price_near(dt, ticker)
    if np.isnan(p0) or np.isnan(p1) or p0 == 0:
        return np.nan
    return (p1 - p0) / p0


def get_volatility(ticker: str, day: pd.Timestamp | str, window: int = 21) -> float:
    dt = pd.to_datetime(day)
    start = (dt - pd.tseries.offsets.BDay(window + 21)).strftime("%Y-%m-%d")
    end = (dt + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            return np.nan
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            data = data["Adj Close"].dropna()
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0].dropna()
        sub = data[:dt]
        returns = sub.pct_change().dropna()
        if len(returns) < window:
            return np.nan
        val = returns.rolling(window).std().iloc[-1]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)
    except Exception:
        return np.nan



_ticker_info_cache: dict[str, dict] = {}


def _get_ticker_info(ticker: str) -> dict:
    if ticker not in _ticker_info_cache:
        try:
            _ticker_info_cache[ticker] = yf.Ticker(ticker).info
        except Exception:
            _ticker_info_cache[ticker] = {}
    return _ticker_info_cache[ticker]


def get_pe_ratio(ticker: str, day: pd.Timestamp | str) -> float:
    try:
        price = get_price_near(day, ticker)
        if np.isnan(price):
            return np.nan
        info = _get_ticker_info(ticker)
        eps = info.get("trailingEps", None)
        if eps is None or eps == 0:
            return np.nan
        return float(price / eps)
    except Exception:
        return np.nan


def get_turnover(ticker: str, day: pd.Timestamp | str, window: int = 10) -> float:
    dt = pd.Timestamp(day).normalize()
    start = (dt - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    end = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, timeout=5)
        if data.empty:
            return np.nan
        if isinstance(data, pd.DataFrame) and "Volume" in data.columns:
            vol_data = data["Volume"].replace(0, np.nan).dropna()
        else:
            vol_data = data.iloc[:, 0].replace(0, np.nan).dropna()
        sub = vol_data[:dt]
        if sub.empty:
            return np.nan
        volume = sub.iloc[-1]
        if isinstance(volume, pd.Series):
            volume = volume.iloc[0]

        info = _get_ticker_info(ticker)
        shares = info.get("sharesOutstanding", None)
        if shares is None or shares == 0 or np.isnan(volume):
            return np.nan

        return float(volume / shares)
    except Exception:
        return np.nan




def get_interest_rate(ticker: str, day: pd.Timestamp | str, df_monthly_rates: pd.DataFrame) -> float:
    econ = get_econ(ticker)
    if econ not in df_monthly_rates.columns:
        return np.nan

    dt = pd.Timestamp(day).tz_localize(None) + pd.tseries.offsets.MonthEnd(0)
    if dt not in df_monthly_rates.index:
        pos = df_monthly_rates.index.get_indexer([dt], method="ffill")[0]
        if pos == -1:
            return np.nan
        dt = df_monthly_rates.index[pos]
    return float(df_monthly_rates.at[dt, econ])


def load_ff_csv(path: str | Path) -> pd.DataFrame:
    try:
        with open(path, "r") as f:
            header_row = 0
            for i, line in enumerate(f):
                if line.strip().startswith(","):
                    header_row = i
                    break

        df = pd.read_csv(path, skiprows=header_row, header=0, index_col=0)
        cols = ["Mkt-RF", "SMB", "HML"]
        df = df.loc[:, df.columns.intersection(cols)].copy()

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        raw = df.index.astype(str).str.strip()
        dates = pd.to_datetime(raw, format="%Y%m", errors="coerce")
        mask = ~dates.isna()

        df2 = df.loc[mask, cols].copy()
        df2.index = dates[mask] + pd.tseries.offsets.MonthEnd(0)
        return df2
    except FileNotFoundError:
        print(f"File not found: {path}")
        return pd.DataFrame(index=pd.to_datetime([]))


def _get_ff_factor(ticker: str, day: pd.Timestamp | str, factor: str, ff_map: dict[str, pd.DataFrame]) -> float:
    country = get_econ(ticker)
    region = _ff_region_map.get(country)
    if region is None or region not in ff_map:
        return np.nan

    df = ff_map[region]
    dt = pd.Timestamp(day).tz_localize(None) + pd.tseries.offsets.MonthEnd(0)
    if dt not in df.index:
        pos = df.index.get_indexer([dt], method="ffill")[0]
        if pos == -1:
            return np.nan
        dt = df.index[pos]
    return float(df.at[dt, factor])


def get_regressors(
    day: pd.Timestamp | str,
    tickers: Iterable[str],
    regressor_names: Iterable[str],
    *,
    rates_df: pd.DataFrame | None = None,
    ff_map: dict[str, pd.DataFrame] | None = None,
) -> np.ndarray:
    ticker_list = list(tickers)
    reg_list = list(regressor_names)
    N, J = len(ticker_list), 1 + len(reg_list)
    X = np.ones((N, J))

    for i, tic in enumerate(ticker_list):
        for j, name in enumerate(reg_list, start=1):
            if name == "interest_rate":
                if rates_df is None:
                    raise ValueError("rates_df must be provided for interest_rate")
                reg_value = get_interest_rate(tic, day, rates_df)
            elif name in {"mkt-rf", "smb", "hml"}:
                if ff_map is None:
                    raise ValueError("ff_map must be provided for FF factors")
                fac = {"mkt-rf": "Mkt-RF", "smb": "SMB", "hml": "HML"}[name]
                reg_value = _get_ff_factor(tic, day, fac, ff_map)
            else:
                reg_fn = {
                    "momentum": get_momentum,
                    "volatility": get_volatility,
                    "pe_ratio": get_pe_ratio,
                    "turnover": get_turnover,
                }[name]
                reg_value = reg_fn(tic, day)
            X[i, j] = _to_float(reg_value)

    return X


def prepare_XPZ(
    tickers: list[str],
    start_date: str,
    end_date: str,
    lookback_weeks: int,
    regressor_names: list[str],
    rates_df: pd.DataFrame | None = None,
    ff_map: dict[str, pd.DataFrame] | None = None,
    compustat_df: pd.DataFrame | str | Path | None = None,
    crsp_df: pd.DataFrame | str | Path | None = None,
    cache_dir: str | Path = "cache",
    price_df: pd.DataFrame | None = None,
    volume_df: pd.DataFrame | None = None,
    return_tickers: bool = False,
    return_dates: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, ...]:
    import gzip, hashlib, json, pickle

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tickers = list(dict.fromkeys(tickers))

    key = {
        "version": "v4_dated_and_lagged",
        "tickers": sorted(tickers),
        "start": start_date,
        "end": end_date,
        "look": lookback_weeks,
        "regs": sorted(regressor_names),
    }
    h = hashlib.md5(json.dumps(key).encode()).hexdigest()[:16]
    cache_file = cache_path / f"XPZ_{h}.pkl.gz"

    if cache_file.exists():
        with gzip.open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        print("Loaded X,P,Z from cache ➜", cache_file.name)
        if len(cached_data) == 5:
            X_c, P_c, Z_c, valid_t_c, valid_d_c = cached_data
            if return_tickers and return_dates:
                return X_c, P_c, Z_c, valid_t_c, valid_d_c
            if return_tickers:
                return X_c, P_c, Z_c, valid_t_c
            if return_dates:
                return X_c, P_c, Z_c, valid_d_c
            return X_c, P_c, Z_c
        elif len(cached_data) == 4:
            X_c, P_c, Z_c, valid_t_c = cached_data
            if return_tickers and return_dates:
                return X_c, P_c, Z_c, valid_t_c, []
            if return_tickers:
                return X_c, P_c, Z_c, valid_t_c
            if return_dates:
                return X_c, P_c, Z_c, []
            return X_c, P_c, Z_c
        else:
            X_c, P_c, Z_c = cached_data
            if return_tickers and return_dates:
                return X_c, P_c, Z_c, tickers, []
            if return_tickers:
                return X_c, P_c, Z_c, tickers
            if return_dates:
                return X_c, P_c, Z_c, []
            return X_c, P_c, Z_c

    dt_start = pd.to_datetime(start_date)
    dt_end = pd.to_datetime(end_date)
    bulk_start = (dt_start - pd.Timedelta(days=220)).strftime("%Y-%m-%d")

    if price_df is not None:
        adj_close = price_df.copy()
        if volume_df is not None:
            volume = volume_df.copy()
        else:
            volume = pd.DataFrame(1.0, index=adj_close.index, columns=adj_close.columns)
    else:
        if yf is None:
            raise ImportError("yfinance is required when price_df is not provided.")
        print(f"Bulk downloading daily prices & volume for {len(tickers)} tickers from {bulk_start} to {end_date}...")
        bulk_data = yf.download(tickers, start=bulk_start, end=end_date, auto_adjust=False, progress=False)

        if isinstance(bulk_data.columns, pd.MultiIndex):
            adj_close = bulk_data["Adj Close"]
            volume = bulk_data["Volume"]
        else:
            adj_close = bulk_data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
            volume = bulk_data[["Volume"]].rename(columns={"Volume": tickers[0]})

    # Ensure columns strictly match tickers order (filtering to available tickers)
    available_tickers = [t for t in tickers if t in adj_close.columns]
    if not available_tickers:
        raise ValueError("None of the requested tickers were found in price data.")
    tickers = available_tickers
    adj_close = adj_close[tickers]
    if volume is not None and not volume.empty:
        volume = volume[[t for t in tickers if t in volume.columns]]

    # Vectorized bulk regressors
    # 1. Momentum: 6-month returns (~126 trading days)
    past_6m = adj_close.shift(126)
    bulk_momentum = (adj_close - past_6m) / past_6m

    # 2. Volatility: 21-day rolling std of daily log returns
    daily_rets = np.log(adj_close / adj_close.shift(1))
    bulk_volatility = daily_rets.rolling(21).std()

    # Pre-process Compustat EPS and CRSP/Compustat Shares
    if isinstance(compustat_df, (str, Path)):
        compustat_df = pd.read_csv(compustat_df)
    if isinstance(crsp_df, (str, Path)):
        crsp_df = pd.read_csv(crsp_df)

    if compustat_df is not None:
        compustat_df["datadate"] = pd.to_datetime(compustat_df["datadate"])

    if crsp_df is not None:
        crsp_df["MthCalDt"] = pd.to_datetime(crsp_df["MthCalDt"])

    weekly_prices = adj_close.loc[start_date:end_date].resample("W-FRI").last()
    weekly_rets = np.log(weekly_prices / weekly_prices.shift(1)).dropna(how="all").T
    weekly_rets = weekly_rets.reindex(index=tickers)

    monthly_prices = adj_close.loc[start_date:end_date].resample("ME").last()
    monthly_log_rets = np.log(monthly_prices / monthly_prices.shift(1)).dropna(how="all").T
    monthly_log_rets = monthly_log_rets.reindex(index=tickers)

    periods = weekly_rets.columns.to_period("M")
    dist_mats, fridays = [], []

    for period, group in weekly_rets.T.groupby(periods):
        if len(group) == 0:
            continue
        end_d = group.index[-1]
        fridays.append(end_d)
        pos = weekly_rets.columns.get_loc(end_d)
        if pos >= lookback_weeks - 1:
            window = weekly_rets.iloc[:, pos - lookback_weeks + 1 : pos + 1].to_numpy()
            corr = np.corrcoef(window)
            dist = 1.0 / (np.abs(corr) + 0.01) - 1.0 / 1.01
            np.fill_diagonal(dist, 0.0)
            dist_mats.append(dist)
        else:
            dist_mats.append(None)

    friday_to_monthend = {f: (f + pd.tseries.offsets.MonthEnd(0)).normalize() for f in fridays}

    X_cols, P_list, Z_list, dates_list = [], [], [], []
    for dist, friday in zip(dist_mats, fridays):
        if dist is None:
            continue
        me_date = friday_to_monthend[friday]
        if me_date not in monthly_log_rets.columns:
            continue

        X_cols.append(monthly_log_rets.loc[tickers, me_date].to_numpy(dtype=float))
        dates_list.append(me_date)

        N = len(tickers)
        J = 1 + len(regressor_names)
        P_mat = np.ones((N, J))

        # Lagging to month t-1: evaluate characteristics strictly using information available at end of month t-1
        prev_me_date = (me_date.replace(day=1) - pd.Timedelta(days=1)) + pd.tseries.offsets.MonthEnd(0)
        prev_month_start = prev_me_date.replace(day=1)

        sub_prices_prev = adj_close[:prev_me_date]
        if not sub_prices_prev.empty:
            prev_dt = sub_prices_prev.index[-1]
            mom_row = bulk_momentum.loc[prev_dt] if prev_dt in bulk_momentum.index else None
            vol_row = bulk_volatility.loc[prev_dt] if prev_dt in bulk_volatility.index else None
            prc_row = adj_close.loc[prev_dt]
            # Average daily volume across all trading days in month t-1
            month_vol_slice = volume.loc[(volume.index >= prev_month_start) & (volume.index <= prev_me_date)]
            if not month_vol_slice.empty:
                avg_daily_vol_row = month_vol_slice.mean(axis=0)
            else:
                avg_daily_vol_row = volume.loc[prev_dt] if (volume is not None and prev_dt in volume.index) else None
        else:
            mom_row = vol_row = prc_row = avg_daily_vol_row = None

        # 3-month (90-day) reporting lag cutoff for Compustat accounting data to eliminate lookahead bias
        compustat_cutoff = prev_me_date - pd.Timedelta(days=90)

        for i, tic in enumerate(tickers):
            for j, name in enumerate(regressor_names, start=1):
                if name == "interest_rate":
                    val = get_interest_rate(tic, prev_me_date, rates_df) if rates_df is not None else np.nan
                elif name in {"mkt-rf", "smb", "hml"}:
                    fac = {"mkt-rf": "Mkt-RF", "smb": "SMB", "hml": "HML"}[name]
                    val = _get_ff_factor(tic, prev_me_date, fac, ff_map) if ff_map is not None else np.nan
                elif name == "momentum":
                    # 126-trading-day return ending at the end of month t-1
                    val = _to_float(mom_row[tic]) if mom_row is not None and tic in mom_row else np.nan
                elif name == "volatility":
                    # 21-day rolling std of daily log returns from month t-1
                    val = _to_float(vol_row[tic]) if vol_row is not None and tic in vol_row else np.nan
                elif name == "pe_ratio":
                    # Price at end of month t-1 divided by EPS reported with >= 3-month lag
                    price_val = _to_float(prc_row[tic]) if prc_row is not None and tic in prc_row else np.nan
                    eps_val = np.nan
                    if compustat_df is not None:
                        sub_c = compustat_df[(compustat_df["tic"] == tic) & (compustat_df["datadate"] <= compustat_cutoff)].sort_values("datadate")
                        if not sub_c.empty:
                            eps_val = _to_float(sub_c.iloc[-1]["epsf12"])
                    val = (price_val / eps_val) if (price_val and eps_val and eps_val != 0) else np.nan
                elif name == "turnover":
                    # Average daily turnover from month t-1 (average daily volume / shares outstanding)
                    v_val = _to_float(avg_daily_vol_row[tic]) if avg_daily_vol_row is not None and tic in avg_daily_vol_row else np.nan
                    shares_val = np.nan
                    # CRSP ShrOut as of month t-1
                    if crsp_df is not None:
                        sub_crsp = crsp_df[(crsp_df["Ticker"] == tic) & (crsp_df["MthCalDt"] <= prev_me_date)].sort_values("MthCalDt")
                        if not sub_crsp.empty:
                            shares_val = _to_float(sub_crsp.iloc[-1]["ShrOut"])
                    # Fallback to Compustat cshoq (with 3-month reporting lag)
                    if (np.isnan(shares_val) or shares_val == 0) and compustat_df is not None:
                        sub_c = compustat_df[(compustat_df["tic"] == tic) & (compustat_df["datadate"] <= compustat_cutoff)].sort_values("datadate")
                        if not sub_c.empty:
                            shares_val = _to_float(sub_c.iloc[-1]["cshoq"]) * 1e6
                    # Fallback for synthetic/offline test data when external databases are not provided
                    if (np.isnan(shares_val) or shares_val == 0) and crsp_df is None and compustat_df is None:
                        shares_val = 1e6
                    val = (v_val / shares_val) if (v_val and shares_val and shares_val != 0) else np.nan
                else:
                    val = np.nan

                P_mat[i, j] = _to_float(val)

        P_list.append(P_mat)
        Z_list.append(dist)


    X_raw = np.column_stack(X_cols)
    P_raw = np.stack(P_list, axis=0)
    Z_raw = np.stack(Z_list, axis=0)

    # Filter out any ticker that produces NaNs in X or P
    nan_in_X = np.isnan(X_raw).any(axis=1) # shape (N,)
    nan_in_P = np.isnan(P_raw).any(axis=(0, 2)) # shape (N,)
    valid_ticker_mask = ~(nan_in_X | nan_in_P)

    valid_count = int(valid_ticker_mask.sum())
    if valid_count < len(tickers):
        removed_tickers = [tickers[i] for i in range(len(tickers)) if not valid_ticker_mask[i]]
        print(f"Removing {len(removed_tickers)} tickers throwing NaNs: {removed_tickers}")

    valid_tickers = [tickers[i] for i in range(len(tickers)) if valid_ticker_mask[i]]

    X = X_raw[valid_ticker_mask, :]
    P = P_raw[:, valid_ticker_mask, :]
    Z = Z_raw[:, valid_ticker_mask, :][:, :, valid_ticker_mask]

    # Cross-sectional Z-score standardization of regressors P (excluding intercept at j=0)
    M, N_valid, J = P.shape
    for t in range(M):
        for j in range(1, J):
            vals = P[t, :, j]
            std_val = np.std(vals)
            if std_val > 1e-8:
                P[t, :, j] = (vals - np.mean(vals)) / std_val
            else:
                P[t, :, j] = 0.0

    print("Applied cross-sectional Z-score standardization to regressors P (excluding intercept).")

    with gzip.open(cache_file, "wb") as f:
        pickle.dump((X, P, Z, valid_tickers, dates_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Built X,P,Z and cached ➜", cache_file.name)

    if return_tickers and return_dates:
        return X, P, Z, valid_tickers, dates_list
    if return_tickers:
        return X, P, Z, valid_tickers
    if return_dates:
        return X, P, Z, dates_list
    return X, P, Z

