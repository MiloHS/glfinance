from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import yfinance as yf

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
