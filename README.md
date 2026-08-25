# GLFinance

This repository contains the maintained, reproducible portfolio and graph learning code for
*Graph Learning for Financial Time Series: A Regressor-Based Approach*.

The public repository is intentionally code-focused. It excludes downloaded
market data, binary cache files, generated figure images, legacy notebook
exports, and private credentials.

## Repository contents

- `src/glfinance/`: maintained graph learning, covariance, portfolio, backtest,
  data, and reporting code;
- `configs/portfolio_walk_forward.yaml`: finalized experimental specification;
- `tests/`: deterministic unit and walk-forward tests; and
- `notebooks/portfolio_walk_forward.ipynb`: a clean Colab entry point that calls
  the maintained package rather than duplicating the implementation.

---

## Portfolio Walk-Forward Experiment

The portfolio experiment:

- learns one graph per week using only information available at that date;
- uses the preceding graph only through a temporal smoothness penalty;
- excludes regressors from portfolio graph learning;
- tunes the covariance window, retained graph modes, and diagonal shrinkage on
  the preceding 52 validation weeks;
- locks the selected hyperparameters for the next 13 unseen weeks; and
- compares Graph-LW and Graph-Sample against full Ledoit-Wolf, sample
  covariance, and equal weight.

### Local Setup & Execution

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebook]'
pytest
glfinance-portfolio --config configs/portfolio_walk_forward.yaml
```

The first experiment run downloads adjusted-close data from Yahoo Finance and a
three-month Treasury-bill series from FRED. Processed inputs are cached under
`data/cache/`; generated results are written under `results/`. Both directories
are ignored by Git because downloaded market data can change or be subject to
redistribution restrictions. Each run writes a manifest with dates, tickers,
package versions, and file checksums.

Use `--data-only` to populate and inspect the cache without solving graphs, and
`--synthetic` for a small offline smoke test.

### Colab

Open `notebooks/portfolio_walk_forward.ipynb`, set `REPOSITORY_PATH` to the
checked-out repository, install the package, and run the configured experiment.
No API key or proprietary dataset is required. SCS is the documented default
solver.

### Output Files

Each run produces:

- `resolved_config.yaml`
- `data_manifest.json`
- `solver_status.csv`
- `selection_history.csv`
- `weekly_performance.csv`
- `weekly_weights.csv`
- `summary.csv`
- `block_summary.csv`
- `bootstrap.csv`
- publication-ready figures under `figures/`

Every reported manuscript value should be traceable to one of these files.

---

## Top 100 Crisis & COVID Experiments

This repository also includes the experiment pipeline for large-scale ($N=100$) market network graph learning and regressor evolution across two major historical macroeconomic crises:
1. **2008 Global Financial Crisis (March 2006 – December 2010, $M=54$ months)**
2. **2020 COVID-19 Liquidity Shock (May 2019 – December 2021, $M=32$ months)**

### Upfront Regressor $Z$-Score Standardization
To eliminate measurement scale discrepancies across raw financial regressors (e.g., P/E ratios in tens versus volatility in small decimals), input regressors in matrix $\mathbf{P}[t]$ are cross-sectionally $Z$-score standardized ($\mu_{t,j} = 0, \sigma_{t,j} = 1$) across all assets $i$ for each month $t$ inside `prepare_XPZ()` in `src/glfinance/data.py`. 

This guarantees that:
- Every learned coefficient $\beta_{j,t}$ directly measures the predicted return response to a **1-standard-deviation change** in factor $j$.
- Market-wide macro factors (Interest Rates, Fama-French Mkt-RF, SMB, HML) have zero cross-sectional variance across stocks, causing IPOPT to assign them **exactly zero ($\beta \equiv 0.0$)**, isolating true stock-level predictive signals (Volatility, Momentum, P/E Ratio, Turnover).

---

## WRDS Data Extraction & Requirements

Because institutional databases like **CRSP** and **Compustat** require paid academic/commercial licenses through Wharton Research Data Services (WRDS), raw data files are omitted from this public repository. 

To reproduce the Top 100 experiments from scratch, researchers should query WRDS as follows:

### 1. CRSP Monthly Stock File (`crsp_crisis.csv` / `crsp_covid.csv`)
- **WRDS Query**: `CRSP -> Monthly Stock File`
- **Required Columns**:
  - `Ticker`: Stock Ticker Symbol
  - `MthCalDt` / `date`: Monthly Calendar Date
  - `ShrOut`: Shares Outstanding (used for turnover calculations)
  - `ALTPRC` / `PRC`: Monthly Closing Price
- **Time Periods**:
  - *2008 Crisis*: March 2005 to December 2010 (includes trailing lookback)
  - *2020 COVID*: May 2018 to December 2021

### 2. Compustat Fundamentals Annual/Quarterly (`crisis_compustat.csv` / `covid_compustat.csv`)
- **WRDS Query**: `Compustat - Capital IQ -> Fundamentals Annual / Quarterly`
- **Required Columns**:
  - `tic`: Stock Ticker Symbol
  - `datadate`: Filing / Period End Date
  - `epsf12`: 12-Month Trailing Earnings Per Share (used for P/E ratios)
  - `cshoq`: Common Shares Outstanding Quarterly (fallback for turnover)

### 3. Kenneth R. French Data Library (`Fama-French Factors/`)
- Download the **Fama-French 3 Factors** (`F-F_Factors_US.csv`) directly from [Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

### 4. FRED Interest Rates
- Set your FRED API key in your terminal environment prior to running:
  ```bash
  export FRED_API_KEY="your_free_fred_api_key"
  ```

---

## Tracked Candidate Ticker JSON Files
The exact universe of top 100 market-cap companies evaluated in each experiment is tracked in JSON format under `cache/`:
- `cache/top_100_tickers_crisis_std.json`: Top 100 candidate tickers for the 2008 Financial Crisis.
- `cache/top_100_tickers_covid_std.json`: Top 100 candidate tickers for the 2020 COVID period.

---

## Top 100 Execution Commands

To execute the generic Top 100 experiment pipeline once WRDS files are placed in the project root:

```bash
# 2008 Financial Crisis Experiment
PYTHONPATH=src python3 -c "
from glfinance.experiments.crisis_top100 import run_crisis_experiment

run_crisis_experiment(
    crsp_path='crsp_crisis.csv',
    compustat_path='crisis_compustat.csv',
    top_n=100,
    start_date='2006-03-01',
    end_date='2010-12-31',
    lookback_weeks=20,
    experiment_name='crisis_std'
)
"

# 2020 COVID Crisis Experiment
PYTHONPATH=src python3 -c "
from glfinance.experiments.crisis_top100 import run_crisis_experiment

run_crisis_experiment(
    crsp_path='crsp_covid.csv',
    compustat_path='covid_compustat.csv',
    top_n=100,
    start_date='2019-01-01',
    end_date='2021-12-31',
    lookback_weeks=20,
    experiment_name='covid_std'
)
"
```

---

## Repository Hygiene

Downloaded market data (`*.csv`, `Fama-French Factors/`), binary cache files (`cache/*.pkl.gz`), generated figure images (`*.png`), and private credentials are excluded by `.gitignore`. Before committing, run:

```bash
git status --short
```


