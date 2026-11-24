# GLFinance

Code for *Graph Learning for Financial Time Series: A Regressor-Based Approach*.
The portfolio experiment uses public data; the crisis experiments also require licensed WRDS inputs.
Downloaded data, generated results, and credentials are not included.

## Setup

From the repository root (Python 3.10 or later):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebook]'
pytest
```

## Portfolio results

The walk-forward experiment selects portfolio parameters on the preceding 52 weeks
and evaluates them over the next 13-week block. Graph estimation uses asset log
returns; validation and evaluation use simple portfolio returns.

Run the baseline first. It downloads Yahoo Finance prices and FRED Treasury-bill
rates to `data/cache/`; no API key is required. Recompute any graphs from the
previous submission: those runs did not apply the graph-magnitude penalty.

```bash
glfinance-portfolio --config configs/portfolio_walk_forward.yaml --output-dir results/portfolio_revision/baseline
```

Then run the following in order. All four commands use [the revision configuration](configs/portfolio_revision.yaml) by default.

```bash
# Compare covariance estimators, correlation constructions, and graphs without the prediction term
python -m glfinance.experiments.portfolio_revision
# Compare components at fixed settings and measure retained return energy
python -m glfinance.experiments.portfolio_revision --components
# Compute bootstrap intervals, variance tests, and block-level figures
python -m glfinance.experiments.portfolio_revision --inference
# Check sensitivity to graph regularization and the dissimilarity offset
python -m glfinance.experiments.portfolio_revision --sensitivity
```

Outputs are in `results/portfolio_revision/`. Start with `baseline/real/summary.csv`
and `comparisons/comparison_summary.csv`; the other analyses write to
`components/`, `inference/`, and `sensitivity/`.

For an offline smoke test, run `glfinance-portfolio --synthetic`. For Colab, use
[the portfolio notebook](notebooks/portfolio_walk_forward.ipynb).

## Top 100 Crisis & COVID Experiments

This repository also includes the experiment pipeline for larger-scale ($N=100$) market network graph learning and regressor evolution across two major historical macroeconomic crises:
1. **2008 Global Financial Crisis (March 2006 – December 2010, $M=54$ months)**
2. **2020 COVID-19 Liquidity Shock (May 2019 – December 2021, $M=32$ months)**

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
Cached candidate lists are in `cache/`:
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
    k=1,
    gammas=(40.0, 1.5e-5, 1.5e-2),
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
    k=1,
    gammas=(40.0, 1.5e-5, 1.5e-2),
    experiment_name='covid_std'
)
"
```

