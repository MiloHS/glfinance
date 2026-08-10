# GLFinance

This repository contains the maintained, reproducible portfolio code for
*Graph Learning for Financial Time Series: A Regressor-Based Approach*.

The public repository is intentionally code-focused. It excludes downloaded
market data, generated results, exploratory robustness runs, legacy notebook
exports, manuscript working files, and reviewer correspondence.

## Repository contents

- `src/glfinance/`: maintained graph learning, covariance, portfolio, backtest,
  data, and reporting code;
- `configs/portfolio_walk_forward.yaml`: finalized experimental specification;
- `tests/`: deterministic unit and walk-forward tests; and
- `notebooks/portfolio_walk_forward.ipynb`: a clean Colab entry point that calls
  the maintained package rather than duplicating the implementation.

## Portfolio experiment

The revised experiment:

- learns one graph per week using only information available at that date;
- uses the preceding graph only through a temporal smoothness penalty;
- excludes regressors from portfolio graph learning;
- tunes the covariance window, retained graph modes, and diagonal shrinkage on
  the preceding 52 validation weeks;
- locks the selected hyperparameters for the next 13 unseen weeks; and
- compares Graph-LW and Graph-Sample against full Ledoit-Wolf, sample
  covariance, and equal weight.

### Local setup

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

## Output files

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

## Repository hygiene

Downloaded data, generated results, local environments, secrets, legacy
notebooks, exploratory experiments, and manuscript working files are excluded
by `.gitignore`. Before committing, run:

```bash
pytest
```

The test suite runs automatically in GitHub Actions. Before pushing, also
review `git status --short` and `git diff --cached --name-only` to confirm that
no local data, results, credentials, or manuscript working files are staged.
