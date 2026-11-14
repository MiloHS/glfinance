from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from glfinance.data import prepare_XPZ


def test_characteristics_are_strictly_lagged_and_no_lookahead() -> None:
    """Verify that regressors P[t] for month t returns do not depend on month t prices/volumes:

    - Momentum is evaluated as of month t-1 (does not contain month t return).
    - Volatility is evaluated over month t-1 (does not contain month t daily returns).
    - Turnover is the average daily volume of month t-1 divided by shares outstanding.
    - Compustat EPS enforces a 90-day (3-month) reporting lag.
    """
    tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("2006-01-01", "2007-06-30", freq="B")

    np.random.seed(42)
    p_aapl = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(dates))))
    p_msft = 50.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(dates))))

    price_df_base = pd.DataFrame({"AAPL": p_aapl, "MSFT": p_msft}, index=dates)
    volume_df_base = pd.DataFrame({"AAPL": 1e6, "MSFT": 2e6}, index=dates)

    with tempfile.TemporaryDirectory() as tmp_dir:
        X1, P1, Z1 = prepare_XPZ(
            tickers=tickers,
            start_date="2006-06-01",
            end_date="2007-05-31",
            lookback_weeks=10,
            regressor_names=["momentum", "volatility", "turnover"],
            price_df=price_df_base,
            volume_df=volume_df_base,
            cache_dir=Path(tmp_dir) / "c1",
        )

        # Now create an altered price_df where we inject an enormous crash and 100x volume
        # ONLY in the final month (May 2007)
        price_df_altered = price_df_base.copy()
        volume_df_altered = volume_df_base.copy()

        may_2007_mask = (price_df_altered.index >= "2007-05-01") & (price_df_altered.index <= "2007-05-31")
        # Halve prices in May 2007 and multiply volume by 100
        price_df_altered.loc[may_2007_mask, :] *= 0.5
        volume_df_altered.loc[may_2007_mask, :] *= 100.0

        X2, P2, Z2 = prepare_XPZ(
            tickers=tickers,
            start_date="2006-06-01",
            end_date="2007-05-31",
            lookback_weeks=10,
            regressor_names=["momentum", "volatility", "turnover"],
            price_df=price_df_altered,
            volume_df=volume_df_altered,
            cache_dir=Path(tmp_dir) / "c2",
        )

        # For the final month (May 2007), return X must be radically different because May prices crashed:
        assert not np.allclose(X1[:, -1], X2[:, -1]), "May 2007 return should reflect the price drop"

        # BUT the regressors P for May 2007 (time index -1) MUST BE IDENTICAL,
        # because May 2007 regressors are strictly lagged to April 2007 (month t-1)!
        np.testing.assert_allclose(
            P1[-1, :, :],
            P2[-1, :, :],
            err_msg="Regressors for month t must NOT be contaminated by month t prices or volume!"
        )


def test_compustat_eps_3month_reporting_lag() -> None:
    """Verify that earnings announcements respect the 3-month (90-day) filing lag."""
    from glfinance.experiments.crisis_top100 import get_eps_from_compustat

    # 1. Controlled synthetic test (runs in CI without external WRDS files)
    with tempfile.TemporaryDirectory() as tmp_dir:
        synth_path = Path(tmp_dir) / "synth_compustat.csv"
        # Two quarterly reports for TEST:
        # Quarter 1: 2007-12-31 with EPS = 1.50
        # Quarter 2: 2008-03-31 with EPS = 2.50
        synth_df = pd.DataFrame({
            "tic": ["TEST", "TEST"],
            "datadate": ["2007-12-31", "2008-03-31"],
            "epsf12": [1.50, 2.50],
        })
        synth_df.to_csv(synth_path, index=False)

        # Query on 2008-04-15:
        # 2008-03-31 was only 15 days ago (< 90 days), so 10-Q is not yet filed.
        # With lag_days=90, the latest available filing is 2007-12-31 (106 days ago >= 90).
        eps_lagged = get_eps_from_compustat("TEST", "2008-04-15", synth_path, lag_days=90)
        assert np.isclose(eps_lagged, 1.50), f"Expected 1.50 with 90-day lag, got {eps_lagged}"

        # If zero lag were used (lookahead bias), it would pick 2.50
        eps_no_lag = get_eps_from_compustat("TEST", "2008-04-15", synth_path, lag_days=0)
        assert np.isclose(eps_no_lag, 2.50), f"Expected 2.50 without lag, got {eps_no_lag}"

        # On 2008-07-01 (92 days after 2008-03-31), Quarter 2 is now available
        eps_available = get_eps_from_compustat("TEST", "2008-07-01", synth_path, lag_days=90)
        assert np.isclose(eps_available, 2.50), f"Expected 2.50 once 90 days elapse, got {eps_available}"

    # 2. Local check on actual WRDS file if present
    repo_root = Path(__file__).parent.parent
    comp_path = repo_root / "crisis_compustat.csv"
    if comp_path.exists():
        eps = get_eps_from_compustat("XOM", "2008-04-15", comp_path, lag_days=90)
        assert not np.isnan(eps)
        comp = pd.read_csv(comp_path)
        comp["datadate"] = pd.to_datetime(comp["datadate"])
        sub = comp[(comp["tic"] == "XOM") & (comp["datadate"] <= pd.to_datetime("2008-04-15") - pd.Timedelta(days=90))]
        expected_eps = sub.sort_values("datadate").iloc[-1]["epsf12"]
        assert np.isclose(eps, expected_eps), f"Expected {expected_eps}, got {eps}"


if __name__ == "__main__":
    test_characteristics_are_strictly_lagged_and_no_lookahead()
    test_compustat_eps_3month_reporting_lag()
    print("All lagged characteristics and reporting lag tests passed successfully!")
