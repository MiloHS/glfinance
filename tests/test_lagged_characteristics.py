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

    repo_root = Path(__file__).parent.parent
    comp_path = repo_root / "crisis_compustat.csv"
    assert comp_path.exists()

    # For day = 2008-04-15:
    # A quarter ending 2008-03-31 has NOT had 90 days elapse (only 15 days), so its 10-Q is not yet filed.
    # The latest available report with a 90-day lag must be from 2007-12-31 or earlier.
    eps = get_eps_from_compustat("XOM", "2008-04-15", comp_path, lag_days=90)
    assert not np.isnan(eps)

    # Directly check Compustat date for XOM
    comp = pd.read_csv(comp_path)
    comp["datadate"] = pd.to_datetime(comp["datadate"])
    sub = comp[(comp["tic"] == "XOM") & (comp["datadate"] <= pd.to_datetime("2008-04-15") - pd.Timedelta(days=90))]
    expected_eps = sub.sort_values("datadate").iloc[-1]["epsf12"]
    assert np.isclose(eps, expected_eps), f"Expected {expected_eps}, got {eps}"


if __name__ == "__main__":
    test_characteristics_are_strictly_lagged_and_no_lookahead()
    test_compustat_eps_3month_reporting_lag()
    print("All lagged characteristics and reporting lag tests passed successfully!")
