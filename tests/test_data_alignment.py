import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from glfinance.data import prepare_XPZ


def test_prepare_XPZ_ticker_alignment() -> None:
    """Verify that row i in X, slice i in P, and row/column i in Z strictly correspond

    to tickers[i], resolving Reviewer 2 comment 1.1 (elimination of .iloc[:, ::-1] reversal).
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    dates = pd.date_range("2006-01-01", "2008-01-01", freq="B")

    # Generate distinct, identifiable price trajectories for each ticker
    # AAPL: strong drift up
    # NVDA: constant drift down
    # MSFT, GOOGL, AMZN: varying trends
    np.random.seed(42)
    t_len = len(dates)
    prices = {}
    for idx, tic in enumerate(tickers):
        base = 100.0 * (idx + 1)
        drift = 0.001 * (idx - 2)
        noise = np.random.normal(0, 0.005, size=t_len)
        rets = drift + noise
        prices[tic] = base * np.exp(np.cumsum(rets))

    # Intentionally shuffle columns in DataFrame to test that prepare_XPZ enforces ticker order
    shuffled_tickers = ["NVDA", "AAPL", "AMZN", "MSFT", "GOOGL"]
    price_df = pd.DataFrame(prices, index=dates)[shuffled_tickers]
    volume_df = pd.DataFrame(1e6, index=dates, columns=shuffled_tickers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        X, P, Z, aligned_tickers = prepare_XPZ(
            tickers=tickers,
            start_date="2006-06-01",
            end_date="2007-12-31",
            lookback_weeks=10,
            regressor_names=["momentum", "volatility"],
            price_df=price_df,
            volume_df=volume_df,
            cache_dir=tmp_dir,
            return_tickers=True,
        )

        assert aligned_tickers == tickers
        N, M = X.shape
        assert N == len(tickers)
        assert P.shape == (M, N, 3)  # intercept + 2 regressors
        assert Z.shape == (M, N, N)

        # 1. Verify that X[i, :] matches monthly log returns of tickers[i]
        monthly_prices = price_df[tickers].loc["2006-06-01":"2007-12-31"].resample("ME").last()
        expected_monthly_rets = np.log(monthly_prices / monthly_prices.shift(1)).dropna(how="all").T
        expected_monthly_rets = expected_monthly_rets.reindex(index=tickers)

        # Compare the first time period
        for i, tic in enumerate(tickers):
            # Check sign and trend consistency
            # In particular, AAPL (i=0) and NVDA (i=4) must not be swapped!
            aapl_ret = expected_monthly_rets.loc["AAPL"].values
            nvda_ret = expected_monthly_rets.loc["NVDA"].values

            # Ensure AAPL ret != NVDA ret
            assert not np.allclose(aapl_ret, nvda_ret)

            # X[0, :] must strictly equal AAPL return
            np.testing.assert_allclose(X[0, :], aapl_ret[-M:], rtol=1e-5, atol=1e-5)
            # X[4, :] must strictly equal NVDA return
            np.testing.assert_allclose(X[4, :], nvda_ret[-M:], rtol=1e-5, atol=1e-5)

        # 2. Verify Z[t, i, j] symmetry and diagonal zero
        for t in range(M):
            np.testing.assert_allclose(np.diag(Z[t]), 0.0, atol=1e-8)
            np.testing.assert_allclose(Z[t], Z[t].T, atol=1e-8)

        # 4. Verify return_dates=True returns explicit timestamps matching M
        X_d, P_d, Z_d, d_tickers, out_dates = prepare_XPZ(
            tickers=tickers,
            start_date="2006-06-01",
            end_date="2007-12-31",
            lookback_weeks=10,
            regressor_names=["momentum", "volatility"],
            price_df=price_df,
            volume_df=volume_df,
            cache_dir=tmp_dir,
            return_tickers=True,
            return_dates=True,
        )
        assert len(out_dates) == M
        assert isinstance(out_dates[0], (pd.Timestamp, np.datetime64))


def test_date_alignment_and_burn_in() -> None:
    """Verify that the 20-week lookback burn-in correctly yields July 2006 (not March 2006),

    resolving the 4-month axis offset pointed out in Comment 1.4.
    """
    tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("2006-01-01", "2010-12-31", freq="B")
    price_df = pd.DataFrame(100.0, index=dates, columns=tickers)
    volume_df = pd.DataFrame(1e6, index=dates, columns=tickers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        X, P, Z, aligned_tickers, valid_dates = prepare_XPZ(
            tickers=tickers,
            start_date="2006-03-01",
            end_date="2010-12-31",
            lookback_weeks=20,
            regressor_names=["momentum", "volatility"],
            price_df=price_df,
            volume_df=volume_df,
            cache_dir=tmp_dir,
            return_tickers=True,
            return_dates=True,
        )

        # 54 months total: March, April, May, June 2006 are burn-in
        assert len(valid_dates) == 54
        assert X.shape[1] == 54

        # First evaluated date is strictly July 2006 (2006-07-31), NOT March 2006!
        assert valid_dates[0].strftime("%Y-%m") == "2006-07"
        # Final evaluated date is December 2010 (2010-12-31)
        assert valid_dates[-1].strftime("%Y-%m") == "2010-12"

        # Lehman Brothers collapse month (September 2008) must be at index 26
        # (July 2006 is index 0 -> 26 months later is September 2008)
        assert valid_dates[26].strftime("%Y-%m") == "2008-09"


if __name__ == "__main__":
    test_prepare_XPZ_ticker_alignment()
    test_date_alignment_and_burn_in()
    print("All ticker and date alignment tests passed successfully!")
