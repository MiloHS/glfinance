from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from glfinance.experiments.crisis_top100 import select_top_market_cap_tickers


def _make_synthetic_universe_data(tmp_path: Path):
    crsp_csv = tmp_path / "crsp_synth.csv"
    comp_csv = tmp_path / "comp_synth.csv"

    # Tickers:
    # 1. Z_MEGA:
    #    - Pre-crisis (2006-01-31): MthPrc = 100, ShrOut = 10 -> MktCap = 1000 (largest)
    #    - Post-crisis (2009-01-31): MthPrc = 10, ShrOut = 10 -> MktCap = 100
    #    - Compustat: positive EPS = 5.0
    # 2. M_NEG_EPS:
    #    - Pre-crisis (2006-01-31): MthPrc = 50, ShrOut = 10 -> MktCap = 500 (second largest)
    #    - Post-crisis (2009-01-31): MthPrc = 20, ShrOut = 10 -> MktCap = 200
    #    - Compustat: NEGATIVE EPS = -10.0 (previously would have been discarded!)
    # 3. A_SMALL_FUTURE_BOOM:
    #    - Pre-crisis (2006-01-31): MthPrc = 10, ShrOut = 10 -> MktCap = 100 (smallest pre-crisis)
    #    - Post-crisis (2009-01-31): MthPrc = 500, ShrOut = 10 -> MktCap = 5000 (huge post-crisis)
    #    - Compustat: positive EPS = 20.0

    crsp_data = pd.DataFrame({
        "Ticker": ["Z_MEGA", "Z_MEGA", "M_NEG_EPS", "M_NEG_EPS", "A_SMALL_FUTURE_BOOM", "A_SMALL_FUTURE_BOOM"],
        "MthCalDt": ["2006-01-31", "2009-01-31", "2006-01-31", "2009-01-31", "2006-01-31", "2009-01-31"],
        "MthPrc": [100.0, 10.0, 50.0, 20.0, 10.0, 500.0],
        "ShrOut": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    })
    crsp_data.to_csv(crsp_csv, index=False)

    comp_data = pd.DataFrame({
        "tic": ["Z_MEGA", "M_NEG_EPS", "A_SMALL_FUTURE_BOOM"],
        "datadate": ["2006-01-31", "2006-01-31", "2006-01-31"],
        "epsf12": [5.0, -10.0, 20.0],
    })
    comp_data.to_csv(comp_csv, index=False)

    dates = pd.date_range("2005-01-01", "2010-12-31", freq="B")
    price_df = pd.DataFrame(
        100.0,
        index=dates,
        columns=["Z_MEGA", "M_NEG_EPS", "A_SMALL_FUTURE_BOOM"],
    )

    return crsp_csv, comp_csv, price_df


def test_universe_selection_rank_ordering_and_pre_crisis_baseline() -> None:
    """Verify that select_top_market_cap_tickers:

    1. Does NOT sort candidates alphabetically (Comment 1.2 part 1).
    2. Ranks purely on pre-crisis market cap without post-crisis lookahead (Comment 1.2 part 2).
    3. Does NOT exclude firms with negative EPS during the crisis (Comment 1.2 part 3).
    """
    # 1. Controlled synthetic test (runs in CI without external WRDS files)
    with tempfile.TemporaryDirectory() as tmp_dir:
        crsp_csv, comp_csv, price_df = _make_synthetic_universe_data(Path(tmp_dir))
        top_tickers = select_top_market_cap_tickers(
            crsp_path=crsp_csv,
            compustat_path=comp_csv,
            top_n=3,
            start_date="2006-03-01",
            end_date="2010-12-31",
            price_df=price_df,
        )

        # Pre-crisis ranking must place Z_MEGA first (1000 MktCap), M_NEG_EPS second (500 MktCap),
        # and A_SMALL_FUTURE_BOOM third (100 MktCap).
        # Alphabetical sorting would have picked A_SMALL_FUTURE_BOOM first.
        # Post-crisis conditioning would have picked A_SMALL_FUTURE_BOOM first (5000 MktCap).
        # Negative EPS filtering would have eliminated M_NEG_EPS.
        assert top_tickers == ["Z_MEGA", "M_NEG_EPS", "A_SMALL_FUTURE_BOOM"], (
            f"Expected ['Z_MEGA', 'M_NEG_EPS', 'A_SMALL_FUTURE_BOOM'], got {top_tickers}"
        )

    # 2. Local check on actual WRDS files if present
    repo_root = Path(__file__).parent.parent
    crsp_path = repo_root / "crsp_crisis.csv"
    comp_path = repo_root / "crisis_compustat.csv"

    if crsp_path.exists() and comp_path.exists():
        top_20 = select_top_market_cap_tickers(
            crsp_path=crsp_path,
            compustat_path=comp_path,
            top_n=20,
            start_date="2006-03-01",
            end_date="2010-12-31",
        )

        assert top_20[0] == "XOM", f"Expected XOM to be largest pre-crisis firm, got {top_20[0]}"
        assert top_20[1] == "GE", f"Expected GE to be second pre-crisis firm, got {top_20[1]}"

        for expected_tic in ["MSFT", "PG", "WMT", "C", "BAC", "JPM"]:
            assert expected_tic in top_20, f"Expected {expected_tic} in top 20 pre-crisis universe"

        is_alphabetical = (top_20 == sorted(top_20))
        assert not is_alphabetical, "Top tickers should be sorted by market cap rank, not alphabetically!"


def test_eps_filtering_removed() -> None:
    """Explicitly verify that a firm with negative trailing EPS during the crisis is retained."""
    # 1. Controlled synthetic test (runs in CI without external WRDS files)
    with tempfile.TemporaryDirectory() as tmp_dir:
        crsp_csv, comp_csv, price_df = _make_synthetic_universe_data(Path(tmp_dir))
        top_2 = select_top_market_cap_tickers(
            crsp_path=crsp_csv,
            compustat_path=comp_csv,
            top_n=2,
            start_date="2006-03-01",
            end_date="2010-12-31",
            price_df=price_df,
        )
        assert "M_NEG_EPS" in top_2, "Firm with negative EPS must be retained in top tickers"

    # 2. Local check on actual WRDS file if present
    repo_root = Path(__file__).parent.parent
    comp_path = repo_root / "crisis_compustat.csv"
    crsp_path = repo_root / "crsp_crisis.csv"
    if comp_path.exists() and crsp_path.exists():
        comp = pd.read_csv(comp_path)
        c_eps = comp[comp["tic"] == "C"]["epsf12"].dropna()
        assert (c_eps < 0).any(), "Citigroup should have negative EPS in Compustat data during 2008"

        top_10 = select_top_market_cap_tickers(
            crsp_path=crsp_path,
            compustat_path=comp_path,
            top_n=10,
            start_date="2006-03-01",
            end_date="2010-12-31",
        )
        assert "C" in top_10, "Citigroup ('C') must NOT be filtered out by negative EPS"


if __name__ == "__main__":
    test_universe_selection_rank_ordering_and_pre_crisis_baseline()
    test_eps_filtering_removed()
    print("All universe selection tests passed successfully!")
