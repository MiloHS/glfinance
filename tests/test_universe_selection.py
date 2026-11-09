from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from glfinance.experiments.crisis_top100 import select_top_market_cap_tickers


def test_universe_selection_rank_ordering_and_pre_crisis_baseline() -> None:
    """Verify that select_top_market_cap_tickers:

    1. Does NOT sort candidates alphabetically (Comment 1.2 part 1).
    2. Ranks purely on pre-crisis market cap without post-crisis lookahead (Comment 1.2 part 2).
    3. Does NOT exclude firms with negative EPS during the crisis (Comment 1.2 part 3).
    """
    repo_root = Path(__file__).parent.parent
    crsp_path = repo_root / "crsp_crisis.csv"
    comp_path = repo_root / "crisis_compustat.csv"

    assert crsp_path.exists(), "crsp_crisis.csv must exist for this test"
    assert comp_path.exists(), "crisis_compustat.csv must exist for this test"

    top_20 = select_top_market_cap_tickers(
        crsp_path=crsp_path,
        compustat_path=comp_path,
        top_n=20,
        start_date="2006-03-01",
        end_date="2010-12-31",
    )

    # 1. Verify not alphabetical (previously ended at MCO or was alphabetical)
    # The first element must be the largest market cap (XOM), NOT AAPL!
    assert top_20[0] == "XOM", f"Expected XOM to be largest pre-crisis firm, got {top_20[0]}"
    assert top_20[1] == "GE", f"Expected GE to be second pre-crisis firm, got {top_20[1]}"

    # Verify presence of major institutions that previously were missing or filtered:
    # MSFT, PG, WMT, C (Citigroup had negative EPS), BAC (Bank of America had negative EPS)
    for expected_tic in ["MSFT", "PG", "WMT", "C", "BAC", "JPM"]:
        assert expected_tic in top_20, f"Expected {expected_tic} in top 20 pre-crisis universe"

    # Verify that the list is NOT monotonically increasing in alphabetical order
    is_alphabetical = (top_20 == sorted(top_20))
    assert not is_alphabetical, "Top tickers should be sorted by market cap rank, not alphabetically!"


def test_eps_filtering_removed() -> None:
    """Explicitly verify that a firm with negative trailing EPS during the crisis is retained."""
    repo_root = Path(__file__).parent.parent
    comp = pd.read_csv(repo_root / "crisis_compustat.csv")
    # Citigroup ('C') suffered massive quarterly losses in 2008: epsf12 < 0
    c_eps = comp[comp["tic"] == "C"]["epsf12"].dropna()
    assert (c_eps < 0).any(), "Citigroup should have negative EPS in Compustat data during 2008"

    top_10 = select_top_market_cap_tickers(
        crsp_path=repo_root / "crsp_crisis.csv",
        compustat_path=repo_root / "crisis_compustat.csv",
        top_n=10,
        start_date="2006-03-01",
        end_date="2010-12-31",
    )
    assert "C" in top_10, "Citigroup ('C') must NOT be filtered out by negative EPS"


if __name__ == "__main__":
    test_universe_selection_rank_ordering_and_pre_crisis_baseline()
    test_eps_filtering_removed()
    print("All universe selection tests passed successfully!")
