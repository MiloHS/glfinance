import tempfile
from pathlib import Path
import numpy as np

from glfinance.experiments.crisis_sector import (
    DEFAULT_REGRESSORS,
    SECTOR_TICKERS,
    algebraic_connectivity,
    compute_scale_free_network_metrics,
    distance_matrix_unweighted_corr,
    frobenius_norm,
    plot_scale_free_comparison,
    plot_sector_metrics_grid,
    plot_spectral_radius_multi,
    slice_sector_from_full_dataset,
    solve_sector_cvxpy,
    spectral_radius,
    time_consistency,
)


def test_us_only_sector_tickers_definitions() -> None:
    assert "Finance" in SECTOR_TICKERS
    assert "Energy" in SECTOR_TICKERS
    assert "Consumer Staples" in SECTOR_TICKERS

    # Confirm US stocks retained
    assert SECTOR_TICKERS["Finance"] == ["C", "BAC", "JPM", "WFC"]
    assert SECTOR_TICKERS["Energy"] == ["XOM", "CVX", "COP"]
    assert SECTOR_TICKERS["Consumer Staples"] == ["PG", "PEP", "KO", "WMT"]

    # Confirm non-US stocks are strictly excluded
    non_us = ["MUFG", "BCS", "TTE", "SHEL", "BP", "UL", "BTI"]
    all_tickers = [t for sec in SECTOR_TICKERS.values() for t in sec]
    for foreign_tic in non_us:
        assert foreign_tic not in all_tickers


def test_four_company_specific_regressors() -> None:
    assert len(DEFAULT_REGRESSORS) == 4
    assert DEFAULT_REGRESSORS == ["momentum", "volatility", "pe_ratio", "turnover"]
    # Ensure no macro factors
    for macro in ["interest_rate", "mkt-rf", "smb", "hml"]:
        assert macro not in DEFAULT_REGRESSORS


def test_slice_sector_from_full_dataset() -> None:
    tickers_full = ["AAPL", "C", "MSFT", "BAC", "JPM", "WFC", "XOM"]
    X_full = np.random.randn(len(tickers_full), 10)
    P_full = np.random.randn(10, len(tickers_full), 5)
    Z_full = np.random.randn(10, len(tickers_full), len(tickers_full))

    sec_tickers = ["C", "BAC", "JPM", "WFC"]
    X_sec, P_sec, Z_sec, tics, _ = slice_sector_from_full_dataset(
        X_full, P_full, Z_full, tickers_full, sec_tickers
    )

    assert X_sec.shape == (4, 10)
    assert P_sec.shape == (10, 4, 5)
    assert Z_sec.shape == (10, 4, 4)
    assert tics == sec_tickers


def test_network_metrics_helpers() -> None:
    # 3-node cycle Laplacian for 2 time steps
    L0 = np.array([
        [2.0, -1.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -1.0, 2.0],
    ])
    L1 = 2.0 * L0
    L = np.stack([L0, L1], axis=0)

    sr = spectral_radius(L)
    assert len(sr) == 2
    np.testing.assert_allclose(sr[0], 3.0, atol=1e-5)
    np.testing.assert_allclose(sr[1], 6.0, atol=1e-5)

    ac = algebraic_connectivity(L)
    assert len(ac) == 2
    np.testing.assert_allclose(ac[0], 3.0, atol=1e-5)
    np.testing.assert_allclose(ac[1], 6.0, atol=1e-5)

    fn = frobenius_norm(L)
    assert len(fn) == 2
    assert fn[1] > fn[0]

    tc = time_consistency(L)
    assert len(tc) == 2
    assert np.isnan(tc[0])
    assert tc[1] > 0.0


def test_distance_matrix_unweighted_corr() -> None:
    signals = np.array([
        [0.05, 0.02, -0.01, 0.03],
        [-0.04, -0.01, 0.02, -0.02],
        [0.01, 0.03, 0.02, 0.01],
    ])
    Z = distance_matrix_unweighted_corr(signals)
    assert Z.shape == (3, 3)
    np.testing.assert_allclose(np.diag(Z), 0.0)
    np.testing.assert_allclose(Z, Z.T)
    assert np.all(Z >= 0.0)


def test_multi_sector_plotting() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        L0 = np.array([
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0],
            [-1.0, -1.0, 2.0],
        ])
        L_dummy = np.stack([L0 * (1 + 0.1 * t) for t in range(24)], axis=0)
        laps = {
            "Finance": L_dummy,
            "Energy": L_dummy * 0.8,
            "Consumer Staples": L_dummy * 0.5,
        }

        fig_path = tmp_path / "test_multi.png"
        plot_spectral_radius_multi(laps, save_path=fig_path, start_year=2006, max_t=24)
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0

        grid_path = tmp_path / "test_grid.png"
        plot_sector_metrics_grid(L_dummy, "Finance", save_path=grid_path, start_year=2006, max_t=24)
        assert grid_path.exists()
        assert grid_path.stat().st_size > 0


def test_solve_sector_cvxpy_synthetic() -> None:
    # Test cvxpy convex solver on small synthetic sector (N=3, M=4)
    np.random.seed(42)
    N, M, J = 3, 4, 5
    X = np.random.randn(N, M) * 0.05
    P = np.ones((M, N, J))
    P[:, :, 1:] = np.random.randn(M, N, J - 1) * 0.1
    Z = np.zeros((M, N, N))
    for t in range(M):
        A = np.random.rand(N, N)
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0.0)
        Z[t] = A

    L_opt, beta_opt, s_hat_opt, obj_val = solve_sector_cvxpy(
        X, P, Z, gammas=(10.0, 1e-4, 1e-2), solver="SCS"
    )
    assert L_opt.shape == (M, N, N)
    assert beta_opt.shape == (J, M)
    assert s_hat_opt.shape == (N, M)

    for t in range(M):
        np.testing.assert_allclose(L_opt[t], L_opt[t].T, atol=1e-4)
        np.testing.assert_allclose(L_opt[t] @ np.ones(N), np.zeros(N), atol=1e-4)
        assert np.all(np.diag(L_opt[t]) >= 0.0)


def test_scale_free_network_metrics_and_plot() -> None:
    # 4-node cycle Laplacian
    L0 = np.array([
        [2.0, -1.0, 0.0, -1.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [-1.0, 0.0, -1.0, 2.0],
    ])
    L = np.stack([L0, L0 * 1.5], axis=0)

    metrics = compute_scale_free_network_metrics(L)
    assert len(metrics["spectral_radius_per_node"]) == 2
    assert len(metrics["frobenius_per_node"]) == 2
    assert len(metrics["average_edge_weight"]) == 2
    assert np.isnan(metrics["relative_structural_change"][0])
    np.testing.assert_allclose(metrics["relative_structural_change"][1], 0.5, atol=1e-5)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        laps = {
            "Finance": L,
            "Energy": np.stack([np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])] * 2, axis=0),
        }
        fig_p = tmp_path / "scale_free_comp.png"
        plot_scale_free_comparison(laps, save_path=fig_p, max_t=2)
        assert fig_p.exists()
        assert fig_p.stat().st_size > 0
