"""Automated zero-lookahead, split-integrity, and purged-CV verification tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import (
    PortfolioConfig,
    compute_performance_metrics,
    run_panel_topk_backtest,
    run_single_asset_backtest,
)
from src.cv import oos_retrain_blocks, purged_dev_folds
from src.data import NSE_IT_UNIVERSE, align_asof_series, load_market_bundle
from src.features import (
    BASE_TECH_FEATURES,
    GLOBAL_MACRO_FEATURES,
    MARKET_SECTOR_FEATURES,
    add_forward_targets,
    apply_trailing_zscore,
    build_stock_features,
)


def test_data_integrity_no_double_splits() -> None:
    stocks, context = load_market_bundle()
    infy = stocks["INFY.NS"]
    assert infy.index.min() <= pd.Timestamp("2007-09-18")
    assert infy.index.max() >= pd.Timestamp("2024-12-30")
    log_ret = np.log(infy["Close"] / infy["Close"].shift(1)).dropna()
    assert float(log_ret.abs().max()) < 0.30, "Unexpected jump in INFY.NS"
    # Verify bonus ex-date 2018-09-04 has no +100% fake jump
    assert abs(float(log_ret.loc["2018-09-04"])) < 0.08
    for sym in NSE_IT_UNIVERSE:
        assert len(stocks[sym]) >= 4200
    assert context["^NSEI"].index.min() <= pd.Timestamp("2007-09-18")


def test_strict_causal_feature_perturbation() -> None:
    """
    Corrupting all stock AND context data strictly AFTER cutoff date `T` must leave
    every engineered feature at `<= T` bit-for-bit identical.
    """
    stocks, context = load_market_bundle(tickers=("INFY.NS",))
    infy = stocks["INFY.NS"].copy()
    cutoff = pd.Timestamp("2020-06-15")

    feat_cols = BASE_TECH_FEATURES + MARKET_SECTOR_FEATURES + GLOBAL_MACRO_FEATURES

    for mode in ("close", "next_open"):
        f_orig = build_stock_features(infy, context, execution_mode=mode)
        f_orig = apply_trailing_zscore(f_orig, feat_cols, window=252, min_periods=126)

        infy_mod = infy.copy()
        infy_mod["Volume"] = infy_mod["Volume"].astype(float)
        infy_mod.loc[infy_mod.index > cutoff, ["Open", "High", "Low", "Close", "Volume"]] *= 9.5
        ctx_mod = {k: v.copy() for k, v in context.items()}
        for k in ctx_mod:
            ctx_mod[k].loc[ctx_mod[k].index > cutoff, ["Open", "High", "Low", "Close"]] *= 7.3

        f_mod = build_stock_features(infy_mod, ctx_mod, execution_mode=mode)
        f_mod = apply_trailing_zscore(f_mod, feat_cols, window=252, min_periods=126)

        a = f_orig.loc[:cutoff, feat_cols]
        b = f_mod.loc[:cutoff, feat_cols]
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=1e-12)


def test_close_mode_blocks_same_day_us_fx() -> None:
    """In `close` execution mode, mutating US/FX on date T must NOT alter features on date T."""
    stocks, context = load_market_bundle(tickers=("INFY.NS",))
    infy = stocks["INFY.NS"].copy()
    target_dt = pd.Timestamp("2021-07-15")

    f1 = build_stock_features(infy, context, execution_mode="close")
    ctx_mod = {k: v.copy() for k, v in context.items()}
    for k in ("INR=X", "^NDX", "INFY", "ACN", "CTSH"):
        if target_dt in ctx_mod[k].index:
            ctx_mod[k].loc[target_dt, "Close"] *= 3.0
    f2 = build_stock_features(infy, ctx_mod, execution_mode="close")

    pd.testing.assert_series_equal(
        f1.loc[target_dt, GLOBAL_MACRO_FEATURES],
        f2.loc[target_dt, GLOBAL_MACRO_FEATURES],
    )


def test_backtest_weight_causality() -> None:
    """Changing future prices/signals > T must not alter backtest weights <= T."""
    stocks, _ = load_market_bundle(tickers=("INFY.NS",))
    infy = stocks["INFY.NS"].loc["2018-01-01":"2021-12-31"].copy()
    rng = np.random.default_rng(42)
    sig = pd.Series(rng.normal(size=len(infy)), index=infy.index)
    cfg = PortfolioConfig(stop_loss_atr_mult=1.5, cooldown_bars=3)

    out1, _ = run_single_asset_backtest(infy, sig, cfg=cfg, execution_mode="next_open")

    cutoff = pd.Timestamp("2020-06-15")
    infy_mod = infy.copy()
    infy_mod.loc[infy_mod.index > cutoff, ["Open", "High", "Low", "Close"]] *= 5.0
    sig_mod = sig.copy()
    sig_mod.loc[sig_mod.index > cutoff] += 10.0

    out2, _ = run_single_asset_backtest(infy_mod, sig_mod, cfg=cfg, execution_mode="next_open")
    pd.testing.assert_series_equal(out1.loc[:cutoff, "weight"], out2.loc[:cutoff, "weight"])


def test_purged_cv_and_oos_blocks_have_zero_overlap() -> None:
    stocks, _ = load_market_bundle(tickers=("INFY.NS",))
    dates = stocks["INFY.NS"].index
    horizon = 10
    for tr_dts, va_dts, tr_cut, va_start in purged_dev_folds(
        dates, oos_start="2022-01-01", n_splits=6, val_bars=252, horizon=horizon, execution_mode="next_open"
    ):
        tr_cut_pos = dates.get_loc(tr_cut)
        va_start_pos = dates.get_loc(va_start)
        # Forward target of last train sample ends at tr_cut_pos + horizon + 1; must be strictly < va_start_pos
        assert tr_cut_pos + horizon + 1 < va_start_pos
        assert tr_dts.max() == tr_cut
        assert va_dts.min() == va_start
        assert va_dts.max() < pd.Timestamp("2022-01-01")

    for tr_cut, blk_start, blk_end in oos_retrain_blocks(
        dates, oos_start="2022-01-01", retrain_every_bars=63, horizon=horizon, execution_mode="next_open"
    ):
        assert dates.get_loc(tr_cut) + horizon + 1 < dates.get_loc(blk_start)
        assert blk_start <= blk_end


def test_forward_target_tail_is_nan_not_zero() -> None:
    stocks, context = load_market_bundle(tickers=("INFY.NS",))
    f = build_stock_features(stocks["INFY.NS"], context, execution_mode="next_open")
    f = add_forward_targets(f, horizon=10, execution_mode="next_open")
    assert f["target_dir"].iloc[-11:].isna().all()
    assert f["target_vol_adj"].iloc[-11:].isna().all()
    assert f["target_triple_barrier"].iloc[-11:].isna().all()


def test_fx_asof_alignment_causality() -> None:
    """Verify INR=X exact alignment: strict < t in close mode, <= t in next_open mode."""
    stocks, context = load_market_bundle(tickers=("INFY.NS",))
    infy = stocks["INFY.NS"]
    fx = context["INR=X"]["Close"]
    target_dt = pd.Timestamp("2021-07-15")
    s_close = align_asof_series(infy.index, fx, allow_same_date=False)
    s_open = align_asof_series(infy.index, fx, allow_same_date=True)
    # On target_dt, close mode must pull prior FX date (< target_dt), not target_dt's own FX bar
    assert s_close.loc[target_dt] == fx.loc[fx.index < target_dt].iloc[-1]
    assert s_open.loc[target_dt] == fx.loc[fx.index <= target_dt].iloc[-1]


def test_cnxit_regime_and_sleeve_backtest_causality() -> None:
    """Verify cnxit_dist_sma200, overlapping sleeves, and rank hysteresis have zero future dependence."""
    stocks, context = load_market_bundle()
    f_infy = build_stock_features(stocks["INFY.NS"], context, execution_mode="next_open")
    assert "cnxit_dist_sma200" in f_infy.columns

    cutoff = pd.Timestamp("2020-06-15")
    ctx_mod = {k: v.copy() for k, v in context.items()}
    ctx_mod["^CNXIT"].loc[ctx_mod["^CNXIT"].index > cutoff, "Close"] *= 4.0
    f_mod = build_stock_features(stocks["INFY.NS"], ctx_mod, execution_mode="next_open")
    pd.testing.assert_series_equal(
        f_infy.loc[:cutoff, "cnxit_dist_sma200"],
        f_mod.loc[:cutoff, "cnxit_dist_sma200"],
    )

    # Panel backtest causality under overlapping_sleeves and rank_hysteresis
    sub_idx = stocks["INFY.NS"].loc["2019-01-01":"2021-12-31"].index
    rng = np.random.default_rng(7)
    preds = pd.DataFrame(rng.normal(size=(len(sub_idx), len(NSE_IT_UNIVERSE))), index=sub_idx, columns=list(NSE_IT_UNIVERSE))
    preds_mod = preds.copy()
    preds_mod.loc[preds_mod.index > cutoff] += 50.0

    stocks_mod = {k: v.copy() for k, v in stocks.items()}
    for k in stocks_mod:
        stocks_mod[k].loc[stocks_mod[k].index > cutoff, ["Open", "High", "Low", "Close"]] *= 3.0

    cfg = PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.02, use_market_regime_filter=True)
    for style in ("overlapping_sleeves", "rank_hysteresis"):
        b1, _ = run_panel_topk_backtest(
            stocks, preds, cfg=cfg, execution_style=style, regime_series=f_infy["cnxit_dist_sma200"]
        )
        b2, _ = run_panel_topk_backtest(
            stocks_mod, preds_mod, cfg=cfg, execution_style=style, regime_series=f_mod["cnxit_dist_sma200"]
        )
        # Note: exec_ret on `cutoff` itself uses Open[cutoff+1..cutoff+2], so test `net_ret` up to cutoff - 2 bars
        safe_end = sub_idx[sub_idx.get_loc(cutoff) - 2]
        pd.testing.assert_series_equal(b1.loc[:safe_end, "net_ret"], b2.loc[:safe_end, "net_ret"])


def test_excess_sharpe_identity() -> None:
    """When weight=1.0 on every bar, cash_adj_cagr equals cagr exactly."""
    idx = pd.bdate_range("2022-01-03", periods=252)
    r = pd.Series(0.0004, index=idx)
    w = pd.Series(1.0, index=idx)
    m = compute_performance_metrics(r, r, w, rf_annual=0.06)
    assert abs(m["cash_adj_cagr"] - m["cagr"]) < 1e-10
    assert abs(m["excess_sharpe"] - m["bh_excess_sharpe"]) < 1e-10


if __name__ == "__main__":
    test_data_integrity_no_double_splits()
    test_strict_causal_feature_perturbation()
    test_close_mode_blocks_same_day_us_fx()
    test_backtest_weight_causality()
    test_purged_cv_and_oos_blocks_have_zero_overlap()
    test_forward_target_tail_is_nan_not_zero()
    test_fx_asof_alignment_causality()
    test_cnxit_regime_and_sleeve_backtest_causality()
    test_excess_sharpe_identity()
    print("ALL ZERO-LEAKAGE & INTEGRITY TESTS PASSED.")
