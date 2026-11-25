"""Stage 1: Strictly pre-2022 (< 2022-01-01) Purged Walk-Forward CV ablation runner for CNN-LSTM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import (
    PortfolioConfig,
    evaluate_signal_quality,
    run_panel_topk_backtest,
    run_single_asset_backtest,
)
from src.cv import build_sequences_panel, build_sequences_single, purged_dev_folds
from src.data import NSE_IT_UNIVERSE, load_market_bundle
from src.features import (
    BASE_TECH_FEATURES,
    CROSS_SECTIONAL_FEATURES,
    GLOBAL_MACRO_FEATURES,
    MARKET_SECTOR_FEATURES,
    add_cross_sectional_features,
    add_forward_targets,
    apply_trailing_zscore,
    build_stock_features,
)
from src.models import CNNLSTMConfig, fit_predict_cnn_lstm_ensemble


def _compute_cs_ic(pred_df: pd.DataFrame, fwd_df: pd.DataFrame) -> float:
    ics = []
    for dt in pred_df.index:
        rp = pred_df.loc[dt].to_numpy(dtype=float)
        rf = fwd_df.loc[dt].to_numpy(dtype=float)
        mask = np.isfinite(rp) & np.isfinite(rf)
        if mask.sum() >= 4 and np.std(rp[mask]) > 0 and np.std(rf[mask]) > 0:
            c, _ = spearmanr(rp[mask], rf[mask])
            if np.isfinite(c):
                ics.append(c)
    return float(np.mean(ics)) if ics else 0.0


def run_single_cv(
    exp_id: str,
    z_infy: pd.DataFrame,
    raw_infy: pd.DataFrame,
    regime_series: pd.Series,
    feature_cols: list[str],
    horizon: int,
    target_col: str,
    seq_len: int,
    mcfg: CNNLSTMConfig,
    execution_mode: str = "next_open",
) -> dict:
    df_t = add_forward_targets(z_infy, horizon=horizon, execution_mode=execution_mode)
    X_all, y_all, d_all = build_sequences_single(df_t, feature_cols, target_col, seq_len)
    folds = list(
        purged_dev_folds(
            raw_infy.index,
            oos_start="2022-01-01",
            n_splits=6,
            val_bars=252,
            horizon=horizon,
            execution_mode=execution_mode,
        )
    )

    preds, dates_va, fold_stats = [], [], []
    is_cls = mcfg.task == "classification"
    for k, (tr_dts, va_dts, _, _) in enumerate(folds):
        m_tr = np.isin(d_all, tr_dts)
        m_va = np.isin(d_all, va_dts)
        sw = None
        if is_cls:
            cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_all[m_tr].astype(int))
            sw = np.where(y_all[m_tr] > 0.5, cw[1], cw[0]).astype(np.float32)

        pv = fit_predict_cnn_lstm_ensemble(
            X_all[m_tr], y_all[m_tr], X_all[m_va], mcfg, base_seed=42 + k * 10, sample_weight=sw
        )
        preds.append(pv)
        dates_va.append(d_all[m_va])

        fwd_va = df_t.loc[d_all[m_va], "fwd_log_ret"]
        if is_cls:
            fold_stats.append(float(roc_auc_score(y_all[m_va], pv)))
        else:
            fold_stats.append(
                evaluate_signal_quality(pd.Series(pv, index=d_all[m_va]), fwd_va)["ic_spearman"]
            )

    p_ser = pd.Series(np.concatenate(preds), index=pd.DatetimeIndex(np.concatenate(dates_va))).sort_index()
    sq = evaluate_signal_quality(p_ser, df_t.loc[p_ser.index, "fwd_log_ret"])

    if target_col == "target_triple_barrier":
        bt_cfg = PortfolioConfig(
            cost_bps_per_side=20.0,
            target_annual_vol=0.14,
            max_leverage=1.0,
            min_hold_bars=1,
            entry_quantile=0.60,
            exit_quantile=0.40,
            stop_loss_atr_mult=1.5 * np.sqrt(horizon / 5.0),
            take_profit_atr_mult=2.0 * np.sqrt(horizon / 5.0),
        )
    else:
        bt_cfg = PortfolioConfig(
            cost_bps_per_side=20.0,
            target_annual_vol=0.14,
            max_leverage=1.0,
            min_hold_bars=min(5, horizon),
            entry_quantile=0.55,
            exit_quantile=0.35,
        )

    _, bt_raw = run_single_asset_backtest(raw_infy, p_ser, cfg=bt_cfg, execution_mode=execution_mode)
    bt_cfg_reg = PortfolioConfig(**{**bt_cfg.__dict__, "use_market_regime_filter": True, "regime_half_weight": 0.25})
    _, bt_reg = run_single_asset_backtest(
        raw_infy, p_ser, cfg=bt_cfg_reg, execution_mode=execution_mode, regime_series=regime_series
    )

    pos_folds = sum(s > (0.50 if is_cls else 0.0) for s in fold_stats)
    row = {
        "exp_id": exp_id,
        "scope": "single_infy",
        "horizon": horizon,
        "target": target_col,
        "seq_len": seq_len,
        "task": mcfg.task,
        "mean_fold_metric": float(np.mean(fold_stats)),
        "pos_folds": f"{pos_folds}/6",
        "pooled_ic": sq["ic_spearman"],
        "cs_ic": np.nan,
        "q5_q1_spread": sq["q_spread"],
        "infy_sharpe": bt_raw["sharpe"],
        "infy_cagr": bt_raw["cagr"],
        "infy_max_dd": bt_raw["max_drawdown"],
        "infy_turn": bt_raw["ann_turnover_nav"],
        "infy_exp": bt_raw["exposure_pct"],
        "infy_reg_sharpe": bt_reg["sharpe"],
        "top2_sharpe": np.nan,
        "top2_cagr": np.nan,
        "top2_max_dd": np.nan,
        "top3_sharpe": np.nan,
        "top3_reg_sharpe": np.nan,
        "bh_infy_sharpe": bt_raw["bh_sharpe"],
    }
    print(
        f"[{exp_id:24s}] fold_stat={row['mean_fold_metric']:+.4f} ({row['pos_folds']}) "
        f"IC={row['pooled_ic']:+.4f} Q5-Q1={row['q5_q1_spread']:+.4f} | "
        f"INFY Sh={row['infy_sharpe']:+.3f} (Reg={row['infy_reg_sharpe']:+.3f}, DD={row['infy_max_dd']:.3f}, Turn={row['infy_turn']:.1f})"
    )
    return row


def run_panel_cv(
    exp_id: str,
    z_panel: dict[str, pd.DataFrame],
    stocks: dict[str, pd.DataFrame],
    regime_series: pd.Series,
    feature_cols: list[str],
    horizon: int,
    target_col: str,
    seq_len: int,
    mcfg: CNNLSTMConfig,
    execution_mode: str = "next_open",
) -> dict:
    t_panel = {sym: add_forward_targets(df, horizon=horizon, execution_mode=execution_mode) for sym, df in z_panel.items()}
    X_all, y_all, d_all, s_all = build_sequences_panel(t_panel, feature_cols, target_col, seq_len)
    folds = list(
        purged_dev_folds(
            stocks["INFY.NS"].index,
            oos_start="2022-01-01",
            n_splits=6,
            val_bars=252,
            horizon=horizon,
            execution_mode=execution_mode,
        )
    )

    fold_ics, fold_infy_ics = [], []
    pred_grid = {sym: [] for sym in NSE_IT_UNIVERSE}
    fwd_grid = {sym: [] for sym in NSE_IT_UNIVERSE}

    for k, (tr_dts, va_dts, _, _) in enumerate(folds):
        m_tr = np.isin(d_all, tr_dts)
        m_va = np.isin(d_all, va_dts)
        pv = fit_predict_cnn_lstm_ensemble(X_all[m_tr], y_all[m_tr], X_all[m_va], mcfg, base_seed=42 + k * 10)

        va_dates_k = d_all[m_va]
        va_syms_k = s_all[m_va]
        va_df = pd.DataFrame({"date": va_dates_k, "sym": va_syms_k, "pred": pv})
        va_df["fwd"] = [
            float(t_panel[sym].loc[dt, "fwd_log_ret"]) for sym, dt in zip(va_syms_k, va_dates_k)
        ]

        fold_ics.append(evaluate_signal_quality(va_df["pred"], va_df["fwd"])["ic_spearman"])
        infy_sub = va_df[va_df["sym"] == "INFY.NS"]
        fold_infy_ics.append(evaluate_signal_quality(infy_sub["pred"], infy_sub["fwd"])["ic_spearman"])

        for sym in NSE_IT_UNIVERSE:
            ss = va_df[va_df["sym"] == sym]
            pred_grid[sym].append(pd.Series(ss["pred"].values, index=pd.DatetimeIndex(ss["date"])))
            fwd_grid[sym].append(pd.Series(ss["fwd"].values, index=pd.DatetimeIndex(ss["date"])))

    pred_df = pd.DataFrame({sym: pd.concat(lst) for sym, lst in pred_grid.items()}).sort_index()
    fwd_df = pd.DataFrame({sym: pd.concat(lst) for sym, lst in fwd_grid.items()}).sort_index()

    cs_ic = _compute_cs_ic(pred_df, fwd_df)
    sq_infy = evaluate_signal_quality(pred_df["INFY.NS"], fwd_df["INFY.NS"])
    sq_panel = evaluate_signal_quality(
        pd.Series(pred_df.to_numpy().ravel()), pd.Series(fwd_df.to_numpy().ravel())
    )

    bt_cfg = PortfolioConfig(
        cost_bps_per_side=20.0,
        target_annual_vol=0.14,
        max_leverage=1.0,
        min_hold_bars=min(5, horizon),
        entry_quantile=0.55,
        exit_quantile=0.35,
    )
    bt_cfg_reg = PortfolioConfig(**{**bt_cfg.__dict__, "use_market_regime_filter": True, "regime_half_weight": 0.25})

    _, bt_infy = run_single_asset_backtest(stocks["INFY.NS"], pred_df["INFY.NS"], cfg=bt_cfg, execution_mode=execution_mode)
    _, bt_infy_reg = run_single_asset_backtest(
        stocks["INFY.NS"], pred_df["INFY.NS"], cfg=bt_cfg_reg, execution_mode=execution_mode, regime_series=regime_series
    )
    rebal = min(5, horizon)
    _, bt_top2 = run_panel_topk_backtest(stocks, pred_df, cfg=bt_cfg, top_k=2, rebalance_every=rebal, execution_mode=execution_mode)
    _, bt_top3 = run_panel_topk_backtest(stocks, pred_df, cfg=bt_cfg, top_k=3, rebalance_every=rebal, execution_mode=execution_mode)
    _, bt_top3_reg = run_panel_topk_backtest(
        stocks, pred_df, cfg=bt_cfg_reg, top_k=3, rebalance_every=rebal, execution_mode=execution_mode, regime_series=regime_series
    )

    pos_folds = sum(x > 0 for x in fold_ics)
    row = {
        "exp_id": exp_id,
        "scope": "panel_7it",
        "horizon": horizon,
        "target": target_col,
        "seq_len": seq_len,
        "task": mcfg.task,
        "mean_fold_metric": float(np.mean(fold_ics)),
        "pos_folds": f"{pos_folds}/6",
        "pooled_ic": sq_panel["ic_spearman"],
        "cs_ic": cs_ic,
        "q5_q1_spread": sq_infy["q_spread"],
        "infy_sharpe": bt_infy["sharpe"],
        "infy_cagr": bt_infy["cagr"],
        "infy_max_dd": bt_infy["max_drawdown"],
        "infy_turn": bt_infy["ann_turnover_nav"],
        "infy_exp": bt_infy["exposure_pct"],
        "infy_reg_sharpe": bt_infy_reg["sharpe"],
        "top2_sharpe": bt_top2["sharpe"],
        "top2_cagr": bt_top2["cagr"],
        "top2_max_dd": bt_top2["max_drawdown"],
        "top3_sharpe": bt_top3["sharpe"],
        "top3_reg_sharpe": bt_top3_reg["sharpe"],
        "bh_infy_sharpe": bt_infy["bh_sharpe"],
    }
    print(
        f"[{exp_id:24s}] fold_IC={row['mean_fold_metric']:+.4f} ({row['pos_folds']}) "
        f"CS_IC={cs_ic:+.4f} INFY_IC={np.mean(fold_infy_ics):+.4f} | "
        f"INFY_Sh={bt_infy['sharpe']:+.3f} (Reg={bt_infy_reg['sharpe']:+.3f}) | "
        f"Top2_Sh={bt_top2['sharpe']:+.3f} (CAGR={bt_top2['cagr']:.3f}, DD={bt_top2['max_drawdown']:.3f}) | "
        f"Top3_Sh={bt_top3['sharpe']:+.3f} (Reg={bt_top3_reg['sharpe']:+.3f})"
    )
    return row


def main() -> None:
    stocks, context = load_market_bundle()
    feat_single = BASE_TECH_FEATURES + MARKET_SECTOR_FEATURES + GLOBAL_MACRO_FEATURES
    feat_panel = feat_single + CROSS_SECTIONAL_FEATURES

    raw_panel = {sym: build_stock_features(stocks[sym], context, execution_mode="next_open") for sym in NSE_IT_UNIVERSE}
    raw_panel = add_cross_sectional_features(raw_panel)
    z_panel = {sym: apply_trailing_zscore(df, feat_panel, window=252, min_periods=126) for sym, df in raw_panel.items()}
    regime_series = raw_panel["INFY.NS"]["nifty_dist_sma200"]

    rows = []

    single_cfg_reg = CNNLSTMConfig(
        conv_filters=16, kernel_size=3, lstm_units=16, dropout=0.35, rec_dropout=0.0,
        l2_reg=1e-3, lr=5e-4, batch_size=64, epochs=25, task="regression", norm_type="layer", n_seeds=3
    )
    single_cfg_cls = CNNLSTMConfig(
        conv_filters=16, kernel_size=3, lstm_units=16, dropout=0.35, rec_dropout=0.0,
        l2_reg=1e-3, lr=5e-4, batch_size=64, epochs=25, task="classification", norm_type="layer", n_seeds=3
    )

    print("=== Stage 1A: Single-Stock INFY CNN-LSTM (Pre-2022 Purged CV) ===")
    rows.append(run_single_cv("S1_H5_dir_cls", z_panel["INFY.NS"], stocks["INFY.NS"], regime_series, feat_single, 5, "target_dir", 20, single_cfg_cls))
    rows.append(run_single_cv("S2_H10_excess_reg", z_panel["INFY.NS"], stocks["INFY.NS"], regime_series, feat_single, 10, "target_excess_vol_adj", 20, single_cfg_reg))
    rows.append(run_single_cv("S3_H10_tb_cls", z_panel["INFY.NS"], stocks["INFY.NS"], regime_series, feat_single, 10, "target_triple_barrier", 20, single_cfg_cls))
    rows.append(run_single_cv("S4_H15_it_excess_reg", z_panel["INFY.NS"], stocks["INFY.NS"], regime_series, feat_single, 15, "target_it_excess_vol_adj", 20, single_cfg_reg))

    panel_cfg = CNNLSTMConfig(
        conv_filters=32, kernel_size=3, lstm_units=32, dropout=0.30, rec_dropout=0.0,
        l2_reg=5e-4, lr=5e-4, batch_size=128, epochs=25, task="regression", norm_type="layer", n_seeds=3
    )

    print("\n=== Stage 1B: 7-Stock NSE IT Panel CNN-LSTM (Pre-2022 Purged CV) ===")
    rows.append(run_panel_cv("P1_H10_vol_adj", z_panel, stocks, regime_series, feat_panel, 10, "target_vol_adj", 20, panel_cfg))
    rows.append(run_panel_cv("P2_H10_excess_nifty", z_panel, stocks, regime_series, feat_panel, 10, "target_excess_vol_adj", 20, panel_cfg))
    rows.append(run_panel_cv("P3_H10_excess_it", z_panel, stocks, regime_series, feat_panel, 10, "target_it_excess_vol_adj", 20, panel_cfg))
    rows.append(run_panel_cv("P4_H15_excess_it", z_panel, stocks, regime_series, feat_panel, 15, "target_it_excess_vol_adj", 20, panel_cfg))
    rows.append(run_panel_cv("P5_H15_excess_nifty", z_panel, stocks, regime_series, feat_panel, 15, "target_excess_vol_adj", 20, panel_cfg))
    rows.append(run_panel_cv("P6_H15_excess_it_s28", z_panel, stocks, regime_series, feat_panel, 15, "target_it_excess_vol_adj", 28, panel_cfg))

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_dir / "stage1_dev_cv_leaderboard.csv", index=False)
    print("\nSaved Stage 1 Purged CV leaderboard to results/stage1_dev_cv_leaderboard.csv")


if __name__ == "__main__":
    main()
