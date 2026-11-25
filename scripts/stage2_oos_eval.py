"""Stage 2: True Out-of-Sample (2022-2024) periodic-retrain evaluation & full diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import (
    PortfolioConfig,
    compute_execution_returns,
    compute_performance_metrics,
    evaluate_signal_quality,
    run_panel_topk_backtest,
    run_single_asset_backtest,
)
from src.cv import (
    build_inference_sequences,
    build_sequences_panel,
    build_sequences_single,
    oos_retrain_blocks,
    purged_dev_folds,
)
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
from src.models import (
    CNNLSTMConfig,
    fit_predict_cnn_lstm_ensemble,
    fit_predict_lgbm_anchor,
    fit_predict_linear_anchor,
)


def _quintile_table(preds: pd.Series, fwd_ret: pd.Series, label: str) -> pd.DataFrame:
    df = pd.concat([preds.rename("pred"), fwd_ret.rename("fwd_ret")], axis=1).dropna()
    df["quintile"] = pd.qcut(df["pred"], q=5, labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"])
    grp = df.groupby("quintile", observed=True)["fwd_ret"].agg(["count", "mean", "std"]).reset_index()
    grp["model"] = label
    return grp


def _cs_rank_table(pred_df: pd.DataFrame, fwd_df: pd.DataFrame, label: str) -> pd.DataFrame:
    ex_df = fwd_df.sub(fwd_df.mean(axis=1), axis=0)
    ranks = pred_df.rank(axis=1, ascending=False, method="first")
    rows = []
    for r in range(1, pred_df.shape[1] + 1):
        mask = ranks == r
        raw_vals = fwd_df.where(mask).stack()
        ex_vals = ex_df.where(mask).stack()
        rows.append(
            {
                "model": label,
                "rank": r,
                "count": int(len(raw_vals)),
                "mean_fwd_ret": float(raw_vals.mean()),
                "mean_excess_vs_it": float(ex_vals.mean()),
                "hit_pos": float((raw_vals > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _yearly_breakdown(df_bt: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for yr, sub in df_bt.groupby(df_bt.index.year):
        sr = sub["net_ret"]
        br = sub["bh_ret"] if "bh_ret" in sub.columns else sub["bh_infy_ret"]
        eq = (1.0 + sr).cumprod()
        beq = (1.0 + br).cumprod()
        ann_v = float(sr.std(ddof=0) * np.sqrt(252.0))
        b_ann_v = float(br.std(ddof=0) * np.sqrt(252.0))
        rows.append(
            {
                "model": model_name,
                "year": int(yr),
                "bars": len(sub),
                "return": float(eq.iloc[-1] - 1.0),
                "bh_return": float(beq.iloc[-1] - 1.0),
                "sharpe": float((sr.mean() * 252.0) / (ann_v + 1e-12)),
                "bh_sharpe": float((br.mean() * 252.0) / (b_ann_v + 1e-12)),
                "max_dd": float((eq / eq.cummax() - 1.0).min()),
                "bh_max_dd": float((beq / beq.cummax() - 1.0).min()),
            }
        )
    return pd.DataFrame(rows)


def _get_prior_infy_dev_signals(
    z_panel: dict[str, pd.DataFrame],
    stocks: dict[str, pd.DataFrame],
    feature_cols: list[str],
    horizon: int,
    target_col: str,
    seq_len: int,
    cfg: CNNLSTMConfig,
    is_panel: bool,
) -> pd.Series:
    t_panel = {sym: add_forward_targets(df, horizon=horizon, execution_mode="next_open") for sym, df in z_panel.items()}
    folds = list(
        purged_dev_folds(
            stocks["INFY.NS"].index, oos_start="2022-01-01", n_splits=6, val_bars=252, horizon=horizon, execution_mode="next_open"
        )
    )
    tr_dts, va_dts, _, _ = folds[-1]
    if is_panel:
        Xp, yp, dp, sp = build_sequences_panel(t_panel, feature_cols, target_col, seq_len)
        m_tr = np.isin(dp, tr_dts)
        m_va = np.isin(dp, va_dts) & (sp == "INFY.NS")
        pv = fit_predict_cnn_lstm_ensemble(Xp[m_tr], yp[m_tr], Xp[m_va], cfg, base_seed=42)
        return pd.Series(pv, index=dp[m_va]).sort_index()
    Xs, ys, ds = build_sequences_single(t_panel["INFY.NS"], feature_cols, target_col, seq_len)
    m_tr = np.isin(ds, tr_dts)
    m_va = np.isin(ds, va_dts)
    sw = None
    if cfg.task == "classification":
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ys[m_tr].astype(int))
        sw = np.where(ys[m_tr] > 0.5, cw[1], cw[0]).astype(np.float32)
    pv = fit_predict_cnn_lstm_ensemble(Xs[m_tr], ys[m_tr], Xs[m_va], cfg, base_seed=42, sample_weight=sw)
    return pd.Series(pv, index=ds[m_va]).sort_index()


def _predict_oos_panel(
    z_panel: dict[str, pd.DataFrame],
    stocks: dict[str, pd.DataFrame],
    feature_cols: list[str],
    horizon: int,
    target_col: str,
    seq_len: int,
    mcfg: CNNLSTMConfig,
    retrain_every: int = 63,
) -> pd.DataFrame:
    t_panel = {sym: add_forward_targets(df, horizon=horizon, execution_mode="next_open") for sym, df in z_panel.items()}
    Xp, yp, dp, _ = build_sequences_panel(t_panel, feature_cols, target_col, seq_len)
    inf_seqs = {sym: build_inference_sequences(z_panel[sym], feature_cols, seq_len) for sym in NSE_IT_UNIVERSE}

    blocks = list(
        oos_retrain_blocks(
            stocks["INFY.NS"].index, oos_start="2022-01-01", retrain_every_bars=retrain_every, horizon=horizon, execution_mode="next_open"
        )
    )
    pred_grid = {sym: [] for sym in NSE_IT_UNIVERSE}
    for b_idx, (tr_cut, blk_start, blk_end) in enumerate(blocks):
        m_tr = dp <= tr_cut
        X_blk_list, meta = [], []
        for sym in NSE_IT_UNIVERSE:
            Xs_i, ds_i = inf_seqs[sym]
            m_blk = (ds_i >= blk_start) & (ds_i <= blk_end)
            X_blk_list.append(Xs_i[m_blk])
            meta.append((sym, ds_i[m_blk]))
        X_blk = np.concatenate(X_blk_list, axis=0)
        p_blk = fit_predict_cnn_lstm_ensemble(Xp[m_tr], yp[m_tr], X_blk, mcfg, base_seed=100 + b_idx * 10)
        offset = 0
        for sym, d_sub in meta:
            n_sub = len(d_sub)
            pred_grid[sym].append(pd.Series(p_blk[offset : offset + n_sub], index=d_sub))
            offset += n_sub

    return pd.DataFrame({sym: pd.concat(lst) for sym, lst in pred_grid.items()}).sort_index()


def _predict_oos_tabular_panel(
    z_panel: dict[str, pd.DataFrame],
    stocks: dict[str, pd.DataFrame],
    feature_cols: list[str],
    horizon: int,
    target_col: str,
    seq_len: int,
    model_type: str,
    retrain_every: int = 63,
) -> pd.DataFrame:
    """Quarterly purged OOS predictions for Ridge or shallow LightGBM on the exact same setup."""
    t_panel = {sym: add_forward_targets(df, horizon=horizon, execution_mode="next_open") for sym, df in z_panel.items()}
    Xp, yp, dp, _ = build_sequences_panel(t_panel, feature_cols, target_col, seq_len)
    inf_seqs = {sym: build_inference_sequences(z_panel[sym], feature_cols, seq_len) for sym in NSE_IT_UNIVERSE}
    blocks = list(
        oos_retrain_blocks(
            stocks["INFY.NS"].index, oos_start="2022-01-01", retrain_every_bars=retrain_every, horizon=horizon, execution_mode="next_open"
        )
    )
    pred_grid = {sym: [] for sym in NSE_IT_UNIVERSE}
    for b_idx, (tr_cut, blk_start, blk_end) in enumerate(blocks):
        m_tr = dp <= tr_cut
        X_blk_list, meta = [], []
        for sym in NSE_IT_UNIVERSE:
            Xs_i, ds_i = inf_seqs[sym]
            m_blk = (ds_i >= blk_start) & (ds_i <= blk_end)
            X_blk_list.append(Xs_i[m_blk])
            meta.append((sym, ds_i[m_blk]))
        X_blk = np.concatenate(X_blk_list, axis=0)
        if model_type == "ridge":
            p_blk = fit_predict_linear_anchor(Xp[m_tr], yp[m_tr], X_blk, task="regression", alpha=2000.0)
        elif model_type == "lgbm":
            p_blk = fit_predict_lgbm_anchor(Xp[m_tr], yp[m_tr], X_blk, seed=100 + b_idx)
        else:
            raise ValueError(model_type)
        offset = 0
        for sym, d_sub in meta:
            n_sub = len(d_sub)
            pred_grid[sym].append(pd.Series(p_blk[offset : offset + n_sub], index=d_sub))
            offset += n_sub
    return pd.DataFrame({sym: pd.concat(lst) for sym, lst in pred_grid.items()}).sort_index()


def _eval_panel_baseline_row(
    exp_id: str,
    pred_df: pd.DataFrame,
    z_panel: dict[str, pd.DataFrame],
    stocks: dict[str, pd.DataFrame],
    reg_nifty: pd.Series,
    horizon: int,
) -> tuple[dict, pd.Series]:
    t_panel = {sym: add_forward_targets(df, horizon=horizon, execution_mode="next_open") for sym, df in z_panel.items()}
    fwd_df = pd.DataFrame({sym: t_panel[sym]["fwd_log_ret"].reindex(pred_df.index) for sym in NSE_IT_UNIVERSE})
    sq_panel = evaluate_signal_quality(pd.Series(pred_df.to_numpy().ravel()), pd.Series(fwd_df.to_numpy().ravel()))
    sq_infy = evaluate_signal_quality(pred_df["INFY.NS"], fwd_df["INFY.NS"])
    cs_ics = []
    for dt in pred_df.index:
        rp, rf = pred_df.loc[dt].to_numpy(float), fwd_df.loc[dt].to_numpy(float)
        m = np.isfinite(rp) & np.isfinite(rf)
        if m.sum() >= 4 and np.std(rp[m]) > 0 and np.std(rf[m]) > 0:
            c, _ = spearmanr(rp[m], rf[m])
            if np.isfinite(c):
                cs_ics.append(c)
    bt_cfg_reg = PortfolioConfig(cost_bps_per_side=20.0, target_annual_vol=0.14, max_leverage=1.0, min_hold_bars=5, use_market_regime_filter=True, regime_half_weight=0.25)
    df_bt, sum_bt = run_panel_topk_backtest(
        stocks, pred_df, cfg=bt_cfg_reg, top_k=3, rebalance_every=5, execution_mode="next_open", regime_series=reg_nifty, execution_style="periodic"
    )
    row = {
        "exp_id": exp_id,
        "panel_ic": sq_panel["ic_spearman"],
        "cs_ic": float(np.mean(cs_ics)),
        "infy_ic": sq_infy["ic_spearman"],
        "q5_q1": sq_panel["q_spread"],
        **sum_bt,
    }
    return row, df_bt["strategy_equity"].rename(exp_id)


def evaluate_oos_panel(
    exp_id: str,
    pred_df: pd.DataFrame,
    z_panel: dict[str, pd.DataFrame],
    stocks: dict[str, pd.DataFrame],
    reg_nifty: pd.Series,
    reg_cnxit: pd.Series,
    feature_cols: list[str],
    horizon: int,
    target_col: str,
    seq_len: int,
    mcfg: CNNLSTMConfig,
    prior_dev_panel: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t_panel = {sym: add_forward_targets(df, horizon=horizon, execution_mode="next_open") for sym, df in z_panel.items()}
    prior_infy = (
        prior_dev_panel["INFY.NS"].tail(252)
        if prior_dev_panel is not None
        else _get_prior_infy_dev_signals(z_panel, stocks, feature_cols, horizon, target_col, seq_len, mcfg, is_panel=True)
    )

    fwd_df = pd.DataFrame({sym: t_panel[sym]["fwd_log_ret"].reindex(pred_df.index) for sym in NSE_IT_UNIVERSE})
    sq_panel = evaluate_signal_quality(pd.Series(pred_df.to_numpy().ravel()), pd.Series(fwd_df.to_numpy().ravel()))
    sq_infy = evaluate_signal_quality(pred_df["INFY.NS"], fwd_df["INFY.NS"])
    cs_ics = []
    for dt in pred_df.index:
        rp, rf = pred_df.loc[dt].to_numpy(float), fwd_df.loc[dt].to_numpy(float)
        m = np.isfinite(rp) & np.isfinite(rf)
        if m.sum() >= 4 and np.std(rp[m]) > 0 and np.std(rf[m]) > 0:
            c, _ = spearmanr(rp[m], rf[m])
            if np.isfinite(c):
                cs_ics.append(c)
    cs_ic = float(np.mean(cs_ics))

    bt_cfg = PortfolioConfig(cost_bps_per_side=20.0, target_annual_vol=0.14, max_leverage=1.0, min_hold_bars=5, entry_quantile=0.55, exit_quantile=0.35)
    bt_cfg_reg = PortfolioConfig(cost_bps_per_side=20.0, target_annual_vol=0.14, max_leverage=1.0, min_hold_bars=5, entry_quantile=0.55, exit_quantile=0.35, use_market_regime_filter=True, regime_half_weight=0.25)
    bt_cfg_slv = PortfolioConfig(cost_bps_per_side=20.0, target_annual_vol=0.14, max_leverage=1.0, rebal_deadband=0.02, use_market_regime_filter=True, regime_half_weight=0.25)

    df_infy_bt, sum_infy = run_single_asset_backtest(stocks["INFY.NS"], pred_df["INFY.NS"], cfg=bt_cfg, execution_mode="next_open", prior_signals=prior_infy)
    df_top3_nifty, sum_top3_nifty = run_panel_topk_backtest(stocks, pred_df, cfg=bt_cfg_reg, top_k=3, rebalance_every=5, execution_mode="next_open", regime_series=reg_nifty, execution_style="periodic")
    df_top3_hyst_cnxit, sum_top3_hyst_cnxit = run_panel_topk_backtest(stocks, pred_df, cfg=bt_cfg_reg, top_k=3, rebalance_every=5, keep_rank=4, execution_mode="next_open", regime_series=reg_cnxit, execution_style="rank_hysteresis")
    df_top3_slv_cnxit, sum_top3_slv_cnxit = run_panel_topk_backtest(stocks, pred_df, cfg=bt_cfg_slv, top_k=3, execution_mode="next_open", regime_series=reg_cnxit, execution_style="overlapping_sleeves", sleeve_horizon=horizon, prior_pred_panel=prior_dev_panel, prior_regime_series=reg_cnxit.loc[reg_cnxit.index < "2022-01-01"] if prior_dev_panel is not None else None)

    cost_rows = []
    for bps in (0.0, 10.0, 20.0, 30.0):
        c_n = PortfolioConfig(**{**bt_cfg_reg.__dict__, "cost_bps_per_side": bps})
        _, s_n = run_panel_topk_backtest(stocks, pred_df, cfg=c_n, top_k=3, rebalance_every=5, execution_mode="next_open", regime_series=reg_nifty, execution_style="periodic")
        _, s_h = run_panel_topk_backtest(stocks, pred_df, cfg=c_n, top_k=3, rebalance_every=5, keep_rank=4, execution_mode="next_open", regime_series=reg_cnxit, execution_style="rank_hysteresis")
        cost_rows.append({"model": exp_id, "bps_per_side": bps, "periodic_nifty_sharpe": s_n["sharpe"], "periodic_nifty_cagr": s_n["cagr"], "hyst_cnxit_sharpe": s_h["sharpe"], "hyst_cnxit_excess_sharpe": s_h["excess_sharpe"], "hyst_cnxit_cagr": s_h["cagr"]})

    q_df = pd.concat([
        _quintile_table(pred_df["INFY.NS"], fwd_df["INFY.NS"], f"{exp_id}_INFY"),
        _quintile_table(pd.Series(pred_df.to_numpy().ravel()), pd.Series(fwd_df.to_numpy().ravel()), f"{exp_id}_Panel"),
    ], ignore_index=True)
    rank_df = _cs_rank_table(pred_df, fwd_df, exp_id)

    yr_df = pd.concat([
        _yearly_breakdown(df_top3_nifty, f"{exp_id}_Top3_NiftyReg"),
        _yearly_breakdown(df_top3_hyst_cnxit, f"{exp_id}_Top3_Hyst_CNXITReg"),
    ], ignore_index=True)

    summary_rows = [
        {"exp_id": f"{exp_id}__INFY", "panel_ic": sq_panel["ic_spearman"], "cs_ic": cs_ic, "infy_ic": sq_infy["ic_spearman"], "q5_q1": sq_infy["q_spread"], **sum_infy},
        {"exp_id": f"{exp_id}__Top3_Periodic_NiftyReg", "panel_ic": sq_panel["ic_spearman"], "cs_ic": cs_ic, "infy_ic": sq_infy["ic_spearman"], "q5_q1": sq_panel["q_spread"], **sum_top3_nifty},
        {"exp_id": f"{exp_id}__Top3_RankHyst_CNXITReg", "panel_ic": sq_panel["ic_spearman"], "cs_ic": cs_ic, "infy_ic": sq_infy["ic_spearman"], "q5_q1": sq_panel["q_spread"], **sum_top3_hyst_cnxit},
        {"exp_id": f"{exp_id}__Top3_Sleeves_CNXITReg", "panel_ic": sq_panel["ic_spearman"], "cs_ic": cs_ic, "infy_ic": sq_infy["ic_spearman"], "q5_q1": sq_panel["q_spread"], **sum_top3_slv_cnxit},
    ]
    curves = pd.DataFrame({
        f"{exp_id}_Top3_Periodic_NiftyReg": df_top3_nifty["strategy_equity"],
        f"{exp_id}_Top3_RankHyst_CNXITReg": df_top3_hyst_cnxit["strategy_equity"],
        f"{exp_id}_Top3_Sleeves_CNXITReg": df_top3_slv_cnxit["strategy_equity"],
        f"{exp_id}_INFY": df_infy_bt["strategy_equity"],
        "INFY_BuyHold": df_infy_bt["bh_equity"],
        "EW_7IT_Sector": df_top3_nifty["ew_sector_equity"],
    })
    return {"rows": summary_rows}, pd.DataFrame(cost_rows), q_df, rank_df, yr_df, curves


def run_oos_single_model(
    exp_id: str,
    z_infy: pd.DataFrame,
    stocks: dict[str, pd.DataFrame],
    reg_cnxit: pd.Series,
    feature_cols: list[str],
    horizon: int,
    target_col: str,
    seq_len: int,
    mcfg: CNNLSTMConfig,
    retrain_every: int = 63,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_infy = stocks["INFY.NS"]
    t_infy = add_forward_targets(z_infy, horizon=horizon, execution_mode="next_open")
    Xs, ys, ds = build_sequences_single(t_infy, feature_cols, target_col, seq_len)
    X_inf, d_inf = build_inference_sequences(z_infy, feature_cols, seq_len)
    blocks = list(
        oos_retrain_blocks(
            raw_infy.index, oos_start="2022-01-01", retrain_every_bars=retrain_every, horizon=horizon, execution_mode="next_open"
        )
    )
    preds = []
    for b_idx, (tr_cut, blk_start, blk_end) in enumerate(blocks):
        m_tr = ds <= tr_cut
        m_blk = (d_inf >= blk_start) & (d_inf <= blk_end)
        sw = None
        if mcfg.task == "classification":
            cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ys[m_tr].astype(int))
            sw = np.where(ys[m_tr] > 0.5, cw[1], cw[0]).astype(np.float32)
        pv = fit_predict_cnn_lstm_ensemble(Xs[m_tr], ys[m_tr], X_inf[m_blk], mcfg, base_seed=100 + b_idx * 10, sample_weight=sw)
        preds.append(pd.Series(pv, index=d_inf[m_blk]))

    p_ser = pd.concat(preds).sort_index()
    prior_infy = _get_prior_infy_dev_signals({"INFY.NS": z_infy}, stocks, feature_cols, horizon, target_col, seq_len, mcfg, is_panel=False)
    fwd = t_infy["fwd_log_ret"].reindex(p_ser.index)
    sq = evaluate_signal_quality(p_ser, fwd)

    bt_cfg = PortfolioConfig(cost_bps_per_side=20.0, target_annual_vol=0.14, max_leverage=1.0, min_hold_bars=min(5, horizon), entry_quantile=0.55, exit_quantile=0.35)
    df_bt, sum_bt = run_single_asset_backtest(raw_infy, p_ser, cfg=bt_cfg, execution_mode="next_open", prior_signals=prior_infy)
    bt_cfg_reg = PortfolioConfig(**{**bt_cfg.__dict__, "use_market_regime_filter": True, "regime_half_weight": 0.25})
    df_reg, sum_reg = run_single_asset_backtest(raw_infy, p_ser, cfg=bt_cfg_reg, execution_mode="next_open", prior_signals=prior_infy, regime_series=reg_cnxit)

    rows = [
        {"exp_id": f"{exp_id}__INFY", "panel_ic": np.nan, "cs_ic": np.nan, "infy_ic": sq["ic_spearman"], "q5_q1": sq["q_spread"], **sum_bt},
        {"exp_id": f"{exp_id}__INFY_CNXITReg", "panel_ic": np.nan, "cs_ic": np.nan, "infy_ic": sq["ic_spearman"], "q5_q1": sq["q_spread"], **sum_reg},
    ]
    return {"rows": rows}, df_bt["strategy_equity"].rename(f"{exp_id}_INFY").to_frame(), _yearly_breakdown(df_bt, f"{exp_id}_INFY"), p_ser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-panel-preds", action="store_true", help="Reuse cached OOS panel prediction CSVs")
    args = parser.parse_args()

    stocks, context = load_market_bundle()
    feat_single = BASE_TECH_FEATURES + MARKET_SECTOR_FEATURES + GLOBAL_MACRO_FEATURES
    feat_panel = feat_single + CROSS_SECTIONAL_FEATURES

    raw_panel = {sym: build_stock_features(stocks[sym], context, execution_mode="next_open") for sym in NSE_IT_UNIVERSE}
    raw_panel = add_cross_sectional_features(raw_panel)
    z_panel = {sym: apply_trailing_zscore(df, feat_panel, window=252, min_periods=126) for sym, df in raw_panel.items()}
    reg_nifty = raw_panel["INFY.NS"]["nifty_dist_sma200"]
    reg_cnxit = raw_panel["INFY.NS"]["cnxit_dist_sma200"]

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_cfg = CNNLSTMConfig(
        conv_filters=32, kernel_size=3, lstm_units=32, dropout=0.30, rec_dropout=0.0,
        l2_reg=5e-4, lr=5e-4, batch_size=128, epochs=25, task="regression", norm_type="layer", n_seeds=5
    )
    single_cfg_reg = CNNLSTMConfig(
        conv_filters=16, kernel_size=3, lstm_units=16, dropout=0.35, rec_dropout=0.0,
        l2_reg=1e-3, lr=5e-4, batch_size=64, epochs=25, task="regression", norm_type="layer", n_seeds=5
    )
    single_cfg_cls = CNNLSTMConfig(
        conv_filters=16, kernel_size=3, lstm_units=16, dropout=0.35, rec_dropout=0.0,
        l2_reg=1e-3, lr=5e-4, batch_size=64, epochs=25, task="classification", norm_type="layer", n_seeds=5
    )

    p2_path = out_dir / "oos_predictions_P2_H10_excess_nifty.csv"
    p5_path = out_dir / "oos_predictions_P5_H15_excess_nifty.csv"
    dev_p2_path = out_dir / "dev_predictions_P2_H10_excess_nifty.csv"

    if args.reuse_panel_preds and p2_path.exists():
        pred_p2 = pd.read_csv(p2_path, index_col=0, parse_dates=True)
    else:
        pred_p2 = _predict_oos_panel(z_panel, stocks, feat_panel, 10, "target_excess_vol_adj", 20, panel_cfg)
        pred_p2.to_csv(p2_path)

    if args.reuse_panel_preds and p5_path.exists():
        pred_p5 = pd.read_csv(p5_path, index_col=0, parse_dates=True)
    else:
        pred_p5 = _predict_oos_panel(z_panel, stocks, feat_panel, 15, "target_excess_vol_adj", 20, panel_cfg)
        pred_p5.to_csv(p5_path)

    prior_dev_p2 = pd.read_csv(dev_p2_path, index_col=0, parse_dates=True) if dev_p2_path.exists() else None

    all_rows, cost_dfs, q_dfs, rank_dfs, yr_dfs, curve_dfs = [], [], [], [], [], []

    r_p2, c_p2, q_p2, rk_p2, y_p2, cv_p2 = evaluate_oos_panel(
        "P2_H10_excess_nifty", pred_p2, z_panel, stocks, reg_nifty, reg_cnxit, feat_panel, 10, "target_excess_vol_adj", 20, panel_cfg, prior_dev_panel=prior_dev_p2
    )
    all_rows.extend(r_p2["rows"]); cost_dfs.append(c_p2); q_dfs.append(q_p2); rank_dfs.append(rk_p2); yr_dfs.append(y_p2); curve_dfs.append(cv_p2)

    r_p5, c_p5, q_p5, rk_p5, y_p5, cv_p5 = evaluate_oos_panel(
        "P5_H15_excess_nifty", pred_p5, z_panel, stocks, reg_nifty, reg_cnxit, feat_panel, 15, "target_excess_vol_adj", 20, panel_cfg
    )
    all_rows.extend(r_p5["rows"]); cost_dfs.append(c_p5); q_dfs.append(q_p5); rank_dfs.append(rk_p5); yr_dfs.append(y_p5)
    curve_dfs.append(cv_p5.drop(columns=["INFY_BuyHold", "EW_7IT_Sector"]))

    # Tabular OOS anchors (Ridge & LightGBM) + 50/50 rank-blend with CNN-LSTM on exact P2_H10 setup
    pred_ridge = _predict_oos_tabular_panel(z_panel, stocks, feat_panel, 10, "target_excess_vol_adj", 20, "ridge")
    pred_lgbm = _predict_oos_tabular_panel(z_panel, stocks, feat_panel, 10, "target_excess_vol_adj", 20, "lgbm")
    pred_blend = 0.5 * pred_p2.rank(axis=1, pct=True) + 0.5 * pred_lgbm.rank(axis=1, pct=True)

    row_ridge, eq_ridge = _eval_panel_baseline_row("Anchor_Ridge_H10__Top3_Periodic_NiftyReg", pred_ridge, z_panel, stocks, reg_nifty, 10)
    row_lgbm, eq_lgbm = _eval_panel_baseline_row("Anchor_LightGBM_H10__Top3_Periodic_NiftyReg", pred_lgbm, z_panel, stocks, reg_nifty, 10)
    row_blend, eq_blend = _eval_panel_baseline_row("Blend_CNNLSTM_LGBM_H10__Top3_Periodic_NiftyReg", pred_blend, z_panel, stocks, reg_nifty, 10)
    all_rows.extend([row_ridge, row_lgbm, row_blend])
    curve_dfs.extend([eq_ridge.to_frame(), eq_lgbm.to_frame(), eq_blend.to_frame()])

    r_s1, cv_s1, y_s1, p_s1 = run_oos_single_model(
        "S1_H5_dir_cls", z_panel["INFY.NS"], stocks, reg_cnxit, feat_single, 5, "target_dir", 20, single_cfg_cls
    )
    all_rows.extend(r_s1["rows"]); curve_dfs.append(cv_s1); yr_dfs.append(y_s1)
    p_s1.to_csv(out_dir / "oos_predictions_S1_H5_dir_cls.csv")

    r_s4, cv_s4, y_s4, _ = run_oos_single_model(
        "S4_H15_it_excess", z_panel["INFY.NS"], stocks, reg_cnxit, feat_single, 15, "target_it_excess_vol_adj", 20, single_cfg_reg
    )
    all_rows.extend(r_s4["rows"]); curve_dfs.append(cv_s4); yr_dfs.append(y_s4)

    oos_idx = cv_p2.index
    sma_sig = raw_panel["INFY.NS"].loc[oos_idx, "dist_sma_200"]
    df_sma, sum_sma = run_single_asset_backtest(
        stocks["INFY.NS"], sma_sig,
        cfg=PortfolioConfig(cost_bps_per_side=20.0, target_annual_vol=0.14, max_leverage=1.0, min_hold_bars=5),
        execution_mode="next_open", signal_mode="threshold_hysteresis", fixed_entry_thresh=0.0, fixed_exit_thresh=-0.02
    )
    all_rows.append({"exp_id": "Benchmark_INFY_200SMA", "panel_ic": np.nan, "cs_ic": np.nan, "infy_ic": np.nan, "q5_q1": np.nan, **sum_sma})
    curve_dfs.append(df_sma["strategy_equity"].rename("Benchmark_INFY_200SMA").to_frame())

    ew_ret = pd.DataFrame({s: compute_execution_returns(stocks[s], "next_open") for s in NSE_IT_UNIVERSE}).reindex(oos_idx).mean(axis=1)
    bh_infy = compute_execution_returns(stocks["INFY.NS"], "next_open").reindex(oos_idx)
    sum_ew = compute_performance_metrics(ew_ret, bh_infy, pd.DataFrame(1.0 / len(NSE_IT_UNIVERSE), index=oos_idx, columns=NSE_IT_UNIVERSE), rf_annual=0.06)
    all_rows.append({"exp_id": "Benchmark_EW_7IT", "panel_ic": np.nan, "cs_ic": np.nan, "infy_ic": np.nan, "q5_q1": np.nan, **sum_ew})

    df_oos = pd.DataFrame(all_rows)
    df_oos.to_csv(out_dir / "stage2_oos_leaderboard.csv", index=False)
    pd.concat(cost_dfs, ignore_index=True).to_csv(out_dir / "oos_cost_sensitivity.csv", index=False)
    pd.concat(q_dfs, ignore_index=True).to_csv(out_dir / "oos_quintiles.csv", index=False)
    pd.concat(rank_dfs, ignore_index=True).to_csv(out_dir / "oos_cs_ranks.csv", index=False)
    pd.concat(yr_dfs, ignore_index=True).to_csv(out_dir / "oos_yearly_breakdown.csv", index=False)
    pd.concat(curve_dfs, axis=1).to_csv(out_dir / "oos_equity_curves.csv")

    cols = ["exp_id", "cagr", "cash_adj_cagr", "ann_vol", "sharpe", "excess_sharpe", "bh_sharpe", "bh_excess_sharpe", "max_drawdown", "ann_turnover_nav"]
    print("\n=== Final OOS Leaderboard (2022-2024, 20 bps/side, rf=6% p.a.) ===")
    print(df_oos[cols].to_string(index=False))


if __name__ == "__main__":
    main()
