"""Stage 1 Execution & Regime Ablation strictly on pre-2022 Dev CV (2015-11..2021-12)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import PortfolioConfig, run_panel_topk_backtest
from src.cv import build_sequences_panel, purged_dev_folds
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


def main() -> None:
    stocks, context = load_market_bundle()
    feat_panel = BASE_TECH_FEATURES + MARKET_SECTOR_FEATURES + GLOBAL_MACRO_FEATURES + CROSS_SECTIONAL_FEATURES
    raw_panel = {sym: build_stock_features(stocks[sym], context, execution_mode="next_open") for sym in NSE_IT_UNIVERSE}
    raw_panel = add_cross_sectional_features(raw_panel)
    z_panel = {sym: apply_trailing_zscore(df, feat_panel, window=252, min_periods=126) for sym, df in raw_panel.items()}
    t_panel = {sym: add_forward_targets(df, horizon=10, execution_mode="next_open") for sym, df in z_panel.items()}

    Xp, yp, dp, sp = build_sequences_panel(t_panel, feat_panel, "target_excess_vol_adj", 20)
    folds = list(
        purged_dev_folds(
            stocks["INFY.NS"].index, oos_start="2022-01-01", n_splits=6, val_bars=252, horizon=10, execution_mode="next_open"
        )
    )
    mcfg = CNNLSTMConfig(
        conv_filters=32, kernel_size=3, lstm_units=32, dropout=0.30, rec_dropout=0.0,
        l2_reg=5e-4, lr=5e-4, batch_size=128, epochs=25, task="regression", norm_type="layer", n_seeds=3
    )

    pred_grid = {sym: [] for sym in NSE_IT_UNIVERSE}
    for k, (tr_dts, va_dts, _, _) in enumerate(folds):
        m_tr = np.isin(dp, tr_dts)
        m_va = np.isin(dp, va_dts)
        pv = fit_predict_cnn_lstm_ensemble(Xp[m_tr], yp[m_tr], Xp[m_va], mcfg, base_seed=42 + k * 10)
        for sym in NSE_IT_UNIVERSE:
            m_s = sp[m_va] == sym
            pred_grid[sym].append(pd.Series(pv[m_s], index=dp[m_va][m_s]))

    pred_df = pd.DataFrame({sym: pd.concat(lst) for sym, lst in pred_grid.items()}).sort_index()
    pred_df.index.name = "date"
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "dev_predictions_P2_H10_excess_nifty.csv")

    reg_nifty = raw_panel["INFY.NS"]["nifty_dist_sma200"]
    reg_cnxit = raw_panel["INFY.NS"]["cnxit_dist_sma200"]
    reg_dual = np.minimum(reg_nifty, reg_cnxit)  # bear if EITHER Nifty50 or NiftyIT < 200-SMA

    configs = [
        ("periodic_NoReg", dict(execution_style="periodic", rebalance_every=5), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.20, use_market_regime_filter=False), None),
        ("periodic_NiftyReg", dict(execution_style="periodic", rebalance_every=5), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.20, use_market_regime_filter=True, regime_half_weight=0.25), reg_nifty),
        ("rankHyst_k3_keep4_NiftyReg", dict(execution_style="rank_hysteresis", rebalance_every=5, keep_rank=4), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.20, use_market_regime_filter=True, regime_half_weight=0.25), reg_nifty),
        ("rankHyst_k3_keep4_CNXITReg", dict(execution_style="rank_hysteresis", rebalance_every=5, keep_rank=4), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.20, use_market_regime_filter=True, regime_half_weight=0.25), reg_cnxit),
        ("sleeves10_NoReg", dict(execution_style="overlapping_sleeves", sleeve_horizon=10), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.02, use_market_regime_filter=False), None),
        ("sleeves10_NiftyReg", dict(execution_style="overlapping_sleeves", sleeve_horizon=10), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.02, use_market_regime_filter=True, regime_half_weight=0.25), reg_nifty),
        ("sleeves10_CNXITReg", dict(execution_style="overlapping_sleeves", sleeve_horizon=10), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.02, use_market_regime_filter=True, regime_half_weight=0.25), reg_cnxit),
        ("sleeves10_DualReg", dict(execution_style="overlapping_sleeves", sleeve_horizon=10), PortfolioConfig(cost_bps_per_side=20.0, rebal_deadband=0.02, use_market_regime_filter=True, regime_half_weight=0.25), reg_dual),
    ]

    rows = []
    for name, kw, pcfg, reg in configs:
        _, sm = run_panel_topk_backtest(stocks, pred_df, cfg=pcfg, top_k=3, execution_mode="next_open", regime_series=reg, **kw)
        rows.append({"exec_variant": name, **sm})

    df_ab = pd.DataFrame(rows)
    df_ab.to_csv(out_dir / "stage1_execution_ablation.csv", index=False)
    cols = ["exec_variant", "cagr", "cash_adj_cagr", "ann_vol", "sharpe", "excess_sharpe", "max_drawdown", "ann_turnover_nav"]
    print("\n=== Stage 1 Pre-2022 Dev CV Execution & Regime Ablation (P2_H10_excess_nifty) ===")
    print(df_ab[cols].to_string(index=False))


if __name__ == "__main__":
    main()
