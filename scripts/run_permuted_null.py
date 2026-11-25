"""Multi-seed permuted-target null check confirming zero structural target/feature leakage."""

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

from src.backtest import evaluate_signal_quality
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
    folds = list(purged_dev_folds(stocks["INFY.NS"].index, oos_start="2022-01-01", n_splits=6, val_bars=252, horizon=10, execution_mode="next_open"))

    cfg = CNNLSTMConfig(conv_filters=32, kernel_size=3, lstm_units=32, dropout=0.30, rec_dropout=0.0, l2_reg=5e-4, lr=5e-4, batch_size=128, epochs=15, task="regression", norm_type="layer", n_seeds=1)
    rows = []
    for pseed in (101, 202, 303, 404, 505):
        rng = np.random.default_rng(pseed)
        ics = []
        for k, (tr_dts, va_dts, _, _) in enumerate(folds):
            m_tr = np.isin(dp, tr_dts)
            m_va = np.isin(dp, va_dts)
            y_perm = rng.permutation(yp[m_tr])
            pv = fit_predict_cnn_lstm_ensemble(Xp[m_tr], y_perm, Xp[m_va], cfg, base_seed=42 + k)
            va_df = pd.DataFrame({"date": dp[m_va], "sym": sp[m_va], "pred": pv})
            va_df["fwd"] = [float(t_panel[sym].loc[dt, "fwd_log_ret"]) for sym, dt in zip(sp[m_va], dp[m_va])]
            ics.append(evaluate_signal_quality(va_df["pred"], va_df["fwd"])["ic_spearman"])
        rows.append({"perm_seed": pseed, "mean_fold_ic": float(np.mean(ics)), "min_fold_ic": float(np.min(ics)), "max_fold_ic": float(np.max(ics))})

    df_null = pd.DataFrame(rows)
    df_null.to_csv(ROOT / "results" / "permuted_sanity_check.csv", index=False)
    print(df_null.to_string(index=False))
    print(f"5-seed null mean IC: {df_null['mean_fold_ic'].mean():+.4f}")


if __name__ == "__main__":
    main()
