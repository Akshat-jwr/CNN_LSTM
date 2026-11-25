"""Purged & embargoed walk-forward sequence splitting and zero-leakage OOS fold generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    train_end_cutoff: pd.Timestamp


def build_sequences_single(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build strictly causal rolling sequences for one asset.

    Sequence at index `t` uses `feature_cols` over `[t - seq_len + 1 .. t]` and targets `df[target_col].iloc[t]`.
    """
    valid = df.dropna(subset=feature_cols + [target_col])
    if len(valid) < seq_len:
        raise ValueError(f"Not enough valid rows ({len(valid)}) for seq_len={seq_len}.")
    X2d = valid[feature_cols].to_numpy(dtype=np.float32)
    y1d = valid[target_col].to_numpy(dtype=np.float32)
    idx = valid.index

    n_seq = len(valid) - seq_len + 1
    X_seq = np.stack([X2d[i : i + seq_len] for i in range(n_seq)], axis=0)
    y_seq = y1d[seq_len - 1 :]
    seq_dates = pd.DatetimeIndex(idx[seq_len - 1 :])
    return X_seq, y_seq, seq_dates


def build_sequences_panel(
    panel: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    Build rolling sequences across a multi-stock panel, keeping per-stock sequence boundaries intact.
    Returns `(X_seq, y_seq, dates, tickers)` sorted chronologically by `dates`.
    """
    xs, ys, dts, syms = [], [], [], []
    for sym, df in panel.items():
        # Handle contiguous non-NaN sequences cleanly per stock
        valid = df.dropna(subset=feature_cols + [target_col])
        if len(valid) < seq_len:
            continue
        X_s, y_s, d_s = build_sequences_single(valid, feature_cols, target_col, seq_len)
        xs.append(X_s)
        ys.append(y_s)
        dts.append(d_s.to_numpy())
        syms.append(np.full(len(y_s), sym, dtype=object))

    if not xs:
        raise ValueError("No valid sequences built from panel.")

    X_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    d_all = pd.DatetimeIndex(np.concatenate(dts))
    s_all = np.concatenate(syms)

    order = np.argsort(d_all.values, kind="mergesort")
    return X_all[order], y_all[order], d_all[order], s_all[order]


def build_inference_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Build causal sequences for prediction dates (does not require forward target to be non-NaN)."""
    valid = df.dropna(subset=feature_cols)
    if len(valid) < seq_len:
        raise ValueError(f"Not enough valid feature rows ({len(valid)}) for seq_len={seq_len}.")
    X2d = valid[feature_cols].to_numpy(dtype=np.float32)
    idx = valid.index
    n_seq = len(valid) - seq_len + 1
    X_seq = np.stack([X2d[i : i + seq_len] for i in range(n_seq)], axis=0)
    seq_dates = pd.DatetimeIndex(idx[seq_len - 1 :])
    return X_seq, seq_dates


def purged_dev_folds(
    unique_dates: pd.DatetimeIndex,
    oos_start: str = "2022-01-01",
    n_splits: int = 6,
    val_bars: int = 252,
    horizon: int = 10,
    execution_mode: str = "next_open",
    embargo_bars: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
    """
    Expanding-window purged CV strictly inside `< oos_start`.

    A training sample `t_tr` has a forward target ending at bar `t_tr + horizon + extra_shift`
    (where `extra_shift = 1` in `next_open` mode because exit is `Open[t + 1 + horizon]`).
    Every training sample whose target end >= `val_start_pos - embargo_bars` is purged.
    """
    dev_dates = pd.DatetimeIndex(sorted(unique_dates[unique_dates < pd.Timestamp(oos_start)]))
    n_dev = len(dev_dates)
    total_val = n_splits * val_bars
    first_val_pos = n_dev - total_val
    extra_shift = 1 if execution_mode == "next_open" else 0
    purge_gap = horizon + extra_shift + embargo_bars
    if first_val_pos <= purge_gap + 252:
        raise ValueError(
            f"Not enough pre-OOS bars ({n_dev}) for {n_splits}x{val_bars} folds and purge={purge_gap}."
        )

    for k in range(n_splits):
        val_start_pos = first_val_pos + k * val_bars
        val_end_pos = val_start_pos + val_bars - 1
        train_cutoff_pos = val_start_pos - purge_gap - 1
        val_start_dt = dev_dates[val_start_pos]
        val_end_dt = dev_dates[val_end_pos]
        train_cutoff_dt = dev_dates[train_cutoff_pos]
        yield dev_dates[: train_cutoff_pos + 1], dev_dates[val_start_pos : val_end_pos + 1], train_cutoff_dt, val_start_dt


def oos_retrain_blocks(
    unique_dates: pd.DatetimeIndex,
    oos_start: str = "2022-01-01",
    retrain_every_bars: int = 63,
    horizon: int = 10,
    execution_mode: str = "next_open",
    embargo_bars: int = 2,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Yield `(train_target_cutoff_dt, block_start_dt, block_end_dt)` for periodic OOS retraining.

    At `block_start_dt`, only training samples `t` whose forward target completes strictly
    before `block_start_dt` (with `embargo_bars` buffer) are eligible for fitting.
    """
    all_dates = pd.DatetimeIndex(sorted(unique_dates))
    oos_mask = all_dates >= pd.Timestamp(oos_start)
    oos_positions = np.where(oos_mask)[0]
    if len(oos_positions) == 0:
        raise ValueError(f"No dates >= {oos_start}")

    extra_shift = 1 if execution_mode == "next_open" else 0
    purge_gap = horizon + extra_shift + embargo_bars

    first_oos = int(oos_positions[0])
    last_oos = int(oos_positions[-1])
    for start_pos in range(first_oos, last_oos + 1, retrain_every_bars):
        end_pos = min(start_pos + retrain_every_bars - 1, last_oos)
        train_cutoff_pos = start_pos - purge_gap - 1
        if train_cutoff_pos < 252:
            raise ValueError("Insufficient training history before first OOS block.")
        yield all_dates[train_cutoff_pos], all_dates[start_pos], all_dates[end_pos]
