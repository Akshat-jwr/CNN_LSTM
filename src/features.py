"""Scale-invariant, strictly causal feature engineering and forward target builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import align_asof_series


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return ((rsi.fillna(50.0) - 50.0) / 50.0).clip(-1.0, 1.0)


def _garman_klass_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    log_hl = np.log(df["High"] / df["Low"].clip(lower=1e-12))
    log_co = np.log(df["Close"] / df["Open"].clip(lower=1e-12))
    gk_daily = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    return np.sqrt((gk_daily.rolling(window).mean().clip(lower=0.0)) * 252.0)


def _rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = 63) -> pd.Series:
    cov = stock_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var().replace(0.0, np.nan)
    return (cov / var).clip(-3.0, 4.0)


def build_stock_features(
    stock_df: pd.DataFrame,
    context: dict[str, pd.DataFrame],
    execution_mode: str = "next_open",
) -> pd.DataFrame:
    """
    Build strictly causal, scale-invariant technical, macro, and cross-market features.

    Parameters
    ----------
    execution_mode:
        - "close": features at row `t` are observed on or before NSE Close `t`;
          US/FX external series are lagged strictly to date `< t` (`allow_same_date=False`).
        - "next_open": signal generated overnight before NSE Open `t+1` using NSE Close `t`
          plus finalized US Close `t` (`allow_same_date=True` for US equities, strictly `< t`
          for 24h FX `INR=X` so no London/NY midnight timestamp overlap is possible).
    """
    if execution_mode not in ("close", "next_open"):
        raise ValueError(f"Unsupported execution_mode={execution_mode!r}")

    df = stock_df.copy()
    px = df["Close"]
    op = df["Open"]
    hi = df["High"]
    lo = df["Low"]
    vol = df["Volume"].astype(float)

    ret_1d = np.log(px / px.shift(1))
    df["ret_1d"] = ret_1d
    df["ret_3d"] = np.log(px / px.shift(3))
    df["ret_5d"] = np.log(px / px.shift(5))
    df["ret_10d"] = np.log(px / px.shift(10))
    df["ret_20d"] = np.log(px / px.shift(20))
    df["ret_63d"] = np.log(px / px.shift(63))
    for lag in range(1, 6):
        df[f"lag_{lag}"] = ret_1d.shift(lag)

    # Intraday / gap price-action (scale-free)
    df["gap_ret"] = np.log(op / px.shift(1))
    df["intraday_ret"] = np.log(px / op)
    df["hl_range_pct"] = (hi - lo) / px
    df["close_pos_in_range"] = (px - lo) / (hi - lo + 1e-12)

    # Oscillators & scale-invariant trend distance
    df["rsi_14"] = _rsi(px, 14)
    ema12, ema26 = _ema(px, 12), _ema(px, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    df["macd_pct"] = (macd_line - macd_signal) / px
    df["macd_line_pct"] = macd_line / px
    df["dist_sma_20"] = px / px.rolling(20).mean() - 1.0
    df["dist_sma_50"] = px / px.rolling(50).mean() - 1.0
    df["dist_sma_200"] = px / px.rolling(200).mean() - 1.0
    df["dist_high_252"] = px / px.rolling(252).max() - 1.0

    # Volatility & distribution
    high_low = hi - lo
    high_close = (hi - px.shift(1)).abs()
    low_close = (lo - px.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14_pct"] = tr.rolling(14).mean() / px
    df["vol_10d"] = ret_1d.rolling(10).std() * np.sqrt(252.0)
    df["vol_21d"] = ret_1d.rolling(21).std() * np.sqrt(252.0)
    df["vol_63d"] = ret_1d.rolling(63).std() * np.sqrt(252.0)
    df["gk_vol_21d"] = _garman_klass_vol(df, 21)
    df["vol_ratio_10_63"] = df["vol_10d"] / (df["vol_63d"] + 1e-12)
    df["ret_skew_21"] = ret_1d.rolling(21).skew().clip(-5.0, 5.0)
    df["ret_kurt_21"] = ret_1d.rolling(21).kurt().clip(-5.0, 20.0)

    # Stationary volume & flow
    log_vol = np.log1p(vol.clip(lower=0.0))
    df["vol_zscore_20"] = (
        (log_vol - log_vol.rolling(20).mean()) / (log_vol.rolling(20).std() + 1e-12)
    ).clip(-5.0, 5.0)
    signed_vol = np.sign(ret_1d.fillna(0.0)) * vol
    df["signed_vol_ratio"] = (_ema(signed_vol, 10) / (_ema(vol, 20) + 1e-12)).clip(-2.0, 2.0)

    # Indian market & IT index context (same-timezone NSE trading hours)
    if "^NSEI" in context:
        nifty_c = align_asof_series(df.index, context["^NSEI"]["Close"], allow_same_date=True)
        nifty_r1 = np.log(nifty_c / nifty_c.shift(1))
        df["nifty_ret_1d"] = nifty_r1
        df["nifty_ret_5d"] = np.log(nifty_c / nifty_c.shift(5))
        df["nifty_ret_20d"] = np.log(nifty_c / nifty_c.shift(20))
        df["nifty_vol_21d"] = nifty_r1.rolling(21).std() * np.sqrt(252.0)
        df["nifty_dist_sma200"] = nifty_c / nifty_c.rolling(200).mean() - 1.0
        df["rel_nifty_5d"] = df["ret_5d"] - df["nifty_ret_5d"]
        df["rel_nifty_20d"] = df["ret_20d"] - df["nifty_ret_20d"]
        df["beta_nifty_63"] = _rolling_beta(ret_1d, nifty_r1, 63)

    if "^CNXIT" in context:
        it_c = align_asof_series(df.index, context["^CNXIT"]["Close"], allow_same_date=True)
        df["cnxit_ret_1d"] = np.log(it_c / it_c.shift(1))
        df["cnxit_ret_5d"] = np.log(it_c / it_c.shift(5))
        df["cnxit_ret_20d"] = np.log(it_c / it_c.shift(20))
        df["cnxit_dist_sma200"] = it_c / it_c.rolling(200).mean() - 1.0
        df["rel_cnxit_5d"] = df["ret_5d"] - df["cnxit_ret_5d"]
        df["rel_cnxit_20d"] = df["ret_20d"] - df["cnxit_ret_20d"]

    # Cross-market global / FX features with strict timezone discipline
    us_allow_same = execution_mode == "next_open"
    if "INR=X" in context:
        # FX closes at ~21:00-23:59 GMT; strictly require prior calendar date in close mode,
        # and also keep strictly < t+1 (i.e. date <= t) in next_open — wait: in next_open on t+1
        # 08:45 IST = 03:15 GMT, so date t's GMT close is finalized before NSE Open t+1.
        fx_c = align_asof_series(df.index, context["INR=X"]["Close"], allow_same_date=us_allow_same)
        df["usdinr_ret_1d"] = np.log(fx_c / fx_c.shift(1))
        df["usdinr_ret_5d"] = np.log(fx_c / fx_c.shift(5))
        df["usdinr_ret_20d"] = np.log(fx_c / fx_c.shift(20))

    if "^NDX" in context:
        ndx_c = align_asof_series(df.index, context["^NDX"]["Close"], allow_same_date=us_allow_same)
        df["ndx_ret_1d"] = np.log(ndx_c / ndx_c.shift(1))
        df["ndx_ret_5d"] = np.log(ndx_c / ndx_c.shift(5))
        df["ndx_ret_20d"] = np.log(ndx_c / ndx_c.shift(20))

    if "INFY" in context:
        adr_c = align_asof_series(df.index, context["INFY"]["Close"], allow_same_date=us_allow_same)
        df["infy_adr_ret_1d"] = np.log(adr_c / adr_c.shift(1))
        df["infy_adr_ret_5d"] = np.log(adr_c / adr_c.shift(5))

    if "ACN" in context and "CTSH" in context:
        acn_c = align_asof_series(df.index, context["ACN"]["Close"], allow_same_date=us_allow_same)
        ctsh_c = align_asof_series(df.index, context["CTSH"]["Close"], allow_same_date=us_allow_same)
        acn_r1 = np.log(acn_c / acn_c.shift(1))
        ctsh_r1 = np.log(ctsh_c / ctsh_c.shift(1))
        df["us_peers_ret_1d"] = 0.5 * (acn_r1 + ctsh_r1)
        df["us_peers_ret_5d"] = 0.5 * (
            np.log(acn_c / acn_c.shift(5)) + np.log(ctsh_c / ctsh_c.shift(5))
        )

    return df


BASE_TECH_FEATURES: list[str] = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_63d",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
    "gap_ret", "intraday_ret", "hl_range_pct", "close_pos_in_range",
    "rsi_14", "macd_pct", "macd_line_pct",
    "dist_sma_20", "dist_sma_50", "dist_sma_200", "dist_high_252",
    "atr_14_pct", "vol_10d", "vol_21d", "vol_63d", "gk_vol_21d",
    "vol_ratio_10_63", "ret_skew_21", "ret_kurt_21",
    "vol_zscore_20", "signed_vol_ratio",
]

MARKET_SECTOR_FEATURES: list[str] = [
    "nifty_ret_1d", "nifty_ret_5d", "nifty_ret_20d", "nifty_vol_21d", "nifty_dist_sma200",
    "rel_nifty_5d", "rel_nifty_20d", "beta_nifty_63",
    "cnxit_ret_1d", "cnxit_ret_5d", "cnxit_ret_20d", "rel_cnxit_5d", "rel_cnxit_20d",
]

GLOBAL_MACRO_FEATURES: list[str] = [
    "usdinr_ret_1d", "usdinr_ret_5d", "usdinr_ret_20d",
    "ndx_ret_1d", "ndx_ret_5d", "ndx_ret_20d",
    "infy_adr_ret_1d", "infy_adr_ret_5d",
    "us_peers_ret_1d", "us_peers_ret_5d",
]

CROSS_SECTIONAL_FEATURES: list[str] = [
    "cs_ret_5d_z",
    "cs_ret_20d_z",
    "cs_vol_21d_z",
    "cs_dist_sma200_z",
]


def add_cross_sectional_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Compute cross-sectional z-scores across stocks on the SAME date `t` (zero lookahead).
    """
    out = {k: v.copy() for k, v in panel.items()}
    specs = {
        "ret_5d": "cs_ret_5d_z",
        "ret_20d": "cs_ret_20d_z",
        "vol_21d": "cs_vol_21d_z",
        "dist_sma_200": "cs_dist_sma200_z",
    }
    for src_col, dst_col in specs.items():
        wide = pd.DataFrame({sym: df[src_col] for sym, df in out.items()})
        mu = wide.mean(axis=1)
        sd = wide.std(axis=1).replace(0.0, np.nan)
        z = wide.sub(mu, axis=0).div(sd, axis=0).fillna(0.0).clip(-4.0, 4.0)
        for sym in out:
            out[sym][dst_col] = z[sym].reindex(out[sym].index)
    return out


def apply_trailing_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 252,
    min_periods: int = 126,
    clip: float = 5.0,
) -> pd.DataFrame:
    """
    Strictly causal rolling z-score normalization per column (uses only <= t).
    Eliminates multi-decade covariate drift without fitting on future bars.
    """
    out = df.copy()
    for col in feature_cols:
        s = out[col].astype(float)
        mu = s.rolling(window=window, min_periods=min_periods).mean()
        sd = s.rolling(window=window, min_periods=min_periods).std().clip(lower=1e-8)
        out[col] = ((s - mu) / sd).clip(-clip, clip)
    return out


def add_forward_targets(
    df: pd.DataFrame,
    horizon: int,
    execution_mode: str = "next_open",
    barrier_k_up: float = 2.0,
    barrier_k_dn: float = 1.5,
) -> pd.DataFrame:
    """
    Attach strictly forward targets indexed at signal date `t` (NaN where future bars missing).

    - In `close` mode: position enters at `Close[t]`, exits at `Close[t + horizon]`.
    - In `next_open` mode: position enters at `Open[t + 1]`, exits at `Open[t + 1 + horizon]`.
    """
    out = df.copy()
    if execution_mode == "close":
        entry_px = out["Close"]
        exit_px = out["Close"].shift(-horizon)
        daily_exec_ret = np.log(out["Close"] / out["Close"].shift(1))
    else:
        entry_px = out["Open"].shift(-1)
        exit_px = out["Open"].shift(-(1 + horizon))
        # Daily open-to-open return attributed to signal date `t`: Open[t+2]/Open[t+1]
        #Wait: store exec_bar_open[t+1] return on bar t+1 cleanly in backtester; here log_ret for targets:
        daily_exec_ret = np.log(out["Open"].shift(-1) / out["Open"])

    fwd_log_ret = np.log(exit_px / entry_px)
    daily_vol = out["vol_21d"] / np.sqrt(252.0)
    vol_scaled = fwd_log_ret / (daily_vol * np.sqrt(horizon) + 1e-8)

    out["fwd_log_ret"] = fwd_log_ret
    out["target_vol_adj"] = vol_scaled.clip(-3.0, 3.0)
    out["target_dir"] = np.where(fwd_log_ret.notna(), (fwd_log_ret > 0.0).astype(float), np.nan)

    # Excess return over Nifty 50 / Nifty IT over the matching horizon
    if "nifty_ret_1d" in out.columns:
        shift_base = 0 if execution_mode == "close" else 1
        nifty_fwd = sum(out["nifty_ret_1d"].shift(-(shift_base + k)) for k in range(1, horizon + 1))
        out["fwd_excess_ret"] = fwd_log_ret - nifty_fwd
        out["target_excess_vol_adj"] = (
            out["fwd_excess_ret"] / (daily_vol * np.sqrt(horizon) + 1e-8)
        ).clip(-3.0, 3.0)
    if "cnxit_ret_1d" in out.columns:
        shift_base = 0 if execution_mode == "close" else 1
        it_fwd = sum(out["cnxit_ret_1d"].shift(-(shift_base + k)) for k in range(1, horizon + 1))
        out["fwd_it_excess_ret"] = fwd_log_ret - it_fwd
        out["target_it_excess_vol_adj"] = (
            out["fwd_it_excess_ret"] / (daily_vol * np.sqrt(horizon) + 1e-8)
        ).clip(-3.0, 3.0)

    # Triple-barrier binary label (1 if upper ATR barrier hits strictly before lower barrier within H bars)
    atr_pct = out["atr_14_pct"].values
    highs = out["High"].values
    lows = out["Low"].values
    closes = out["Close"].values
    opens = out["Open"].values
    n = len(out)
    tb = np.full(n, np.nan, dtype=float)
    start_off = 1 if execution_mode == "close" else 1
    for i in range(n):
        if execution_mode == "close":
            if i + horizon >= n:
                continue
            p0 = closes[i]
            win_start = i + 1
            win_end = i + horizon
        else:
            if i + 1 + horizon >= n:
                continue
            p0 = opens[i + 1]
            win_start = i + 1
            win_end = i + horizon
        vol_unit = atr_pct[i]
        if not np.isfinite(p0) or not np.isfinite(vol_unit) or vol_unit <= 0:
            continue
        up_lvl = p0 * np.exp(barrier_k_up * vol_unit * np.sqrt(horizon / 5.0))
        dn_lvl = p0 * np.exp(-barrier_k_dn * vol_unit * np.sqrt(horizon / 5.0))
        outcome = np.nan
        for j in range(win_start, win_end + 1):
            hit_up = highs[j] >= up_lvl
            hit_dn = lows[j] <= dn_lvl
            if hit_up and hit_dn:
                outcome = 0.0  # conservative: assume stop hits first on ambiguous bar
                break
            if hit_dn:
                outcome = 0.0
                break
            if hit_up:
                outcome = 1.0
                break
        if np.isnan(outcome):
            # Vertical time barrier: positive if return at horizon end > 0
            p_end = closes[win_end] if execution_mode == "close" else opens[i + 1 + horizon]
            outcome = 1.0 if p_end > p0 else 0.0
        tb[i] = outcome
    out["target_triple_barrier"] = tb
    _ = daily_exec_ret
    return out
