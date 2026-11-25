"""Leakage-free single-asset and sector-portfolio backtest engines + diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class PortfolioConfig:
    cost_bps_per_side: float = 20.0      # NSE cash delivery realistic cost (~20 bps/side)
    rf_annual: float = 0.06              # Conservative Indian 91d T-bill / liquid ETF proxy (~6% p.a.)
    target_annual_vol: float = 0.14      # Annual vol target for scaling exposure
    max_leverage: float = 1.0            # Cash equity delivery: no leverage > 1.0
    vol_window: int = 21
    rebal_deadband: float = 0.20         # Ignore vol-sizing tweaks smaller than 20% NAV
    min_hold_bars: int = 5               # Minimum holding window to align with multi-day signal
    entry_quantile: float = 0.55         # Trailing causal quantile to enter long (regression/cls)
    exit_quantile: float = 0.35          # Hysteresis exit quantile (below -> cash)
    trailing_calib_bars: int = 252       # Strict past-only window for quantile calibration
    stop_loss_atr_mult: float | None = None
    take_profit_atr_mult: float | None = None
    cooldown_bars: int = 3
    use_market_regime_filter: bool = False  # Optional: gate exposure when regime_series < 0
    regime_half_weight: float = 0.50        # Weight multiplier in bear regime (0.0=cash, 0.5=half)


def compute_execution_returns(
    price_df: pd.DataFrame,
    execution_mode: str = "next_open",
) -> pd.Series:
    """
    Compute the 1-bar earnable simple return corresponding to a signal emitted on date `t`.

    - `next_open`: signal at `t` sets position entered at `Open[t+1]`, earning `Open[t+2]/Open[t+1] - 1`.
    - `close`: signal at `t` sets position entered at `Close[t]`, earning `Close[t+1]/Close[t] - 1`.
    """
    if execution_mode == "next_open":
        r = price_df["Open"].shift(-2) / price_df["Open"].shift(-1) - 1.0
    elif execution_mode == "close":
        r = price_df["Close"].shift(-1) / price_df["Close"] - 1.0
    else:
        raise ValueError(f"Unknown execution_mode={execution_mode!r}")
    return r.rename("exec_ret")


def compute_performance_metrics(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    weights: pd.DataFrame | pd.Series,
    rf_annual: float = 0.06,
) -> dict[str, float | int | str]:
    """
    Compute annualized performance, drawdown, active hit-rate, % NAV turnover,
    and both raw (`rf=0`) and institutional risk-free-adjusted (`excess_sharpe`, `cash_adj_cagr`) metrics.
    """
    sr = strat_ret.dropna().astype(float)
    br = bench_ret.reindex(sr.index).fillna(0.0).astype(float)
    if len(sr) < 2:
        raise ValueError("Not enough bars to compute metrics.")

    w = weights.reindex(sr.index).fillna(0.0)
    if isinstance(w, pd.Series):
        w = w.to_frame()

    eq = (1.0 + sr).cumprod()
    bh_eq = (1.0 + br).cumprod()

    years = len(sr) / 252.0
    total_ret = float(eq.iloc[-1] - 1.0)
    bh_total_ret = float(bh_eq.iloc[-1] - 1.0)
    cagr = float((eq.iloc[-1]) ** (1.0 / max(years, 1e-6)) - 1.0)
    bh_cagr = float((bh_eq.iloc[-1]) ** (1.0 / max(years, 1e-6)) - 1.0)

    ann_vol = float(sr.std(ddof=0) * np.sqrt(252.0))
    bh_ann_vol = float(br.std(ddof=0) * np.sqrt(252.0))
    sharpe = float((sr.mean() * 252.0) / (ann_vol + 1e-12))
    bh_sharpe = float((br.mean() * 252.0) / (bh_ann_vol + 1e-12))

    # Institutional risk-free-adjusted accounting: idle cash (1 - gross_exp) earns rf_daily,
    # and Sharpe subtracts rf_daily from both strategy and Buy & Hold identically.
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    gross_exp = w.abs().sum(axis=1)
    idle_cash = (1.0 - gross_exp).clip(lower=0.0)
    sr_with_cash = sr + idle_cash * rf_daily
    eq_cash = (1.0 + sr_with_cash).cumprod()
    cash_adj_cagr = float((eq_cash.iloc[-1]) ** (1.0 / max(years, 1e-6)) - 1.0)
    excess_sr = sr_with_cash - rf_daily
    excess_br = br - rf_daily
    excess_ann_vol = float(sr_with_cash.std(ddof=0) * np.sqrt(252.0))
    excess_sharpe = float((excess_sr.mean() * 252.0) / (excess_ann_vol + 1e-12))
    bh_excess_sharpe = float((excess_br.mean() * 252.0) / (bh_ann_vol + 1e-12))

    downside = sr[sr < 0.0]
    down_vol = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252.0)) if len(downside) > 0 else 0.0
    sortino = float((sr.mean() * 252.0) / (down_vol + 1e-12))

    max_dd = float((eq / eq.cummax() - 1.0).min())
    bh_max_dd = float((bh_eq / bh_eq.cummax() - 1.0).min())
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-9 else 0.0

    active_mask = gross_exp > 1e-6
    exposure_pct = float(active_mask.mean())
    active_sr = sr[active_mask]
    active_hit_rate = float((active_sr > 0.0).mean()) if len(active_sr) > 0 else 0.0

    dw = w.diff().fillna(w.iloc[0]).abs().sum(axis=1)
    total_one_way_turnover = float(dw.sum())
    ann_turnover_pct = float(total_one_way_turnover / max(years, 1e-6))
    rebalance_days = int((dw > 1e-6).sum())

    return {
        "window_start": str(sr.index[0].date()),
        "window_end": str(sr.index[-1].date()),
        "bars": int(len(sr)),
        "total_return": total_ret,
        "bh_total_return": bh_total_ret,
        "cagr": cagr,
        "cash_adj_cagr": cash_adj_cagr,
        "bh_cagr": bh_cagr,
        "ann_vol": ann_vol,
        "bh_ann_vol": bh_ann_vol,
        "sharpe": sharpe,
        "excess_sharpe": excess_sharpe,
        "bh_sharpe": bh_sharpe,
        "bh_excess_sharpe": bh_excess_sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "bh_max_drawdown": bh_max_dd,
        "exposure_pct": exposure_pct,
        "active_hit_rate": active_hit_rate,
        "ann_turnover_nav": ann_turnover_pct,
        "rebalance_days": rebalance_days,
    }


def evaluate_signal_quality(
    preds: pd.Series,
    fwd_returns: pd.Series,
    n_bins: int = 5,
) -> dict[str, float]:
    """Compute Spearman IC and quantile return monotonicity strictly on aligned signal/target pairs."""
    df = pd.concat([preds.rename("p"), fwd_returns.rename("y")], axis=1).dropna()
    if len(df) < 30 or df["p"].nunique() < 3:
        return {"ic_spearman": 0.0, "q_spread": 0.0}
    ic, _ = spearmanr(df["p"], df["y"])
    try:
        bins = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")
        q_means = df.groupby(bins)["y"].mean()
        spread = float(q_means.iloc[-1] - q_means.iloc[0])
    except Exception:
        spread = 0.0
    return {"ic_spearman": float(ic) if np.isfinite(ic) else 0.0, "q_spread": spread}


def run_single_asset_backtest(
    price_df: pd.DataFrame,
    signal_series: pd.Series,
    cfg: PortfolioConfig,
    execution_mode: str = "next_open",
    prior_signals: pd.Series | None = None,
    regime_series: pd.Series | None = None,
    signal_mode: str = "quantile_hysteresis",
    fixed_entry_thresh: float = 0.0,
    fixed_exit_thresh: float = -0.10,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """
    Strictly causal single-asset backtest (no future quantile peeking, no stop-loss overwrite).
    """
    exec_ret = compute_execution_returns(price_df, execution_mode=execution_mode)
    idx = signal_series.index.intersection(exec_ret.dropna().index)
    sig_vals = signal_series.reindex(idx).astype(float)
    ret_vals = exec_ret.reindex(idx).astype(float)

    close_ret = np.log(price_df["Close"] / price_df["Close"].shift(1))
    roll_vol = (close_ret.rolling(cfg.vol_window).std() * np.sqrt(252.0)).reindex(idx).ffill().fillna(0.20)
    atr_pct = (
        (price_df["High"] - price_df["Low"]).rolling(14).mean() / price_df["Close"]
    ).reindex(idx).ffill().fillna(0.02)

    if prior_signals is not None and len(prior_signals) > 0:
        combined_sig = pd.concat([prior_signals.astype(float), sig_vals])
        combined_sig = combined_sig[~combined_sig.index.duplicated(keep="last")].sort_index()
    else:
        combined_sig = sig_vals

    q_hi = (
        combined_sig.shift(1)
        .rolling(cfg.trailing_calib_bars, min_periods=30)
        .quantile(cfg.entry_quantile)
        .reindex(idx)
    )
    q_lo = (
        combined_sig.shift(1)
        .rolling(cfg.trailing_calib_bars, min_periods=30)
        .quantile(cfg.exit_quantile)
        .reindex(idx)
    )

    n = len(idx)
    weights = np.zeros(n, dtype=float)
    raw_state = 0
    hold_bars = 0
    cooldown = 0
    entry_close = np.nan
    prev_w = 0.0
    closes = price_df["Close"].reindex(idx).to_numpy(dtype=float)

    for i in range(n):
        s_i = float(sig_vals.iloc[i])

        stopped = False
        if raw_state == 1 and i > 0:
            hold_bars += 1
            trade_log_ret = np.log(max(closes[i], 1e-12) / max(entry_close, 1e-12))
            atr_i = float(atr_pct.iloc[i])
            if cfg.stop_loss_atr_mult is not None and trade_log_ret <= -cfg.stop_loss_atr_mult * atr_i:
                raw_state = 0
                hold_bars = 0
                cooldown = cfg.cooldown_bars
                stopped = True
            elif cfg.take_profit_atr_mult is not None and trade_log_ret >= cfg.take_profit_atr_mult * atr_i:
                raw_state = 0
                hold_bars = 0
                cooldown = 1
                stopped = True

        if cooldown > 0:
            cooldown -= 1

        if not stopped and cooldown == 0:
            if signal_mode == "quantile_hysteresis":
                hi_i = float(q_hi.iloc[i]) if np.isfinite(q_hi.iloc[i]) else fixed_entry_thresh
                lo_i = float(q_lo.iloc[i]) if np.isfinite(q_lo.iloc[i]) else fixed_exit_thresh
            else:
                hi_i = fixed_entry_thresh
                lo_i = fixed_exit_thresh

            if raw_state == 0:
                if s_i >= hi_i:
                    raw_state = 1
                    hold_bars = 0
                    entry_close = closes[i]
            else:
                if hold_bars >= cfg.min_hold_bars and s_i <= lo_i:
                    raw_state = 0
                    hold_bars = 0
                    entry_close = np.nan

        if raw_state == 0:
            target_w = 0.0
        else:
            vol_i = max(float(roll_vol.iloc[i]), 1e-4)
            target_w = min(cfg.target_annual_vol / vol_i, cfg.max_leverage)
            if cfg.use_market_regime_filter and regime_series is not None:
                reg_val = float(regime_series.reindex(idx).iloc[i])
                if np.isfinite(reg_val) and reg_val < 0.0:
                    target_w *= cfg.regime_half_weight

        if prev_w > 0.0 and target_w > 0.0 and abs(target_w - prev_w) < cfg.rebal_deadband:
            w_i = prev_w
        else:
            w_i = target_w

        weights[i] = w_i
        prev_w = w_i

    w_ser = pd.Series(weights, index=idx, name="weight")
    dw = w_ser.diff().fillna(w_ser.iloc[0]).abs()
    cost = (cfg.cost_bps_per_side / 1e4) * dw
    gross_ret = w_ser * ret_vals
    net_ret = gross_ret - cost
    bh_ret = ret_vals.copy()

    out = pd.DataFrame(
        {
            "signal": sig_vals,
            "weight": w_ser,
            "exec_ret": ret_vals,
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "bh_ret": bh_ret,
            "strategy_equity": (1.0 + net_ret).cumprod(),
            "bh_equity": (1.0 + bh_ret).cumprod(),
        },
        index=idx,
    )
    summary = compute_performance_metrics(net_ret, bh_ret, w_ser, rf_annual=cfg.rf_annual)
    return out, summary


def run_panel_topk_backtest(
    stocks: dict[str, pd.DataFrame],
    pred_panel: pd.DataFrame,
    cfg: PortfolioConfig,
    top_k: int = 3,
    rebalance_every: int = 5,
    execution_mode: str = "next_open",
    regime_series: pd.Series | None = None,
    min_score: float = -np.inf,
    execution_style: str = "periodic",
    sleeve_horizon: int = 10,
    keep_rank: int = 4,
    prior_pred_panel: pd.DataFrame | None = None,
    prior_regime_series: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """
    Strictly causal cross-sectional Top-K long-only IT sector portfolio.

    `execution_style`:
      - "periodic": re-ranks every `rebalance_every` bars, holding top `top_k`.
      - "overlapping_sleeves": allocates `1 / sleeve_horizon` daily to that day's Top-K
        basket and holds each sleeve for `sleeve_horizon` days (rolling causal average of
        daily Top-K target weights — zero new parameters, eliminates weekly churn spikes).
      - "rank_hysteresis": holds existing positions as long as their cross-sectional rank
        is `<= keep_rank` (`4`), filling open slots with highest-ranked non-held stocks.
    """
    if execution_style not in ("periodic", "overlapping_sleeves", "rank_hysteresis"):
        raise ValueError(f"Unknown execution_style={execution_style!r}")

    tickers = list(pred_panel.columns)
    exec_rets = pd.DataFrame(
        {sym: compute_execution_returns(stocks[sym], execution_mode=execution_mode) for sym in tickers}
    )
    idx = pred_panel.index.intersection(exec_rets.dropna(how="any").index)
    preds = pred_panel.reindex(idx)
    rets = exec_rets.reindex(idx)

    # Portfolio-level vol scaling strictly from finalized Close-to-Close basket returns <= t
    close_rets = pd.DataFrame(
        {sym: np.log(stocks[sym]["Close"] / stocks[sym]["Close"].shift(1)) for sym in tickers}
    )
    ew_close_ret = close_rets.mean(axis=1)
    basket_vol_full = (ew_close_ret.rolling(cfg.vol_window).std() * np.sqrt(252.0)).ffill().fillna(0.20)
    ew_ret = rets.mean(axis=1)

    if execution_style == "overlapping_sleeves":
        if prior_pred_panel is not None and len(prior_pred_panel) > 0:
            full_preds = pd.concat([prior_pred_panel[tickers], preds])
            full_preds = full_preds[~full_preds.index.duplicated(keep="last")].sort_index()
        else:
            full_preds = preds
        if prior_regime_series is not None and regime_series is not None:
            full_reg = pd.concat([prior_regime_series, regime_series])
            full_reg = full_reg[~full_reg.index.duplicated(keep="last")].sort_index()
        else:
            full_reg = regime_series

        daily_targets = pd.DataFrame(0.0, index=full_preds.index, columns=tickers)
        for dt in full_preds.index:
            row_scores = full_preds.loc[dt].to_numpy(dtype=float)
            valid_mask = np.isfinite(row_scores) & (row_scores > min_score)
            vol_dt = float(basket_vol_full.reindex([dt]).ffill().fillna(0.20).iloc[0])
            gross_cap = min(cfg.target_annual_vol / max(vol_dt, 1e-4), cfg.max_leverage)
            if cfg.use_market_regime_filter and full_reg is not None:
                reg_val = float(full_reg.reindex([dt]).iloc[0])
                if np.isfinite(reg_val) and reg_val < 0.0:
                    gross_cap *= cfg.regime_half_weight
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 0 and gross_cap > 0.0:
                chosen = valid_idx[np.argsort(row_scores[valid_idx])[::-1][:top_k]]
                daily_targets.loc[dt, [tickers[j] for j in chosen]] = gross_cap / len(chosen)

        sleeve_weights = daily_targets.rolling(sleeve_horizon, min_periods=1).mean().reindex(idx)
        weights = pd.DataFrame(0.0, index=idx, columns=tickers)
        current_w = np.zeros(len(tickers), dtype=float)
        for i in range(len(idx)):
            target_w = sleeve_weights.iloc[i].to_numpy(dtype=float)
            if i == 0 or np.abs(target_w - current_w).sum() >= cfg.rebal_deadband:
                current_w = target_w
            weights.iloc[i] = current_w
    else:
        basket_vol = basket_vol_full.reindex(idx).ffill().fillna(0.20)
        weights = pd.DataFrame(0.0, index=idx, columns=tickers)
        current_w = np.zeros(len(tickers), dtype=float)
        held_indices: list[int] = []

        for i in range(len(idx)):
            if i % rebalance_every == 0:
                row_scores = preds.iloc[i].to_numpy(dtype=float)
                valid_mask = np.isfinite(row_scores) & (row_scores > min_score)
                gross_cap = min(cfg.target_annual_vol / max(float(basket_vol.iloc[i]), 1e-4), cfg.max_leverage)
                if cfg.use_market_regime_filter and regime_series is not None:
                    reg_val = float(regime_series.reindex(idx).iloc[i])
                    if np.isfinite(reg_val) and reg_val < 0.0:
                        gross_cap *= cfg.regime_half_weight

                target_w = np.zeros(len(tickers), dtype=float)
                valid_idx = np.where(valid_mask)[0]
                if len(valid_idx) > 0 and gross_cap > 0.0:
                    order = valid_idx[np.argsort(row_scores[valid_idx])[::-1]]
                    if execution_style == "rank_hysteresis":
                        rank_map = {int(sym_idx): rank_pos + 1 for rank_pos, sym_idx in enumerate(order)}
                        kept = [idx_j for idx_j in held_indices if rank_map.get(idx_j, 999) <= keep_rank]
                        for cand in order:
                            if len(kept) >= top_k:
                                break
                            if int(cand) not in kept:
                                kept.append(int(cand))
                        chosen = np.array(kept[:top_k], dtype=int)
                        held_indices = list(chosen)
                    else:
                        chosen = order[:top_k]
                    target_w[chosen] = gross_cap / len(chosen)
                else:
                    held_indices = []

                if np.abs(target_w - current_w).sum() >= cfg.rebal_deadband:
                    current_w = target_w
            weights.iloc[i] = current_w

    dw = weights.diff().fillna(weights.iloc[0]).abs().sum(axis=1)
    cost = (cfg.cost_bps_per_side / 1e4) * dw
    gross_ret = (weights * rets).sum(axis=1)
    net_ret = gross_ret - cost

    bh_infy = rets["INFY.NS"] if "INFY.NS" in rets.columns else ew_ret
    out = pd.DataFrame(
        {
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "bh_infy_ret": bh_infy,
            "ew_sector_ret": ew_ret,
            "strategy_equity": (1.0 + net_ret).cumprod(),
            "bh_equity": (1.0 + bh_infy).cumprod(),
            "ew_sector_equity": (1.0 + ew_ret).cumprod(),
        },
        index=idx,
    )
    summary = compute_performance_metrics(net_ret, bh_infy, weights, rf_annual=cfg.rf_annual)
    ew_bh_eq = (1.0 + ew_ret).cumprod()
    ew_years = len(ew_ret) / 252.0
    summary["ew_sector_cagr"] = float(ew_bh_eq.iloc[-1] ** (1.0 / max(ew_years, 1e-6)) - 1.0)
    summary["ew_sector_sharpe"] = float(
        (ew_ret.mean() * 252.0) / (ew_ret.std(ddof=0) * np.sqrt(252.0) + 1e-12)
    )
    rf_daily = (1.0 + cfg.rf_annual) ** (1.0 / 252.0) - 1.0
    summary["ew_sector_excess_sharpe"] = float(
        ((ew_ret - rf_daily).mean() * 252.0) / (ew_ret.std(ddof=0) * np.sqrt(252.0) + 1e-12)
    )
    return out, summary
