"""Leakage-free market data loading, caching, and sanity verification."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

NSE_IT_UNIVERSE: tuple[str, ...] = (
    "INFY.NS",
    "TCS.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "COFORGE.NS",
    "MPHASIS.NS",
)

CONTEXT_TICKERS: tuple[str, ...] = (
    "^NSEI",
    "^CNXIT",
    "INR=X",
    "^NDX",
    "INFY",
    "ACN",
    "CTSH",
)

DEFAULT_START = "2007-09-17"  # First date with native ^NSEI and ^CNXIT coverage
DEFAULT_END = "2025-01-01"    # Exclusive end in yfinance -> covers through 2024-12-31


def _flatten_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to single-level OHLCV columns indexed by UTC-naive date."""
    if raw is None or raw.empty:
        raise ValueError("Downloaded DataFrame is empty.")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0.0)
    return df


def validate_ohlcv(
    df: pd.DataFrame,
    ticker: str,
    max_abs_log_ret: float = 0.40,
) -> None:
    """Assert corporate-action integrity and absence of corrupted price spikes."""
    if df.empty:
        raise ValueError(f"{ticker}: empty DataFrame.")
    if not df.index.is_monotonic_increasing:
        raise AssertionError(f"{ticker}: index is not monotonic increasing.")
    if df.index.has_duplicates:
        raise AssertionError(f"{ticker}: index contains duplicate timestamps.")
    for col in ("Open", "High", "Low", "Close"):
        if (df[col] <= 0).any():
            raise AssertionError(f"{ticker}: non-positive values found in {col}.")
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    worst = float(log_ret.abs().max())
    if worst >= max_abs_log_ret:
        worst_dt = log_ret.abs().idxmax()
        raise AssertionError(
            f"{ticker}: abnormal single-day |log_ret|={worst:.4f} on {worst_dt.date()} "
            f"(>= {max_abs_log_ret}); possible unadjusted split/data error."
        )


def load_ticker(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Path | str | None = Path("data/cache"),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load total-return (auto_adjust=True) daily OHLCV for a ticker, caching locally."""
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        safe_name = ticker.replace("^", "IDX_").replace("=", "_")
        cache_path = cache_root / f"{safe_name}_{start}_{end}.csv"
        if cache_path.exists() and not force_refresh:
            df = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            validate_ohlcv(df, ticker)
            return df

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = _flatten_ohlcv(raw)
    validate_ohlcv(df, ticker)
    if cache_path is not None:
        df.to_csv(cache_path)
    return df


def align_asof_series(
    target_index: pd.DatetimeIndex,
    source_series: pd.Series,
    allow_same_date: bool,
    max_staleness_days: int = 10,
) -> pd.Series:
    """
    Strictly causal as-of alignment onto `target_index`.

    Parameters
    ----------
    allow_same_date:
        - False: only matches source dates strictly `< target_date` (use for US/FX when
          generating signals on NSE date `t` before or at NSE close `t`).
        - True: matches source dates `<= target_date` (use for Indian indices trading
          simultaneously with NSE, or US/FX when generating signals before NSE Open `t+1`).
    """
    tgt = pd.DataFrame({"date": pd.DatetimeIndex(target_index)}).sort_values("date")
    src = (
        source_series.dropna()
        .rename("val")
        .to_frame()
        .reset_index()
    )
    src.columns = ["src_date", "val"]
    src["src_date"] = pd.to_datetime(src["src_date"]).dt.tz_localize(None)
    src = src.sort_values("src_date")

    merged = pd.merge_asof(
        tgt,
        src,
        left_on="date",
        right_on="src_date",
        direction="backward",
        allow_exact_matches=allow_same_date,
        tolerance=pd.Timedelta(days=max_staleness_days),
    )
    out = pd.Series(merged["val"].values, index=tgt["date"], name=source_series.name)
    return out.reindex(target_index)


def load_market_bundle(
    tickers: Iterable[str] = NSE_IT_UNIVERSE,
    context_tickers: Iterable[str] = CONTEXT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Path | str | None = Path("data/cache"),
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Download/load and validate all NSE IT stocks and macro/sector context series."""
    stocks: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        stocks[ticker] = load_ticker(
            ticker, start=start, end=end, cache_dir=cache_dir, force_refresh=force_refresh
        )
    context: dict[str, pd.DataFrame] = {}
    for ticker in context_tickers:
        context[ticker] = load_ticker(
            ticker, start=start, end=end, cache_dir=cache_dir, force_refresh=force_refresh
        )
    return stocks, context
