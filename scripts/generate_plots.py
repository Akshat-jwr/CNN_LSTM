"""Generate publication-quality OOS equity, drawdown, cost-sensitivity, rank/quintile, and yearly plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def plot_equity_and_drawdown() -> None:
    eq = pd.read_csv(RESULTS / "oos_equity_curves.csv", parse_dates=["date"], index_col="date")
    lb = pd.read_csv(RESULTS / "stage2_oos_leaderboard.csv").set_index("exp_id")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    sh_per = lb.loc["P2_H10_excess_nifty__Top3_Periodic_NiftyReg", "sharpe"]
    sh_lgb = lb.loc["Anchor_LightGBM_H10__Top3_Periodic_NiftyReg", "sharpe"]
    sh_s1 = lb.loc["S1_H5_dir_cls__INFY", "sharpe"]
    sh_ew = lb.loc["Benchmark_EW_7IT", "sharpe"]
    sh_sma = lb.loc["Benchmark_INFY_200SMA", "sharpe"]
    sh_bh = lb.loc["Benchmark_INFY_200SMA", "bh_sharpe"]

    series_specs = [
        ("P2_H10_excess_nifty_Top3_Periodic_NiftyReg", f"CNN-LSTM Panel Top-3 + Nifty Regime (Sh={sh_per:+.2f})", "#0b5394", "-", 2.4),
        ("Anchor_LightGBM_H10__Top3_Periodic_NiftyReg", f"LightGBM Anchor Top-3 + Nifty Regime (Sh={sh_lgb:+.2f})", "#ff7f0e", "-.", 1.6),
        ("S1_H5_dir_cls_INFY", f"CNN-LSTM Single-Stock INFY (H=5, Sh={sh_s1:+.2f})", "#2ca02c", "-", 2.0),
        ("EW_7IT_Sector", f"Equal-Weight 7-IT Benchmark (Sh={sh_ew:+.2f})", "#17becf", "--", 1.5),
        ("Benchmark_INFY_200SMA", f"200-SMA Trend Benchmark INFY (Sh={sh_sma:+.2f})", "#9467bd", ":", 1.5),
        ("INFY_BuyHold", f"INFY Buy & Hold Benchmark (Sh={sh_bh:+.2f})", "#555555", "--", 1.7),
    ]
    for col, label, color, ls, lw in series_specs:
        if col in eq.columns:
            s = eq[col].dropna()
            ax1.plot(s.index, s.values, label=label, color=color, linestyle=ls, linewidth=lw)
            dd = s / s.cummax() - 1.0
            ax2.plot(dd.index, dd.values * 100.0, label=label, color=color, linestyle=ls, linewidth=1.2)

    ax1.set_title("True Out-of-Sample Walk-Forward Performance (2022-2024, Quarterly Retrain, 20 bps/side)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative Equity (Rebased to 1.0)")
    ax1.legend(loc="upper left", fontsize=8.8, ncol=2)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(RESULTS / "oos_equity_curves.png", dpi=220)
    plt.close(fig)


def plot_cost_sensitivity() -> None:
    df = pd.read_csv(RESULTS / "oos_cost_sensitivity.csv")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for model_name, sub in df.groupby("model"):
        ax.plot(sub["bps_per_side"], sub["hyst_cnxit_sharpe"], marker="o", linewidth=2.2, label=f"{model_name} (Rank-Hyst + CNXIT Regime)")
        ax.plot(sub["bps_per_side"], sub["periodic_nifty_sharpe"], marker="s", linestyle="--", linewidth=1.8, label=f"{model_name} (Periodic + Nifty Regime)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_title("OOS Sharpe Sensitivity to Transaction Costs (2022-2024)", fontsize=12, fontweight="bold")
    ax.set_xlabel("One-Way Transaction Cost (bps per side)")
    ax.set_ylabel("Annualized Sharpe Ratio (rf=0)")
    ax.set_xticks([0, 10, 20, 30])
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "oos_cost_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_quintiles() -> None:
    rk = pd.read_csv(RESULTS / "oos_cs_ranks.csv")
    q = pd.read_csv(RESULTS / "oos_quintiles.csv")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    sub_r = rk[rk["model"] == "P2_H10_excess_nifty"]
    colors_r = ["#d62728" if v < 0 else "#0b5394" for v in sub_r["mean_excess_vs_it"]]
    axes[0].bar(sub_r["rank"].astype(str), sub_r["mean_excess_vs_it"] * 100.0, color=colors_r, alpha=0.88, edgecolor="black")
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_title("P2_H10: Cross-Sectional Rank 1..7 Excess vs IT Mean (%)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Daily Predicted Rank (1 = Best, 7 = Worst)")
    axes[0].set_ylabel("Mean 10d Excess Log Return (%)")

    sub_q = q[q["model"] == "P2_H10_excess_nifty_Panel"]
    colors_q = ["#d62728" if v < 0 else "#1f77b4" for v in sub_q["mean"]]
    axes[1].bar(sub_q["quintile"], sub_q["mean"] * 100.0, color=colors_q, alpha=0.88, edgecolor="black")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_title("P2_H10: Pooled Panel Quintile Forward Return (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mean 10d Forward Log Return (%)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("OOS Cross-Sectional Rank & Quintile Monotonicity (2022-2024)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "oos_quintiles.png", dpi=200)
    plt.close(fig)


def plot_yearly_breakdown() -> None:
    df = pd.read_csv(RESULTS / "oos_yearly_breakdown.csv")
    keep = ["P2_H10_excess_nifty_Top3_NiftyReg", "S1_H5_dir_cls_INFY"]
    sub = df[df["model"].isin(keep)]
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    rename_map = {
        "P2_H10_excess_nifty_Top3_NiftyReg": "Panel Top-3 + Nifty Regime (H=10)",
        "S1_H5_dir_cls_INFY": "Single INFY (H=5)",
    }

    piv_ret = sub.pivot(index="year", columns="model", values="return") * 100.0
    piv_ret["INFY_BuyHold"] = sub.groupby("year")["bh_return"].first() * 100.0
    piv_ret = piv_ret.rename(columns=rename_map)
    piv_ret.plot(kind="bar", ax=ax1, edgecolor="black", color=["#0b5394", "#2ca02c", "#777777"])
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_title("OOS Annual Net Return (%) vs INFY Buy & Hold", fontweight="bold")
    ax1.set_ylabel("Return (%)")
    ax1.tick_params(axis="x", rotation=0)

    piv_sh = sub.pivot(index="year", columns="model", values="sharpe")
    piv_sh["INFY_BuyHold"] = sub.groupby("year")["bh_sharpe"].first()
    piv_sh = piv_sh.rename(columns=rename_map)
    piv_sh.plot(kind="bar", ax=ax2, edgecolor="black", color=["#0b5394", "#2ca02c", "#777777"])
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_title("OOS Annual Sharpe Ratio vs INFY Buy & Hold", fontweight="bold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(RESULTS / "oos_yearly_breakdown.png", dpi=200)
    plt.close(fig)


def main() -> None:
    plot_equity_and_drawdown()
    plot_cost_sensitivity()
    plot_quintiles()
    plot_yearly_breakdown()
    print("Generated all OOS diagnostic plots in results/.")


if __name__ == "__main__":
    main()
