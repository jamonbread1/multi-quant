# -*- coding: utf-8 -*-
"""杠杆倍数 vs 年化 研究 v1.0 (S1 海龟, 引擎 v1.1: 含强平 + 盘中回撤)

问题: 杠杆倍数如何影响年化收益, 代价是多少 (盘中回撤、强平风险)?
预先登记的杠杆档位 (不做网格寻优): 1.0 / 1.5(冻结版) / 2.0 / 3.0 / 5.0 / 10.0
口径:
  - 净口径: 权益复利扣 taker 0.05%/边 + 资金费 0.004%/8h
  - 逐仓强平 MMR 0.5%, 强平价 = 入场 * (1 -/+ (1/LEV - MMR))
  - 回撤: 盘中最差 MTM (eq_worst) 与 平仓权益 (eq_curve) 两种口径都报告
  - 全样本: 从 2020-01-27 (引擎预热后首个交易日) 起 30U
  - OOS: 从 2024-01-01 起 30U 重启 (预热数据取切分前 198 根)
数据: research/oos/data/eth4h_from_fmz_1h.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_engine_v1_1 as E  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01")
START_U = 30.0
LEVS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
WARM = 198


def load_k():
    k = pd.read_csv(os.path.join(HERE, "data", "eth4h_from_fmz_1h.csv"))
    k["dt"] = pd.to_datetime(k["dt"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]].reset_index(drop=True)


def _mdd(series):
    """回撤序列: 仅统计有效 (非 NaN) 区间; 峰值 <=0 之后视为已爆仓, 回撤记为 -100%."""
    x = np.asarray(series, dtype=float)
    valid = ~np.isnan(x)
    x = np.maximum(x[valid], 0.0)  # 权益不能为负 (强平后归零)
    peak = np.maximum.accumulate(x)
    dd = np.where(peak > 0, x / np.where(peak > 0, peak, 1.0) - 1.0, -1.0)
    return float(dd.min())


def stats(tr, eq_curve, eq_worst, n_liq, years):
    eq_end = float(eq_curve[-1])
    mdd_real = _mdd(eq_curve)
    mdd_intra = _mdd(eq_worst)
    mult = eq_end / START_U
    cagr = (mult ** (1 / years) - 1) if mult > 0 else -1.0
    calmar = cagr / abs(mdd_intra) if mdd_intra < 0 else np.nan
    return dict(end_u=eq_end, mult=mult, cagr=cagr, mdd_real=mdd_real, mdd_intra=mdd_intra,
                calmar=calmar, n_liq=n_liq, n_trades=len(tr),
                ruined=bool(eq_end <= 0.0))


def run_window(k, start_idx, lev, trade_from=None):
    kk = k.iloc[start_idx:].reset_index(drop=True)
    tr, eq, eqw = E.run(kk, don_n=48, lev=lev)
    n_liq = getattr(E.run, "last_n_liq", 0)
    if trade_from is not None and len(tr):
        tr = tr[pd.to_datetime(tr.entry_time) >= trade_from]
    first = pd.to_datetime(tr.entry_time).min() if len(tr) else kk.dt.iloc[WARM]
    last = pd.to_datetime(tr.exit_time).max() if len(tr) else kk.dt.iloc[-1]
    years = max((last - first).days / 365.25, 1e-9)
    return tr, eq, eqw, n_liq, years


def main():
    k = load_k()
    split_idx = int(k.index[k.dt >= SPLIT][0])
    rows = []
    for lev in LEVS:
        tr, eq, eqw, n_liq, years = run_window(k, 0, lev)
        full = stats(tr, eq, eqw, n_liq, years)
        # OOS 窗口: 从切分前 WARM 根开始, 使引擎恰好在切分日开始交易 (30U 重启)
        tro, eqo, eqwo, nlo, yo = run_window(k, split_idx - WARM, lev, trade_from=SPLIT)
        oos = stats(tro, eqo, eqwo, nlo, yo)
        rows.append(dict(lev=lev, **{f"full_{x}": y for x, y in full.items()},
                         **{f"oos_{x}": y for x, y in oos.items()}))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(HERE, "leverage_summary_v1_0.csv"), index=False, encoding="utf-8")

    def pct(x):
        return f"{x*100:+.1f}%"

    L = ["# 杠杆倍数 vs 年化 研究报告 v1.0\n",
         "**脚本**: `research/oos/leverage_v1_0.py`；引擎 `s1_engine_v1_1.py`（v1.0 的执行语义 + 强平 + 盘中回撤）  ",
         "**口径**: 净口径（扣 0.05%/边 + 资金费），逐仓强平 MMR 0.5%，30U 起始；OOS 为 2024-01-01 起重启  ",
         "**回撤**: `盘中` = 持仓期间最差 MTM（主口径）；`平仓` = 仅平仓后权益（v1.0 口径，偏乐观）\n",
         "## 1. 全样本（2020-01 ~ 2026-10）\n",
         "| 杠杆 | 期末 | 年化 | 盘中最大回撤 | 平仓口径回撤 | Calmar | 强平次数 | 是否爆仓 |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        L.append(f"| {r.lev:.1f}x | {r.full_end_u:.1f}U ({r.full_mult:.2f}x) | {pct(r.full_cagr)} | "
                 f"{r.full_mdd_intra*100:.1f}% | {r.full_mdd_real*100:.1f}% | {r.full_calmar:.2f} | "
                 f"{int(r.full_n_liq)} | {'是' if r.full_ruined else '否'} |")
    L.append("\n## 2. 样本外（2024-01 ~ 2026-10，30U 重启）\n")
    L.append("| 杠杆 | 期末 | 年化 | 盘中最大回撤 | Calmar | 强平次数 | 是否爆仓 |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in S.iterrows():
        L.append(f"| {r.lev:.1f}x | {r.oos_end_u:.1f}U ({r.oos_mult:.2f}x) | {pct(r.oos_cagr)} | "
                 f"{r.oos_mdd_intra*100:.1f}% | {r.oos_calmar:.2f} | {int(r.oos_n_liq)} | {'是' if r.oos_ruined else '否'} |")

    # 作图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(S.lev, S.full_cagr * 100, "o-", label="full sample")
        ax[0].plot(S.lev, S.oos_cagr * 100, "s--", label="OOS 2024+")
        ax[0].set_xlabel("leverage (x)")
        ax[0].set_ylabel("annualized return (%)")
        ax[0].set_title("Annualized return vs leverage (net)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(S.lev, S.full_mdd_intra * 100, "o-", color="tab:red", label="full, intrabar MDD")
        ax[1].plot(S.lev, S.oos_mdd_intra * 100, "s--", color="tab:orange", label="OOS, intrabar MDD")
        ax[1].set_xlabel("leverage (x)")
        ax[1].set_ylabel("max drawdown (%)")
        ax[1].set_title("Max drawdown vs leverage")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "leverage_curve_v1_0.png"), dpi=110)
        L.append("\n![leverage](leverage_curve_v1_0.png)\n")
    except Exception as e:  # noqa: BLE001
        L.append(f"\n(作图失败: {e})\n")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "leverage_report_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
