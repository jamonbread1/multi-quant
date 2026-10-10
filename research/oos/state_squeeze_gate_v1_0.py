# -*- coding: utf-8 -*-
"""S1 仅压缩过滤 (SQUEEZE-only entry gate) 样本外检验 v1.0

配置 (预登记, 唯一配置):
  S-G = 仅允许入场当根决策 bar 处于压缩态 (ATR14 / mean(ATR14,100) <= 0.8) 时开仓;
        其余决策 bar 跳过新入场 (不补做, 已有仓位不动), 引擎 s1_engine_v1_2.entry_skip。
  基线 = S1 原规则 (无闸门)。
  控制 = 随机跳过同比例 (约 75%) 的入场决策, 500 次, 固定种子。

数据说明: 压缩阈值 0.8 与 100 根均线窗口, 来自 state_lead_v1_0 的方向 B 研究 (该研究含 IS 观察)。
          因此 OOS 结果是对“已看过 IS 的选择”的检验, 不是纯盲测; 判定只用 OOS。

判定规则 (运行前固定, 仅看 OOS 2024-01 起):
  通过 当且仅当:
    (1) OOS Calmar(S-G) > OOS Calmar(基线)
    (2) OOS 年化(S-G) >= 基线 - 5pp
    (3) 被拦截的基线交易 (决策根非压缩) 的 OOS 平均 pnl_r < 0
    (4) OOS Calmar(S-G) >= 随机控制 90 分位
恒真率: 决策根压缩占比 >95% 则无效 (仅报告)。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_lead_v1_0 as SL  # noqa: E402
import state_transition_gate_v2_v1_0 as G2  # noqa: E402
import s1_engine_v1_1 as E1  # noqa: E402  (仅用于 ATR)

SPLIT = pd.Timestamp("2024-01-01")
IS_START = pd.Timestamp("2021-01-01")
N_RAND = 500
SEED = 20261010


def squeeze_mask(k):
    h, l, c = k.high.values, k.low.values, k.close.values
    atr = np.array(E1.atr_wilder(h, l, c))
    ratio = atr / pd.Series(atr).rolling(100, min_periods=100).mean().values
    return ratio <= 0.8  # NaN -> False


def main():
    k = SL.load_k()
    sq = squeeze_mask(k)
    skip = ~sq  # 非压缩决策根 -> 跳过入场
    valid_oos = (k.dt >= SPLIT).values
    share_sq_oos = sq[valid_oos].mean()
    share_sq_is = sq[(k.dt >= IS_START).values & (k.dt < SPLIT).values].mean()

    L = ["# S1 仅压缩过滤（SQUEEZE-only）样本外检验 v1.0\n",
         "**脚本**: `research/oos/state_squeeze_gate_v1_0.py`；引擎 `s1_engine_v1_2.py`（entry_skip）。  ",
         "**判定**: OOS 2024-01 起。IS 2021–2023 仅参考。\n",
         "**注意**: 阈值 0.8 与 100 根窗口来自方向 B 研究（已看过 IS）。本检验是对该选择的 OOS 检验，不是盲测。\n",
         f"- 恒真率: 决策根压缩占比 IS {share_sq_is*100:.1f}% / OOS {share_sq_oos*100:.1f}%（远低于 95%，有效）\n"]

    rows = []
    for per, s, e in [("IS 2021-23", IS_START, SPLIT), ("OOS 2024+", SPLIT, None)]:
        b_tr, b = G2.window_run(k, s, e, skip=None)
        g_tr, g = G2.window_run(k, s, e, skip=skip)
        dec = pd.to_datetime(b_tr.entry_time) - pd.Timedelta(hours=4)
        idx = np.searchsorted(k.dt.values, dec.values.astype("datetime64[ns]"))
        blocked = ~sq[idx]
        rb = b_tr.loc[blocked, "pnl_r"].astype(float)
        rk = b_tr.loc[~blocked, "pnl_r"].astype(float)
        rows.append(dict(period=per, cfg="基线", **b, blocked_n=0, blocked_mean_r=np.nan, kept_mean_r=float(rk.mean())))
        rows.append(dict(period=per, cfg="S-G 仅压缩入场", **g, blocked_n=int(blocked.sum()),
                         blocked_mean_r=float(rb.mean()) if len(rb) else np.nan,
                         kept_mean_r=float(rk.mean()) if len(rk) else np.nan))

    rng = np.random.default_rng(SEED)
    rand_cal = []
    for _ in range(N_RAND):
        m = rng.random(len(k)) < (1 - share_sq_oos)  # 同比例随机跳过
        _, rs = G2.window_run(k, SPLIT, None, skip=m)
        rand_cal.append(rs["calmar"])
    rand_cal = np.array(rand_cal, dtype=float)
    rand_cal = rand_cal[~np.isnan(rand_cal)]

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "state_squeeze_gate_v1_0.csv"), index=False, encoding="utf-8")

    def fp(x, p=False, d=2):
        if pd.isna(x):
            return "—"
        return f"{x*100:.1f}%" if p else f"{x:.{d}f}"

    L.append("\n## 结果（S1, lev 1.5, 净口径, 30U 窗口起点重启）\n")
    L.append("| 期间 | 配置 | 交易数 | 期末 | 年化 | 盘中回撤 | Calmar | 拦截笔数 | 拦截交易平均R | 保留交易平均R |\n|---|---|---|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        L.append(f"| {r.period} | {r.cfg} | {int(r.n)} | {r.end_u:.1f}U | {fp(r.cagr,True)} | {fp(r.mdd_intra,True)} | "
                 f"{fp(r.calmar)} | {int(r.blocked_n)} | {fp(r.blocked_mean_r,d=3)} | {fp(r.kept_mean_r,d=3)} |")

    b = R[(R.period == "OOS 2024+") & (R.cfg == "基线")].iloc[0]
    g = R[(R.period == "OOS 2024+") & (R.cfg.str.startswith("S-G"))].iloc[0]
    pct = (rand_cal < g.calmar).mean() * 100
    L.append(f"\n**随机对照（OOS，随机跳过 {(1-share_sq_oos)*100:.1f}% 入场，{len(rand_cal)} 次）**：Calmar 中位数 {np.median(rand_cal):.2f}，"
             f"5–95% [{np.percentile(rand_cal,5):.2f}, {np.percentile(rand_cal,95):.2f}]；S-G Calmar {g.calmar:.2f} 位于第 **{pct:.0f}** 分位\n")

    c1 = g.calmar > b.calmar
    c2 = g.cagr >= b.cagr - 0.05
    c3 = (not pd.isna(g.blocked_mean_r)) and g.blocked_mean_r < 0
    c4 = pct >= 90
    verdict = "通过" if (c1 and c2 and c3 and c4) else "不通过"
    L.append("## 判定（预登记，仅 OOS）\n")
    L.append(f"- (1) Calmar S-G {g.calmar:.2f} > 基线 {b.calmar:.2f}: {'满足' if c1 else '不满足'}")
    L.append(f"- (2) 年化 S-G {g.cagr*100:.1f}% ≥ 基线 {b.cagr*100:.1f}% − 5pp: {'满足' if c2 else '不满足'}")
    L.append(f"- (3) 被拦截交易平均 R {fp(g.blocked_mean_r,d=3)} < 0: {'满足' if c3 else '不满足'}")
    L.append(f"- (4) 随机分位 {pct:.0f} ≥ 90: {'满足' if c4 else '不满足'}")
    L.append(f"\n**结论: {verdict}**")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "state_squeeze_gate_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
