# -*- coding: utf-8 -*-
"""v1.1 TRANSITION 闸门 v2 检验 (改进版: 仅拦截新入场, 不强制平仓) v1.0

相对 v1 闸门 (state_transition_oos_v1_0.py) 的改动:
  - 引擎 s1_engine_v1_2 的 entry_skip: 决策根 TRANSITION 时跳过新入场 (不补做), 已有仓位不动。
  - v1 的 allow 会强制平仓, 把「拦截哪几笔」与「提前平仓改变后续结构」混在一起, 结论不可归因, 故废弃。
  - 增加随机对照: 随机跳过相同比例的决策根 (500 次), 检验 TRANSITION 是否优于“随便少开几笔”。

配置 (预登记, 共 2 个; 上限 16):
  G1 = 仅拦截新入场, 决策根为 TRANSITION (主配置)
  C0 = 基线 (无闸门)
  控制 = 随机 mask, 与 G1 同比例 (500 seeds, 固定随机种子)

判定规则 (运行前固定; 仅看 OOS 2024-01 起; IS 只作参考):
  G1 通过 当且仅当同时满足:
    (1) OOS Calmar(G1) > OOS Calmar(C0)
    (2) OOS CAGR(G1) >= OOS CAGR(C0) - 5pp
    (3) 被拦截的基线交易 (决策根为 TRANSITION) 的 OOS 平均 pnl_r < 0
    (4) OOS Calmar(G1) 高于随机对照分布的 90 分位数
  恒真率: TRANSITION 占决策根 >95% 则无效 (本次远低于, 仅报告)。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_engine_v1_2 as E  # noqa: E402
import state_lead_v1_0 as SL  # noqa: E402
from leverage_v1_0 import _mdd  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01")
IS_START = pd.Timestamp("2021-01-01")
START_U = 30.0
WARM = 198
N_RAND = 500
SEED = 20261010


def stats(eq, eqw, years, n):
    end = float(eq[-1])
    mult = end / START_U
    cagr = mult ** (1 / years) - 1 if mult > 0 else -1.0
    mdd_i = _mdd(eqw)
    calmar = cagr / abs(mdd_i) if mdd_i < 0 else np.nan
    return dict(end_u=end, cagr=cagr, mdd_intra=mdd_i, calmar=calmar, n=n)


def window_run(k, start_ts, end_ts=None, skip=None):
    i0 = int(np.searchsorted(k.dt.values, np.datetime64(start_ts))) - WARM
    kk = k.iloc[i0:].reset_index(drop=True)
    sk = None if skip is None else skip[i0:]
    tr, eq, eqw = E.run(kk, don_n=48, lev=1.5, entry_skip=sk)
    tr = tr[pd.to_datetime(tr.entry_time) >= start_ts]
    if end_ts is not None:
        tr = tr[pd.to_datetime(tr.entry_time) < end_ts]
    tr = tr.reset_index(drop=True)
    first = pd.to_datetime(tr.entry_time).min()
    last = pd.to_datetime(tr.exit_time).max()
    years = (last - first).days / 365.25
    return tr, stats(eq, eqw, years, len(tr))


def main():
    k = SL.load_k()
    lab = pd.read_csv(os.path.join(HERE, "data", "state_4h_wf_v1_1_causal.csv"))
    lab["dt"] = pd.to_datetime(lab["dt"])
    st = lab.set_index("dt")["state_v1_1"].reindex(k.dt).values
    valid = ~pd.isna(st)
    trans = (st == "TRANSITION")
    skip = trans.copy()  # 决策根 TRANSITION -> 跳过新入场

    L = ["# v1.1 TRANSITION 闸门 v2 检验（仅拦截新入场）v1.0\n",
         "**脚本**: `research/oos/state_transition_gate_v2_v1_0.py`；引擎 `s1_engine_v1_2.py`（entry_skip）。  ",
         "**IS** = 2021–2023（参考），**OOS** = 2024-01 起（判定）。\n",
         "## 设计修正（相对 v1 闸门）\n",
         "- v1 用 `allow`：决策根为 TRANSITION 时把 want 置 0，已有仓位被强制平仓，之后重新开仓。交易结构改变，结论无法归因到“拦截入场”本身。",
         "- v2 用 `entry_skip`：只跳过新入场（不补做），已有仓位按原规则管理。\n"]

    # 恒真率
    L.append(f"- 恒真率: TRANSITION 占决策根 IS {trans[valid & (k.dt<SPLIT).values].mean()*100:.1f}% / "
             f"OOS {trans[valid & (k.dt>=SPLIT).values].mean()*100:.1f}%（远低于 95%）\n")

    rows = []
    for per, s, e in [("IS 2021-23", IS_START, SPLIT), ("OOS 2024+", SPLIT, None)]:
        b_tr, b = window_run(k, s, e, skip=None)
        g_tr, g = window_run(k, s, e, skip=skip)
        dec = pd.to_datetime(b_tr.entry_time) - pd.Timedelta(hours=4)
        blocked = trans[np.searchsorted(k.dt.values, dec.values.astype("datetime64[ns]"))]
        rb = b_tr.loc[blocked, "pnl_r"].astype(float)
        rows.append(dict(period=per, cfg="C0 基线", **b, blocked_n=0, blocked_mean_r=np.nan))
        rows.append(dict(period=per, cfg="G1 拦截TRANSITION入场", **g,
                         blocked_n=int(blocked.sum()),
                         blocked_mean_r=float(rb.mean()) if len(rb) else np.nan))

    # 随机对照 (OOS, 同比例)
    rng = np.random.default_rng(SEED)
    share = trans[valid & (k.dt >= SPLIT).values].mean()
    oos_mask = (k.dt >= SPLIT).values
    cal_r, cagr_r = [], []
    for _ in range(N_RAND):
        m = np.zeros(len(k), bool)
        pick = rng.random(len(k)) < share
        m[pick] = True
        m &= valid
        _, rs = window_run(k, SPLIT, None, skip=m)
        cal_r.append(rs["calmar"])
        cagr_r.append(rs["cagr"])
    cal_r = np.array(cal_r, dtype=float)
    cal_r = cal_r[~np.isnan(cal_r)]

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "state_transition_gate_v2_v1_0.csv"), index=False, encoding="utf-8")

    def fp(x, p=False, d=2):
        if pd.isna(x):
            return "—"
        return f"{x*100:.1f}%" if p else f"{x:.{d}f}"

    L.append("## 结果\n")
    L.append("| 期间 | 配置 | 交易数 | 期末 | 年化 | 盘中回撤 | Calmar | 拦截笔数 | 拦截交易平均 R |\n|---|---|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        L.append(f"| {r.period} | {r.cfg} | {int(r.n)} | {r.end_u:.1f}U | {fp(r.cagr,True)} | {fp(r.mdd_intra,True)} | "
                 f"{fp(r.calmar)} | {int(r.blocked_n)} | {fp(r.blocked_mean_r,d=3)} |")

    b = R[(R.period == "OOS 2024+") & (R.cfg.str.startswith("C0"))].iloc[0]
    g = R[(R.period == "OOS 2024+") & (R.cfg.str.startswith("G1"))].iloc[0]
    pct = (cal_r < g.calmar).mean() * 100
    L.append(f"\n**随机对照（OOS，同比例 {share*100:.1f}% 随机跳过入场，{len(cal_r)} 次）**：")
    L.append(f"- 随机对照 Calmar 中位数 {np.median(cal_r):.2f}，5%–95% 区间 [{np.percentile(cal_r,5):.2f}, {np.percentile(cal_r,95):.2f}]")
    L.append(f"- G1 的 Calmar {g.calmar:.2f} 位于随机分布的 **{pct:.0f}** 分位\n")

    c1 = g.calmar > b.calmar
    c2 = g.cagr >= b.cagr - 0.05
    c3 = (not pd.isna(g.blocked_mean_r)) and g.blocked_mean_r < 0
    c4 = pct >= 90
    verdict = "通过" if (c1 and c2 and c3 and c4) else "不通过"
    L.append("## 判定（预登记规则，仅 OOS）\n")
    L.append(f"- (1) Calmar G1 {g.calmar:.2f} > C0 {b.calmar:.2f}: {'满足' if c1 else '不满足'}")
    L.append(f"- (2) 年化 G1 {g.cagr*100:.1f}% ≥ C0 {b.cagr*100:.1f}% − 5pp: {'满足' if c2 else '不满足'}")
    L.append(f"- (3) 被拦截交易平均 R {fp(g.blocked_mean_r,d=3)} < 0: {'满足' if c3 else '不满足'}")
    L.append(f"- (4) G1 Calmar 分位 {pct:.0f} ≥ 90: {'满足' if c4 else '不满足'}")
    L.append(f"\n**结论: {verdict}**")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "state_transition_gate_v2_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
