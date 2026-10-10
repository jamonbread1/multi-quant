# -*- coding: utf-8 -*-
"""v1.1 TRANSITION 样本外检验 v1.0

数据: state_4h_wf_v1_1_causal.csv (由 state_transition_repro_v1_0.py 生成, walk-forward 因果标签, 2021-01 起)
IS = 2021-01 ~ 2023-12 (仅用于观察), OOS = 2024-01 起 (判定用)

检验 A: 领先/滞后 (与 state_lead_v1_0 相同口径: 新行情的首次 48 根突破)
检验 B: 预登记的单一配置 — S1 海龟 (lev 1.5, 净口径), 闸门 = 决策根为 TRANSITION 时禁止开仓并平仓 (引擎 allow 语义)

预登记判定规则 (运行前固定, 不据结果修改):
  B 通过 当且仅当同时满足:
    (1) OOS Calmar(闸门) > OOS Calmar(基线)
    (2) OOS CAGR(闸门) >= OOS CAGR(基线) - 5 个百分点
    (3) 被闸门拦截的基线交易, 平均 pnl_r < 0 (闸门确实拦截了亏损交易)
  否则不通过。IS 只作参考, 不参与判定。
恒真率检查: 报告 TRANSITION 在决策根上的占比; 若 >95% 则闸门无效。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_engine_v1_1 as E  # noqa: E402
import state_lead_v1_0 as SL  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01")
IS_START = pd.Timestamp("2021-01-01")
START_U = 30.0
WARM = 198


def load_labels():
    s = pd.read_csv(os.path.join(HERE, "data", "state_4h_wf_v1_1_causal.csv"))
    s["dt"] = pd.to_datetime(s["dt"])
    return s.set_index("dt")


def stats(eq_curve, eq_worst, years, n_trades):
    from leverage_v1_0 import _mdd
    end = float(eq_curve[-1])
    mult = end / START_U
    cagr = mult ** (1 / years) - 1 if mult > 0 else -1.0
    mdd_i = _mdd(eq_worst)
    mdd_r = _mdd(eq_curve)
    calmar = cagr / abs(mdd_i) if mdd_i < 0 else np.nan
    return dict(end_u=end, cagr=cagr, mdd_intra=mdd_i, mdd_real=mdd_r, calmar=calmar, n=n_trades)


def window_run(k, lab, start_ts, allow=None, end_ts=None):
    """从 start_ts 开始 30U 重启的窗口回测 (预热 WARM 根, 切分日开始交易)."""
    i0 = int(np.searchsorted(k.dt.values, np.datetime64(start_ts))) - WARM
    kk = k.iloc[i0:].reset_index(drop=True)
    al = None
    if allow is not None:
        al = allow[i0:]
    tr, eq, eqw = E.run(kk, don_n=48, lev=1.5, allow=al)
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
    lab = load_labels()
    st = lab["state_v1_1"].reindex(k.dt).values  # 与 k 对齐 (2021 前为 NaN)
    L = ["# v1.1 TRANSITION 样本外检验 v1.0\n",
         "**脚本**: `research/oos/state_transition_oos_v1_0.py`；标签 `data/state_4h_wf_v1_1_causal.csv`（WF 因果）。  ",
         "**IS** = 2021–2023（参考），**OOS** = 2024-01 起（判定）。\n"]

    # ---- A: 领先/滞后 ----
    up_ev, dn_ev = SL.events(k)
    up_f, dn_f = SL.fresh(up_ev), SL.fresh(dn_ev)
    trans = (st == "TRANSITION")
    valid = ~pd.isna(st)
    L.append("## A. 领先/滞后（新行情首次突破，决策 bar 上的 TRANSITION 提升度）\n")
    L.append("| 方向 | 期间 | 事件数 | 基准 P(TRANSITION) | lift k=0 | lift k=1 | lift k=2 | lift k=4 |\n|---|---|---|---|---|---|---|---|")
    for side, ev in [("上行", up_f), ("下行", dn_f)]:
        for per, m in [("IS 2021-23", valid & (k.dt < SPLIT).values), ("OOS 2024+", valid & (k.dt >= SPLIT).values)]:
            idx = np.where(ev & m)[0]
            idx = idx[idx >= 5]
            base = trans[m].mean()
            lifts = []
            for kk in [0, 1, 2, 4]:
                p = trans[idx - kk].mean() if len(idx) else np.nan
                lifts.append(p / base if base > 0 else np.nan)
            L.append(f"| {side} | {per} | {len(idx)} | {base*100:.1f}% | " +
                     " | ".join(f"{x:.2f}" for x in lifts) + " |")

    # 同时给出 WF TREND (v1.0 同时期) 作参照: 用 v1.0 标签
    # ---- B: 闸门 ----
    L.append("\n## B. 预登记闸门配置：S1 (lev 1.5, 净口径), 决策根 TRANSITION → 禁止开仓并平仓\n")
    # 恒真率检查 (决策根 = 入场 bar 的前一根)
    allow = np.full(len(k), 2, dtype=int)
    allow[trans] = 0
    share_oos = trans[valid & (k.dt >= SPLIT).values].mean()
    share_is = trans[valid & (k.dt < SPLIT).values].mean()
    L.append(f"- 恒真率检查: TRANSITION 占决策根 IS {share_is*100:.1f}% / OOS {share_oos*100:.1f}%（阈值 95%，远低于，通过）\n")

    rows = []
    for per, start, end in [("IS 2021-23", IS_START, SPLIT), ("OOS 2024+", SPLIT, None)]:
        base_tr, base = window_run(k, lab, start, allow=None, end_ts=end)
        gate_tr, gate = window_run(k, lab, start, allow=allow, end_ts=end)
        # 被拦截的基线交易: 决策根 (entry_time - 4h) 为 TRANSITION
        dec_dt = pd.to_datetime(base_tr.entry_time) - pd.Timedelta(hours=4)
        blocked = trans[np.searchsorted(k.dt.values, dec_dt.values.astype("datetime64[ns]"))]
        r_block = base_tr.loc[blocked, "pnl_r"].astype(float)
        r_keep = base_tr.loc[~blocked, "pnl_r"].astype(float)
        rows.append(dict(period=per, cfg="基线", **base))
        rows.append(dict(period=per, cfg="闸门(TRANSITION)", **gate))
        rows[-2]["blocked_n"] = int(blocked.sum())
        rows[-2]["blocked_mean_r"] = float(r_block.mean()) if len(r_block) else np.nan
        rows[-2]["kept_mean_r"] = float(r_keep.mean()) if len(r_keep) else np.nan
        rows[-2]["blocked_win"] = float((r_block > 0).mean()) if len(r_block) else np.nan
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "state_transition_gate_v1_0.csv"), index=False, encoding="utf-8")

    def f(x, p=False, d=2):
        if pd.isna(x):
            return "—"
        return f"{x*100:.1f}%" if p else f"{x:.{d}f}"

    L.append("| 期间 | 配置 | 交易数 | 期末 | 年化 | 盘中回撤 | Calmar |\n|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        L.append(f"| {r.period} | {r.cfg} | {int(r.n)} | {r.end_u:.1f}U | {f(r.cagr,True)} | {f(r.mdd_intra,True)} | {f(r.calmar)} |")
    L.append("\n被闸门拦截的基线交易（决策根为 TRANSITION）：\n")
    L.append("| 期间 | 拦截笔数 | 拦截交易平均 R | 保留交易平均 R | 拦截交易胜率 |\n|---|---|---|---|---|")
    for per in ["IS 2021-23", "OOS 2024+"]:
        r = R[(R.period == per) & (R.cfg == "基线")].iloc[0]
        L.append(f"| {per} | {int(r.blocked_n)} | {f(r.blocked_mean_r,d=3)} | {f(r.kept_mean_r,d=3)} | {f(r.blocked_win,True)} |")

    # 判定 (预登记规则)
    b = R[(R.period == "OOS 2024+") & (R.cfg == "基线")].iloc[0]
    g = R[(R.period == "OOS 2024+") & (R.cfg == "闸门(TRANSITION)")].iloc[0]
    c1 = g.calmar > b.calmar
    c2 = g.cagr >= b.cagr - 0.05
    c3 = (b.blocked_mean_r < 0) if not pd.isna(b.blocked_mean_r) else False
    verdict = "通过" if (c1 and c2 and c3) else "不通过"
    L.append("\n## 判定（预登记规则，仅 OOS）\n")
    L.append(f"- (1) OOS Calmar 闸门 {g.calmar:.2f} > 基线 {b.calmar:.2f}: {'满足' if c1 else '不满足'}")
    L.append(f"- (2) OOS 年化 闸门 {g.cagr*100:.1f}% ≥ 基线 {b.cagr*100:.1f}% − 5pp: {'满足' if c2 else '不满足'}")
    L.append(f"- (3) 被拦截交易平均 R {b.blocked_mean_r:.3f} < 0: {'满足' if c3 else '不满足'}")
    L.append(f"\n**结论: {verdict}**")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "state_transition_oos_v1_0.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
