# -*- coding: utf-8 -*-
"""方向 B: 状态的领先/滞后 研究 v1.0

问题: 市场状态相对「行情启动」是领先还是滞后? 能否作为分流的提前信号?
事件: 48 根 4H 通道突破 (与 S1 触发一致, 不加 ADX 门控)
  上行事件: close[i] > max(high[i-48:i])    下行事件: close[i] < min(low[i-48:i])
候选指标 (全部因果, 只用 i 及之前的数据; 每个指标都给出 on/off 布尔序列):
  WF_TREND   WF 状态 = 同向趋势 (上行 TREND_UP / 下行 TREND_DOWN)
  WF_NONCALM WF 状态 != RANGE_CALM (任何非平静状态)
  ADX15      ADX(14) >= 15 (S1 门控)
  ADX25      ADX(14) >= 25
  ER20       效率比 |c_i - c_{i-20}| / sum|dc| >= 0.3, 且方向同向
  SQUEEZE    ATR14 / mean(ATR14, 100) <= 0.8  (压缩, 突破前的先兆)
  EXPAND     ATR14 / mean(ATR14, 100) >= 1.2  (扩张)
  NARROW     (max20 高 - min20 低) / ATR14 <= 6 (通道收窄)
指标:
  lift(k)  = P(on at i-k | 事件) / P(on)  , k=0 (突破当根), 1, 2, 4  ——  >1 表示事件前该指标已偏向 on (领先)
  first_on = 相对突破的首次 on 的 bar 偏移 (中位数, 范围 [-10, +20]), 负数=突破前已 on
切分: IS(<2024) / OOS(>=2024) 分开报告 lift, 检验稳定性
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")
sys.path.insert(0, HERE)
import s1_engine_v1_0 as E  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01")


def load_k():
    k = pd.read_csv(os.path.join(HERE, "data", "eth4h_from_fmz_1h.csv"))
    k["dt"] = pd.to_datetime(k["dt"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]].reset_index(drop=True)


def load_wf(dts):
    w = pd.read_csv(os.path.join(REPO, "WF_4H_rule_states.csv"))
    w["dt"] = pd.to_datetime(w["dt"])
    return w.set_index("dt")["state"].reindex(dts).values


def indicators(k, wf):
    h, l, c = k.high.values, k.low.values, k.close.values
    atr = np.array(E.atr_wilder(h, l, c))
    adx = np.array(E.adx_wilder(h, l, c))
    n = len(c)
    atr_mean = pd.Series(atr).rolling(100, min_periods=100).mean().values
    ratio = atr / atr_mean
    dc = pd.Series(c).diff().abs().rolling(20).sum().values
    er = np.abs(pd.Series(c).diff(20).values) / dc
    sign20 = np.sign(pd.Series(c).diff(20).values)
    hi20 = pd.Series(h).rolling(20).max().values
    lo20 = pd.Series(l).rolling(20).min().values
    width = (hi20 - lo20) / atr
    ind_up = {
        "WF_TREND": wf == "TREND_UP",
        "WF_NONCALM": (wf != "RANGE_CALM") & pd.notna(wf),
        "ADX15": adx >= 15,
        "ADX25": adx >= 25,
        "ER20": (er >= 0.3) & (sign20 > 0),
        "SQUEEZE": ratio <= 0.8,
        "EXPAND": ratio >= 1.2,
        "NARROW": width <= 6,
    }
    ind_dn = dict(ind_up)
    ind_dn["WF_TREND"] = wf == "TREND_DOWN"
    ind_dn["WF_NONCALM"] = (wf != "RANGE_CALM") & pd.notna(wf)
    ind_dn["ER20"] = (er >= 0.3) & (sign20 < 0)
    return ind_up, ind_dn


def events(k):
    h, l, c = k.high.values, k.low.values, k.close.values
    n = len(c)
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    hh = pd.Series(h).shift(1).rolling(48).max().values
    ll = pd.Series(l).shift(1).rolling(48).min().values
    up[48:] = c[48:] > hh[48:]
    dn[48:] = c[48:] < ll[48:]
    return up, dn


def lift_table(ind, ev, dts_mask, lags=(0, 1, 2, 4)):
    """ev: 事件 bool 数组. 返回 {指标: {k: lift}} 以及 base."""
    out = {}
    idx = np.where(ev & dts_mask)[0]
    idx = idx[idx >= 30]
    base_mask = dts_mask.copy()
    for name, arr in ind.items():
        arr = np.asarray(arr, dtype=bool)
        base = arr[base_mask].mean()
        row = {}
        for kk in lags:
            vals = arr[idx - kk]
            p = vals.mean() if len(vals) else np.nan
            row[f"lift_k{kk}"] = p / base if base > 0 else np.nan
            row[f"p_k{kk}"] = p
        row["base"] = base
        row["n_events"] = len(idx)
        out[name] = row
    return out


def first_on(ind, ev, offsets=range(-10, 21)):
    """相对事件的首次 on 偏移 (中位数), 只统计在窗口内首次 on 的事件."""
    idx = np.where(ev)[0]
    res = {}
    for name, arr in ind.items():
        arr = np.asarray(arr, dtype=bool)
        leads = []
        for i in idx:
            if i - 10 < 0 or i + 20 >= len(arr):
                continue
            seg = arr[i - 10:i + 21]
            on = np.where(seg)[0]
            if len(on) == 0:
                continue
            leads.append(on[0] - 10)  # 第一个 on 的偏移
        res[name] = (float(np.median(leads)) if leads else np.nan, len(leads), len(idx))
    return res


def fresh(ev, cool=24):
    """新行情的第一次突破: 前 cool 根内没有同向突破."""
    out = np.zeros(len(ev), bool)
    last = -10**9
    for i in np.where(ev)[0]:
        if i - last > cool:
            out[i] = True
        last = i
    return out


def main():
    k = load_k()
    wf = load_wf(k.dt)
    ind_up, ind_dn = indicators(k, wf)
    up_ev, dn_ev = events(k)
    up_fresh, dn_fresh = fresh(up_ev), fresh(dn_ev)
    is_mask = (k.dt < SPLIT).values
    oos_mask = (k.dt >= SPLIT).values
    all_mask = np.ones(len(k), bool)

    L = ["# 方向 B：状态领先/滞后研究 v1.0\n",
         "**脚本**: `research/oos/state_lead_v1_0.py`；事件 = 48 根 4H 通道突破（上行 / 下行分别统计）  ",
         "**lift(k)** = P(指标在突破前第 k 根为 on | 突破) / P(指标为 on)。>1 表示该指标在突破前已偏向 on（领先），≈1 表示无关，<1 表示突破之后才 on（滞后）。\n"]
    summary_rows = []
    L.append("\n## 新行情的首次突破（前 24 根内无同向突破，剔除持续性突破）\n")
    L.append("> 这是检验“状态是否领先行情启动”的主要口径。持续性突破（趋势中的新高/新低）会让状态天然偏 on，造成虚高的 lift。\n")
    for side, ind, ev in [("上行·新行情", ind_up, up_fresh), ("下行·新行情", ind_dn, dn_fresh)]:
        lt = lift_table(ind, ev, all_mask)
        n_ev = next(iter(lt.values()))["n_events"]
        L.append(f"\n### {side}（事件 n={n_ev}）\n")
        L.append("| 指标 | 基准 P(on) | lift k=0 | lift k=1 | lift k=2 | lift k=4 |\n|---|---|---|---|---|---|")
        for name, r in lt.items():
            L.append(f"| {name} | {r['base']*100:.1f}% | {r['lift_k0']:.2f} | {r['lift_k1']:.2f} | "
                     f"{r['lift_k2']:.2f} | {r['lift_k4']:.2f} |")
            summary_rows.append(dict(side=side, period="全样本", indicator=name, **r))
        fo = first_on(ind, ev)
        L.append(f"\n首次 on 中位偏移（4H bar）：" + "；".join(
            f"{nm}={'' if pd.isna(md) else f'{md:+.0f}'}" for nm, (md, _, _) in fo.items()) + "\n")
    for side, ind, ev in [("上行突破", ind_up, up_ev), ("下行突破", ind_dn, dn_ev)]:
        L.append(f"\n## {side}\n")
        for period, m in [("全样本", all_mask), ("IS (<2024)", is_mask), ("OOS (>=2024)", oos_mask)]:
            lt = lift_table(ind, ev, m)
            n_ev = next(iter(lt.values()))["n_events"]
            L.append(f"\n### {side} · {period}（事件 n={n_ev}）\n")
            L.append("| 指标 | 基准 P(on) | lift k=0 (突破当根) | lift k=1 | lift k=2 | lift k=4 |\n|---|---|---|---|---|---|")
            for name, r in lt.items():
                L.append(f"| {name} | {r['base']*100:.1f}% | {r['lift_k0']:.2f} | {r['lift_k1']:.2f} | "
                         f"{r['lift_k2']:.2f} | {r['lift_k4']:.2f} |")
                summary_rows.append(dict(side=side, period=period, indicator=name, **r))
        fo = first_on(ind, ev)
        L.append(f"\n### {side} · 相对突破的首次 on 偏移（中位数，单位 4H bar；负数=突破前已 on）\n")
        L.append("| 指标 | 首次 on 中位偏移 | 有效事件数 | 总事件数 |\n|---|---|---|---|")
        for name, (med, nv, nt) in fo.items():
            m_txt = "" if pd.isna(med) else f"{med:+.0f}"
            L.append(f"| {name} | {m_txt} | {nv} | {nt} |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "state_lead_report_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    pd.DataFrame(summary_rows).to_csv(os.path.join(HERE, "state_lead_lift_v1_0.csv"), index=False, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
