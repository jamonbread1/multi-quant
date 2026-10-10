# -*- coding: utf-8 -*-
"""市场状态分流研究 v1.0 —— 两个方向

方向 B: 状态本身是否可靠 (因果 WF 状态, 开仓/判断只用当根收盘及之前数据)
  B1 各状态之后 1 根 / 6 根 / 30 根 (4h / 1d / 5d) 的前瞻收益分布, IS 与 OOS 分开看
  B2 状态持续时长与切换频率 (路由成本的代理: 切换越频繁, 策略被切来切去越多)
方向 A: 状态 → 策略组 的路由是否优于不路由 (引擎 S1, 净口径, 已对账)
  预先登记 5 组路由规则 (按经济逻辑设定, 不由结果反推):
    R0 不路由 (基线)
    R1 趋势对齐: TREND_UP 只做多, TREND_DOWN 只做空, RANGE_CALM / HIGH_VOL 不限制
    R2 趋势对齐 + HIGH_VOL 空仓
    R3 只保留趋势组: TREND_UP 只做多, TREND_DOWN 只做空, RANGE_CALM / HIGH_VOL 空仓
    R4 反向对照: TREND_UP 只做空, TREND_DOWN 只做多 (期望显著更差, 用于检验对齐是否有意义)
  评估: IS (<2024) / OOS (>=2024) / 全样本; 以及每个状态下的 S1 交易表现。
数据: WF_4H_rule_states.csv (因果), eth4h_from_fmz_1h.csv (与引擎一致)。
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
START_U = 30.0
N_BOOT = 2000
SEED = 20261010
STATES = ["TREND_UP", "TREND_DOWN", "RANGE_CALM", "HIGH_VOL"]


def load_k():
    k = pd.read_csv(os.path.join(HERE, "data", "eth4h_from_fmz_1h.csv"))
    k["dt"] = pd.to_datetime(k["dt"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]].reset_index(drop=True)


def load_wf():
    w = pd.read_csv(os.path.join(REPO, "WF_4H_rule_states.csv"))
    w["dt"] = pd.to_datetime(w["dt"])
    return w.set_index("dt")["state"]


# ---------------- 方向 B ----------------
def state_validity(k, st):
    close = k["close"].values
    states = st.reindex(k["dt"]).values
    rows = []
    for h, name in [(1, "4h"), (6, "1d"), (30, "5d")]:
        fwd = np.full(len(close), np.nan)
        fwd[:-h] = close[h:] / close[:-h] - 1  # 从 t 收盘到 t+h 收盘, t 时刻状态已知
        for period, mask in [("IS", k["dt"] < SPLIT), ("OOS", k["dt"] >= SPLIT)]:
            for s in ["ALL"] + STATES:
                base = mask.values & ~np.isnan(fwd)
                m = (base if s == "ALL" else (base & (states == s)))
                x = fwd[m]
                if len(x) < 5:
                    rows.append(dict(h=name, period=period, state=s, n=len(x), mean=np.nan, t=np.nan, pos=np.nan))
                    continue
                t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
                rows.append(dict(h=name, period=period, state=s, n=len(x), mean=x.mean(), t=t, pos=(x > 0).mean()))
    B1 = pd.DataFrame(rows)

    # B2 持续时长与切换
    s = pd.Series(states).dropna()
    runs = (s != s.shift()).cumsum()
    lengths = s.groupby(runs).size()
    switches = (s != s.shift()).sum()
    B2 = dict(bars=len(s), switches=int(switches), per_year=float(switches / ((k.dt.iloc[-1] - k.dt.iloc[0]).days / 365.25)),
              median_run_bars=float(lengths.median()), mean_run_bars=float(lengths.mean()),
              share={x: float((s == x).mean()) for x in STATES})
    return B1, B2


# ---------------- 方向 A ----------------
def allow_from_rule(states, rule):
    """states: 与 k 对齐的状态数组 (缺失为 None). 返回 allow 数组 (+1/-1/0/2)."""
    out = np.full(len(states), 2, dtype=int)
    for i, s in enumerate(states):
        if s is None:
            out[i] = 0
            continue
        out[i] = rule.get(s, 2)
    return out


RULES = [
    ("R0 不路由(基线)", {}),
    ("R1 趋势对齐", {"TREND_UP": 1, "TREND_DOWN": -1}),
    ("R2 趋势对齐+HIGH_VOL空仓", {"TREND_UP": 1, "TREND_DOWN": -1, "HIGH_VOL": 0}),
    ("R3 只保留趋势组", {"TREND_UP": 1, "TREND_DOWN": -1, "RANGE_CALM": 0, "HIGH_VOL": 0}),
    ("R4 反向对照", {"TREND_UP": -1, "TREND_DOWN": 1}),
]


def window(tr, start, end):
    sub = tr[(tr.exit_time >= start) & (tr.exit_time < end)] if len(tr) else tr
    if len(sub) == 0:
        return dict(n=0)
    ret = sub.pnl_net.values / sub.equity_entry.values
    eq = np.concatenate([[START_U], START_U * np.cumprod(1 + ret)])
    peak = np.maximum.accumulate(eq)
    years = max((pd.Timestamp(sub.exit_time.iloc[-1]) - pd.Timestamp(sub.entry_time.iloc[0])).days / 365.25, 1e-9)
    r = sub.pnl_r.values
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
    rng = np.random.default_rng(SEED)
    boots = np.array([rng.choice(r, len(r), replace=True).mean() for _ in range(N_BOOT)])
    return dict(n=len(sub), win=float((sub.pnl_net > 0).mean()), mean_r=float(r.mean()),
                lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), t=float(t),
                mult=float(eq[-1] / START_U), cagr=float((eq[-1] / START_U) ** (1 / years) - 1),
                mdd=float((eq / peak - 1).min()))


def main():
    k = load_k()
    st = load_wf()
    states = st.reindex(k["dt"]).values  # 对齐到 bar 的开盘时间 dt (该 bar 的状态在其收盘时确定)
    # 信号根 cl 的状态: 引擎在 j 根开盘时用 cl=j-1 的 want, 故 allow 也按 cl 对齐 -> 直接用 states[cl]
    B1, B2 = state_validity(k, st)

    # 路由: allow 数组按 bar 索引 (引擎 allow[cl])
    states_list = [None if (isinstance(x, float) and np.isnan(x)) else x for x in states]

    res = []
    trades_store = {}
    for name, rule in RULES:
        allow = allow_from_rule(states_list, rule)
        tr, _ = E.run(k, don_n=48, adx_min=15, allow=allow)
        tr["entry_time"] = pd.to_datetime(tr.entry_time)
        tr["exit_time"] = pd.to_datetime(tr.exit_time)
        trades_store[name] = tr
        end = k.dt.iloc[-1] + pd.Timedelta(seconds=1)
        res.append(dict(rule=name, full=window(tr, k.dt.iloc[0], end),
                        IS=window(tr, k.dt.iloc[0], SPLIT), OOS=window(tr, SPLIT, end)))

    # 每状态的 S1(R0) 交易表现 (按信号状态归因)
    tr0 = trades_store["R0 不路由(基线)"].copy()
    tr0["sig_state"] = [st.get(pd.Timestamp(t) - pd.Timedelta(hours=4), None) for t in tr0.entry_time]
    lift_rows = []
    sig_all = pd.Series([x for x in tr0.sig_state.values if x is not None])
    bar_share = pd.Series(states_list).dropna().value_counts(normalize=True)
    for s in STATES:
        ent_share = float((sig_all == s).mean())
        lift_rows.append(dict(state=s, bar_share=float(bar_share.get(s, 0)), entry_share=ent_share,
                              lift=ent_share / float(bar_share.get(s, np.nan))))
    per_state = []
    for s in STATES:
        g = tr0[tr0.sig_state == s]
        for period, sub in [("IS", g[g.exit_time < SPLIT]), ("OOS", g[g.exit_time >= SPLIT])]:
            per_state.append(dict(state=s, period=period, n=len(sub),
                                  mean_r=float(sub.pnl_r.mean()) if len(sub) else np.nan,
                                  sum_u=float(sub.pnl_net.sum()) if len(sub) else np.nan))

    # ---------- 报告 ----------
    def fmt_w(w, label):
        if w.get("n", 0) == 0:
            return f"| {label} | 0 | | | | | | |"
        return (f"| {label} | {w['n']} | {w['win']*100:.1f}% | {w['mean_r']:+.3f} [{w['lo']:+.2f},{w['hi']:+.2f}] | "
                f"{w['t']:+.2f} | {w['mult']:.2f}x | {w['cagr']*100:+.1f}% | {w['mdd']*100:.1f}% |")

    L = ["# 市场状态分流研究报告 v1.0\n",
         "**脚本**: `research/oos/state_routing_v1_0.py`；引擎 `s1_engine_v1_0.py`（已对账）；状态 `WF_4H_rule_states.csv`（因果）  ",
         f"**切分**: {SPLIT.date()}；**口径**: 净口径（扣 0.05%/边 + 资金费），30U 起始；窗口内从 30U 重新复利\n",
         "## 方向 B：状态的前瞻预测力（因果）\n",
         "前瞻收益 = t 收盘到 t+h 收盘；状态在 t 收盘时已知。t 为单样本 t 统计量（未做多重检验修正）。\n"]
    L.append("| 周期 | 前瞻 | 状态 | n | 均值 | t | 上涨占比 |\n|---|---|---|---|---|---|---|")
    L.append("> `ALL` 为无条件基准（同期全部 bar），状态的价值应看相对 ALL 的超额，而非绝对正负。")
    for _, r in B1.iterrows():
        mean = "" if pd.isna(r['mean']) else f"{r['mean']*100:+.2f}%"
        t = "" if pd.isna(r['t']) else f"{r['t']:+.2f}"
        pos = "" if pd.isna(r['pos']) else f"{r['pos']*100:.1f}%"
        L.append(f"| {r['period']} | {r['h']} | {r['state']} | {r['n']} | {mean} | {t} | {pos} |")
    L.append("")
    L.append("### B2 状态稳定性\n")
    L.append(f"- 切换次数: {B2['switches']}（约 {B2['per_year']:.0f} 次/年）")
    L.append(f"- 持续中位数: {B2['median_run_bars']:.0f} 根 4H bar（约 {B2['median_run_bars']*4/24:.1f} 天）")
    L.append("- 各状态占比: " + "，".join(f"{k_}={v*100:.1f}%" for k_, v in B2['share'].items()) + "\n")

    L.append("## 方向 A：路由规则 vs 基线（S1 净口径）\n")
    L.append("| 规则 | 笔数 | 胜率 | 均笔R [95%CI] | t | 30U 期末 | 年化 | 最大回撤 |\n|---|---|---|---|---|---|---|---|")
    for r in res:
        L.append(fmt_w(r["full"], f"{r['rule']} · 全样本"))
        L.append(fmt_w(r["IS"], f"{r['rule']} · IS"))
        L.append(fmt_w(r["OOS"], f"{r['rule']} · OOS"))
    L.append("\n## 基线 S1 进场状态提升度（进场占比 / K 线占比，>1 表示该状态更易触发进场）\n")
    L.append("| 状态 | K 线占比 | 进场占比 | 提升度 |\n|---|---|---|---|")
    for x in lift_rows:
        L.append(f"| {x['state']} | {x['bar_share']*100:.1f}% | {x['entry_share']*100:.1f}% | {x['lift']:.2f} |")
    L.append("\n## 基线 S1 按信号状态归因（R0）\n")
    L.append("| 状态 | 周期 | 笔数 | 均笔R | 净盈亏U |\n|---|---|---|---|---|")
    for p in per_state:
        mr = "" if pd.isna(p['mean_r']) else f"{p['mean_r']:+.3f}"
        su = "" if pd.isna(p['sum_u']) else f"{p['sum_u']:+.1f}"
        L.append(f"| {p['state']} | {p['period']} | {p['n']} | {mr} | {su} |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "state_routing_report_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    pd.DataFrame(per_state).to_csv(os.path.join(HERE, "state_routing_per_state_v1_0.csv"), index=False)
    B1.to_csv(os.path.join(HERE, "state_validity_B1_v1_0.csv"), index=False)
    print(txt)


if __name__ == "__main__":
    main()
