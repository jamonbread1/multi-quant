# -*- coding: utf-8 -*-
"""S1 提速变体 v1.0 —— 验证「入场慢 → 收益慢」假设.

动机 (用户): 做状态机是因为觉得收益速度太慢, 错过很多机会。
机会捕获分析显示: S1 进场中位延迟约 8 天 (≈ don48 通道长度), 大波段捕获率约 16%。
假设: 缩短入场通道能更早进场, 从而提升捕获与复利速度。

预先登记的 4 组设置 (运行前固定, 不做网格寻优):
  V0  don48 / ADX15  基线 (冻结版参数)
  V1  don24 / ADX15  入场通道减半 (exit = 12)
  V2  don16 / ADX15  入场通道 16 根 (≈2.7 天, exit = 8)
  V3  don48 / ADX0   基线但关闭 ADX 门控 (检验门控是否在压制早期进场)
评估: 全样本 / IS (<2024) / OOS (>=2024), 同时报告进场延迟与大波段捕获。
注意: 上述 V1–V3 是基于「提速」假设选择的, 不是从结果中挑出; 所有结果全部列出。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_engine_v1_0 as E  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01")
START_U = 30.0
VARIANTS = [
    ("V0 don48/ADX15 (基线)", dict(don_n=48, adx_min=15)),
    ("V1 don24/ADX15", dict(don_n=24, adx_min=15)),
    ("V2 don16/ADX15", dict(don_n=16, adx_min=15)),
    ("V3 don48/ADX0 (关门控)", dict(don_n=48, adx_min=0)),
]
THR = 0.25


def load_k():
    k = pd.read_csv(os.path.join(HERE, "data", "eth4h_from_fmz_1h.csv"))
    k["dt"] = pd.to_datetime(k["dt"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]]


def zigzag_legs(close, thr):
    legs, pivot, ext, trend = [], 0, 0, 0
    for i in range(1, len(close)):
        p = close[i]
        if trend == 0:
            if p >= close[pivot] * (1 + thr):
                trend, ext = 1, i
            elif p <= close[pivot] * (1 - thr):
                trend, ext = -1, i
        elif trend == 1:
            if p > close[ext]:
                ext = i
            elif p <= close[ext] * (1 - thr):
                legs.append((pivot, ext, 1))
                pivot, ext, trend = ext, i, -1
        else:
            if p < close[ext]:
                ext = i
            elif p >= close[ext] * (1 + thr):
                legs.append((pivot, ext, -1))
                pivot, ext, trend = ext, i, 1
    return legs


def leg_capture(tr, k):
    """对每条大波段腿, 计算 S1 同向持仓覆盖、进场延迟、对数收益捕获."""
    close = k["close"].values
    times = k["dt"].values
    lr = np.diff(np.log(close), prepend=np.nan)
    pos = np.zeros(len(k))
    if len(tr):
        a = np.searchsorted(times, pd.to_datetime(tr.entry_time).values)
        b = np.searchsorted(times, pd.to_datetime(tr.exit_time).values)
        for ai, bi, s in zip(a, b, tr.side.map({"long": 1, "short": -1}).values):
            pos[ai:bi] = s
    rows = []
    for a, b, d in zigzag_legs(close, THR):
        if b <= a:
            continue
        leg = np.log(close[b] / close[a]) * d
        seg = slice(a + 1, b + 1)
        inn = pos[seg] == d
        cap = np.nansum(np.where(inn, lr[seg], 0.0)) * d
        idx = np.where(inn)[0]
        lag = (times[a + 1 + idx[0]] - times[a]) / np.timedelta64(1, "D") if len(idx) else np.nan
        rows.append(dict(leg_lr=leg, captured_lr=cap, lag=lag, missed=len(idx) == 0, up=d > 0))
    return pd.DataFrame(rows)


def window_stats(tr, start, end, label, k):
    sub = tr[(tr.exit_time >= start) & (tr.exit_time < end)] if len(tr) else tr  # noqa
    if len(sub) == 0:
        return dict(label=label, n=0)
    r = sub.pnl_r.values
    # 窗口内重新从 30U 复利: 逐笔收益率 = 净盈亏 / 开仓权益 (仓位按历史权益定, 不能用绝对金额从 30U 累加)
    ret = sub.pnl_net.values / sub.equity_entry.values
    eq = np.concatenate([[START_U], START_U * np.cumprod(1 + ret)])
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    years = max((pd.Timestamp(sub.exit_time.iloc[-1]) - pd.Timestamp(sub.entry_time.iloc[0])).days / 365.25, 1e-9)
    mult = eq[-1] / START_U
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
    return dict(label=label, n=len(sub), win=float((sub.pnl_net > 0).mean()), mean_r=float(r.mean()),
                t=float(t), end_u=float(eq[-1]), mult=float(mult),
                cagr=float(mult ** (1 / years) - 1), mdd=mdd)


def fmt(s):
    if s.get("n", 0) == 0:
        return f"| {s['label']} | 0 | | | | | | |"
    return (f"| {s['label']} | {s['n']} | {s['win']*100:.1f}% | {s['mean_r']:+.3f} | {s['t']:+.2f} | "
            f"{s['end_u']:.0f}U ({s['mult']:.2f}x) | {s['cagr']*100:+.1f}% | {s['mdd']*100:.1f}% |")


HDR = ("| 设置 | 笔数 | 胜率 | 均笔R | t | 期末 | 年化 | 最大回撤 |\n|---|---|---|---|---|---|---|---|")


def main():
    k = load_k()
    start, end = k.dt.iloc[0], k.dt.iloc[-1] + pd.Timedelta(seconds=1)
    lines = ["# S1 提速变体 v1.0 报告\n",
             "**脚本**: `research/oos/s1_speed_variants_v1_0.py`，引擎 `s1_engine_v1_0.py`（已对账，见 `s1_engine_recon_v1_0.md`）  ",
             f"**数据**: `research/oos/data/eth4h_from_fmz_1h.csv`（ETH 4H，2019-12 ~ 2026-10）  ",
             f"**切分**: {SPLIT.date()}；**口径**: 权益复利扣 taker 0.05%/边 + 资金费；pnl_u 为毛利\n",
             "> 4 组设置在运行前登记，全部列出，不做选择。\n"]
    all_rows = []
    for name, p in VARIANTS:
        tr, _ = E.run(k, **p)
        tr["entry_time"] = pd.to_datetime(tr.entry_time)
        tr["exit_time"] = pd.to_datetime(tr.exit_time)
        cap = leg_capture(tr, k)
        lag_med = cap.lag.median()
        cap_ratio = cap.captured_lr.sum() / cap.leg_lr.sum()
        full = window_stats(tr, start, end, "全样本", k)
        is_ = window_stats(tr, start, SPLIT, "IS (<2024)", k)
        oos = window_stats(tr, SPLIT, end, "OOS (>=2024)", k)
        all_rows.append(dict(variant=name, lag_med=lag_med, cap_ratio=cap_ratio,
                             miss=int(cap.missed.sum()), legs=len(cap), **{f"full_{x}": y for x, y in full.items() if x != "label"},
                             **{f"oos_{x}": y for x, y in oos.items() if x != "label"}))
        lines.append(f"\n## {name}\n")
        lines.append(f"- 大波段进场中位延迟: **{lag_med:.1f} 天**；大波段对数收益捕获（按腿加权）: **{cap_ratio*100:.1f}%**；"
                     f"完全错过的腿: {int(cap.missed.sum())}/{len(cap)}\n")
        lines.append(HDR)
        lines += [fmt(full), fmt(is_), fmt(oos)]
    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(HERE, "s1_speed_variants_summary_v1_0.csv"), index=False, encoding="utf-8")
    lines.append("\n## 汇总（全部登记设置）\n")
    lines.append("| 设置 | 进场中位延迟(天) | 捕获率 | 全样本期末 | 全样本年化 | 全样本MDD | OOS 期末 | OOS 年化 | OOS MDD |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in all_rows:
        lines.append(f"| {r['variant']} | {r['lag_med']:.1f} | {r['cap_ratio']*100:.1f}% | "
                     f"{r['full_end_u']:.0f}U ({r['full_mult']:.2f}x) | {r['full_cagr']*100:+.1f}% | {r['full_mdd']*100:.1f}% | "
                     f"{r['oos_end_u']:.0f}U ({r['oos_mult']:.2f}x) | {r['oos_cagr']*100:+.1f}% | {r['oos_mdd']*100:.1f}% |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "S1_speed_variants_report_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
