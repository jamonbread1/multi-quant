# -*- coding: utf-8 -*-
"""S3 状态门控 样本外检验 (WF 因果状态) v1.0

与 s3_gate_oos_v1_0.py 的区别:
  - 状态来源: WF_4H_rule_states.csv (walk-forward 规则引擎, 4 状态, 无 TRANSITION)
    而非 S1_trades_with_states.csv 中的 hybrid 状态 (可能含前视).
  - 对齐: 开仓时刻 E 能用到的最新已收盘 4H bar 是 dt = E-4h (dt 为 K 线开盘时间, 收盘在 dt+4h).
    主口径 = E-4h (因果). 敏感性口径 = E (含一根未收盘 bar, 仅作对照, 说明前视的影响).
规则 (同 v1.0, 只用 IS 确定): 排除 IS 内均笔 R < 0 的状态.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")
TRADES = os.path.join(REPO, "S1_trades_with_states.csv")
WF = os.path.join(REPO, "WF_4H_rule_states.csv")
OUT_MD = os.path.join(HERE, "S3_gate_WF_OOS_report_v1_0.md")

SPLIT = pd.Timestamp("2024-01-01")
START_U = 30.0
N_BOOT = 2000
SEED = 20261010


def load_trades():
    t = pd.read_csv(TRADES)
    t["e"] = pd.to_datetime(t["entry_time_utc"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["ret_eq"] = t["pnl_u"] / t["equity_entry"]
    t["period"] = np.where(t["exit_time"] < SPLIT, "IS", "OOS")
    return t


def mdd(eq):
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def summarize(sub, label):
    if len(sub) == 0:
        return None
    r = sub["pnl_r"].values
    eq = np.concatenate([[START_U], START_U * np.cumprod(1 + sub["ret_eq"].values)])
    rng = np.random.default_rng(SEED)
    boots = np.array([rng.choice(r, len(r), replace=True).mean() for _ in range(N_BOOT)])
    t_stat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
    return dict(label=label, n=len(sub), win=float((sub.pnl_u > 0).mean()), mean_r=float(r.mean()),
                lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                t=float(t_stat), sum_u=float(sub.pnl_u.sum()), mult=float(eq[-1] / START_U), mdd=mdd(eq))


def row(s):
    return (f"| {s['label']} | {s['n']} | {s['win']*100:.1f}% | {s['mean_r']:+.3f} | "
            f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['t']:+.2f} | {s['sum_u']:+.1f} | "
            f"{s['mult']:.2f}x | {s['mdd']*100:.1f}% |")


HDR = ("| 方案 | 笔数 | 胜率 | 均笔R | 95% CI | t | 净盈亏U | 30U 期末 | 最大回撤 |\n"
       "|---|---|---|---|---|---|---|---|---|")


def run(t, state_col, tag, lines):
    is_t, oos_t = t[t.period == "IS"], t[t.period == "OOS"]
    is_stats = is_t.groupby(state_col)["pnl_r"].agg(["size", "mean"])
    excluded = sorted(is_stats.index[is_stats["mean"] < 0].tolist())
    gated = lambda df: df[~df[state_col].isin(excluded)]
    lines.append(f"\n## {tag}\n")
    lines.append(f"IS 排除的状态: **{', '.join(excluded) if excluded else '（无）'}**\n")
    lines.append("IS 分状态（定规则用）：\n")
    lines.append("| 状态 | IS 笔数 | IS 均笔R |\n|---|---|---|")
    for s, r in is_stats.sort_values("mean").iterrows():
        lines.append(f"| {s} | {int(r['size'])} | {r['mean']:+.3f} |")
    lines.append("\nOOS 检验:\n")
    lines.append(HDR)
    for s in [summarize(oos_t, "OOS 全部交易"), summarize(gated(oos_t), "OOS 门控后")]:
        if s:
            lines.append(row(s))
    lines.append("\n全样本（描述性，含 IS，非有效性证据）:\n")
    lines.append(HDR)
    for s in [summarize(t, "全样本 全部交易"), summarize(gated(t), "全样本 门控后")]:
        if s:
            lines.append(row(s))
    return excluded


def main():
    t = load_trades()
    w = pd.read_csv(WF)
    w["dt"] = pd.to_datetime(w["dt"])
    ws = w.set_index("dt")["state"]

    lines = ["# S3 状态门控 样本外检验报告（WF 因果状态）v1.0\n",
             "**脚本**: `research/oos/s3_gate_oos_wf_v1_0.py`  ",
             "**状态**: `WF_4H_rule_states.csv`（walk-forward 规则引擎，4 状态）  ",
             f"**切分点**: {SPLIT.date()}；规则只用 IS 确定  ",
             "**主口径**: 状态取开仓前最后一根已收盘 4H bar（dt = E-4h）。"
             "**敏感性**: dt = E（含一根未收盘 bar，存在前视）。"]

    t["wf_state"] = ws.reindex(t.e - pd.Timedelta(hours=4)).values
    t["wf_state_lookahead"] = ws.reindex(t.e).values
    cov = t["wf_state"].notna().mean()
    lines.append(f"\n覆盖率: {cov*100:.1f}%（183 笔中有 WF 状态的比例）\n")

    run(t, "wf_state", "主口径（因果）：E-4h", lines)
    run(t, "wf_state_lookahead", "敏感性（前视）：E", lines)

    # 原 hybrid 状态结果, 作对照
    run(t, "state_entry", "对照：原 hybrid 状态（S1_trades_with_states）", lines)

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
