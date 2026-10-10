# -*- coding: utf-8 -*-
"""S3 状态门控 样本外检验 v1.0

问题: 「排除某状态的交易」这条门控在样本外是否仍有效?
数据: S1_trades_with_states.csv (183 笔 FMZ 真实日志 × 开仓时状态标签)
规则 (只用样本内 IS = 平仓时间 < 2024-01-01 确定, 运行前固定):
    排除 IS 内均笔 R < 0 的状态。
    注意: 此规则在全样本上发现 (v1.1 脚本), 本次只是把它作为「先在 IS 定规则, 再测 OOS」的检验。
评估: 对 OOS (>=2024) 比较「全部交易」与「门控后」的笔数、均 R、t、bootstrap CI、
      净盈亏、30U 逐笔复利期末与最大回撤。全样本的门控结果仅作描述性对照。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MR_REPO", "/home/user/multi-quant")
SRC = os.path.join(ROOT, "S1_trades_with_states.csv")
OUT_MD = os.path.join(HERE, "S3_gate_OOS_report_v1_0.md")

SPLIT = pd.Timestamp("2024-01-01")
START_U = 30.0
N_BOOT = 2000
SEED = 20261010


def load():
    t = pd.read_csv(SRC)
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["entry_time"] = pd.to_datetime(t["entry_time"])
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
                t=float(t_stat), sum_u=float(sub.pnl_u.sum()), mult=float(eq[-1] / START_U),
                mdd=mdd(eq))


def row(s):
    return (f"| {s['label']} | {s['n']} | {s['win']*100:.1f}% | {s['mean_r']:+.3f} | "
            f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['t']:+.2f} | {s['sum_u']:+.1f} | "
            f"{s['mult']:.2f}x | {s['mdd']*100:.1f}% |")


HDR = ("| 方案 | 笔数 | 胜率 | 均笔R | 95% CI | t | 净盈亏U | 30U 期末 | 最大回撤 |\n"
       "|---|---|---|---|---|---|---|---|---|")


def main():
    t = load()
    is_t = t[t.period == "IS"]
    oos_t = t[t.period == "OOS"]

    # --- 1. 仅用 IS 定规则 ---
    is_stats = is_t.groupby("state_entry")["pnl_r"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "mean_r"})
    excluded = sorted(is_stats.index[is_stats["mean_r"] < 0].tolist())
    print("IS 排除的状态:", excluded)

    gated = lambda df: df[~df.state_entry.isin(excluded)]

    # --- 2. 样本外检验 ---
    oos_rows = [summarize(oos_t, "OOS 全部交易 (基准)"),
                summarize(gated(oos_t), f"OOS 门控后 (排除 {','.join(excluded)})")]
    oos_rows = [r for r in oos_rows if r]

    # --- 3. 分状态 OOS 明细 ---
    by_state = []
    for s, g in oos_t.groupby("state_entry"):
        ss = summarize(g, s)
        by_state.append(ss)
    by_state = sorted(by_state, key=lambda x: x["mean_r"])

    # --- 4. 描述性: 全样本门控 (与项目既有 457U→943U 对照) ---
    full_rows = [summarize(t, "全样本 全部交易"),
                 summarize(gated(t), f"全样本 门控后 (排除 {','.join(excluded)})")]

    # --- 5. IS 对照 ---
    is_rows = [summarize(is_t, "IS 全部交易"), summarize(gated(is_t), "IS 门控后")]

    lines = []
    lines.append("# S3 状态门控 样本外检验报告 v1.0\n")
    lines.append("**脚本**: `research/oos/s3_gate_oos_v1_0.py`  ")
    lines.append("**数据**: `S1_trades_with_states.csv`（183 笔 FMZ 真实日志 × 开仓状态）  ")
    lines.append(f"**切分点**: {SPLIT.date()}；**规则只用 IS 确定**：排除 IS 内均笔 R < 0 的状态  ")
    lines.append(f"**IS 排除的状态**: {', '.join(excluded)}\n")

    lines.append("## 1. IS 内各状态表现（用于定规则）\n")
    lines.append("| 状态 | IS 笔数 | IS 均笔R | 处理 |")
    lines.append("|---|---|---|---|")
    for s, r in is_stats.sort_values("mean_r").iterrows():
        lines.append(f"| {s} | {int(r.n)} | {r.mean_r:+.3f} | {'排除' if s in excluded else '保留'} |")
    lines.append("")

    lines.append("## 2. 样本外检验（核心结论）\n")
    lines.append(HDR)
    lines += [row(r) for r in oos_rows]
    lines.append("")

    lines.append("## 3. OOS 分状态明细（描述性）\n")
    lines.append(HDR)
    lines += [row(r) for r in by_state]
    lines.append("")

    lines.append("## 4. 对照：IS 与全样本\n")
    lines.append(HDR)
    lines += [row(r) for r in is_rows + full_rows]
    lines.append("")
    lines.append("> 全样本门控的结果包含 IS 定规则所用的数据，属于样本内，只作描述性对照，不作为有效性证据。\n")

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
