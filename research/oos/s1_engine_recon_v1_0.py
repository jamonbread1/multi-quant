# -*- coding: utf-8 -*-
"""S1 引擎对账 v1.0: 研究引擎 vs FMZ 真实日志 (S1_trades_with_states.csv)."""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_engine_v1_0 as E  # noqa: E402

REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")


def load_k():
    # FMZ 回测所用数据起点为 2019-12-25 (预热 198 根后首笔开仓于 2020-01-28, 与日志一致)
    k = pd.read_csv(os.path.join(HERE, "data", "eth4h_from_fmz_1h.csv"))
    k["dt"] = pd.to_datetime(k["dt"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]]


def main():
    k = load_k()
    tr, _ = E.run(k, don_n=48)
    log = pd.read_csv(os.path.join(REPO, "S1_trades_with_states.csv"))
    log["entry_time_utc"] = pd.to_datetime(log["entry_time_utc"])
    log["exit_time_utc"] = pd.to_datetime(log["exit_time_utc"])
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    m = tr.merge(
        log[["entry_time_utc", "side", "exit_time_utc", "entry_px", "exit_px", "pnl_u"]],
        left_on=["entry_time", "side"], right_on=["entry_time_utc", "side"],
        how="outer", suffixes=("_eng", "_log"), indicator=True)
    both = m[m["_merge"] == "both"]
    e_only = m[m["_merge"] == "left_only"]
    l_only = m[m["_merge"] == "right_only"]
    exit_match = (both["exit_time"] == both["exit_time_utc"]).mean() if len(both) else np.nan
    px_rel = (both["entry_px_eng"] - both["entry_px_log"]).abs() / both["entry_px_log"]

    me = E.metrics(tr)
    lg = log.assign(exit_time=log.exit_time_utc, entry_time=log.entry_time_utc)
    ml = E.metrics(lg, col="pnl_u")
    out = [
        f"- 日志交易: {len(log)} 笔; 引擎交易: {len(tr)} 笔",
        f"- 入场时间+方向 精确匹配: {len(both)} 笔 ({len(both)/len(log)*100:.1f}% 日志)",
        f"- 仅引擎有: {len(e_only)} 笔; 仅日志有: {len(l_only)} 笔",
        f"- 匹配笔中离场时间一致率: {exit_match*100:.1f}%",
        f"- 匹配笔入场价相对偏差: 中位数 {px_rel.median()*100:.3f}%, 最大 {px_rel.max()*100:.2f}%",
        f"- 引擎终值: {me['end_u']:.1f}U ({me['mult']:.2f}x), CAGR {me['cagr']*100:.1f}%, MDD {me['mdd']*100:.1f}%",
        f"- 日志终值: {ml['end_u']:.1f}U ({ml['mult']:.2f}x), CAGR {ml['cagr']*100:.1f}%, MDD {ml['mdd']*100:.1f}%",
    ]
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "s1_engine_recon_v1_0.md"), "w", encoding="utf-8") as f:
        f.write("# S1 引擎对账 v1.0\n\n" + txt + "\n")
    m.to_csv(os.path.join(HERE, "s1_engine_vs_log_v1_0.csv"), index=False)


if __name__ == "__main__":
    main()
