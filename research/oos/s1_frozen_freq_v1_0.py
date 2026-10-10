# -*- coding: utf-8 -*-
"""S1 冻结版复核 + 交易频率实验 v1.0

成本口径 = 冻结版声明 (turtle_4h_ETH_frozen_lev15.py 冻结头 / 绩效表):
  taker 0.05%/边 + 滑点 0.03%/边 + 资金费 0.004%/8h; 强平 MMR 0.5% (逐仓近似)。
引擎: s1_engine_v1_3.py (默认参数 = 冻结版逻辑)。

配置 (预登记, 共 5 个, 上限 16):
  F0  冻结版 lev1.5 (don48 / ADX15 / 止损后需等信号回 0 才重入)           ← 基准
  F0b 冻结版 lev0.65 (冻结版另一档, 仅对照)
  F1  止损后允许立即重入 (reentry_block=False)                           ← 增加信号
  F2  去 ADX 门控 (adx_min=0)                                            ← 增加信号
  F3  don24 更快突破 (出场 12)                                          ← 增加信号

判定 (预登记, 仅 OOS 与全样本同时看):
  "提高交易频率有改善" 当且仅当某配置 (F1/F2/F3) 同时满足:
    全样本 CAGR > F0 且 全样本 Calmar >= F0 且 OOS CAGR >= F0 的 OOS CAGR
  全样本 = 2020-01 起; OOS = 2024-01 起 30U 重启。回撤为盘中最差 MTM (主口径)。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_engine_v1_3 as E  # noqa: E402
import state_lead_v1_0 as SL  # noqa: E402
from leverage_v1_0 import _mdd  # noqa: E402

START_U = 30.0
SPLIT = pd.Timestamp("2024-01-01")
WARM = 198
FEE, SLIP = 0.0005, 0.0003

CFG = {
    "F0 冻结版 lev1.5": dict(lev=1.5, don_n=48, adx_min=15, reentry_block=True),
    "F0b 冻结版 lev0.65": dict(lev=0.65, don_n=48, adx_min=15, reentry_block=True),
    "F1 止损后立即重入": dict(lev=1.5, don_n=48, adx_min=15, reentry_block=False),
    "F2 去ADX门控": dict(lev=1.5, don_n=48, adx_min=0, reentry_block=True),
    "F3 don24 更快突破": dict(lev=1.5, don_n=24, adx_min=15, reentry_block=True),
}


def run_cfg(k, p, start_ts=None, end_ts=None):
    if start_ts is None:
        kk, first_ts = k, None
    else:
        i0 = int(np.searchsorted(k.dt.values, np.datetime64(start_ts))) - WARM
        kk = k.iloc[i0:].reset_index(drop=True)
    tr, eq, eqw = E.run(kk.copy(), don_n=p["don_n"], lev=p["lev"], adx_min=p["adx_min"],
                        fee_side=FEE, slip_side=SLIP, fund=True, liq=True, mmr=0.005,
                        reentry_block=p["reentry_block"])
    if start_ts is not None:
        tr = tr[pd.to_datetime(tr.entry_time) >= start_ts]
    if end_ts is not None:
        tr = tr[pd.to_datetime(tr.entry_time) < end_ts]
    tr = tr.reset_index(drop=True)
    if len(tr):
        first = pd.to_datetime(tr.entry_time).min()
        last = pd.to_datetime(tr.exit_time).max()
        years = max((last - first).days / 365.25, 1e-9)
    else:
        years = 1.0
    end_eq = float(eq[-1]) if start_ts is None else float(eq[-1])
    mult = end_eq / START_U
    cagr = mult ** (1 / years) - 1 if mult > 0 else -1.0
    mdd_i = _mdd(eqw)
    mdd_r = _mdd(eq)
    calmar = cagr / abs(mdd_i) if mdd_i < 0 else np.nan
    wr = (tr.pnl_net > 0).mean() if len(tr) else np.nan
    return dict(n=len(tr), win=wr, end_u=end_eq, cagr=cagr, mdd_intra=mdd_i, mdd_real=mdd_r,
                calmar=calmar, liq=getattr(E.run, "last_n_liq", 0))


def main():
    k = SL.load_k()
    rows = []
    for name, p in CFG.items():
        full = run_cfg(k, p)
        oos = run_cfg(k, p, start_ts=SPLIT)
        rows.append(dict(cfg=name, **{f"full_{x}": y for x, y in full.items()},
                         **{f"oos_{x}": y for x, y in oos.items()}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "s1_frozen_freq_v1_0.csv"), index=False, encoding="utf-8")

    def p_(x):
        return "—" if pd.isna(x) else f"{x*100:.1f}%"

    def f2(x):
        return "—" if pd.isna(x) else f"{x:.2f}"

    L = ["# S1 冻结版复核与交易频率实验 v1.0\n",
         "**脚本**: `research/oos/s1_frozen_freq_v1_0.py`；引擎 `s1_engine_v1_3.py`（默认 = 冻结版逻辑）。  ",
         "**成本**: taker 0.05%/边 + 滑点 0.03%/边 + 资金费 0.004%/8h（冻结头口径）。  ",
         "**回撤**: 盘中最差 MTM（主口径）。OOS = 2024-01 起 30U 重启。\n",
         "## 结果\n",
         "| 配置 | 全样本笔数 | 胜率 | 期末 | CAGR | 盘中MDD | Calmar | OOS笔数 | OOS CAGR | OOS MDD | OOS Calmar |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r.cfg} | {int(r.full_n)} | {p_(r.full_win)} | {r.full_end_u:.1f}U | {p_(r.full_cagr)} | "
                 f"{p_(r.full_mdd_intra)} | {f2(r.full_calmar)} | {int(r.oos_n)} | {p_(r.oos_cagr)} | "
                 f"{p_(r.oos_mdd_intra)} | {f2(r.oos_calmar)} |")

    b = R[R.cfg.str.startswith("F0 ")].iloc[0]
    L.append("\n## 判定（预登记）\n")
    any_ok = False
    for _, r in R[R.cfg.str.match(r"F[123] ")].iterrows():
        c1 = r.full_cagr > b.full_cagr
        c2 = r.full_calmar >= b.full_calmar
        c3 = r.oos_cagr >= b.oos_cagr
        ok = c1 and c2 and c3
        any_ok |= ok
        L.append(f"- {r.cfg}: 全样本CAGR {'>' if c1 else '≤'} F0 · 全样本Calmar {'≥' if c2 else '<'} F0 · OOS CAGR {'≥' if c3 else '<'} F0 → {'满足' if ok else '不满足'}")
    L.append(f"\n**结论: {'有改善' if any_ok else '无改善（更多信号未带来更好结果）'}**")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "s1_frozen_freq_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
