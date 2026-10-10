# -*- coding: utf-8 -*-
"""短周期趋势 (1H 突破 + 短持仓) 预登记检验 v1.0

动机: 上一轮 F1-F3 只是在 4H 上增加 S1 的信号, 没有测试真正的高频策略。
      本轮把同一套突破逻辑搬到 1H, 周期更短, 检验高频是否能在冻结成本下改善结果。

成本口径 = 冻结版: taker 0.05%/边 + 滑点 0.03%/边 + 资金费 0.004%/8h (按 1H 折算), 杠杆 1.5, 强平 MMR 0.5%。
引擎: s1_engine_v1_3.py (周期无关; 资金费常量按 1H 临时覆盖, 仅在本脚本内生效)。

配置 (预登记, 共 5 个, 上限 16):
  F0    冻结版 4H don48 (基准, 与上一轮相同)
  H24   1H don24 / 出场 12
  H48   1H don48 / 出场 24
  H96   1H don96 / 出场 48
  H192  1H don192 / 出场 96 (= 4H don48 的 1H 等价窗口, 对照)
其它参数 (ADX15, 3ATR 吊灯, 9% 止损封顶, 止损后需信号回零再入场) 与冻结版一致, 不做扫描。

数据: 1H = ETH-USDT-SWAP_1H.csv (OKX 1H, 2020-01-01 起, 无缺口); 4H 基准沿用上一轮数据。

判定 (预登记, 与上一轮一致, 仅 OOS 与全样本同时看):
  H* "有改善" 当且仅当某配置同时满足:
    全样本 CAGR > F0 且 全样本 Calmar >= F0 且 OOS CAGR >= F0 的 OOS CAGR
  Calmar 用盘中最差 MTM 回撤。OOS = 2024-01 起 30U 重启。
同时报告: 交易笔数/年、单笔均净收益、成本占毛收益比 (成本效率), 用于判断高频为何成立或失败。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")
sys.path.insert(0, HERE)
import s1_engine_v1_3 as E  # noqa: E402
import state_lead_v1_0 as SL  # noqa: E402
from leverage_v1_0 import _mdd  # noqa: E402

START_U = 30.0
SPLIT = pd.Timestamp("2024-01-01")
FEE, SLIP = 0.0005, 0.0003
FUND_4H = 0.00004 * 4 / 8
FUND_1H = 0.00004 * 1 / 8


def load_1h():
    k = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_1H.csv"))
    k["dt"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]].reset_index(drop=True)


CFG = {
    "F0 冻结版 4H don48": dict(tf="4H", don_n=48),
    "H24 1H don24": dict(tf="1H", don_n=24),
    "H48 1H don48": dict(tf="1H", don_n=48),
    "H96 1H don96": dict(tf="1H", don_n=96),
    "H192 1H don192 (4H等价)": dict(tf="1H", don_n=192),
}


def run_cfg(k4, k1, p, start_ts=None):
    if p["tf"] == "4H":
        k, fund = k4, FUND_4H
    else:
        k, fund = k1, FUND_1H
    E.FUND_PER_BAR = fund
    if start_ts is None:
        kk = k
    else:
        warm = p["don_n"] + 150
        i0 = max(0, int(np.searchsorted(k.dt.values, np.datetime64(start_ts))) - warm)
        kk = k.iloc[i0:].reset_index(drop=True)
    tr, eq, eqw = E.run(kk.copy(), don_n=p["don_n"], lev=1.5, adx_min=15,
                        fee_side=FEE, slip_side=SLIP, fund=True, liq=True, mmr=0.005)
    if start_ts is not None:
        tr = tr[pd.to_datetime(tr.entry_time) >= start_ts]
    tr = tr.reset_index(drop=True)
    n = len(tr)
    if n:
        first = pd.to_datetime(tr.entry_time).min()
        last = pd.to_datetime(tr.exit_time).max()
        years = max((last - first).days / 365.25, 1e-9)
    else:
        years = 1.0
    end_eq = float(eq[-1])
    mult = end_eq / START_U
    cagr = mult ** (1 / years) - 1 if mult > 0 else -1.0
    mdd_i = _mdd(eqw)
    calmar = cagr / abs(mdd_i) if mdd_i < 0 else np.nan
    # 成本效率: 手续费+资金费 vs 毛收益 (毛收益 = pnl_u; 成本 = 毛 - 净)
    gross = float(tr.pnl_u.sum()) if n else np.nan
    cost = float((tr.pnl_u - tr.pnl_net).sum()) if n else np.nan
    return dict(n=n, trades_per_yr=(n / years if n else np.nan), win=(tr.pnl_net > 0).mean() if n else np.nan,
                avg_net_pct=(tr.pnl_net / tr.equity_entry).mean() * 100 if n else np.nan,
                cost_over_gross=(cost / gross if gross and gross > 0 else np.nan),
                end_u=end_eq, cagr=cagr, mdd_intra=mdd_i, calmar=calmar)


def main():
    k4 = SL.load_k()
    k1 = load_1h()
    rows = []
    for name, p in CFG.items():
        full = run_cfg(k4, k1, p)
        oos = run_cfg(k4, k1, p, start_ts=SPLIT)
        rows.append(dict(cfg=name, **{f"full_{x}": y for x, y in full.items()},
                         **{f"oos_{x}": y for x, y in oos.items()}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "s1_short_trend_v1_0.csv"), index=False, encoding="utf-8")

    def p_(x):
        return "—" if pd.isna(x) else f"{x*100:.1f}%"

    def f2(x):
        return "—" if pd.isna(x) else f"{x:.2f}"

    L = ["# 短周期趋势（1H 突破 + 短持仓）预登记检验 v1.0\n",
         "**脚本**: `research/oos/s1_short_trend_v1_0.py`；引擎 `s1_engine_v1_3.py`。  ",
         "**成本**: 冻结版（手续费 0.05% + 滑点 0.03% 每边，资金费按周期折算），LEV 1.5，强平 MMR 0.5%。  ",
         "**回撤**: 盘中最差 MTM。OOS = 2024-01 起 30U 重启。\n",
         "## 结果\n",
         "| 配置 | 全样本笔数 | 笔/年 | 胜率 | 单笔均净收益 | 成本/毛收益 | 期末 | CAGR | 盘中MDD | Calmar | OOS CAGR | OOS MDD | OOS Calmar |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r.cfg} | {int(r.full_n)} | {r.full_trades_per_yr:.0f} | {p_(r.full_win)} | "
                 f"{'—' if pd.isna(r.full_avg_net_pct) else f'{r.full_avg_net_pct:+.2f}%'} | "
                 f"{p_(r.full_cost_over_gross)} | {r.full_end_u:.1f}U | {p_(r.full_cagr)} | {p_(r.full_mdd_intra)} | "
                 f"{f2(r.full_calmar)} | {p_(r.oos_cagr)} | {p_(r.oos_mdd_intra)} | {f2(r.oos_calmar)} |")

    b = R[R.cfg.str.startswith("F0")].iloc[0]
    L.append("\n## 判定（预登记）\n")
    any_ok = False
    for _, r in R[~R.cfg.str.startswith("F0")].iterrows():
        c1 = r.full_cagr > b.full_cagr
        c2 = r.full_calmar >= b.full_calmar
        c3 = r.oos_cagr >= b.oos_cagr
        ok = c1 and c2 and c3
        any_ok |= ok
        L.append(f"- {r.cfg}: 全样本CAGR {'>' if c1 else '≤'} F0 · 全样本Calmar {'≥' if c2 else '<'} F0 · OOS CAGR {'≥' if c3 else '<'} F0 → {'满足' if ok else '不满足'}")
    L.append(f"\n**结论: {'高频有改善' if any_ok else '高频未改善（预登记条件均不满足）'}**")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "s1_short_trend_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
