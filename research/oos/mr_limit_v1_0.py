# -*- coding: utf-8 -*-
"""1H 限价挂单 (双向区间挂单) 预登记检验 v1.0

动机 (用户): ETH 短线大量震荡, 回归类策略在短线只会被手续费和扫损吃掉。
             改用限价挂单: 不追价、以挂单价成交, 并可能使用 maker 费率。

机制 (单标的, 1H, 同一时间最多一个仓位, 名义 = 权益 x 1.5):
  - 空仓时: 以决策 bar cl 的收盘价 c[cl] 挂 买限价 = c*(1-d) 与 卖限价 = c*(1+d), 有效 V 根 K 线。
            挂单期间不重复挂单; 过期或成交后重新挂单。
  - 买单成交: 第 k 根 low[k] <= 买价; 成交价 = min(open[k], 买价)。卖单对称 (high[k] >= 卖价, 成交价 = max(open[k], 卖价))。
  - 同一根 K 线两侧都触及: 选成交价离开盘价更近的一侧 (保守处理)。
  - 成交当根不做出场判定 (次根起判定)。
  - 出场 (按顺序): 止损 (市价, 成交 min(open, stop) / max(open, stop)) > 止盈 (限价, 成交 tp 或更优开盘价) > 时间止损 (满 T 根, 开盘价)。
    同一根 K 线止损与止盈同时触及: 按止损 (保守)。

成本:
  maker 假设 (主口径, 未核实): 挂单成交与止盈成交 = 0.02% 每边; 止损/时间止损 = taker 0.05% + 滑点 0.03%。
  敏感性 (无 maker 折扣): 所有成交 = 0.05% + 0.03% = 0.08% 每边。
  资金费: 0.004%/8h, 按持仓 1H bar 计。

配置 (预登记, 4 个; 累计 16 个以内):
  L1  d=0.5%, tp=1.0%     L2  d=0.5%, tp=2.0%
  L3  d=1.0%, tp=1.0%     L4  d=1.0%, tp=2.0%
  共用: V=6 根有效, 止损 3%, 时间止损 24 根。

判定 (预登记, 仅看 maker 主口径):
  "限价挂单有正期望" 当且仅当某配置同时满足:
    (1) 全样本单笔平均净收益 > 0 且 OOS 单笔平均净收益 > 0
    (2) 多头与空头各自的全样本单笔平均净收益 > 0
    (3) 成本 / 毛收益 < 100%
  敏感性结果 (无 maker 折扣) 只报告, 不参与判定。OOS = 2024-01 起 30U 重启。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")
sys.path.insert(0, HERE)
from leverage_v1_0 import _mdd  # noqa: E402
import s1_engine_v1_1 as E1  # noqa: E402  (仅用 atr_wilder 作为占位, 本策略不使用 ATR)

START_U = 30.0
SPLIT = pd.Timestamp("2024-01-01")
LEV = 1.5
V = 6
SL = 0.03
T = 24
FUND_1H = 0.00004 / 8
MAKER = 0.0002
TAKER_SLIP = 0.0005 + 0.0003
WARM = 50


def load_1h():
    k = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_1H.csv"))
    k["dt"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]].reset_index(drop=True)


def run(k, d, tp, maker_mode=True, start_ts=None, sl=SL):
    o, h, l, c = (k[x].values for x in ["open", "high", "low", "close"])
    n = len(c)
    start = WARM
    if start_ts is not None:
        start = max(WARM, int(np.searchsorted(k.dt.values, np.datetime64(start_ts))))
    eq = START_U
    eq_curve = np.full(n, np.nan)
    eq_worst = np.full(n, np.nan)
    eq_curve[:start] = eq
    eq_worst[:start] = eq
    trades = []
    orders_placed = 0
    fills = 0
    pos = 0
    entry = qty = stop = tp_px = 0.0
    ent_j = 0
    entry_eq = eq
    eb = es = None  # 挂单价
    expire = -1
    fee_entry = MAKER if maker_mode else TAKER_SLIP
    fee_tp = MAKER if maker_mode else TAKER_SLIP
    fee_mkt = TAKER_SLIP

    def close_pos(j, px, reason, fee_exit):
        nonlocal eq, pos
        gross = qty * (px - entry) * pos
        fees = qty * entry * fee_entry + qty * px * fee_exit
        fund = qty * entry * FUND_1H * (j - ent_j)
        net = gross - fees - fund
        eq += net
        trades.append(dict(entry_time=k.dt.iloc[ent_j], exit_time=k.dt.iloc[j], side=pos,
                           entry_px=entry, exit_px=px, bars=j - ent_j, gross=gross,
                           cost=fees + fund, net=net, equity_entry=entry_eq, reason=reason))
        pos = 0

    for j in range(start, n):
        cl = j - 1
        if pos != 0:
            if pos > 0:
                if l[j] <= stop:
                    close_pos(j, min(o[j], stop), "stop", fee_mkt)
                elif h[j] >= tp_px:
                    close_pos(j, max(o[j], tp_px), "tp", fee_tp)
                elif j - ent_j >= T:
                    close_pos(j, o[j], "time", fee_mkt)
            else:
                if h[j] >= stop:
                    close_pos(j, max(o[j], stop), "stop", fee_mkt)
                elif l[j] <= tp_px:
                    close_pos(j, min(o[j], tp_px), "tp", fee_tp)
                elif j - ent_j >= T:
                    close_pos(j, o[j], "time", fee_mkt)
        else:
            # 1) 挂单成交判定 (本根 j)
            buy_hit = eb is not None and l[j] <= eb and j <= expire
            sell_hit = es is not None and h[j] >= es and j <= expire
            if buy_hit or sell_hit:
                if buy_hit and sell_hit:
                    side = 1 if abs(o[j] - eb) <= abs(o[j] - es) else -1
                else:
                    side = 1 if buy_hit else -1
                px = min(o[j], eb) if side > 0 else max(o[j], es)
                pos = side
                entry = px
                ent_j = j
                entry_eq = eq
                qty = eq * LEV / px
                if sl is None:  # 批次2: 无止损 (多头 stop=0 永不触发, 空头 stop=1e18 永不触发)
                    stop = 0.0 if side > 0 else 1e18
                else:
                    stop = entry * (1 - side * sl)
                tp_px = entry * (1 + side * tp)
                eb = es = None
                fills += 1
            elif eb is None and es is None or j > expire:
                # 2) 无有效挂单时, 以 cl 收盘价挂双向限价
                eb = c[cl] * (1 - d)
                es = c[cl] * (1 + d)
                expire = j + V - 1
                orders_placed += 1
        eq_curve[j] = eq
        if pos != 0:
            adv = (l[j] - entry) if pos > 0 else (entry - h[j])
            eq_worst[j] = min(eq, eq + qty * adv)
        else:
            eq_worst[j] = eq
    tr = pd.DataFrame(trades)
    return tr, eq_curve, eq_worst, orders_placed, fills


def metrics(tr, eq_worst, orders, fills):
    n = len(tr)
    if n == 0:
        return dict(n=0, orders=orders, fills=fills)
    first = pd.to_datetime(tr.entry_time).min()
    last = pd.to_datetime(tr.exit_time).max()
    years = max((last - first).days / 365.25, 1e-9)
    eqs = START_U * np.cumprod(1 + tr.net.values / tr.equity_entry.values)
    end_eq = float(eqs[-1])
    mult = end_eq / START_U
    cagr = mult ** (1 / years) - 1 if mult > 0 else -1.0
    mdd_real = _mdd(np.concatenate([[START_U], eqs]))
    mdd_i = _mdd(eq_worst)
    cal = cagr / abs(mdd_i) if mdd_i < 0 else np.nan
    gross = tr.gross.sum()
    cost = tr.cost.sum()
    pct = tr.net / tr.equity_entry * 100
    lg = pct[tr.side > 0]
    sh = pct[tr.side < 0]
    return dict(n=n, n_per_yr=n / years, fill_rate=fills / orders if orders else np.nan,
                tp_share=(tr.reason == "tp").mean(), win=(tr.net > 0).mean(),
                avg_net=pct.mean(), long_avg=lg.mean() if len(lg) else np.nan,
                short_avg=sh.mean() if len(sh) else np.nan,
                cost_over_gross=(cost / gross) if gross > 0 else np.nan,
                end_u=end_eq, cagr=cagr, mdd_realized=mdd_real, mdd_intra=mdd_i, calmar=cal,
                avg_bars=tr.bars.mean())


CFG = {
    "L1 d0.5% tp1.0%": dict(d=0.005, tp=0.010),
    "L2 d0.5% tp2.0%": dict(d=0.005, tp=0.020),
    "L3 d1.0% tp1.0%": dict(d=0.010, tp=0.010),
    "L4 d1.0% tp2.0%": dict(d=0.010, tp=0.020),
    # 批次2 (追加登记): 同挂单, 无止损, 仅止盈 + 24 根时间止损
    "N1 d0.5% tp1.0% 无止损": dict(d=0.005, tp=0.010, sl=None),
    "N2 d0.5% tp2.0% 无止损": dict(d=0.005, tp=0.020, sl=None),
    "N3 d1.0% tp1.0% 无止损": dict(d=0.010, tp=0.010, sl=None),
    "N4 d1.0% tp2.0% 无止损": dict(d=0.010, tp=0.020, sl=None),
}


def fmt_p(x):
    return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"


def fmt_pp(x):
    return "—" if (x is None or pd.isna(x)) else f"{x:+.2f}%"


def fmt_2(x):
    return "—" if (x is None or pd.isna(x)) else f"{x:.2f}"


def main():
    k = load_1h()
    rows = []
    for name, cfg in CFG.items():
        for mode, maker in [("maker", True), ("taker", False)]:
            sl = cfg.get("sl", SL)
            tr, eqc, eqw, orders, fills = run(k, cfg["d"], cfg["tp"], maker_mode=maker, sl=sl)
            full = metrics(tr, eqw, orders, fills)
            tro, eqo, eqwo, ordo, fillo = run(k, cfg["d"], cfg["tp"], maker_mode=maker, start_ts=SPLIT, sl=sl)
            tro = tro[pd.to_datetime(tro.entry_time) >= SPLIT].reset_index(drop=True)
            oos = metrics(tro, eqwo, ordo, fillo)
            rows.append(dict(cfg=name, mode=mode, **{f"full_{x}": y for x, y in full.items()},
                             **{f"oos_{x}": y for x, y in oos.items()}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "mr_limit_v1_0.csv"), index=False, encoding="utf-8")

    L = ["# 1H 限价挂单（双向区间挂单）预登记检验 v1.0\n",
         "**脚本**: `research/oos/mr_limit_v1_0.py`（独立引擎）。  ",
         "**主口径 maker**：挂单成交与止盈 0.02%/边（**假设，未核实**，沙箱无法访问 OKX）；止损与时间止损 0.05% + 滑点 0.03%。  ",
         "**敏感性 taker**：全部 0.05% + 0.03%（无 maker 折扣）。资金费 0.004%/8h。  ",
         "**OOS** = 2024-01 起 30U 重启。\n",
         "## 结果\n",
         "| 配置 | 口径 | 挂单成交率 | 笔/年 | 止盈占比 | 胜率 | 单笔均净 | 多头 | 空头 | 成本/毛收益 | CAGR | 盘中MDD | OOS 笔数 | OOS 单笔均净 | OOS CAGR |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r.cfg} | {r['mode']} | {fmt_p(r.full_fill_rate)} | {r.full_n_per_yr:.0f} | {fmt_p(r.full_tp_share)} | "
                 f"{fmt_p(r.full_win)} | {fmt_pp(r.full_avg_net)} | {fmt_pp(r.full_long_avg)} | {fmt_pp(r.full_short_avg)} | "
                 f"{fmt_p(r.full_cost_over_gross)} | {fmt_p(r.full_cagr)} | {fmt_p(r.full_mdd_intra)} | "
                 f"{int(r.oos_n)} | {fmt_pp(r.oos_avg_net)} | {fmt_p(r.oos_cagr)} |")

    L.append("\n## 判定（预登记，仅 maker 主口径）\n")
    any_ok = False
    for name in CFG:
        r = R[(R.cfg == name) & (R["mode"] == "maker")].iloc[0]
        c1 = (r.full_avg_net > 0) and (r.oos_avg_net > 0)
        c2 = (not pd.isna(r.full_long_avg)) and (not pd.isna(r.full_short_avg)) and r.full_long_avg > 0 and r.full_short_avg > 0
        c3 = (not pd.isna(r.full_cost_over_gross)) and r.full_cost_over_gross < 1.0
        ok = c1 and c2 and c3
        any_ok |= ok
        L.append(f"- {name}: (1) 两段单笔均净>0 {'满足' if c1 else '不满足'} · (2) 多空均>0 {'满足' if c2 else '不满足'} · "
                 f"(3) 成本/毛<100% {'满足' if c3 else '不满足'} → **{'通过' if ok else '未通过'}**")
    L.append(f"\n**结论: {'限价挂单有正期望（maker 假设需核实后再采信）' if any_ok else '限价挂单未通过预登记条件'}**")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "mr_limit_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
