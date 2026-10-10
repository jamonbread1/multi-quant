# -*- coding: utf-8 -*-
"""1H 限价入场 + 非固定出场 (动态/追踪) 预登记检验 v1.0

设计依据 (搜索整理, 不作为证据):
  - r/algotrading 等论坛: 追踪止损适合趋势; 均值回归常用时间止损或“回到参考点即出场”。
  - GitHub crypto-watcher #221 的检验原则: 出场规则必须与 (a) 同入场的纯时间出场 和 (b) 同出场的随机入场 比较,
    否则无法区分收益来自入场还是出场。
  - Elite/Reddit 讨论: 吊灯止损的改进常常很小, 且可能与已有 ATR/ADX 过滤重复。

入场 (固定, 沿用 mr_limit_v1_0): 双向限价挂单, 距收盘 d=0.5%, 有效 6 根; 成交价 = 挂单价或更差开盘价。
  同根两侧都触及: 选离开盘价更近的一侧。成交当根不做出场判定。

出场 (6 种, 预登记):
  X0  纯时间出场: 满 24 根, 开盘价平仓。无任何价格止损 (对照)。
  X1  吊灯 ATR22, m=3: 多头 stop_t = max(stop_{t-1}, 入场后最高价(至 cl) - 3*ATR22(cl)); 空头对称。
  X2  吊灯 ATR22, m=2.5 (同 X1, 更紧)。
  X3  Donchian 追踪: 多头 stop = max(stop, 最近 12 根已收盘 bar 的最低价); 空头对称。
  X4  均值回归出场: 多头 z20(cl) >= 0 出场, 空头 z20(cl) <= 0 出场。
  X5  保本 + 吊灯: 收盘浮盈 >= 初始风险 R 后, stop = max(stop, 入场 *(1+0.15%)); 之后 吊灯 m=3。
  所有 X1-X5 共用: 初始灾难性止损 = 入场 - 3*ATR14(cl), 空头对称 (ATR 比例, 非固定百分比); 满 24 根时间出场。

成本: 挂单成交 maker 0.02%/边 (主口径, 假设未核实); 所有出场为市价 (taker 0.05% + 滑点 0.03%)。
      敏感性: 全部 taker 0.08%/边。资金费 0.004%/8h。杠杆 1.5。

对照 (按检验原则):
  C-time  = X0 (同入场的纯时间出场)。
  C-rand  = 同出场规则, 入场改为随机: 每根空仓 bar 以概率 p 随机开多/空, 开盘价成交 (3 个随机种子平均)。
           p 取使随机交易数约等于真实入场数的值。比较口径: 毛收益 (排除成本差异)。

判定 (预登记, 仅 maker 主口径; 对每个 X1-X5):
  通过 当且仅当同时满足:
    P1 全样本与 OOS 的单笔平均净收益均 > 0;
    P2 全样本单笔平均净收益 > X0 的全样本单笔平均净收益 (优于纯时间出场);
    P3 全样本 (真实入场 - 随机入场) 的单笔平均毛收益 > 0 (入场携带信息, 不只是出场在制造收益)。
OOS = 2024-01 起 30U 重启。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")
sys.path.insert(0, HERE)
from leverage_v1_0 import _mdd  # noqa: E402
import s1_engine_v1_1 as E1  # noqa: E402

START_U = 30.0
SPLIT = pd.Timestamp("2024-01-01")
LEV = 1.5
D = 0.005
V = 6
T = 24
FUND_1H = 0.00004 / 8
MAKER = 0.0002
TAKER_SLIP = 0.0008
WARM = 220
SEEDS = [11, 22, 33]
BE_BUF = 0.0015
DON_M = 12


def load_1h():
    k = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_1H.csv"))
    k["dt"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]].reset_index(drop=True)


EXITS = {
    "X0 纯时间": dict(kind="time"),
    "X1 吊灯ATR22 m3": dict(kind="chand", m=3.0),
    "X2 吊灯ATR22 m2.5": dict(kind="chand", m=2.5),
    "X3 Donchian12 追踪": dict(kind="don", M=DON_M),
    "X4 均值回归出场": dict(kind="mean"),
    "X5 保本+吊灯m3": dict(kind="be"),
}


def zscore20(c):
    s = pd.Series(c)
    m = s.rolling(20).mean()
    sd = s.rolling(20).std()
    return ((s - m) / (sd + 1e-12)).values


def simulate(k, exit_cfg, maker_mode=True, start_ts=None, random_p=None, seed=None):
    o, h, l, c = (k[x].values for x in ["open", "high", "low", "close"])
    n = len(c)
    atr14 = np.array(E1.atr_wilder(h, l, c, 14))
    atr22 = np.array(E1.atr_wilder(h, l, c, 22))
    z = zscore20(c)
    start = WARM
    if start_ts is not None:
        start = max(WARM, int(np.searchsorted(k.dt.values, np.datetime64(start_ts))))
    rng = np.random.default_rng(seed) if random_p is not None else None
    fee_entry = MAKER if maker_mode else TAKER_SLIP
    fee_exit = TAKER_SLIP
    kind = exit_cfg["kind"]
    st = dict(eq=START_U, pos=0, entry=0.0, qty=0.0, stop=0.0, ent_j=0, entry_eq=START_U,
              R=0.0, hh=0.0, ll=0.0, be_done=False, eb=None, es=None, expire=-1)
    eq_worst = np.full(n, np.nan)
    eq_worst[:start] = START_U
    trades = []

    def do_open(j, side, fill):
        st["pos"] = side
        st["entry"] = fill
        st["ent_j"] = j
        st["entry_eq"] = st["eq"]
        st["qty"] = st["eq"] * LEV / fill
        st["hh"] = h[j]
        st["ll"] = l[j]
        st["be_done"] = False
        d0 = atr14[j - 1] * 3.0
        st["R"] = d0
        st["stop"] = fill - side * d0  # 灾难性止损 (ATR 比例)
        if kind == "time":
            st["stop"] = 0.0 if side > 0 else 1e18

    def do_close(j, px, reason):
        side = st["pos"]
        gross = st["qty"] * (px - st["entry"]) * side
        fees = st["qty"] * st["entry"] * fee_entry + st["qty"] * px * fee_exit
        fund = st["qty"] * st["entry"] * FUND_1H * (j - st["ent_j"])
        net = gross - fees - fund
        st["eq"] += net
        trades.append(dict(entry_time=k.dt.iloc[st["ent_j"]], exit_time=k.dt.iloc[j], side=side,
                           entry_px=st["entry"], exit_px=px, bars=j - st["ent_j"], gross=gross,
                           cost=fees + fund, net=net, equity_entry=st["entry_eq"], reason=reason))
        st["pos"] = 0

    for j in range(start, n):
        cl = j - 1
        if st["pos"] != 0:
            side = st["pos"]
            # 1) 更新追踪 (只用 <= cl)
            if kind in ("chand", "be"):
                m = exit_cfg.get("m", 3.0)
                if side > 0:
                    st["stop"] = max(st["stop"], st["hh"] - m * atr22[cl])
                else:
                    st["stop"] = min(st["stop"], st["ll"] + m * atr22[cl])
            if kind == "don":
                M = exit_cfg["M"]
                if side > 0:
                    st["stop"] = max(st["stop"], l[cl - M + 1:cl + 1].min())
                else:
                    st["stop"] = min(st["stop"], h[cl - M + 1:cl + 1].max())
            if kind == "be" and not st["be_done"]:
                fav = (c[cl] - st["entry"]) * side
                if fav >= st["R"]:
                    st["be_done"] = True
                    lvl = st["entry"] * (1 + side * BE_BUF)
                    st["stop"] = max(st["stop"], lvl) if side > 0 else min(st["stop"], lvl)
            # 2) 本根出场判定
            if side > 0:
                if kind != "time" and l[j] <= st["stop"]:
                    do_close(j, min(o[j], st["stop"]), "stop")
                elif kind == "mean" and z[cl] >= 0:
                    do_close(j, o[j], "mean")
                elif j - st["ent_j"] >= T:
                    do_close(j, o[j], "time")
            else:
                if kind != "time" and h[j] >= st["stop"]:
                    do_close(j, max(o[j], st["stop"]), "stop")
                elif kind == "mean" and z[cl] <= 0:
                    do_close(j, o[j], "mean")
                elif j - st["ent_j"] >= T:
                    do_close(j, o[j], "time")
            if st["pos"] != 0:
                st["hh"] = max(st["hh"], h[j])
                st["ll"] = min(st["ll"], l[j])
        else:
            if random_p is not None:
                if rng.random() < random_p:
                    side = 1 if rng.random() < 0.5 else -1
                    do_open(j, side, o[j])
            else:
                buy_hit = st["eb"] is not None and l[j] <= st["eb"] and j <= st["expire"]
                sell_hit = st["es"] is not None and h[j] >= st["es"] and j <= st["expire"]
                if buy_hit or sell_hit:
                    if buy_hit and sell_hit:
                        side = 1 if abs(o[j] - st["eb"]) <= abs(o[j] - st["es"]) else -1
                    else:
                        side = 1 if buy_hit else -1
                    fill = min(o[j], st["eb"]) if side > 0 else max(o[j], st["es"])
                    st["eb"] = st["es"] = None
                    do_open(j, side, fill)
                elif st["eb"] is None or j > st["expire"]:
                    st["eb"] = c[cl] * (1 - D)
                    st["es"] = c[cl] * (1 + D)
                    st["expire"] = j + V - 1
        # 更新 hh/ll 也用于开仓当根以后 (入场当根的 h/l 已计入 hh/ll 初始值)
        if st["pos"] != 0:
            eq_worst[j] = min(st["eq"], st["eq"] + st["qty"] * ((l[j] - st["entry"]) if st["pos"] > 0 else (st["entry"] - h[j])))
        else:
            eq_worst[j] = st["eq"]
    return pd.DataFrame(trades), eq_worst


def metrics(tr, eq_worst):
    n = len(tr)
    if n == 0:
        return dict(n=0)
    first = pd.to_datetime(tr.entry_time).min()
    last = pd.to_datetime(tr.exit_time).max()
    years = max((last - first).days / 365.25, 1e-9)
    eqs = START_U * np.cumprod(1 + tr.net.values / tr.equity_entry.values)
    end = float(eqs[-1])
    mult = end / START_U
    cagr = mult ** (1 / years) - 1 if mult > 0 else -1.0
    mdd_i = _mdd(eq_worst)
    cal = cagr / abs(mdd_i) if mdd_i < 0 else np.nan
    pct = tr.net / tr.equity_entry * 100
    gpct = tr.gross / tr.equity_entry * 100
    return dict(n=n, n_per_yr=n / years, avg_net=pct.mean(), avg_gross=gpct.mean(),
                win=(tr.net > 0).mean(), avg_bars=tr.bars.mean(),
                stop_share=(tr.reason == "stop").mean(), mean_share=(tr.reason == "mean").mean(),
                time_share=(tr.reason == "time").mean(),
                end_u=end, cagr=cagr, mdd_intra=mdd_i, calmar=cal)


def fmt_p(x):
    return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"


def fmt_pp(x):
    return "—" if (x is None or pd.isna(x)) else f"{x:+.3f}%"


def fmt_2(x):
    return "—" if (x is None or pd.isna(x)) else f"{x:.2f}"


def main():
    k = load_1h()
    rows = []
    for name, cfg in EXITS.items():
        tr, eqw = simulate(k, cfg, maker_mode=True)
        full = metrics(tr, eqw)
        tro, eqwo = simulate(k, cfg, maker_mode=True, start_ts=SPLIT)
        tro = tro[pd.to_datetime(tro.entry_time) >= SPLIT].reset_index(drop=True)
        oos = metrics(tro, eqwo)
        # 随机入场对照 (同出场): 概率使交易数约等于真实
        n_bars = len(k) - WARM
        p = max(1e-4, full["n"] / n_bars)
        rg = []
        for sd in SEEDS:
            trr, _ = simulate(k, cfg, maker_mode=True, random_p=p, seed=sd)
            rg.append(dict(n=len(trr), gross=(trr.gross / trr.equity_entry * 100).mean() if len(trr) else np.nan))
        rand_gross = float(np.nanmean([x["gross"] for x in rg]))
        rand_n = float(np.mean([x["n"] for x in rg]))
        # taker 敏感性
        tr_t, eqw_t = simulate(k, cfg, maker_mode=False)
        ft = metrics(tr_t, eqw_t)
        rows.append(dict(cfg=name, **{f"full_{x}": y for x, y in full.items()},
                         **{f"oos_{x}": y for x, y in oos.items()},
                         rand_gross=rand_gross, rand_n=rand_n, taker_avg_net=ft.get("avg_net", np.nan),
                         taker_n=ft.get("n", 0)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "dyn_exit_v1_0.csv"), index=False, encoding="utf-8")

    base = R[R.cfg.str.startswith("X0")].iloc[0]
    L = ["# 1H 限价入场 + 非固定出场 预登记检验 v1.0\n",
         "**脚本**: `research/oos/dyn_exit_v1_0.py`（独立引擎）。  ",
         "**入场**: 双向限价挂单，距收盘 0.5%，有效 6 根（沿用上一轮）。  ",
         "**成本**: 挂单成交 maker 0.02%/边（**假设，未核实**）；出场市价 0.05% + 滑点 0.03%；资金费 0.004%/8h；杠杆 1.5。  ",
         "**对照**: X0 纯时间出场；随机入场（同出场，3 种子平均，交易数匹配真实）。  ",
         "**OOS** = 2024-01 起 30U 重启。\n",
         "## 结果（maker 主口径）\n",
         "| 出场 | 笔/年 | 胜率 | 单笔均净 | 单笔均毛 | 止损/均值/时间占比 | 平均持仓(bar) | CAGR | 盘中MDD | OOS 单笔均净 | OOS CAGR | 随机入场 毛/笔 | 真实-随机 毛 | taker 单笔均净 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r.cfg} | {r.full_n_per_yr:.0f} | {fmt_p(r.full_win)} | {fmt_pp(r.full_avg_net)} | {fmt_pp(r.full_avg_gross)} | "
                 f"{fmt_p(r.full_stop_share)}/{fmt_p(r.full_mean_share)}/{fmt_p(r.full_time_share)} | {r.full_avg_bars:.1f} | "
                 f"{fmt_p(r.full_cagr)} | {fmt_p(r.full_mdd_intra)} | {fmt_pp(r.oos_avg_net)} | {fmt_p(r.oos_cagr)} | "
                 f"{fmt_pp(r.rand_gross)} | {fmt_pp(r.full_avg_gross - r.rand_gross)} | {fmt_pp(r.taker_avg_net)} |")

    L.append("\n## 判定（预登记，maker 主口径）\n")
    any_ok = False
    for _, r in R[~R.cfg.str.startswith("X0")].iterrows():
        p1 = (r.full_avg_net > 0) and (r.oos_avg_net > 0)
        p2 = r.full_avg_net > base.full_avg_net
        p3 = (r.full_avg_gross - r.rand_gross) > 0
        ok = p1 and p2 and p3
        any_ok |= ok
        L.append(f"- {r.cfg}: P1 两段净>0 {'满足' if p1 else '不满足'} · P2 优于纯时间出场 {'满足' if p2 else '不满足'} · "
                 f"P3 真实入场优于随机入场(毛) {'满足' if p3 else '不满足'} → **{'通过' if ok else '未通过'}**")
    L.append(f"\n**结论: {'存在通过预登记条件的动态出场（需独立复核）' if any_ok else '所有非固定出场均未通过预登记条件'}**")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "dyn_exit_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
