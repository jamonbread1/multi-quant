# -*- coding: utf-8 -*-
"""1H 均值回归 (短线非趋势) 预登记检验 v1.0

动机: 趋势指标在短周期上存在滞后, 且短线趋势本身不明显。本轮改用非趋势的均值回归信号。

引擎 (独立, 不复用 S1 引擎):
  - 信号: 决策 bar cl (已收盘) 上计算, 下一根 bar 开盘成交 (j = cl+1, 成交价 = open[j])。
  - 多空双向, 同一时间最多一个仓位。
  - 仓位名义 = 权益 x 1.5 (与冻结版 LEV 一致); 强平 MMR 0.5% (本策略止损远小于强平距离, 未触发)。
  - 出场 (先到先出, 按 bar 顺序判定):
      a) 止损: 3 x ATR14(cl) 与 9% 封顶的较小值; 触发价 = min(open, stop) (多) / max(open, stop) (空)
      b) 均值回归出场: 决策 bar cl 上 z 回到 0 (或 RSI 回到 50), 成交价 = open[j]
      c) 时间止损: 持仓满 24 根 bar, 成交价 = open[j]
  - 成本: 手续费 0.05% + 滑点 0.03% 每边 (名义); 资金费 0.004%/8h 按持仓 bar 数计 (1H)。
  - 回撤: 盘中最差 MTM (持仓期间用 bar 不利极值)。

配置 (预登记, 共 4 个, 上限 16):
  M1  z20, |z| >= 2.0
  M2  z20, |z| >= 2.5
  M3  z50, |z| >= 2.0
  M4  RSI14, RSI <= 25 做多 / >= 75 做空 (出场 RSI 回到 50)

判定 (预登记, 仅 OOS 与全样本同时看):
  "短线均值回归成立" 当且仅当某配置同时满足:
    (1) 全样本 CAGR > 0 且 OOS CAGR > 0
    (2) 全样本单笔平均净收益 > 0 (按权益百分比计)
    (3) 多头与空头各自的单笔平均净收益都 > 0 (避免单边 beta 冒充 alpha)
    (4) 成本 / 毛收益 < 100%
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
import s1_engine_v1_1 as E1  # noqa: E402  (仅用 atr_wilder)

START_U = 30.0
SPLIT = pd.Timestamp("2024-01-01")
FEE, SLIP = 0.0005, 0.0003
FUND_1H = 0.00004 / 8
LEV = 1.5
MAX_STOP = 0.09
STOP_MULT = 3.0
TIME_STOP = 24
WARM = 200


def load_1h():
    k = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_1H.csv"))
    k["dt"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    return k[["dt", "open", "high", "low", "close"]].reset_index(drop=True)


def rsi(c, n=14):
    d = pd.Series(c).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    rs = up / (dn + 1e-12)
    return (100 - 100 / (1 + rs)).values


def zscore(c, n):
    s = pd.Series(c)
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return ((s - m) / (sd + 1e-12)).values


def signals(cfg, k):
    c = k.close.values
    if cfg["kind"] == "z":
        z = zscore(c, cfg["n"])
        long_sig = z <= -cfg["th"]
        short_sig = z >= cfg["th"]
        exit_long = z >= 0
        exit_short = z <= 0
    else:
        r = rsi(c, 14)
        long_sig = r <= cfg["lo"]
        short_sig = r >= cfg["hi"]
        exit_long = r >= 50
        exit_short = r <= 50
    return long_sig, short_sig, exit_long, exit_short


def run(k, cfg, start_ts=None):
    o, h, l, c = (k[x].values for x in ["open", "high", "low", "close"])
    atr = np.array(E1.atr_wilder(h, l, c))
    ls, ss, xl, xs = signals(cfg, k)
    use_stop = cfg.get("stop", True)
    n = len(c)
    start = WARM
    if start_ts is not None:
        start = max(WARM, int(np.searchsorted(k.dt.values, np.datetime64(start_ts))))
    eq = START_U
    eq_curve = np.full(n, np.nan)
    eq_worst = np.full(n, np.nan)
    trades = []
    pos = 0
    entry = qty = stop = 0.0
    ent_j = 0
    entry_eq = eq
    eq_curve[:start] = eq
    eq_worst[:start] = eq
    for j in range(start, n):
        cl = j - 1
        # 1) 持仓: 先判止损 (bar j 内), 再判信号/时间出场 (开盘成交)
        if pos != 0:
            if pos > 0:
                hit = l[j] <= stop
                if hit:
                    px = min(o[j], stop)
                    reason = "stop"
                else:
                    px = None
            else:
                hit = h[j] >= stop
                if hit:
                    px = max(o[j], stop)
                    reason = "stop"
                else:
                    px = None
            if px is None:
                held = j - ent_j
                sig_exit = (pos > 0 and xl[cl]) or (pos < 0 and xs[cl])
                if sig_exit:
                    px, reason = o[j], "mean"
                elif held >= TIME_STOP:
                    px, reason = o[j], "time"
            if px is not None:
                gross = qty * (px - entry) * pos
                fees = qty * (entry + px) * (FEE + SLIP)
                fund = qty * entry * FUND_1H * (j - ent_j)
                net = gross - fees - fund
                eq += net
                trades.append(dict(entry_time=k.dt.iloc[ent_j], exit_time=k.dt.iloc[j],
                                   side=pos, entry_px=entry, exit_px=px, bars=j - ent_j,
                                   gross=gross, cost=fees + fund, net=net,
                                   equity_entry=entry_eq, reason=reason))
                pos = 0
        # 2) 开仓 (用 cl 的信号, j 开盘成交)
        if pos == 0 and eq > 0:
            d = 0
            if ls[cl]:
                d = 1
            elif ss[cl]:
                d = -1
            if d != 0 and np.isfinite(atr[cl]):
                entry = o[j]
                if use_stop:
                    dist = min(STOP_MULT * atr[cl], MAX_STOP * entry)
                    stop = entry - d * dist
                else:  # 批次2: 无止损 (多头 stop=0 永不触发, 空头 stop=1e18 永不触发)
                    stop = 0.0 if d > 0 else 1e18
                qty = eq * LEV / entry
                pos = d
                ent_j = j
                entry_eq = eq
        eq_curve[j] = eq
        if pos != 0:
            adverse = (l[j] - entry) if pos > 0 else (entry - h[j])
            eq_worst[j] = min(eq, eq + qty * adverse)
        else:
            eq_worst[j] = eq
    # 收尾: 未平仓按最后收盘计 (仅记录, 不计入)
    tr = pd.DataFrame(trades)
    return tr, eq_curve, eq_worst


def metrics(tr, eq_curve, eq_worst, start_ts=None, end_ts=None):
    if end_ts is not None and len(tr):
        tr = tr[pd.to_datetime(tr.entry_time) < end_ts]
    n = len(tr)
    if n == 0:
        return dict(n=0)
    first = pd.to_datetime(tr.entry_time).min()
    last = pd.to_datetime(tr.exit_time).max()
    years = max((last - first).days / 365.25, 1e-9)
    # 权益: 以交易序列复利 (与 eq_curve 一致, 此处用逐笔净收益复利)
    eqs = START_U * np.cumprod(1 + tr.net.values / tr.equity_entry.values)
    end_eq = float(eqs[-1])
    mult = end_eq / START_U
    cagr = mult ** (1 / years) - 1 if mult > 0 else -1.0
    mdd_i = _mdd(np.concatenate([[START_U], eqs]))  # 逐笔权益路径的回撤 (已实现)
    intra = _mdd(eq_worst)  # eq_worst 在窗口起点之前恒为 START_U, 不影响回撤
    cal = cagr / abs(intra) if intra < 0 else np.nan
    gross = tr.gross.sum()
    cost = tr.cost.sum()
    return dict(n=n, trades_per_yr=n / years, win=(tr.net > 0).mean(),
                avg_net_pct=(tr.net / tr.equity_entry).mean() * 100,
                long_avg_pct=(tr[tr.side > 0].net / tr[tr.side > 0].equity_entry).mean() * 100 if (tr.side > 0).any() else np.nan,
                short_avg_pct=(tr[tr.side < 0].net / tr[tr.side < 0].equity_entry).mean() * 100 if (tr.side < 0).any() else np.nan,
                cost_over_gross=(cost / gross) if gross > 0 else np.nan,
                end_u=end_eq, cagr=cagr, mdd_realized=mdd_i, mdd_intra=intra, calmar=cal)


CFG = {
    "M1 z20 |z|>=2.0": dict(kind="z", n=20, th=2.0),
    "M2 z20 |z|>=2.5": dict(kind="z", n=20, th=2.5),
    "M3 z50 |z|>=2.0": dict(kind="z", n=50, th=2.0),
    "M4 RSI14 25/75": dict(kind="rsi", lo=25, hi=75),
    # 批次2 (追加登记): 同信号, 无止损, 仅均值回归出场 + 24 根时间止损
    "N1 z20 |z|>=2.0 无止损": dict(kind="z", n=20, th=2.0, stop=False),
    "N2 z20 |z|>=2.5 无止损": dict(kind="z", n=20, th=2.5, stop=False),
    "N3 z50 |z|>=2.0 无止损": dict(kind="z", n=50, th=2.0, stop=False),
    "N4 RSI14 25/75 无止损": dict(kind="rsi", lo=25, hi=75, stop=False),
}


def main():
    k = load_1h()
    rows = []
    for name, cfg in CFG.items():
        tr, eqc, eqw = run(k, cfg)
        full = metrics(tr, eqc, eqw, k.dt.iloc[WARM])
        # OOS: 从 2024 起重启 (重新跑引擎, 不截断)
        tro, eqo, eqwo = run(k, cfg, start_ts=SPLIT)
        tro = tro[pd.to_datetime(tro.entry_time) >= SPLIT].reset_index(drop=True)
        oos = metrics(tro, eqo, eqwo, SPLIT)
        rows.append(dict(cfg=name, **{f"full_{x}": y for x, y in full.items()},
                         **{f"oos_{x}": y for x, y in oos.items()}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "mr1h_v1_0.csv"), index=False, encoding="utf-8")

    # 参考: 买入持有 (同期)
    bh_full = (k.close.iloc[-1] / k.close.iloc[WARM]) ** (365.25 / ((k.dt.iloc[-1] - k.dt.iloc[WARM]).days)) - 1
    bh_oos_idx = int(np.searchsorted(k.dt.values, np.datetime64(SPLIT)))
    bh_oos = (k.close.iloc[-1] / k.close.iloc[bh_oos_idx]) ** (365.25 / ((k.dt.iloc[-1] - SPLIT).days)) - 1

    def p_(x):
        return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

    def pp(x):
        return "—" if (x is None or pd.isna(x)) else f"{x:+.2f}%"

    def f2(x):
        return "—" if (x is None or pd.isna(x)) else f"{x:.2f}"

    L = ["# 1H 均值回归（短线非趋势）预登记检验 v1.0\n",
         "**脚本**: `research/oos/mr1h_v1_0.py`（独立引擎）。  ",
         "**成本**: 手续费 0.05% + 滑点 0.03% 每边，资金费 0.004%/8h；名义 = 权益 × 1.5。  ",
         "**回撤**: 逐笔权益的已实现回撤 与 盘中最差 MTM（主口径）。OOS = 2024-01 起 30U 重启。\n",
         f"**参考（同期买入持有 ETH）**: 全样本年化 {p_(bh_full)}，OOS 年化 {p_(bh_oos)}。\n",
         "## 结果\n",
         "| 配置 | 笔/年 | 胜率 | 单笔均净 | 多头均净 | 空头均净 | 成本/毛收益 | 期末 | CAGR | 盘中MDD | Calmar | OOS CAGR | OOS 单笔均净 | OOS 多/空均净 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r.cfg} | {r.full_trades_per_yr:.0f} | {p_(r.full_win)} | {pp(r.full_avg_net_pct)} | "
                 f"{pp(r.full_long_avg_pct)} | {pp(r.full_short_avg_pct)} | {p_(r.full_cost_over_gross)} | "
                 f"{r.full_end_u:.1f}U | {p_(r.full_cagr)} | {p_(r.full_mdd_intra)} | {f2(r.full_calmar)} | "
                 f"{p_(r.oos_cagr)} | {pp(r.oos_avg_net_pct)} | {pp(r.oos_long_avg_pct)} / {pp(r.oos_short_avg_pct)} |")

    L.append("\n## 判定（预登记）\n")
    any_ok = False
    for _, r in R.iterrows():
        c1 = (r.full_cagr > 0) and (r.oos_cagr > 0)
        c2 = r.full_avg_net_pct > 0
        c3 = (not pd.isna(r.full_long_avg_pct)) and (not pd.isna(r.full_short_avg_pct)) and r.full_long_avg_pct > 0 and r.full_short_avg_pct > 0
        c4 = (not pd.isna(r.full_cost_over_gross)) and r.full_cost_over_gross < 1.0
        ok = c1 and c2 and c3 and c4
        any_ok |= ok
        L.append(f"- {r.cfg}: (1) 两段 CAGR>0 {'满足' if c1 else '不满足'} · (2) 单笔均净>0 {'满足' if c2 else '不满足'} · "
                 f"(3) 多空均>0 {'满足' if c3 else '不满足'} · (4) 成本/毛<100% {'满足' if c4 else '不满足'} → **{'通过' if ok else '未通过'}**")
    L.append(f"\n**结论: {'短线均值回归成立（需独立复核）' if any_ok else '短线均值回归未通过预登记条件'}**")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "mr1h_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
