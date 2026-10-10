# -*- coding: utf-8 -*-
"""
外部 Pine 策略 "Grok/Claude Turtle Trend Pro Strategy (HA)" 的逐行复刻回测 v1.0。
来源: 用户上传, origin/main 提交 062568e。代码语义照抄, 不做任何参数或逻辑修改。

执行语义 (按 Pine 默认 process_orders_on_close=false 处理):
  * 信号在 bar t 收盘时判定 (所有 ta.* 指标只用到 t 及之前的数据)。
  * 市价单 (strategy.entry / strategy.close) 在 bar t+1 开盘成交, 价格 = open +/- 2 ticks。
  * strategy.exit 的止损价在 bar t 收盘时按 bar t 的值设定 (close -/+ 2*ATR), 从 bar t+1 起生效;
    bar t+1 内: 若开盘已越过止损则按开盘价 (加滑点) 成交, 否则若盘中触及则按止损价 (加滑点) 成交。
  * 同一根 bar 内先处理开盘市价单, 再检查止损。止损只对开仓时的同一持仓有效 (反手后失效)。
  * 仓位 = 权益的 10% (default_qty_type=percent_of_equity, default_qty_value=10), 权益在成交前按开盘价标记。
  * 佣金 0.075% 按成交额收取 (每边); 滑点 2 ticks, mintick 取 0.01 (假设, 见报告)。
  * 无杠杆, 无资金费。pyramiding=0。

与 TradingView 的已知差异 (无法在本地完全复现):
  * 数据源: 本地为 OKX ETH-USDT-SWAP 日线 (UTC+8 日界, 即 UTC 16:00), Pine 作者用 Binance 日线 (UTC 0:00)。
  * ta.ema 的种子取前 3 个有效值的均值 (Pine 实现细节未公开, 此处为近似)。
  * percent_of_equity 的仓位计算时点按成交时开盘价近似。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# ── 输入参数 (与 Pine input 默认值一致) ──
INITIAL_CAPITAL = 10000.0
QTY_PCT = 0.10
COMMISSION = 0.075 / 100
MINTICK = 0.01
SLIP = 2 * MINTICK
ENTRY_LEN = 20
EXIT_LEN = 15
ATR_LEN = 20
ATR_MULT = 2.0
ADX_LEN = 14
ADX_SMOOTH_EMA = 3
ADX_THRESH = 20
COOLDOWN = 5
USE_HA = True


def rma(x, n):
    """Pine ta.rma: 前 n 个有效值的 SMA 作为种子, 之后 alpha=1/n 递推。"""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < n:
        return out
    s0 = valid[n - 1]
    out[s0] = np.mean(x[valid[:n]])
    for i in range(s0 + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def ema(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < n:
        return out
    s0 = valid[n - 1]
    out[s0] = np.mean(x[valid[:n]])
    a = 2.0 / (n + 1)
    for i in range(s0 + 1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def heikin_ashi(o, h, l, c):
    n = len(c)
    hc = (o + h + l + c) / 4.0
    ho = np.full(n, np.nan)
    ho[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
    hh = np.maximum.reduce([h, ho, hc])
    hl = np.minimum.reduce([l, ho, hc])
    return ho, hh, hl, hc


def indicators(df):
    o, h, l, c = (df[x].values.astype(float) for x in ("open", "high", "low", "close"))
    ho, hh, hl, hc = heikin_ashi(o, h, l, c)
    calc_h, calc_l = (hh, hl) if USE_HA else (h, l)
    d = {}
    d["entryHighest"] = pd.Series(calc_h).rolling(ENTRY_LEN).max().shift(1).values
    d["entryLowest"] = pd.Series(calc_l).rolling(ENTRY_LEN).min().shift(1).values
    d["exitLowest"] = pd.Series(calc_l).rolling(EXIT_LEN).min().shift(1).values
    d["exitHighest"] = pd.Series(calc_h).rolling(EXIT_LEN).max().shift(1).values
    d["calcHigh"], d["calcLow"] = calc_h, calc_l
    # ATR (regular prices), ta.atr(20) = rma(tr, 20)
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    tr[0] = h[0] - l[0]
    d["atr"] = rma(tr, ATR_LEN)
    # ta.dmi(14, 14)
    up = np.r_[np.nan, h[1:] - h[:-1]]
    dn = np.r_[np.nan, l[:-1] - l[1:]]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    tr_s = rma(tr, ADX_LEN)
    p_s = rma(plus_dm, ADX_LEN)
    m_s = rma(minus_dm, ADX_LEN)
    di_p = 100 * p_s / tr_s
    di_m = 100 * m_s / tr_s
    dx = 100 * np.abs(di_p - di_m) / (di_p + di_m)
    adx = rma(dx, ADX_LEN)
    d["adxSmoothed"] = ema(adx, ADX_SMOOTH_EMA)
    d["isTrending"] = d["adxSmoothed"] > ADX_THRESH
    return pd.DataFrame(d, index=df.index)


def simulate(df, ind, start_idx, end_idx=None):
    o = df.open.values.astype(float)
    h = df.high.values.astype(float)
    l = df.low.values.astype(float)
    c = df.close.values.astype(float)
    dts = df.dt.values
    n = len(df) if end_idx is None else end_idx + 1

    cash = INITIAL_CAPITAL
    pos = 0.0          # 有符号张数 (ETH)
    entry_px = 0.0
    entry_i = -1
    entry_cost_acc = 0.0
    stop_px = np.nan   # 由上一根 bar 收盘设定的止损价
    stop_side = 0      # 止损所属持仓方向
    stop_pos_id = -1   # 止损所属持仓的编号 (反手或平仓后失效)
    pos_id = 0
    last_trade_bar = 0
    pending = []       # 下一根开盘待执行的市价单: (kind, side)
    trades = []
    eq_close = np.full(len(df), np.nan)

    for i in range(start_idx, n):
        # ── (A) 开盘: 处理上一根 bar 收盘挂出的市价单 ──
        eq_open = cash + pos * (o[i] - entry_px) if pos != 0 else cash
        for kind, side in pending:
            if kind == "entry":
                if pos > 0 and side > 0:
                    continue  # 已经是多头, 不加仓 (pos<=0 条件)
                if pos < 0 and side < 0:
                    continue
                # 先平掉反向仓位 (反手), 再开新仓
                if pos != 0 and ((side > 0 and pos < 0) or (side < 0 and pos > 0)):
                    fpx = o[i] - SLIP if pos > 0 else o[i] + SLIP
                    pnl = pos * (fpx - entry_px)
                    comm = abs(pos) * fpx * COMMISSION
                    cash += pnl - comm
                    trades.append(dict(entry_time=dts[entry_i], exit_time=dts[i],
                                       side="long" if pos > 0 else "short",
                                       entry_px=entry_px, exit_px=fpx, qty=abs(pos),
                                       pnl_net=pnl - comm - entry_cost_acc, reason="reverse"))
                    pos = 0.0
                    entry_cost_acc = 0.0
                    stop_side = 0
                fpx = o[i] + SLIP if side > 0 else o[i] - SLIP
                qty = QTY_PCT * eq_open / fpx
                comm = qty * fpx * COMMISSION
                cash -= comm
                entry_cost_acc = comm
                pos = side * qty
                entry_px = fpx
                entry_i = i
                pos_id += 1
            elif kind == "close":
                if pos == 0:
                    continue
                if (side > 0 and pos > 0) or (side < 0 and pos < 0):
                    fpx = o[i] - SLIP if pos > 0 else o[i] + SLIP
                    pnl = pos * (fpx - entry_px)
                    comm = abs(pos) * fpx * COMMISSION
                    cash += pnl - comm
                    trades.append(dict(entry_time=dts[entry_i], exit_time=dts[i],
                                       side="long" if pos > 0 else "short",
                                       entry_px=entry_px, exit_px=fpx, qty=abs(pos),
                                       pnl_net=pnl - comm - entry_cost_acc, reason="signal"))
                    pos = 0.0
                    entry_cost_acc = 0.0
                    stop_side = 0
        pending = []

        # ── (B) 盘中: 检查止损 (由上一根收盘设定, 仅对同一持仓有效) ──
        if pos != 0 and stop_side != 0 and not np.isnan(stop_px) and stop_pos_id == pos_id:
            if pos > 0:
                if o[i] <= stop_px:
                    fpx = o[i] - SLIP
                elif l[i] <= stop_px:
                    fpx = stop_px - SLIP
                else:
                    fpx = None
                if fpx is not None:
                    pnl = pos * (fpx - entry_px)
                    comm = abs(pos) * fpx * COMMISSION
                    cash += pnl - comm
                    trades.append(dict(entry_time=dts[entry_i], exit_time=dts[i], side="long",
                                       entry_px=entry_px, exit_px=fpx, qty=abs(pos),
                                       pnl_net=pnl - comm - entry_cost_acc, reason="stop"))
                    pos = 0.0; entry_cost_acc = 0.0; stop_side = 0
            else:
                if o[i] >= stop_px:
                    fpx = o[i] + SLIP
                elif h[i] >= stop_px:
                    fpx = stop_px + SLIP
                else:
                    fpx = None
                if fpx is not None:
                    pnl = pos * (fpx - entry_px)
                    comm = abs(pos) * fpx * COMMISSION
                    cash += pnl - comm
                    trades.append(dict(entry_time=dts[entry_i], exit_time=dts[i], side="short",
                                       entry_px=entry_px, exit_px=fpx, qty=abs(pos),
                                       pnl_net=pnl - comm - entry_cost_acc, reason="stop"))
                    pos = 0.0; entry_cost_acc = 0.0; stop_side = 0

        # ── (C) 收盘: 计算信号, 挂出下一根 bar 的市价单与止损 ──
        eq_close[i] = cash + pos * (c[i] - entry_px) if pos != 0 else cash
        if i < 1 or np.isnan(ind["entryHighest"].iat[i]) or np.isnan(ind["entryLowest"].iat[i]) \
           or np.isnan(ind["exitLowest"].iat[i]) or np.isnan(ind["exitHighest"].iat[i]) \
           or np.isnan(ind["atr"].iat[i]):
            continue
        long_break = ind["calcHigh"][i] > ind["entryHighest"].iat[i]
        short_break = ind["calcLow"][i] < ind["entryLowest"].iat[i]
        long_exit = ind["calcLow"][i] < ind["exitLowest"].iat[i]
        short_exit = ind["calcHigh"][i] > ind["exitHighest"].iat[i]
        adx_ok = bool(ind["isTrending"].iat[i]) if not pd.isna(ind["isTrending"].iat[i]) else False
        cooldown_ok = (i - last_trade_bar) >= COOLDOWN
        long_cond = long_break and adx_ok and cooldown_ok
        short_cond = short_break and adx_ok and cooldown_ok

        # strategy.entry: 注意 pos 此时为收盘时的持仓
        if long_cond and pos <= 0:
            pending.append(("entry", +1))
            last_trade_bar = i
        if short_cond and pos >= 0:
            pending.append(("entry", -1))
            last_trade_bar = i
        if pos > 0 and long_exit:
            pending.append(("close", +1))
        if pos < 0 and short_exit:
            pending.append(("close", -1))

        # strategy.exit 止损: 收盘时 pos>0 / pos<0 时设定, 从下一根 bar 生效
        if pos > 0:
            stop_px = c[i] - ATR_MULT * ind["atr"].iat[i]
            stop_side = +1
            stop_pos_id = pos_id
        elif pos < 0:
            stop_px = c[i] + ATR_MULT * ind["atr"].iat[i]
            stop_side = -1
            stop_pos_id = pos_id
        else:
            stop_side = 0

    # 期末按收盘价强制平仓 (仅用于统计, 与 TradingView 的未平仓浮动盈亏口径一致)
    if pos != 0:
        fpx = c[n - 1]
        pnl = pos * (fpx - entry_px)
        comm = abs(pos) * fpx * COMMISSION
        trades.append(dict(entry_time=dts[entry_i], exit_time=dts[n - 1],
                           side="long" if pos > 0 else "short", entry_px=entry_px,
                           exit_px=fpx, qty=abs(pos), pnl_net=pnl - comm - entry_cost_acc,
                           reason="open_at_end"))
    return pd.DataFrame(trades), eq_close


def stats(trades, eq, dts, start_ts, end_ts):
    mask = (dts >= np.datetime64(start_ts)) & (dts <= np.datetime64(end_ts))
    e = eq[mask]
    e = e[~np.isnan(e)]
    if len(e) == 0:
        return {}
    peak = np.maximum.accumulate(e)
    mdd = float((e / peak - 1).min())
    yrs = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).days / 365.25
    final = float(e[-1])
    tr = trades[(pd.to_datetime(trades.exit_time) >= pd.Timestamp(start_ts))] if len(trades) else trades
    n = len(tr)
    wins = int((tr.pnl_net > 0).sum()) if n else 0
    return dict(start=start_ts, end=end_ts, final_equity=round(final, 2),
                return_pct=round((final / INITIAL_CAPITAL - 1) * 100, 2),
                cagr_pct=round(((final / INITIAL_CAPITAL) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else np.nan,
                mdd_close_pct=round(mdd * 100, 2), trades=n,
                win_rate_pct=round(wins / n * 100, 1) if n else np.nan,
                avg_trade_pct_of_equity=round(float(tr.pnl_net.mean() / INITIAL_CAPITAL * 100), 3) if n else np.nan)


def main():
    df = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_1D.csv"))
    df["dt"] = pd.to_datetime(df["datetime"].str.replace(r"\+00:00$", "", regex=True))
    df = df.sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
    ind = indicators(df)
    dts = df.dt.values
    results = []
    for label, start in [("作者窗口 2025-01-01~2025-12-02", "2025-01-01"),
                         ("全样本 2020-01-01~2026-10-09", "2020-01-01")]:
        start_idx = int(np.searchsorted(dts, np.datetime64(start)))
        # 指标从 2020 起算 (与 TradingView 预热一致), 交易从 start_idx 开始 (flat 起步)
        end = "2025-12-02" if label.startswith("作者") else "2026-10-09"
        end_idx = int(np.searchsorted(dts, np.datetime64(end), side="right")) - 1
        trades, eq = simulate(df, ind, start_idx, end_idx)
        s = stats(trades, eq, dts, start, end)
        s["window"] = label
        results.append(s)
        tr = trades[pd.to_datetime(trades.exit_time) >= pd.Timestamp(start)] if len(trades) else trades
        tr.to_csv(os.path.join(HERE, f"pine_turtle_trades_{end}_v1_0.csv"), index=False)
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(HERE, "pine_turtle_replica_v1_0.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
