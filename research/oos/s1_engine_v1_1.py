# -*- coding: utf-8 -*-
"""S1 研究引擎 v1.1 —— 在 v1.0 基础上新增 (执行语义不变, 对账结果与 v1.0 一致):
  L1 强平模拟 (逐仓口径): 多头强平价 = 入场价 * (1 - 1/LEV + MMR), 空头对称;
     强平价在止损之前被触及 (同一根 bar 的 low/high) 则按强平处理, 损失全部入场权益; 权益归零后停止交易.
  L2 盘中最差权益路径 eq_worst: 持仓期间用该 bar 的不利极值计算 MTM, 用于回撤统计 (v1.0 仅用平仓后权益).
  MMR = 0.5% (维持保证金率, 项目备忘口径).
  返回 (trades, eq_curve, eq_worst); run.last_n_liq 为强平次数.
"""

import math
import numpy as np
import pandas as pd

STOP_MULT = 3.0
MAX_STOP_PCT = 0.09
LEV = 1.5
FEE_SIDE = 0.0005          # taker 每边 (对账: 日志权益隐含每边约 0.05%, 不含滑点)
SLIP_SIDE = 0.0003         # 滑点每边 (契约假设, 日志权益中未见)
FUND_PER_BAR = 0.00004 * 4 / 8  # 每 4H bar
START_U = 30.0


def atr_wilder(h, l, c, n=14):
    al = 2.0 / (n + 1.0)
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    out = np.empty(len(c))
    s = tr[0]
    for i in range(len(c)):
        s = tr[i] if i == 0 else tr[i] * al + s * (1 - al)
        out[i] = s
    return out


def adx_wilder(h, l, c, n=14):
    """与源码 adx_wilder 逐下标一致 (span-ewm ATR, alpha=1/n 平滑 DM/DX)."""
    al_span = 2.0 / (n + 1.0)
    al = 1.0 / n
    out = np.full(len(c), np.nan)
    atrx = h[0] - l[0]
    sp = sm = 0.0
    adx = np.nan
    for i in range(1, len(c)):
        upm = h[i] - h[i - 1]
        dnm = l[i - 1] - l[i]
        plus = upm if (upm > dnm and upm > 0) else 0.0
        minus = dnm if (dnm > upm and dnm > 0) else 0.0
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        atrx = tr * al_span + atrx * (1 - al_span)
        sp = plus * al + sp * (1 - al)
        sm = minus * al + sm * (1 - al)
        if atrx != 0:
            pdi, mdi = 100 * sp / atrx, 100 * sm / atrx
            if pdi + mdi != 0:
                dx = 100 * abs(pdi - mdi) / (pdi + mdi)
                adx = dx if adx != adx else dx * al + adx * (1 - al)
        out[i] = adx
    return out


def want_series(h, l, c, adx, n, m, adx_min):
    """逐根计算 want (状态机), want[i] 只用到 [0..i]."""
    N = len(c)
    want = np.zeros(N, dtype=int)
    pos = 0
    for i in range(N):
        if i < n:
            want[i] = 0
            continue
        up = h[i - n:i].max()
        dn = l[i - n:i].min()
        upx = h[i - m:i].max()
        dnx = l[i - m:i].min()
        a = adx[i]
        gate = (a == a and a >= adx_min) or (a != a and 0.0 >= adx_min)
        if pos <= 0 and gate and c[i] > up:
            pos = 1
        elif pos >= 0 and gate and c[i] < dn:
            pos = -1
        if pos == 1 and c[i] < dnx:
            pos = 0
        elif pos == -1 and c[i] > upx:
            pos = 0
        want[i] = pos
    return want


MMR = 0.005


def run(df, don_n=48, don_exit=None, adx_min=15, stop_mult=STOP_MULT, lev=LEV, start_u=START_U,
        fee_side=FEE_SIDE, slip_side=0.0, fund=True, allow=None, liq=True, mmr=MMR):
    """allow: 可选, 长度=bar 数的数组, 取值 +1(仅允许多) / -1(仅允许空) / 0(禁止开仓) / 2(不限制).
    按信号根 cl 的值过滤 want: 不允许的方向视为 want=0 (已有仓位则在该根信号离场)."""
    """df: 列 dt(UTC naive), open, high, low, close. 返回 (trades DataFrame, equity 序列)."""
    if don_exit is None:
        don_exit = don_n // 2
    o, h, l, c = (df[x].values.astype(float) for x in ("open", "high", "low", "close"))
    dts = df["dt"].values
    atr = atr_wilder(h, l, c)
    adx = adx_wilder(h, l, c)
    want = want_series(h, l, c, adx, don_n, don_exit, adx_min)

    eq = start_u
    eq_curve = np.full(len(c), np.nan)
    eq_worst = np.full(len(c), np.nan)
    n_liq = 0
    trades = []
    pos = 0          # +1 / -1 / 0
    entry = qty = stop = risk = 0.0
    ent_bar = -1
    ent_time = None
    block = False
    block_sig = 0
    cum_fund = 0.0
    entry_eq = 0.0

    def close_pos(j, px, reason):
        nonlocal pos, eq, cum_fund
        sgn = pos
        gross = qty * (px - entry) * sgn
        # pnl_u: 毛利 (与 FMZ 日志 pnl_u 口径一致 = 数量 x 价差)
        # net : 权益口径 (扣手续费/滑点/资金费), 用于复利 (对账: 日志权益变动 = 毛利 - 费用)
        fees = qty * (entry + px) * (fee_side + slip_side)
        fund_cost = qty * entry * cum_fund if fund else 0.0
        pnl = gross
        net = gross - fees - fund_cost
        eq += net
        trades.append(dict(
            entry_time=ent_time, exit_time=dts[j], side="long" if sgn > 0 else "short",
            entry_px=entry, exit_px=px, qty=qty, equity_entry=entry_eq,
            pnl_u=pnl, pnl_net=net, risk=risk, pnl_r=(pnl / (qty * risk)) if risk > 0 else np.nan,
            exit_reason=reason, bars=j - ent_bar,
        ))
        pos = 0
        cum_fund = 0.0

    warmup = don_n + 150  # 源码 WARMUP = DON_N + 150 根 (ADX/ATR 收敛)
    for j in range(1, len(c)):
        if j < warmup:
            eq_curve[j] = eq
            eq_worst[j] = eq
            continue
        if eq <= 0:
            eq_curve[j] = 0.0
            eq_worst[j] = 0.0
            continue
        cl = j - 1  # 最后收盘 bar
        px_sig = c[cl]  # 源码 on_tick: px = c[-1], 入场/信号离场/仓位/止损距离均用此价
        px_mkt = o[j]   # 止损离场为市价单, 对账显示成交价 = 第 j 根开盘价附近
        if pos != 0:
            cum_fund += FUND_PER_BAR
            sd = min(stop_mult * atr[cl], MAX_STOP_PCT * px_sig)
            if liq and cl != ent_bar:
                if pos > 0:
                    liq_px = entry * (1 - 1 / lev + mmr)
                    liq_hit = l[cl] <= liq_px
                else:
                    liq_px = entry * (1 + 1 / lev - mmr)
                    liq_hit = h[cl] >= liq_px
                if liq_hit:
                    n_liq += 1
                    trades.append(dict(
                        entry_time=ent_time, exit_time=dts[j], side="long" if pos > 0 else "short",
                        entry_px=entry, exit_px=liq_px, qty=qty, equity_entry=entry_eq,
                        pnl_u=-entry_eq, pnl_net=-entry_eq, risk=risk,
                        pnl_r=(-entry_eq / (qty * risk)) if risk > 0 else np.nan,
                        exit_reason="liq", bars=j - ent_bar))
                    eq = eq - entry_eq
                    pos = 0
                    cum_fund = 0.0
                    block, block_sig = True, want[cl]
                    eq_curve[j] = eq
                    eq_worst[j] = eq
                    continue
            if pos > 0:
                # 源码: stop = max(stop, h[-2] - sd), h[-2] 为最后收盘 bar 的前一根
                stop = max(stop, h[cl - 1] - sd)
                if l[cl] <= stop and cl != ent_bar:
                    close_pos(j, px_mkt, "stop")
                    block, block_sig = True, want[cl]
                    eq_curve[j] = eq
                    continue
            else:
                stop = min(stop, l[cl - 1] + sd)
                if h[cl] >= stop and cl != ent_bar:
                    close_pos(j, px_mkt, "stop")
                    block, block_sig = True, want[cl]
                    eq_curve[j] = eq
                    continue

        wnt = want[cl]
        if allow is not None:
            a = allow[cl]
            if a == 0 or (a == 1 and wnt < 0) or (a == -1 and wnt > 0):
                wnt = 0
        if block and wnt != block_sig:
            block = False
        if pos != 0 and wnt != pos:
            close_pos(j, px_sig, "signal")
        if pos == 0 and wnt != 0 and not block:
            q = eq * lev / px_sig
            if q > 0:
                dist = min(stop_mult * atr[cl], MAX_STOP_PCT * px_sig)
                pos, entry, qty = wnt, px_sig, q
                risk, stop = dist, px_sig - wnt * dist
                ent_bar, ent_time = cl, dts[j]
                entry_eq = eq
                cum_fund = 0.0
        eq_curve[j] = eq  # 已实现权益
        if pos > 0:
            eq_worst[j] = min(eq, eq + qty * (l[j] - entry))
        elif pos < 0:
            eq_worst[j] = min(eq, eq + qty * (entry - h[j]))
        else:
            eq_worst[j] = eq
    run.last_n_liq = n_liq
    return pd.DataFrame(trades), eq_curve, eq_worst


def metrics(tr, start_u=START_U, col="pnl_net"):
    if len(tr) == 0:
        return {}
    eq = start_u + np.cumsum(tr[col].values)
    eq = np.concatenate([[start_u], eq])
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    years = (pd.Timestamp(tr.exit_time.iloc[-1]) - pd.Timestamp(tr.entry_time.iloc[0])).days / 365.25
    mult = eq[-1] / start_u
    return dict(n=len(tr), end_u=float(eq[-1]), mult=float(mult),
                cagr=float(mult ** (1 / max(years, 1e-9)) - 1), mdd=mdd,
                win=float((tr[col] > 0).mean()), mean_r=float(tr.pnl_r.mean()))
