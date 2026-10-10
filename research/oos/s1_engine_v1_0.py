# -*- coding: utf-8 -*-
"""S1 海龟 研究引擎 v1.0 —— 复刻 turtle_4h_ETH_frozen_lev15.py 的执行语义, 用于对账与参数对照.

执行语义 (源自 FMZ 策略源码):
  - 信号: 第 j-1 根 4H bar 收盘后, 用 [0..j-1] 的数据计算 want (don 状态机 + ADX 门控)
  - 执行: 第 j 根 bar 开始时, 以第 j-1 根收盘价 px_sig 成交 (源码 px = c[-1]; 对账: 入场价 100% = 前一根收盘)
    (日志时间戳 = 该 bar 开盘时间 UTC)
  - 吊灯止损: sd = min(STOP*ATR[j-1], MAX_STOP*px)
      吊灯抬升用 j-2 根的 High (多) / Low (空) (源码 h[-2]); 触发用 j-1 根的 Low (多) / High (空)
      入场当根 (j-1 == 入场信号根) 不做触发判定
      成交价: 第 j 根开盘价市价成交 (对账: 日志止损离场价 100% 落在 open[j] 的 0.1% 以内)
  - 信号离场: want != cur 时在 open[j] 平仓; 随后若 want != 0 且 未 block 则同棒反手
  - 止损离场后 block=True, 直到 want 变化 (不同信号) 才允许重入
  - 仓位: 名义 = 权益 * LEV, 数量不取整 (与交易所张数取整的差异见对账报告)
  - 成本: 权益复利扣 taker 0.05%/边 (对账依据), 滑点 0.03%/边 为契约假设 (默认关闭, slip_side=0), 资金费 0.004%/8h 按持仓计提

用法: 对账 + 参数对照 (见 run_s1_engine_check_v1_0.py)
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


def run(df, don_n=48, don_exit=None, adx_min=15, stop_mult=STOP_MULT, lev=LEV, start_u=START_U,
        fee_side=FEE_SIDE, slip_side=0.0, fund=True):
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
            continue
        cl = j - 1  # 最后收盘 bar
        px_sig = c[cl]  # 源码 on_tick: px = c[-1], 入场/信号离场/仓位/止损距离均用此价
        px_mkt = o[j]   # 止损离场为市价单, 对账显示成交价 = 第 j 根开盘价附近
        if pos != 0:
            cum_fund += FUND_PER_BAR
            sd = min(stop_mult * atr[cl], MAX_STOP_PCT * px_sig)
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
        eq_curve[j] = eq if pos == 0 else eq  # 已实现权益 (不含未平盈亏)
    return pd.DataFrame(trades), eq_curve


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
