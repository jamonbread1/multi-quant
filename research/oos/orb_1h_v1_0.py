# -*- coding: utf-8 -*-
"""ORB 1H 预登记检验 v1.0 —— 严格按 orb_1h_prereg_v1_0.md 实现。
UTC 日界开盘区间突破, 下一根 bar 开盘入场, 对侧区间止损, 当日收盘离场, 每日最多 1 笔。
成本: 往返 0.16% 名义 (taker 0.05% + 滑点 0.03% 双边); 杠杆 1.5; 起始 30U; 逐笔复利重放。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RT_COST = 0.0016
LEV = 1.5
START_U = 30.0
OOS_START = pd.Timestamp("2024-01-01")
EXTRA_STRESS = 0.0006   # P3: 滑点加倍后额外的往返成本 (每边额外 0.03%)
CFGS = {"O1": (1, False), "O2": (4, False), "O3": (1, True), "O4": (4, True)}
WIN = 30                # 波动过滤回看天数 (预登记)


def load():
    k = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_1H.csv"))
    k["dt"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    k = k.sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
    k["day"] = k.dt.dt.floor("D")
    k["hour"] = k.dt.dt.hour
    # 只保留完整 24 根 bar 的日子
    cnt = k.groupby("day").size()
    full = cnt[cnt == 24].index
    return k[k.day.isin(full)].reset_index(drop=True)


def days_arrays(k):
    out = []
    for d, g in k.groupby("day", sort=True):
        out.append(dict(day=d, o=g.open.values, h=g.high.values, l=g.low.values,
                        c=g.close.values, dts=g.dt.values))
    return out


def trade(D, j, sgn, K):
    """j: 突破 bar 索引 (收盘突破), 入场 j+1 开盘; sgn: +1/-1. 返回 (gross_ret, entry_idx)."""
    RH = D["h"][:K].max()
    RL = D["l"][:K].min()
    stop = RL if sgn > 0 else RH
    e_i = j + 1
    e = D["o"][e_i]
    ex = None
    for i in range(e_i, 24):
        o_, h_, l_, c_ = D["o"][i], D["h"][i], D["l"][i], D["c"][i]
        if sgn > 0:
            if o_ <= stop:
                ex = o_; break
            if l_ <= stop:
                ex = stop; break
        else:
            if o_ >= stop:
                ex = o_; break
            if h_ >= stop:
                ex = stop; break
        if i == 23:
            ex = c_
    return sgn * (ex / e - 1.0)


def run_config(days, K, filt):
    recs = []
    widths = {}  # 日期 -> 本日 K 根区间宽度 (用于过滤)
    eligible = 0
    sigd = 0
    hist = []
    for D in days:
        RH = D["h"][:K].max(); RL = D["l"][:K].min()
        w = (RH - RL) / D["o"][0]
        ok = True
        if filt:
            if len(hist) < WIN:
                ok = False
            else:
                ok = w >= np.median(hist[-WIN:])
        hist.append(w)
        if len(hist) > WIN * 2:
            hist = hist[-WIN * 2:]
        eligible += 1
        if not ok:
            continue
        j_found = None; sgn = 0
        for j in range(K, 23):   # 信号 bar 必须在 K..22 (入场 <=23)
            c = D["c"][j]
            if c > RH:
                sgn = 1; j_found = j; break
            if c < RL:
                sgn = -1; j_found = j; break
        if j_found is None:
            continue
        sigd += 1
        g = trade(D, j_found, sgn, K)
        recs.append(dict(day=D["day"], sgn=sgn, gross=g, j=j_found, width=w,
                         seeded=j_found, K=K))
    return pd.DataFrame(recs), eligible, sigd


def replay(df, cost, start=None):
    d = df if start is None else df[df.day >= start]
    eq = START_U; eqs = [eq]
    for r in d.gross.values:
        eq = eq * (1 + LEV * (r - cost))
        eqs.append(eq)
    eqs = np.array(eqs)
    peak = np.maximum.accumulate(eqs)
    mdd = float((eqs / peak - 1).min())
    yrs = max((d.day.max() - d.day.min()).days / 365.25, 1e-9) if len(d) else np.nan
    mult = eqs[-1] / START_U
    cagr = mult ** (1 / yrs) - 1 if len(d) and mult > 0 else np.nan
    return dict(n=len(d), end_u=eqs[-1], cagr=cagr, mdd_real=mdd)


def control_random(days, K, filt, trades_days, seeds=200):
    """C1: 与配置相同日期/入场时点, 方向随机 (每个种子一次)"""
    rng = np.random.default_rng(0)
    res = []
    by_day = {D["day"]: D for D in days}
    base, _, _ = run_config(days, K, filt)
    for s in range(seeds):
        gs = []
        for _, r in base.iterrows():
            D = by_day[r.day]
            sg = 1 if rng.random() < 0.5 else -1
            gs.append(trade(D, int(r.j), sg, K))
        res.append(np.mean(gs) if gs else np.nan)
    return float(np.nanmean(res))


def control_time(days, K, base):
    """C2: 同日、入场 bar = K (区间结束后首根)、方向取区间内方向 sign(close[K-1]-open[0])"""
    by_day = {D["day"]: D for D in days}
    gs = []
    for _, r in base.iterrows():
        D = by_day[r.day]
        sg = 1 if D["c"][K - 1] >= D["o"][0] else -1
        gs.append(trade(D, K - 1, sg, K))  # 入场 = K 开盘
    return float(np.mean(gs)) if gs else np.nan


def main():
    k = load()
    days = days_arrays(k)
    rows = []
    detail = []
    for name, (K, filt) in CFGS.items():
        tr, elig, sigd = run_config(days, K, filt)
        tr["cfg"] = name
        detail.append(tr)
        full = replay(tr, RT_COST)
        oos_tr = tr[tr.day >= OOS_START]
        is_tr = tr[tr.day < OOS_START]
        oos = replay(tr, RT_COST, start=OOS_START)
        oos_stress = replay(tr, RT_COST + EXTRA_STRESS, start=OOS_START)
        c1 = control_random(days, K, filt, tr)
        c2 = control_time(days, K, tr[tr.day >= OOS_START]) if len(oos_tr) else np.nan
        row = dict(cfg=name, K=K, filter=filt, signal_days=sigd, eligible_days=elig,
                   signal_rate=round(sigd / elig, 3),
                   n_full=full["n"], n_is=len(is_tr), n_oos=oos["n"],
                   win_full=round(float((tr.gross - RT_COST > 0).mean()), 3),
                   mean_net_full=round(float((tr.gross - RT_COST).mean()) * 100, 3),
                   mean_net_is=round(float((is_tr.gross - RT_COST).mean()) * 100, 3) if len(is_tr) else np.nan,
                   mean_net_oos=round(float((oos_tr.gross - RT_COST).mean()) * 100, 3) if len(oos_tr) else np.nan,
                   mean_gross_oos=round(float(oos_tr.gross.mean()) * 100, 3) if len(oos_tr) else np.nan,
                   C1_random_gross_oos=round(c1 * 100, 3),
                   C2_time_gross_oos=round(c2 * 100, 3),
                   mean_net_oos_stress=round(float((oos_tr.gross - RT_COST - EXTRA_STRESS).mean()) * 100, 3) if len(oos_tr) else np.nan,
                   full_end_u=round(full["end_u"], 2), full_cagr=round(full["cagr"], 3),
                   full_mdd_real=round(full["mdd_real"], 3),
                   oos_end_u=round(oos["end_u"], 2), oos_cagr=round(oos["cagr"], 3),
                   oos_mdd_real=round(oos["mdd_real"], 3))
        row["P1"] = bool(row["mean_net_full"] > 0 and row["mean_net_oos"] > 0)
        row["P2"] = bool(row["mean_gross_oos"] > row["C1_random_gross_oos"])
        row["P3"] = bool(row["mean_net_oos_stress"] > 0)
        row["P4"] = bool(row["signal_rate"] < 0.9)
        row["pass"] = bool(row["P1"] and row["P2"] and row["P3"] and row["P4"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "orb_1h_v1_0.csv"), index=False)
    pd.concat(detail).to_csv(os.path.join(HERE, "orb_1h_trades_v1_0.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
