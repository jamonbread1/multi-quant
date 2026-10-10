# -*- coding: utf-8 -*-
"""资金费逆向信号 试跑 v1.0 (非判定性). 见 funding_pilot_prereg_v1_0.md."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RT = 0.0016
WIN = 90
CFGS = {"P1": (1.5, 24), "P2": (1.5, 72), "P3": (2.0, 24), "P4": (2.0, 72)}


def load():
    f = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_funding.csv"))
    f["t"] = pd.to_datetime(f["datetime"].str.replace(r"\+00:00$", "", regex=True))
    f = f.sort_values("t").reset_index(drop=True)
    k = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_1H.csv"))
    k["dt"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    k = k.sort_values("dt").drop_duplicates("dt").set_index("dt")
    return f, k


def simulate(f, k, theta, H, rng=None):
    r = f.funding_rate.values
    mu = pd.Series(r).shift(1).rolling(WIN, min_periods=WIN).mean().values
    sd = pd.Series(r).shift(1).rolling(WIN, min_periods=WIN).std().values
    z = (r - mu) / sd
    trades = []
    busy_until = -1
    for i in range(len(f)):
        if i <= busy_until or np.isnan(z[i]):
            continue
        if z[i] > theta:
            sgn = -1
        elif z[i] < -theta:
            sgn = 1
        else:
            continue
        if rng is not None:
            sgn = 1 if rng.random() < 0.5 else -1
        t0 = f.t.iloc[i]
        t1 = t0 + pd.Timedelta(hours=H)
        if t1 > k.index[-1] or t0 not in k.index or t1 not in k.index:
            continue
        ep = k.loc[t0, "open"]
        xp = k.loc[t1, "open"]
        gross = sgn * (xp / ep - 1)
        # 持仓期间的资金费: 结算点 j 满足 t0 < t_j <= t1
        fund = 0.0
        j = i + 1
        while j < len(f) and f.t.iloc[j] <= t1:
            fund += sgn * r[j]  # 多头付 r, 空头收 r -> 计入 pnl 为 -sgn*r; 这里 fund 是 pnl 的贡献
            j += 1
        fund = -fund
        busy_until = j - 1
        trades.append(dict(t0=t0, t1=t1, sgn=sgn, z=z[i], gross=gross, funding=fund,
                           net=gross + fund - RT))
    return pd.DataFrame(trades)


def main():
    f, k = load()
    rows = []
    det = []
    for name, (theta, H) in CFGS.items():
        tr = simulate(f, k, theta, H)
        rng = np.random.default_rng(0)
        rand_means = [simulate(f, k, theta, H, rng=np.random.default_rng(s))["gross"].mean()
                      for s in range(50)]
        n_days = (f.t.iloc[-1] - f.t.iloc[WIN]).days
        tr["cfg"] = name
        det.append(tr)
        rows.append(dict(cfg=name, theta=theta, hold_h=H, n_trades=len(tr),
                         trades_per_day=round(len(tr) / max(n_days, 1), 3),
                         n_long=int((tr.sgn > 0).sum()) if len(tr) else 0,
                         n_short=int((tr.sgn < 0).sum()) if len(tr) else 0,
                         mean_gross_pct=round(tr.gross.mean() * 100, 3) if len(tr) else np.nan,
                         mean_funding_pct=round(tr.funding.mean() * 100, 4) if len(tr) else np.nan,
                         mean_net_pct=round(tr.net.mean() * 100, 3) if len(tr) else np.nan,
                         win=round(float((tr.net > 0).mean()), 3) if len(tr) else np.nan,
                         random_dir_gross_pct=round(float(np.nanmean(rand_means)) * 100, 3)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "funding_pilot_v1_0.csv"), index=False)
    pd.concat(det).to_csv(os.path.join(HERE, "funding_pilot_trades_v1_0.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.to_string())
    print("days", n_days)


if __name__ == "__main__":
    main()
