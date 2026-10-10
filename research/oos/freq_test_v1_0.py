# -*- coding: utf-8 -*-
"""交易频率预登记检验 v1.0 (冻结成本口径, LEV 1.5, 基于 s1_engine_v1_3).
配置 (预登记, 共 4 个):
  F0 冻结基线: reentry_block=True, ADX15, don48
  F1 止损/强平后即可再入场: reentry_block=False
  F2 ADX0: adx_min=0
  F3 don_n=24
成本: fee_side=0.0005, slip_side=0.0003, 资金费开启.
全样本: 2019-12-25 播种 (与官方日志首笔 2020-01-28 对齐), 统计窗口 2020-01 起; IS: 2020-01~2023-12; OOS: 2024-01 起 30U 重启.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_engine_v1_3 as E
from state_lead_v1_0 import load_k
from leverage_v1_0 import stats as _stats

FEE, SLIP, LEV = 0.0005, 0.0003, 1.5
CFG = [
    ("F0 冻结基线", dict(don_n=48, adx_min=15, reentry_block=True)),
    ("F1 止损后即可再入", dict(don_n=48, adx_min=15, reentry_block=False)),
    ("F2 ADX0", dict(don_n=48, adx_min=0, reentry_block=True)),
    ("F3 don24", dict(don_n=24, adx_min=15, reentry_block=True)),
]
k = load_k()
k = k[k.dt >= "2019-12-25"].reset_index(drop=True)

def seg(start, end=None):
    x = k[k.dt >= start]
    if end is not None:
        x = x[x.dt < end]
    return x.reset_index(drop=True)

def go(df, cfg):
    tr, eq, eqw = E.run(df, lev=LEV, fee_side=FEE, slip_side=SLIP, fund=True, **cfg)
    n_liq = getattr(E.run, "last_n_liq", 0)
    return tr, eq, eqw, n_liq

def years_of(tr, t0):
    if len(tr) == 0:
        return 1e-9
    last = pd.to_datetime(tr.exit_time).max()
    return max((last - pd.Timestamp(t0)).days / 365.25, 1e-9)

rows = []
for name, cfg in CFG:
    # 全样本 (2020-01 起, 与绩效表一致): 预热期内信号不交易由引擎 want 决定
    full_df = seg("2019-12-25")  # 与官方日志一致: 2019-12-25 播种, 首笔 2020-01-28
    tr, eq, eqw, nl = go(full_df, cfg)
    st_full = _stats(tr, eq, eqw, nl, years_of(tr, "2020-01-03"))
    # IS: 2020-01 起至 2023-12 末, 独立 30U
    is_df = seg("2019-12-25", "2024-01-01")
    tri, eqi, eqwi, nli = go(is_df, cfg)
    st_is = _stats(tri, eqi, eqwi, nli, years_of(tri, "2020-01-03"))
    # OOS: 2024-01 起 30U 重启
    oos_df = seg("2024-01-01")
    tro, eqo, eqwo, nlo = go(oos_df, cfg)
    st_oos = _stats(tro, eqo, eqwo, nlo, years_of(tro, "2024-01-01"))
    rows.append(dict(cfg=name,
        full_n=st_full["n_trades"], full_end_u=round(st_full["end_u"], 2),
        full_win=round(float((tr.pnl_net > 0).mean()), 3) if len(tr) else np.nan,
        full_cagr=round(st_full["cagr"], 3), full_mdd_intra=round(st_full["mdd_intra"], 3),
        full_calmar=round(st_full["calmar"], 2), full_liq=st_full["n_liq"],
        is_n=st_is["n_trades"], is_end_u=round(st_is["end_u"], 2), is_cagr=round(st_is["cagr"], 3),
        is_mdd_intra=round(st_is["mdd_intra"], 3), is_calmar=round(st_is["calmar"], 2),
        oos_n=st_oos["n_trades"], oos_end_u=round(st_oos["end_u"], 2),
        oos_win=round(float((tro.pnl_net > 0).mean()), 3) if len(tro) else np.nan,
        oos_cagr=round(st_oos["cagr"], 3), oos_mdd_intra=round(st_oos["mdd_intra"], 3),
        oos_calmar=round(st_oos["calmar"], 2), oos_liq=st_oos["n_liq"]))
    tr.to_csv(os.path.join(HERE, f"freq_trades_{name.split()[0]}_v1_0.csv"), index=False)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "freq_test_v1_0.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(out.T.to_string())
