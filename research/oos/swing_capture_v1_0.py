# -*- coding: utf-8 -*-
"""机会捕获分析 v1.0 —— S1 海龟到底错过了多少大行情?

动机 (用户陈述): 做状态机是因为觉得收益速度太慢, 错过很多机会。
本脚本把「错过的机会」量化:

定义 (运行前固定, 不寻优):
  - 大波段 = ETH 4H 收盘价的 zigzag 腿, 幅度 >= 25% (用于事后度量, 含前瞻, 仅作测量尺)。
  - 对每条腿统计:
      * 腿内 S1 同向持仓占比 (coverage)
      * 首次同向进场相对腿起点的延迟 (天)
      * 对数收益捕获率 = S1 同向持仓期间的 ETH 对数收益 之和 / 腿的对数收益
        (未考虑杠杆与成本, 只衡量方向性捕获)
      * 完全错过的腿 (无同向持仓)
  - 同时统计: 全部腿的收益占比, 以及按腿方向 (up/down) 拆分。
数据: ETH-USDT-SWAP_4H.csv (UTC), S1_trades_with_states.csv (UTC 开平仓时间)。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")
K = os.path.join(REPO, "ETH-USDT-SWAP_4H.csv")
TR = os.path.join(REPO, "S1_trades_with_states.csv")
OUT_MD = os.path.join(HERE, "swing_capture_report_v1_0.md")
OUT_CSV = os.path.join(HERE, "swing_legs_v1_0.csv")

THR = 0.25  # zigzag 阈值


def zigzag_legs(close, thr):
    """标准 zigzag: 返回 (start_idx, end_idx, direction) 腿列表, direction +1 上涨腿 / -1 下跌腿."""
    legs = []
    pivot = 0          # 上一个确认的转折点
    ext = 0            # 当前趋势的极值点
    trend = 0          # 0 未定, +1 上涨中, -1 下跌中
    for i in range(1, len(close)):
        p = close[i]
        if trend == 0:
            if p >= close[pivot] * (1 + thr):
                trend, ext = 1, i
            elif p <= close[pivot] * (1 - thr):
                trend, ext = -1, i
        elif trend == 1:
            if p > close[ext]:
                ext = i
            elif p <= close[ext] * (1 - thr):
                legs.append((pivot, ext, +1))
                pivot, ext, trend = ext, i, -1
        else:  # trend == -1
            if p < close[ext]:
                ext = i
            elif p >= close[ext] * (1 + thr):
                legs.append((pivot, ext, -1))
                pivot, ext, trend = ext, i, 1
    return legs


def main():
    k = pd.read_csv(K)
    k["dt"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    close = k["close"].values
    times = k["dt"].values

    t = pd.read_csv(TR)
    t["e"] = pd.to_datetime(t["entry_time_utc"])
    t["x"] = pd.to_datetime(t["exit_time_utc"])
    t["sgn"] = np.where(t["side"] == "long", 1, -1)

    # 每根 bar 的持仓方向 (开仓 bar 起算, 平仓 bar 前一根为止)
    pos = np.zeros(len(k))
    t_idx = np.searchsorted(times, t["e"].values)
    x_idx = np.searchsorted(times, t["x"].values)
    for (a, b, s) in zip(t_idx, x_idx, t["sgn"].values):
        pos[a:b] = s
    logret = np.diff(np.log(close), prepend=np.nan)  # bar i 的收益 = 从 i-1 到 i

    legs = zigzag_legs(close, THR)
    rows = []
    for (a, b, d) in legs:
        if b <= a:
            continue
        leg_lr = np.log(close[b] / close[a]) * d  # 腿的方向性对数收益 (正值 = 顺势)
        seg = slice(a + 1, b + 1)
        in_dir = pos[seg] == d
        coverage = in_dir.mean()
        # 捕获: 同向持仓期间的 ETH 对数收益 * 方向
        captured = np.nansum(np.where(in_dir, logret[seg], 0.0)) * d
        first = np.where(in_dir)[0]
        if len(first):
            entry_lag_d = (times[a + 1 + first[0]] - times[a]) / np.timedelta64(1, "D")
        else:
            entry_lag_d = np.nan
        rows.append(dict(
            start=pd.Timestamp(times[a]), end=pd.Timestamp(times[b]),
            direction="up" if d > 0 else "down",
            move_pct=float(close[b] / close[a] - 1),
            days=(times[b] - times[a]) / np.timedelta64(1, "D"),
            leg_lr=float(leg_lr), coverage=float(coverage),
            captured_lr=float(captured), capture_ratio=float(captured / leg_lr) if leg_lr > 0 else np.nan,
            entry_lag_days=float(entry_lag_d) if not np.isnan(entry_lag_d) else np.nan,
            missed=bool(not in_dir.any()),
        ))
    L = pd.DataFrame(rows)
    L.to_csv(OUT_CSV, index=False, encoding="utf-8")

    def block(df, name):
        if len(df) == 0:
            return f"\n### {name}\n\n无样本\n"
        w = df.dropna(subset=["capture_ratio"])
        return (f"\n### {name}（n={len(df)}）\n\n"
                f"| 指标 | 数值 |\n|---|---|\n"
                f"| 完全错过的腿 | {int(df.missed.sum())} / {len(df)} ({df.missed.mean()*100:.0f}%) |\n"
                f"| 平均持仓覆盖率 | {df.coverage.mean()*100:.1f}% |\n"
                f"| 平均对数收益捕获率（腿级，中位数） | {w.capture_ratio.median()*100:.1f}% |\n"
                f"| 平均对数收益捕获率（按腿对数收益加权） | {w.captured_lr.sum()/w.leg_lr.sum()*100:.1f}% |\n"
                f"| 进场延迟中位数（天，仅有进场的腿） | {df.entry_lag_days.median():.1f} |\n"
                f"| 进场延迟 75 分位（天） | {df.entry_lag_days.quantile(0.75):.1f} |\n"
                f"| 腿的对数收益合计 | {df.leg_lr.sum():.2f} |\n"
                f"| S1 同向捕获的对数收益合计 | {df.captured_lr.sum():.2f} |\n")

    lines = ["# 机会捕获分析报告 v1.0（S1 vs ETH 大波段）\n",
             f"**脚本**: `research/oos/swing_capture_v1_0.py`  ",
             f"**大波段定义**: ETH 4H 收盘价 zigzag 腿, 幅度 >= {THR*100:.0f}%（事后测量尺，含前瞻）  ",
             f"**样本**: {k.dt.iloc[0].date()} ~ {k.dt.iloc[-1].date()}，共 {len(legs)} 条腿\n",
             "> 注意: 大波段本身由事后 zigzag 定义，所有腿都是「已经发生的大行情」。"
             "度量的是 S1 在这些行情里抓到多少，而不是可以事前识别这些腿。\n"]
    lines.append(block(L, "全部大波段"))
    lines.append(block(L[L.direction == "up"], "上涨腿（顺势做多）"))
    lines.append(block(L[L.direction == "down"], "下跌腿（顺势做空）"))
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
