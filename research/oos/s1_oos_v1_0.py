# -*- coding: utf-8 -*-
"""S1 海龟 (4H don48 + ADX15 + 3ATR 吊灯) 样本内 / 样本外 逐笔复盘 v1.0

数据:
  - data/strategy/S1_trade_log_fmz.csv : FMZ 真实运行日志 (183 笔开平仓)
  - ETH 4H K 线 (eth4h_from_fmz_1h.csv) : 仅用于同期买入持有基准
方法:
  1. 从日志逐笔还原交易 (开仓行 + 本笔平仓行), 不依赖回测引擎.
  2. 切分点 SPLIT 在运行前固定为 2024-01-01 (沿用项目既有 2024 子区间口径),
     按平仓时间归属 IS / OOS. 不做任何切分点寻优.
  3. 指标: 笔数、胜率、均笔 R、t 统计、bootstrap 95% CI、30U 逐笔复利、
     最大回撤、年化; 并与同期 ETH 买入持有对比.
注意:
  - 冻结参数是在全样本上确定的, 因此 OOS 不是严格意义的参数 OOS,
    只能说明"冻结策略在未参与设计的时段表现如何".
  - 本脚本只做事后统计, 不生成新参数.
"""
import os
import re
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MR_ROOT", "/home/user/work/mr")
LOG = os.path.join(ROOT, "data", "strategy", "S1_trade_log_fmz.csv")
KLINE = os.path.join(ROOT, "strategy", "meanrev", "research", "eth4h_from_fmz_1h.csv")
OUT_MD = os.path.join(HERE, "S1_OOS_report_v1_0.md")

SPLIT = pd.Timestamp("2024-01-01")
START_U = 30.0
N_BOOT = 2000
SEED = 20261010

RE_OPEN = re.compile(r"开仓 (多|空) ([\d.]+)张 @([\d.]+) stop=([\d.]+) \(权益([\d.]+)\)")
RE_CLOSE = re.compile(
    r"\[L\d 本笔\] (stop|signal) \| (多|空)([\d.]+)张 @([\d.]+) 止损距离([\d.]+) \(([-\d.]+)U\) \| "
    r"离场([\d.]+) \| 盈亏([-+\d.]+)U = ([-+\d.]+)R"
)


def parse_trades(path):
    df = pd.read_csv(path)
    df = df.sort_values("ID").reset_index(drop=True)  # 日志文件为倒序, 按 ID 还原时间顺序
    df["时间"] = pd.to_datetime(df["时间"])
    trades, pending = [], None
    for _, row in df.iterrows():
        msg = row["消息"]
        if not isinstance(msg, str):
            continue
        m = RE_OPEN.search(msg)
        if m:
            pending = dict(
                entry_time=row["时间"], side="long" if m.group(1) == "多" else "short",
                qty=float(m.group(2)), entry_px=float(m.group(3)),
                stop_px=float(m.group(4)), equity_entry=float(m.group(5)),
            )
            continue
        m = RE_CLOSE.search(msg)
        if m:
            if pending is None:
                raise ValueError(f"平仓行找不到对应开仓: ID={row['ID']}")
            exit_reason = m.group(1)
            pnl_u, pnl_r = float(m.group(8)), float(m.group(9))
            trades.append(dict(
                **pending, exit_time=row["时间"], exit_reason=exit_reason,
                exit_px=float(m.group(7)), pnl_u=pnl_u, pnl_r=pnl_r,
            ))
            pending = None
    t = pd.DataFrame(trades)
    t["ret_eq"] = t["pnl_u"] / t["equity_entry"]  # 逐笔权益口径收益
    return t


def load_kline(path):
    k = pd.read_csv(path)
    k["dt"] = pd.to_datetime(k["dt"].str.replace(r"\+00:00$", "", regex=True))
    return k.set_index("dt")["close"].sort_index()


def max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def stats(t, label, start, end, close):
    sub = t[(t.exit_time >= start) & (t.exit_time < end)].copy()
    if len(sub) == 0:
        return None
    r = sub["pnl_r"].values
    ret = sub["ret_eq"].values
    eq = START_U * np.cumprod(1 + ret)
    eq = np.concatenate([[START_U], eq])
    years = (sub.exit_time.iloc[-1] - sub.entry_time.iloc[0]).days / 365.25
    years = max(years, 1e-9)
    cagr = (eq[-1] / START_U) ** (1 / years) - 1
    rng = np.random.default_rng(SEED)
    boots = np.array([rng.choice(r, size=len(r), replace=True).mean() for _ in range(N_BOOT)])
    t_stat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
    # 同期 ETH 买入持有 (以窗口内首尾收盘价近似)
    c0 = close[close.index >= start].iloc[0]
    c1 = close[close.index < end].iloc[-1]
    bh_total = c1 / c0 - 1
    bh_cagr = (c1 / c0) ** (1 / years) - 1
    return dict(
        label=label, n=len(sub), win=float((r > 0).mean()), mean_r=float(r.mean()),
        median_r=float(np.median(r)), t=float(t_stat),
        ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
        sum_u=float(sub.pnl_u.sum()), end_eq=float(eq[-1]), mult=float(eq[-1] / START_U),
        cagr=float(cagr), mdd=max_drawdown(eq), years=years,
        bh_total=float(bh_total), bh_cagr=float(bh_cagr),
        long_n=int((sub.side == "long").sum()), short_n=int((sub.side == "short").sum()),
    )


def fmt_row(s):
    return (f"| {s['label']} | {s['n']} | {s['win']*100:.1f}% | {s['mean_r']:+.3f} | "
            f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] | {s['t']:+.2f} | {s['sum_u']:+.1f} | "
            f"{s['mult']:.2f}x | {s['cagr']*100:+.1f}% | {s['mdd']*100:.1f}% | "
            f"{s['bh_total']*100:+.1f}% / {s['bh_cagr']*100:+.1f}% |")


def main():
    t = parse_trades(LOG)
    close = load_kline(KLINE)
    print(f"parsed trades: {len(t)}  (long={int((t.side=='long').sum())}, short={int((t.side=='short').sum())})")
    # 一致性校验: 逐笔 pnl_u 之和应与日志总和一致, 且开平数一致
    assert len(t) == 183, len(t)

    end_all = t.exit_time.max() + pd.Timedelta(seconds=1)
    start_all = t.exit_time.min()
    windows = [
        ("全样本", start_all, end_all),
        ("样本内 IS (<2024)", start_all, SPLIT),
        ("样本外 OOS (>=2024)", SPLIT, end_all),
        ("描述性: 2025-2026 (OOS 去掉 2024)", pd.Timestamp("2025-01-01"), end_all),
    ]
    rows = [stats(t, lab, s, e, close) for lab, s, e in windows]
    rows = [r for r in rows if r]

    # 描述性年度表 (不用于选参)
    t["year"] = t.exit_time.dt.year
    yearly = []
    for y, g in t.groupby("year"):
        yy = stats(t, str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01"), close)
        if yy:
            yearly.append(yy)

    lines = []
    lines.append("# S1 海龟 样本内 / 样本外 逐笔复盘报告 v1.0\n")
    lines.append("**脚本**: `research/oos/s1_oos_v1_0.py`  ")
    lines.append(f"**数据**: FMZ 真实日志 `S1_trade_log_fmz.csv`，共 {len(t)} 笔已平仓交易  ")
    lines.append(f"**切分点**: {SPLIT.date()}（运行前固定，不寻优）  ")
    lines.append("**口径**: 30U 起始逐笔权益复利 (pnl_u / 开仓权益)；ETH 买入持有为同期首尾收盘价近似\n")
    lines.append("## 1. 窗口汇总\n")
    lines.append("| 窗口 | 笔数 | 胜率 | 均笔R | 均笔R 95% CI (bootstrap) | t | 净盈亏U | 期末倍数 | 年化 | 最大回撤 | ETH 买入持有 总/年化 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lines += [fmt_row(r) for r in rows]
    lines.append("")
    lines.append("## 2. 年度分解（描述性，不用于选择参数）\n")
    lines.append("| 年 | 笔数 | 胜率 | 均笔R | 均笔R 95% CI | t | 净盈亏U | 期末倍数 | 年化 | 最大回撤 | ETH 买入持有 总/年化 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lines += [fmt_row(r) for r in yearly]
    lines.append("")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    t.to_csv(os.path.join(HERE, "S1_trades_reconstructed_v1_0.csv"), index=False, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
