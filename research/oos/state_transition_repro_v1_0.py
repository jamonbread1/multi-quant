# -*- coding: utf-8 -*-
"""v1.1 (TRANSITION) 因果复现 v1.0

目标: 在 ETH 4H 上因果地复现 rules_v1_1 (含 TRANSITION) 与 v1.0 (无 TRANSITION) 的状态标签,
     供后续 OOS 检验使用。

源码: vendor_mr/ (来自 market_regime -git.zip, 未修改)
协议 (预先定义, 不做寻优):
  - 特征: features.build_features (因果, 只用 <=t 数据)
  - walk-forward: 每年 1 月 1 日重拟合阈值, 阈值只用 Y 年之前的数据 (expanding window)
      trend_thresh = quantile(|trend_score|, 0.68)
      vol_high     = quantile(vol_score, 0.75)
      vol_crisis   = quantile(vol_score, 0.90)
    标签从 2021-01-01 开始 (首次拟合窗口为 2020 全年)
  - 迟滞: confirm_bars=3, 状态机跨年连续 (不重置)
  - 置信度校准不参与标签, 本脚本不计算

校验:
  V1 因果性: 截断数据重算的特征 == 全样本特征 (随机 3 个截点)
  V2 等价性: 与 RuleBasedDetector(v1.1)._decode 在常数阈值下逐 bar 一致 (TRANSITION 开)
  V3 对照: 与仓库 WF_4H_rule_states.csv 的一致率 (该文件生成脚本未找到, 协议未知, 仅作参考)
"""
import os
import sys
import importlib
import types
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MR_REPO", "/home/user/multi-quant")
VENDOR = os.path.join(HERE, "vendor_mr")


def _load_pkg():
    """以 stub 包方式导入 vendor_mr/market_regime 的子模块, 跳过 pipeline/hmm/gmm 等重依赖。"""
    root = os.path.join(VENDOR, "market_regime")
    pkg = types.ModuleType("mr")
    pkg.__path__ = [root]
    sys.modules["mr"] = pkg
    det = types.ModuleType("mr.detectors")
    det.__path__ = [os.path.join(root, "detectors")]
    sys.modules["mr.detectors"] = det
    F = importlib.import_module("mr.features")
    T = importlib.import_module("mr.taxonomy")
    R11 = importlib.import_module("mr.detectors.rules_v1_1")
    return F, T, R11


F, T, R11 = _load_pkg()

SPLIT = pd.Timestamp("2024-01-01")
REFIT_YEARS = list(range(2021, 2027))
CONFIRM = 3
TRANS_LOW, TRANS_VOL = 0.6, 0.8


def load_4h():
    k = pd.read_csv(os.path.join(REPO, "ETH-USDT-SWAP_4H.csv"))
    k["datetime"] = pd.to_datetime(k["datetime"].str.replace(r"\+00:00$", "", regex=True))
    k = k.rename(columns={"datetime": "dt"})
    return k.reset_index(drop=True)


def scores(X):
    """与 RuleBasedDetector._compute_scores 完全一致 (复用原类方法)。"""
    det = R11.RuleBasedDetector()
    return det._compute_scores(X)


def decode(trend, vol, ret20, adx, t_th, v_hi, v_cr, transition=True):
    """逐 bar 解码, 阈值为逐 bar 数组 (支持 walk-forward 分段重拟合)。
    与 rules_v1_1.RuleBasedDetector._decode 的逻辑一致; transition=False 即 v1.0 决策树。"""
    n = len(trend)
    states = np.empty(n, dtype=object)
    cur = "RANGE_CALM"
    pending, pending_cnt = None, 0
    for i in range(n):
        t, v = trend[i], vol[i]
        if np.isnan(t) or np.isnan(v):
            states[i] = cur
            continue
        th, vh, vc = t_th[i], v_hi[i], v_cr[i]
        tl = th * TRANS_LOW
        vl = vh * TRANS_VOL
        if v >= vc and (np.isnan(ret20[i]) or ret20[i] < 0):
            cand = "HIGH_VOL"
        elif v >= vh and abs(t) < th:
            cand = "HIGH_VOL"
        elif t >= th and adx[i] >= 18:
            cand = "TREND_UP"
        elif t <= -th and adx[i] >= 18:
            cand = "TREND_DOWN"
        elif transition and abs(t) < tl and vl <= v < vh:
            cand = "TRANSITION"
        else:
            cand = "RANGE_CALM"
        if cand == cur:
            pending, pending_cnt = None, 0
        else:
            if pending == cand:
                pending_cnt += 1
            else:
                pending, pending_cnt = cand, 1
            if pending_cnt >= CONFIRM:
                cur = cand
                pending, pending_cnt = None, 0
        states[i] = cur
    return states


def fit_thresholds(sc, lo, hi):
    """在 [lo, hi) 上拟合阈值 (与 rules_v1_1.fit 的 calibrate=True 分支一致)。"""
    t = sc["trend_score"].iloc[lo:hi].abs()
    v = sc["vol_score"].iloc[lo:hi].dropna()
    return float(t.quantile(0.68)), float(v.quantile(0.75)), float(v.quantile(0.90))


def wf_labels(k, feats, transition=True):
    """walk-forward 标签: 逐年重拟合阈值, 状态机跨年连续。返回 (states Series, 逐bar阈值表)。"""
    sc = scores(feats)
    dts = k["dt"]
    n = len(k)
    trend = sc["trend_score"].values
    vol = sc["vol_score"].values
    ret20 = feats["ret_20"].values
    adx = feats["adx_14"].values
    t_arr = np.full(n, np.nan)
    vh_arr = np.full(n, np.nan)
    vc_arr = np.full(n, np.nan)
    bounds = []
    for Y in REFIT_YEARS:
        ts = pd.Timestamp(f"{Y}-01-01")
        t_end = pd.Timestamp(f"{Y + 1}-01-01")
        idx_end = int(np.searchsorted(dts.values, np.datetime64(ts)))
        idx_next = int(np.searchsorted(dts.values, np.datetime64(t_end)))
        # 拟合窗口: 从 2020-01-01 到 Y 年前一根 bar (全部在 Y 之前)
        th, vh, vc = fit_thresholds(sc, 0, idx_end)
        t_arr[idx_end:idx_next] = th
        vh_arr[idx_end:idx_next] = vh
        vc_arr[idx_end:idx_next] = vc
        bounds.append((Y, idx_end, idx_next, th, vh, vc))
    start = bounds[0][1]
    states = np.full(n, None, dtype=object)
    st = decode(trend[start:], vol[start:], ret20[start:], adx[start:],
                t_arr[start:], vh_arr[start:], vc_arr[start:], transition=transition)
    states[start:] = st
    out = pd.Series(states, index=dts.values)
    return out, bounds, sc


def main():
    k = load_4h()
    # build_features 需要 OHLCV 列; 直接用原表 (索引为整数)
    base = k[["open", "high", "low", "close", "vol", "vol_ccy_quote"]].astype(float).reset_index(drop=True)
    feats = F.build_features(base, bar="4H")
    feats.index = k.dt.values

    L = ["# v1.1 (TRANSITION) 因果复现 — 校验与标签\n",
         "**脚本**: `research/oos/state_transition_repro_v1_0.py`；源码 `vendor_mr/`（来自 market_regime -git.zip，未修改）。\n"]

    # V1 因果性
    L.append("## V1 因果性（截断重算 == 全样本）\n")
    ok_all = True
    for cut in [3000, 8000, 12000]:
        trunc = F.build_features(base.iloc[:cut].reset_index(drop=True), bar="4H")
        a = trunc.iloc[-1].values.astype(float)
        b = feats.iloc[cut - 1].values.astype(float)
        same = np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12)
        ok_all &= same
        L.append(f"- 截点 {cut}（{k.dt.iloc[cut-1]}）: {'通过' if same else '**不一致**'}")
    L.append(f"\n结论: {'通过' if ok_all else '未通过'}\n")

    # V2 等价性 (2024 窗口 + 常数阈值, 与原类 _decode 比较)
    sc = scores(feats)
    lo, hi = 0, int(np.searchsorted(k.dt.values, np.datetime64(SPLIT)))
    th, vh, vc = fit_thresholds(sc, lo, hi)
    det = R11.RuleBasedDetector()
    det.trend_thresh_, det.vol_high_pct_, det.vol_crisis_pct_ = th, vh, vc
    det.scores_ = sc
    orig_states, _, _ = det._decode(feats, sc)
    n = len(feats)
    mine = decode(sc["trend_score"].values, sc["vol_score"].values, feats["ret_20"].values,
                  feats["adx_14"].values, np.full(n, th), np.full(n, vh), np.full(n, vc), transition=True)
    eq = float((np.asarray(orig_states, dtype=object) == mine).mean())
    L.append("## V2 等价性（常数阈值，全样本，与 RuleBasedDetector._decode 比较）\n")
    L.append(f"- 阈值: trend={th:.4f}, vol_high={vh:.4f}, vol_crisis={vc:.4f}")
    L.append(f"- 逐 bar 一致率: **{eq*100:.4f}%**（n={n}）\n")

    # walk-forward 标签
    s11, bounds, _ = wf_labels(k, feats, transition=True)
    s10, _, _ = wf_labels(k, feats, transition=False)
    L.append("## 逐年重拟合阈值（expanding，只用 Y 年之前的数据）\n")
    L.append("| 年 | trend_thresh | vol_high | vol_crisis |\n|---|---|---|---|")
    for Y, a, b, th_, vh_, vc_ in bounds:
        L.append(f"| {Y} | {th_:.4f} | {vh_:.4f} | {vc_:.4f} |")

    lab = pd.DataFrame({"dt": k.dt.values, "close": k.close.values,
                        "state_v1_1": s11.values, "state_v1_0": s10.values})
    lab = lab[lab.dt >= pd.Timestamp("2021-01-01")].reset_index(drop=True)
    lab.to_csv(os.path.join(HERE, "data", "state_4h_wf_v1_1_causal.csv"), index=False, encoding="utf-8")

    # 状态分布 & 切换
    L.append("\n## 标签分布（WF 2021-01 ~ 2026-10）\n")
    L.append("| 状态 | v1.0 占比 | v1.1 占比 |\n|---|---|---|")
    for s in ["TREND_UP", "TREND_DOWN", "RANGE_CALM", "HIGH_VOL", "TRANSITION"]:
        p0 = (lab.state_v1_0 == s).mean() * 100
        p1 = (lab.state_v1_1 == s).mean() * 100
        L.append(f"| {s} | {p0:.1f}% | {p1:.1f}% |")
    sw = (lab.state_v1_1 != lab.state_v1_1.shift()).sum()
    yrs = (lab.dt.iloc[-1] - lab.dt.iloc[0]).days / 365.25
    L.append(f"\n- v1.1 切换次数 {sw}（约 {sw/yrs:.0f} 次/年）")
    # 运行长度
    runs = (lab.state_v1_1 != lab.state_v1_1.shift()).cumsum()
    med_run = lab.groupby(runs).size().median()
    L.append(f"- v1.1 中位持续 {med_run:.0f} 根 4H bar（约 {med_run*4/24:.1f} 天）")
    tr_runs = lab[lab.state_v1_1 == "TRANSITION"].groupby(runs[lab.state_v1_1 == "TRANSITION"]).size()
    L.append(f"- TRANSITION 段数 {len(tr_runs)}，中位长度 {tr_runs.median():.0f} 根（{tr_runs.median()*4:.0f} 小时）" if len(tr_runs) else "- TRANSITION 段数 0")

    # V3 对照仓库 WF 文件
    wf = pd.read_csv(os.path.join(REPO, "WF_4H_rule_states.csv"))
    wf["dt"] = pd.to_datetime(wf["dt"])
    m = lab.merge(wf[["dt", "state"]], on="dt", how="inner")
    agree10 = (m.state_v1_0 == m.state).mean() * 100
    agree11 = (m.state_v1_1 == m.state).mean() * 100
    L.append("\n## V3 与仓库 WF_4H_rule_states.csv 的对照（协议未知，仅参考）\n")
    L.append(f"- 重叠 bar {len(m)}；本复现 v1.0 与仓库文件一致率 **{agree10:.1f}%**；v1.1 一致率 {agree11:.1f}%")
    L.append("- 注: 仓库文件只含 4 状态（无 TRANSITION），其拟合协议未找到。一致率低说明协议不同，不影响本研究的因果性结论，但 v1.0 标签不应与仓库文件混用。")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(HERE, "state_transition_repro_v1_0.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
