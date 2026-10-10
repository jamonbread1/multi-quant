"""
规则引擎检测器 (RuleBasedDetector) v1.1
=========================================
在 v1.0 基础上修复两个经 walk-forward 验证确认的缺陷。

版本变更 (相对 v1.0, 见 docs/状态信号研究结论_证伪链条_v1.0_20261010.md):
    [FIX1] TRANSITION 结构上不可达
        v1.0 决策树为 HIGH_VOL > TREND_UP > TREND_DOWN > RANGE_CALM,
        无 TRANSITION 分支; 该状态仅在 hybrid 三检测器 1-1-1 分裂时降级产生。
        walk-forward 实测占比 0%, 导致聚合器的「降敞口」机制形同虚设。
        本版新增 F1 分支: 趋势方向未确认(|trend_score| < t_th*0.6)且波动抬升
        (vol_score 处于 v_hi*0.8 与 v_hi 之间) → TRANSITION。
        该条件在语义上对应「趋势不明确 + 波动开始抬升」的不确定区, 是设计文档中
        TRANSITION 的原始定义。

    [FIX2] 置信度不校准
        v1.0 的 conf = 0.5 + 0.25*strength 是固定映射, 与实际预测正确性无关。
        walk-forward 实测: 置信度五分位对应的逐bar收益非单调
        (1.88 / 0.21 / -2.30 / 2.31 / 7.63 bp), 聚合器不能据此风控。
        本版改为「状态后验正确率」校准: conf = 该状态的历史后验方向命中率,
        由 fit() 在样本内实测 (因果: 只用 t 时刻及之前的数据判定, 命中率用
        t+1 的已实现方向衡量), 再按 strength 做小幅调节。

口径契约 (与 v1.0 完全一致, 未改动任何已验证的执行语义):
    趋势分 trend_score = 方向项(对称) × ADX 确认门
    波动分 vol_score   = 0.7*vol_pct_252 + 0.3*vol_ratio 修正
    迟滞: 连续 confirm_bars 根才允许切换
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..taxonomy import States
from .base import BaseDetector, make_result


class RuleBasedDetector(BaseDetector):
    name = "rules"
    VERSION = "1.1"

    def __init__(
        self,
        trend_thresh: Optional[float] = None,
        confirm_bars: int = 3,
        vol_crisis_pct: Optional[float] = None,
        vol_high_pct: Optional[float] = None,
        calibrate: bool = True,
        trans_low_frac: float = 0.6,      # [FIX1] 趋势未确认阈值 = t_th * 该系数
        trans_vol_frac: float = 0.8,      # [FIX1] 波动抬升下界 = v_hi * 该系数
        calibrate_conf: bool = True,      # [FIX2] 是否校准置信度
    ):
        self.trend_thresh = trend_thresh
        self.confirm_bars = confirm_bars
        self.vol_crisis_pct = vol_crisis_pct
        self.vol_high_pct = vol_high_pct
        self.calibrate = calibrate
        self.trans_low_frac = trans_low_frac
        self.trans_vol_frac = trans_vol_frac
        self.calibrate_conf = calibrate_conf
        self.scores_: Optional[pd.DataFrame] = None
        self.state_conf_: Optional[dict] = None

    # ------------------------------------------------------------------
    def _compute_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """与 v1.0 逐字一致, 未改动任何分项权重。"""
        sc = pd.DataFrame(index=X.index)
        direction = (
            0.45 * X["sma60_dist"].clip(-3, 3)
            + 0.35 * (X["trend_slope"] * X["trend_r2"]).clip(-3, 3)
            + 0.20 * X["hh_hl_score"].clip(-1, 1)
        )
        adx_gate = 0.35 + 0.65 * ((X["adx_14"] - 12) / 18).clip(0, 1)
        sc["trend_score"] = direction * adx_gate
        sc["vol_score"] = (
            0.7 * X["vol_pct_252"].fillna(0.5)
            + 0.3 * (X["vol_ratio"] - 1).clip(-1, 3) / 3
        ).clip(0, 1)
        sc["mr_score"] = (
            0.5 * (1 - X["trend_r2"]).clip(0, 1)
            + 0.5 * (1 - X["var_ratio_5"].clip(0.5, 1.5) / 1.5)
        )
        return sc

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_anchor: Optional[pd.Series] = None) -> "RuleBasedDetector":
        sc = self._compute_scores(X)
        self.scores_ = sc
        if self.calibrate:
            t = sc["trend_score"].abs()
            self.trend_thresh_ = float(t.quantile(0.68)) if self.trend_thresh is None else self.trend_thresh
            v = sc["vol_score"].dropna()
            self.vol_high_pct_ = float(v.quantile(0.75)) if self.vol_high_pct is None else self.vol_high_pct
            self.vol_crisis_pct_ = float(v.quantile(0.90)) if self.vol_crisis_pct is None else self.vol_crisis_pct
        else:
            self.trend_thresh_ = float(self.trend_thresh or 0.5)
            self.vol_high_pct_ = float(self.vol_high_pct or 0.65)
            self.vol_crisis_pct_ = float(self.vol_crisis_pct or 0.92)

        # [FIX2] 校准各状态的后验正确率 (在 fit 窗口内实测, 因果)
        if self.calibrate_conf:
            raw = self._decode(X, sc)
            self.state_conf_ = self._calibrate_confidence(X, raw)
        return self

    # ------------------------------------------------------------------
    def _decode(self, X: pd.DataFrame, sc: pd.DataFrame):
        """返回 (states, strength, switched) 三个数组, detect 与校准共用。"""
        n = len(X)
        trend = sc["trend_score"].values
        vol = sc["vol_score"].values
        ret20 = X["ret_20"].values
        adx = X["adx_14"].values

        t_th = self.trend_thresh_
        v_hi = self.vol_high_pct_
        v_cr = self.vol_crisis_pct_
        # [FIX1] TRANSITION 区间的两个边界
        t_lo = t_th * self.trans_low_frac
        v_lo = v_hi * self.trans_vol_frac

        states = np.full(n, "RANGE_CALM", dtype=object)
        strength = np.zeros(n)
        switched = np.zeros(n, dtype=bool)

        cur = "RANGE_CALM"
        pending = None
        pending_cnt = 0
        for i in range(n):
            t, v = trend[i], vol[i]
            if np.isnan(t) or np.isnan(v):
                states[i] = cur
                strength[i] = 0.3
                continue

            if v >= v_cr and (np.isnan(ret20[i]) or ret20[i] < 0):
                cand, s = "HIGH_VOL", v
            elif v >= v_hi and abs(t) < t_th:
                cand, s = "HIGH_VOL", v
            elif t >= t_th and adx[i] >= 18:
                cand, s = "TREND_UP", t / (t_th + 1e-12) - 1
            elif t <= -t_th and adx[i] >= 18:
                cand, s = "TREND_DOWN", -t / (t_th + 1e-12) - 1
            elif abs(t) < t_lo and v_lo <= v < v_hi:       # [FIX1] 新分支
                cand, s = "TRANSITION", 1.0 - abs(t) / (t_lo + 1e-12)
            else:
                cand, s = "RANGE_CALM", 1 - abs(t) / (t_th + 1e-12)

            if cand == cur:
                pending, pending_cnt = None, 0
                states[i] = cur
                strength[i] = s
            else:
                if pending == cand:
                    pending_cnt += 1
                else:
                    pending, pending_cnt = cand, 1
                if pending_cnt >= self.confirm_bars:
                    cur = cand
                    pending, pending_cnt = None, 0
                    states[i] = cur
                    strength[i] = s
                    switched[i] = True
                else:
                    states[i] = cur
                    strength[i] = 0.4
        return states, strength, switched

    # ------------------------------------------------------------------
    def _calibrate_confidence(self, X: pd.DataFrame,
                              decoded) -> dict:
        """
        [FIX2] 用「状态后验方向命中率」替代 v1.0 的固定映射。

        定义: 状态 s 在 t 时刻判定, 若 t+1 的实际方向与该状态的语义方向一致,
              则该次判定正确。命中率即该状态的「后验正确率」。
        语义方向: TREND_UP/HIGH_VOL(上涨修复) → 涨; TREND_DOWN → 跌;
                  RANGE_CALM/TRANSITION → 不判方向(用 |收益| 幅度作区分度)。

        注意: 命中率使用 t+1 的已实现收益衡量, 但这仅用于 fit 期的置信度标定,
              不参与状态判定, 因此不构成前视偏差 (状态本身仍只由 <=t 的特征决定)。
        """
        states, strength, _ = decoded
        if "ret_1" not in X.columns:
            return {}
        ret_next = X["ret_1"].values
        conf: dict = {}
        for s in States.names():
            m = states == s
            n_s = int(m.sum())
            if n_s < 30:
                conf[s] = 0.5
                continue
            idx = np.where(m)[0]
            idx = idx[idx + 1 < len(ret_next)]
            r = ret_next[idx + 1]
            r = r[~np.isnan(r)]
            if len(r) < 20:
                conf[s] = 0.5
                continue
            if s == "TREND_DOWN":
                hit = float((r < 0).mean())
            elif s in ("TREND_UP", "HIGH_VOL"):
                hit = float((r > 0).mean())
            else:
                # 无方向状态: 用「绝对收益是否超过中位数」衡量区分度
                hit = float((np.abs(r) > np.median(np.abs(r))).mean())
            # 命中率映射到 [0.35, 0.95], 避免极端值
            conf[s] = float(np.clip(0.35 + hit * 0.6, 0.35, 0.95))
        return conf

    # ------------------------------------------------------------------
    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        sc = self.scores_ if self.scores_ is not None else self._compute_scores(X)
        if not hasattr(self, "trend_thresh_"):
            self.fit(X)

        states, strength, switched = self._decode(X, sc)
        n = len(X)

        conf = np.zeros(n)
        base = self.state_conf_ or {}
        for i in range(n):
            s = states[i]
            base_c = base.get(str(s), 0.5)
            if np.isnan(strength[i]):
                conf[i] = 0.3
                continue
            # [FIX2] v1.1.1: 以标定值为基准, 按 strength 做连续调节。
            #   v1.1.0 曾让无方向状态恒取标定值, 导致置信度退化为状态级常数
            #   (五分位只剩3档, 无法区分同一状态内的强弱) —— 现改为所有状态
            #   都随 strength 连续变动, 保证置信度是 bar 级的连续量。
            adj = np.clip((strength[i] - 0.5) * 0.12, -0.12, 0.12)
            conf[i] = float(np.clip(base_c + adj, 0.3, 0.95))
        # 切换当根置信度打折 (刚换状态, 确定性低)
        conf[switched] *= 0.9
        conf = np.clip(conf, 0.0, 1.0)

        res = make_result(X.index, list(states), conf)
        res["trend_score"] = sc["trend_score"].values
        res["vol_score"] = sc["vol_score"].values
        res["detector_version"] = self.VERSION
        return res