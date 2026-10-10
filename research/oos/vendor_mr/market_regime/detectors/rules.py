"""
规则引擎检测器 (RuleBasedDetector)
=================================
可解释的基线: 由趋势分、波动分合成状态, 并用「迟滞状态机」抑制标签抖动。

    趋势分 trend_score = 方向项(对称) × ADX 确认门
        方向项 = 0.45*sma60_dist + 0.35*trend_slope*trend_r2 + 0.20*hh_hl_score
        ADX门 = 0.35 + 0.65*clip((adx-12)/18, 0, 1)      # 趋势强度不足时整体衰减
    波动分 vol_score   = 0.7*vol_pct_252 + 0.3*vol_ratio 修正

决策规则 (带迟滞):
  - 波动分 >= crisis 且 20期收益为负 -> HIGH_VOL (危机下跌)
  - 波动分 >= high 且趋势不强        -> HIGH_VOL (高波动盘整)
  - 趋势分 >= +t 且 ADX>=18          -> TREND_UP
  - 趋势分 <= -t 且 ADX>=18          -> TREND_DOWN
  - 其余                             -> RANGE_CALM
  - 连续 confirm_bars 根才允许切换 (迟滞, 抑制抖动)

阈值默认在 fit() 时用样本内分位数校准 (calibrate=True), 也可手动指定。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..taxonomy import States
from .base import BaseDetector, make_result


class RuleBasedDetector(BaseDetector):
    name = "rules"

    def __init__(
        self,
        trend_thresh: Optional[float] = None,
        confirm_bars: int = 3,
        vol_crisis_pct: Optional[float] = None,
        vol_high_pct: Optional[float] = None,
        calibrate: bool = True,
    ):
        self.trend_thresh = trend_thresh
        self.confirm_bars = confirm_bars
        self.vol_crisis_pct = vol_crisis_pct
        self.vol_high_pct = vol_high_pct
        self.calibrate = calibrate
        self.scores_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def _compute_scores(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return self

    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        sc = self.scores_ if self.scores_ is not None else self._compute_scores(X)
        if not hasattr(self, "trend_thresh_"):
            self.fit(X)

        n = len(X)
        states = np.full(n, "RANGE_CALM", dtype=object)
        conf = np.zeros(n)

        trend = sc["trend_score"].values
        vol = sc["vol_score"].values
        ret20 = X["ret_20"].values
        adx = X["adx_14"].values

        t_th = self.trend_thresh_
        v_hi = self.vol_high_pct_
        v_cr = self.vol_crisis_pct_

        cur = "RANGE_CALM"
        pending = None
        pending_cnt = 0
        for i in range(n):
            t, v = trend[i], vol[i]
            if np.isnan(t) or np.isnan(v):
                states[i] = cur
                conf[i] = 0.3
                continue

            if v >= v_cr and (np.isnan(ret20[i]) or ret20[i] < 0):
                cand = "HIGH_VOL"
                strength = v
            elif v >= v_hi and abs(t) < t_th:
                cand = "HIGH_VOL"
                strength = v
            elif t >= t_th and adx[i] >= 18:
                cand = "TREND_UP"
                strength = t / (t_th + 1e-12) - 1
            elif t <= -t_th and adx[i] >= 18:
                cand = "TREND_DOWN"
                strength = -t / (t_th + 1e-12) - 1
            else:
                cand = "RANGE_CALM"
                strength = 1 - abs(t) / (t_th + 1e-12)

            if cand == cur:
                pending, pending_cnt = None, 0
                states[i] = cur
                conf[i] = float(np.clip(0.5 + 0.25 * min(2.0, max(0.0, strength)), 0, 1))
            else:
                if pending == cand:
                    pending_cnt += 1
                else:
                    pending, pending_cnt = cand, 1
                if pending_cnt >= self.confirm_bars:
                    cur = cand
                    pending, pending_cnt = None, 0
                    states[i] = cur
                    conf[i] = float(np.clip(0.45 + 0.25 * min(2.0, max(0.0, strength)), 0, 0.9))
                else:
                    states[i] = cur
                    conf[i] = 0.4

        res = make_result(X.index, list(states), conf)
        res["trend_score"] = sc["trend_score"].values
        res["vol_score"] = sc["vol_score"].values
        return res
