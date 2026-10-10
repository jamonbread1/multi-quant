"""
检测器基类与状态映射工具
========================
所有市场状态检测器的统一接口:

    detector.fit(features, y_anchor=None) -> self
    detector.detect(features) -> pd.DataFrame[state, confidence, ...]

ML 检测器输出隐状态/聚类 id, 通过 map_latent_to_canonical() 映射到规范状态:
  1) 语义质心规则 (主): 按 (趋势方向, 波动水平) 质心确定性映射;
  2) 锚点精修 (辅): 规则检测器高置信样本在模糊区 (语义判为 RANGE_CALM) 提供修正。
保证多检测器结果可比、可共识。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from ..taxonomy import States


class BaseDetector(ABC):
    """市场状态检测器接口。"""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y_anchor: Optional[pd.Series] = None) -> "BaseDetector":
        """在特征矩阵 X 上拟合。y_anchor 为可选的规范标签锚点 (来自规则检测器)。"""

    @abstractmethod
    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        返回 DataFrame(index 与 X 相同), 至少包含列:
          state:      规范状态名 (States)
          confidence: [0,1] 置信度
        """

    def fit_detect(self, X: pd.DataFrame, train_end: Optional[int] = None,
                   y_anchor: Optional[pd.Series] = None) -> pd.DataFrame:
        if train_end is not None:
            self.fit(X.iloc[:train_end], y_anchor.iloc[:train_end] if y_anchor is not None else None)
        else:
            self.fit(X, y_anchor)
        return self.detect(X)


def make_result(index, states: list, confidence: np.ndarray,
                extra: Optional[dict] = None) -> pd.DataFrame:
    out = pd.DataFrame(
        {"state": pd.Categorical(states, categories=States.names()),
         "confidence": np.asarray(confidence, dtype=float)},
        index=index,
    )
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def map_latent_to_canonical(
    latent: np.ndarray,
    X: pd.DataFrame,
    y_anchor: Optional[pd.Series],
    vol_col: str = "vol_20",
    ret_col: str = "ret_1",
    trend_col: str = "trend_slope",
) -> dict:
    """
    把 ML 隐状态 id 映射到规范状态名。

    打分 = 锚点占比 + λ * 语义亲和度:
      锚点占比:   该隐状态内规则标签的占比 (可解释基线的投票)
      语义亲和度: 质心 (趋势方向, 漂移符号, 波动水平) 与规范状态的贴合度
    然后用匈牙利算法做「单射」分配 (每个规范状态至多对应一个隐状态),
    从机制上防止多隐状态坍缩到同一标签。

    返回 {latent_id: canonical_name}
    """
    from scipy.optimize import linear_sum_assignment

    latent = np.asarray(latent)
    anchor_vals = None
    if y_anchor is not None:
        a = pd.Series(y_anchor).reset_index(drop=True)
        if len(a) == len(latent):
            anchor_vals = a

    df = pd.DataFrame({
        "latent": latent,
        "vol": X[vol_col].values if vol_col in X.columns else np.nan,
        "ret": X[ret_col].values if ret_col in X.columns else np.nan,
        "trend": X[trend_col].values if trend_col in X.columns else np.nan,
    })
    df = df.dropna(subset=["latent"])

    stats = df.groupby("latent")[["vol", "ret", "trend"]].mean()
    med_vol = float(stats["vol"].median())
    med_trend = float(df["trend"].abs().median())
    gate = max(0.15, 0.5 * med_trend)
    lam = 0.35

    # 锚点占比
    anchor_share: dict = {}
    if anchor_vals is not None:
        for lid, grp in df.groupby("latent"):
            a = anchor_vals.loc[grp.index].dropna()
            a = a[a != "TRANSITION"]
            if len(a) >= 15:
                vc = a.value_counts(normalize=True)
                anchor_share[lid] = vc.to_dict()

    labels = ["TREND_UP", "TREND_DOWN", "HIGH_VOL", "RANGE_CALM"]
    lats = list(stats.index)
    S = np.zeros((len(lats), len(labels)))
    for i, lid in enumerate(lats):
        row = stats.loc[lid]
        tu = float(np.clip(row["trend"] / gate, 0, 1)) * float(row["ret"] > 0)
        td = float(np.clip(-row["trend"] / gate, 0, 1)) * float(row["ret"] < 0)
        hv = float(np.clip((row["vol"] - 1.15 * med_vol) / (0.35 * med_vol + 1e-12), 0, 1)) \
            if med_vol else 0.0
        rc = float(np.clip(1 - abs(row["trend"]) / gate, 0, 1)) * float(row["vol"] <= 1.15 * med_vol)
        sem = {"TREND_UP": tu, "TREND_DOWN": td, "HIGH_VOL": hv, "RANGE_CALM": rc}
        for j, lab in enumerate(labels):
            S[i, j] = anchor_share.get(lid, {}).get(lab, 0.0) + lam * sem[lab]

    r_ind, c_ind = linear_sum_assignment(-S)
    mapping = {}
    for i, j in zip(r_ind, c_ind):
        mapping[lats[i]] = labels[j]
    # 未被分配的隐状态 (行数>列数时) 取该行最优
    for i, lid in enumerate(lats):
        if lid not in mapping:
            mapping[lid] = labels[int(np.argmax(S[i]))]
    return mapping
