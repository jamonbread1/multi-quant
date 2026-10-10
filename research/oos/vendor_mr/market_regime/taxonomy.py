"""
市场状态分类体系 (State Taxonomy)
================================
多风格策略聚合系统的核心契约之一: 统一、可解释、可扩展的状态定义。

规范状态 (canonical states) —— 所有检测器最终都映射到这组标签:

  TREND_UP      上升趋势        趋势/动量风格占优
  TREND_DOWN    下降趋势        趋势/对冲/防御风格占优
  RANGE_CALM    低波动震荡      均值回归/套利/持币收息(carry)占优
  HIGH_VOL      高波动/危机     降杠杆、防御、短久期策略占优
  TRANSITION    过渡/不确定     低置信度, 聚合器应降低总敞口

每个状态附带:
  - code: 稳定的整数编码 (供向量化/模型输入)
  - style_hint: 默认的风格权重建议 (供 StyleAllocator 初始化)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class StateDef:
    code: int
    name: str
    desc: str
    style_hint: Dict[str, float] = field(default_factory=dict)


class States:
    """规范状态注册表。"""

    TREND_UP = StateDef(
        code=1, name="TREND_UP", desc="上升趋势",
        style_hint={"trend": 0.55, "breakout": 0.25, "carry": 0.10,
                    "mean_reversion": 0.00, "defensive": 0.10},
    )
    TREND_DOWN = StateDef(
        code=2, name="TREND_DOWN", desc="下降趋势",
        style_hint={"trend": 0.45, "breakout": 0.20, "carry": 0.05,
                    "mean_reversion": 0.10, "defensive": 0.20},
    )
    RANGE_CALM = StateDef(
        code=3, name="RANGE_CALM", desc="低波动震荡",
        style_hint={"trend": 0.05, "breakout": 0.10, "carry": 0.35,
                    "mean_reversion": 0.40, "defensive": 0.10},
    )
    HIGH_VOL = StateDef(
        code=4, name="HIGH_VOL", desc="高波动/危机",
        style_hint={"trend": 0.25, "breakout": 0.25, "carry": 0.00,
                    "mean_reversion": 0.10, "defensive": 0.40},
    )
    TRANSITION = StateDef(
        code=5, name="TRANSITION", desc="过渡/不确定",
        style_hint={"trend": 0.15, "breakout": 0.15, "carry": 0.15,
                    "mean_reversion": 0.15, "defensive": 0.40},
    )

    ALL = [TREND_UP, TREND_DOWN, RANGE_CALM, HIGH_VOL, TRANSITION]
    BY_NAME = {s.name: s for s in ALL}
    BY_CODE = {s.code: s for s in ALL}
    # 交易性状态(可积极配置) vs 防御性状态
    ACTIVE = ["TREND_UP", "TREND_DOWN", "RANGE_CALM", "HIGH_VOL"]

    @classmethod
    def names(cls) -> list:
        return [s.name for s in cls.ALL]

    @classmethod
    def to_code(cls, labels) -> pd.Series:
        s = pd.Series(labels)
        return s.map({n: st.code for n, st in cls.BY_NAME.items()})

    @classmethod
    def from_code(cls, codes) -> pd.Series:
        s = pd.Series(codes)
        return s.map({st.code: st.name for st in cls.ALL})


# 五大风格的描述 (多风格策略聚合系统的标准风格池)
STYLE_PROFILES = {
    "trend": {
        "desc": "趋势/动量跟踪 (MA、通道、时序动量)",
        "typical": ["MA交叉", "Donchian通道", "时序动量TSMOM"],
    },
    "mean_reversion": {
        "desc": "均值回归 (布林带回归、配对价差、振荡器)",
        "typical": ["布林带回归", "RSI反转", "资金费率套利回归"],
    },
    "breakout": {
        "desc": "波动突破 (波动率收缩后突破、事件驱动)",
        "typical": ["波动率挤压突破", "高低点突破"],
    },
    "carry": {
        "desc": "资金费率/期限结构收益 (carry / funding harvest)",
        "typical": ["正资金费率收割", "期现基差"],
    },
    "defensive": {
        "desc": "防御/现金/低敞口 (降杠杆、对冲、观望)",
        "typical": ["现金", "减仓", "跨所对冲"],
    },
}

# 风格在各状态下的默认目标权重 (行=状态, 列=风格), 由 style_hint 生成
def default_regime_style_matrix() -> pd.DataFrame:
    rows = {s.name: s.style_hint for s in States.ALL}
    return pd.DataFrame(rows).T[list(STYLE_PROFILES)]
