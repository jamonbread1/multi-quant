"""
特征工程 (Feature Engineering)
==============================
所有特征均为「因果特征」: t 时刻特征只使用 <=t 的数据, 无前视偏差。

特征族:
  收益/波动  ret_*, vol_*, vol_of_vol, parkinson, down_vol_share
  趋势强度  trend_slope, trend_r2, sma_dist, adx, hh_hl_score
  均值回归  var_ratio, autocorr_1, range_pos
  分布形态  skew_*, kurt_*
  风险状态  dd_120, vol_pct_252 (滚动分位), jump_flag
  资金面    funding_* (可选, 合并自资金费率历史)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

# 年化系数 (加密市场 7x24)
ANN = {
    "1H": np.sqrt(365 * 24),
    "4H": np.sqrt(365 * 6),
    "1D": np.sqrt(365),
    "1d": np.sqrt(365),
}


def _rolling_rank_pct(x: pd.Series, window: int) -> pd.Series:
    """因果滚动分位: 当前值在过去 window 个观测中的分位 (含当前)。"""
    def _rank_last(a: np.ndarray) -> float:
        return float((a[:-1] <= a[-1]).mean())
    return x.rolling(window, min_periods=max(20, window // 4)).apply(_rank_last, raw=True)


def _slope_r2(y: np.ndarray):
    """对 0..n-1 做 OLS, 返回 (斜率, R²)。"""
    n = len(y)
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    denom = (dx * dx).sum()
    slope = (dx * dy).sum() / (denom + EPS)
    ss_tot = (dy * dy).sum()
    ss_res = ((y - (ym + slope * dx)) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    return slope, max(0.0, min(1.0, r2))


def _rolling_slope_r2(logp: pd.Series, window: int):
    sl = logp.rolling(window).apply(lambda a: _slope_r2(a)[0], raw=True)
    r2 = logp.rolling(window).apply(lambda a: _slope_r2(a)[1], raw=True)
    return sl, r2


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    rs = up / (dn + EPS)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, min_periods=n).mean() / (atr + EPS)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, min_periods=n).mean() / (atr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.ewm(alpha=1 / n, min_periods=n).mean()
    return adx


def _hurst_rs(logp: pd.Series, window: int = 96) -> pd.Series:
    """滚动 R/S Hurst 指数 (因果)。H>0.5 趋势持续, H<0.5 均值回归。"""
    def _rs(a: np.ndarray) -> float:
        y = a - a.mean()
        if len(y) < 8 or np.allclose(y, 0):
            return 0.5
        z = np.cumsum(y)
        rs = (z.max() - z.min()) / (a.std() + EPS)
        n = len(a)
        if rs <= 0:
            return 0.5
        return float(min(1.0, max(0.0, np.log(rs + EPS) / np.log(n))))
    return logp.rolling(window, min_periods=window // 2).apply(_rs, raw=True)


def build_features(df: pd.DataFrame, bar: str = "1H",
                   with_hurst: bool = True) -> pd.DataFrame:
    """
    输入: OHLCV DataFrame (open/high/low/close/vol[_ccy_quote], datetime)
    输出: 特征 DataFrame (同 index), 前若干行为 NaN (暖机期)。
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high, low = df["high"].astype(float), df["low"].astype(float)
    logp = np.log(close)

    # ---- 收益 ----
    ret = logp.diff()
    out["ret_1"] = ret
    out["ret_5"] = logp.diff(5)
    out["ret_20"] = logp.diff(20)

    # ---- 波动 ----
    out["vol_20"] = ret.rolling(20).std()
    out["vol_60"] = ret.rolling(60).std()
    out["vol_of_vol"] = out["vol_20"].rolling(60).std()
    out["vol_ratio"] = out["vol_20"] / (out["vol_60"] + EPS)
    park = np.log(high / low) ** 2 / (4 * np.log(2.0))
    out["parkinson_20"] = np.sqrt(park.rolling(20).mean())
    # 下行波动占比 (semivol ratio)
    dn = ret.clip(upper=0).pow(2).rolling(20).mean()
    up = ret.clip(lower=0).pow(2).rolling(20).mean()
    out["down_vol_share"] = np.sqrt(dn) / (np.sqrt(dn) + np.sqrt(up) + EPS)
    out["vol_pct_252"] = _rolling_rank_pct(out["vol_20"], 252)

    # ---- 趋势 ----
    win = 40
    slope, r2 = _rolling_slope_r2(logp, win)
    # 以「窗口内漂移 / 窗口内波动」无量纲化: slope*win / (vol*sqrt(win)) = slope*sqrt(win)/vol
    out["trend_slope"] = slope * np.sqrt(win) / (out["vol_20"] + EPS)
    out["trend_r2"] = r2
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    out["sma20_dist"] = (close / ma20 - 1) / (out["vol_20"] * np.sqrt(20) + EPS)
    out["sma60_dist"] = (close / ma60 - 1) / (out["vol_20"] * np.sqrt(60) + EPS)
    out["sma120_dist"] = (close / ma120 - 1) / (out["vol_20"] * np.sqrt(120) + EPS)
    out["adx_14"] = _adx(df, 14)

    # 高点/低点结构: 近20根收盘价与前20根的高低点关系
    hh = high.rolling(20).max()
    ll = low.rolling(20).min()
    hh_prev = high.shift(20).rolling(20).max()
    ll_prev = low.shift(20).rolling(20).min()
    out["hh_hl_score"] = (
        (hh > hh_prev).astype(float) - (ll < ll_prev).astype(float)
    )

    # ---- 均值回归结构 ----
    # 方差比 VR(5): >1 趋势性, <1 均值回归
    var1 = ret.rolling(60).var()
    var5 = logp.diff(5).rolling(60).var() / 5.0
    out["var_ratio_5"] = var5 / (var1 + EPS)
    out["autocorr_1"] = ret.rolling(60).corr(ret.shift(1))
    rng = (high.rolling(20).max() - low.rolling(20).min())
    out["range_pos"] = (close - low.rolling(20).min()) / (rng + EPS)

    # ---- 分布形态 ----
    out["skew_20"] = ret.rolling(20).skew()
    out["kurt_20"] = ret.rolling(20).kurt()

    # ---- 风险 ----
    roll_high = close.rolling(120).max()
    out["dd_120"] = close / roll_high - 1
    # 跳跃: |ret| > 5 * bipower 波动
    bp = (ret.abs() * ret.shift(1).abs()).rolling(20).mean() * (np.pi / 2)
    bipower = np.sqrt(bp)
    out["jump_z"] = ret.abs() / (bipower * 5 + EPS)

    if with_hurst:
        out["hurst_96"] = _hurst_rs(logp, 96)

    # ---- 量能 (对数化, z 后处理) ----
    if "vol_ccy_quote" in df.columns:
        v = df["vol_ccy_quote"].astype(float)
    else:
        v = df["vol"].astype(float)
    out["log_volume"] = np.log1p(v)
    out["volume_z"] = (out["log_volume"] - out["log_volume"].rolling(120).mean()) / (
        out["log_volume"].rolling(120).std() + EPS
    )

    return out


def attach_funding(features: pd.DataFrame, ohlcv: pd.DataFrame,
                   funding: pd.DataFrame) -> pd.DataFrame:
    """把资金费率按K线时间对齐合并 (费率是 8h 结算制, 前向填充到每根K线)。"""
    if funding is None or funding.empty:
        return features
    f = funding.copy()
    f = f.sort_values("funding_time")
    ts = ohlcv["ts"].values if "ts" in ohlcv.columns else (
        ohlcv["datetime"].astype("int64") // 10 ** 6).values
    f_idx = pd.Series(f["funding_rate"].values,
                      index=pd.Index(f["funding_time"].values, name="ts"))
    # 合并: 每根K线取「最近一次已发生的结算费率」(reindex + ffill, 向量化)
    all_ts = np.union1d(ts, f_idx.index.values)
    s = f_idx.reindex(all_ts).sort_index().ffill()
    merged = pd.Series(ts, index=features.index).map(s)
    out = features.copy()
    out["funding_rate"] = merged.values
    out["funding_ma_3"] = pd.Series(out["funding_rate"]).rolling(3, min_periods=1).mean().values
    out["funding_ma_21"] = pd.Series(out["funding_rate"]).rolling(21, min_periods=1).mean().values
    return out


def standardize_train(feature_df: pd.DataFrame, feature_cols: list,
                      train_end: int) -> pd.DataFrame:
    """
    用 [0, train_end) 的均值/标准差做 z-score (训练窗口统计量), 返回副本。
    用于 ML 检测器; 生产中可改为滚动/扩展 z-score。
    """
    out = feature_df[feature_cols].astype(float).copy()
    mu = out.iloc[:train_end].mean()
    sd = out.iloc[:train_end].std().replace(0, np.nan)
    return (out - mu) / (sd + EPS)
