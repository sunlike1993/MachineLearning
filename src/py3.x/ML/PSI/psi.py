"""
PSI (Population Stability Index) 特征稳定性指数计算模块

PSI 用于衡量特征分布在两个数据集之间的偏移程度，常用于：
  - 模型监控（比较训练集 vs 生产数据）
  - 特征漂移检测（比较基准期 vs 当前期）

PSI 计算公式：
  PSI = Σ (Actual_i% - Expected_i%) × ln(Actual_i% / Expected_i%)

PSI 解读标准：
  PSI < 0.1    : 分布无显著变化，特征稳定
  0.1 ≤ PSI < 0.2 : 分布有轻微变化，需关注
  PSI ≥ 0.2    : 分布发生显著偏移，建议重新训练模型
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 数值型特征 PSI
# ──────────────────────────────────────────────────────────────────────────────

def calc_psi_numeric(
    expected: Union[np.ndarray, pd.Series],
    actual: Union[np.ndarray, pd.Series],
    bins: Union[int, List[float]] = 10,
    eps: float = 1e-6,
) -> Tuple[float, pd.DataFrame]:
    """
    计算数值型特征的 PSI。

    参数
    ----
    expected : array-like
        基准分布（如训练集），作为分箱依据。
    actual : array-like
        当前分布（如测试集/生产数据）。
    bins : int 或 list of float
        分箱数量（整数）或自定义分箱边界列表。
    eps : float
        平滑常数，防止除零或 log(0)。

    返回
    ----
    psi_value : float
        总 PSI 值。
    detail_df : pd.DataFrame
        各分箱的详细信息，包含 expected_pct、actual_pct、psi_bin 等列。
    """
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    # 去除 NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("expected 或 actual 去除 NaN 后为空，无法计算 PSI。")

    # 生成分箱边界
    if isinstance(bins, (list, np.ndarray)):
        breakpoints = np.array(bins)
    else:
        breakpoints = np.nanpercentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

    # 确保边界覆盖全域
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # 统计各箱频次
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # 转换为占比，并平滑
    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

    # 按箱计算 PSI
    psi_bins = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_value = float(np.sum(psi_bins))

    # 构造明细表
    labels = [
        f"({breakpoints[i]:.4g}, {breakpoints[i+1]:.4g}]"
        for i in range(len(breakpoints) - 1)
    ]
    detail_df = pd.DataFrame({
        "bin": labels,
        "expected_count": expected_counts,
        "actual_count": actual_counts,
        "expected_pct": expected_pct,
        "actual_pct": actual_pct,
        "psi_bin": psi_bins,
    })

    return psi_value, detail_df


# ──────────────────────────────────────────────────────────────────────────────
# 类别型特征 PSI
# ──────────────────────────────────────────────────────────────────────────────

def calc_psi_categorical(
    expected: Union[np.ndarray, pd.Series],
    actual: Union[np.ndarray, pd.Series],
    eps: float = 1e-6,
) -> Tuple[float, pd.DataFrame]:
    """
    计算类别型特征的 PSI。

    参数
    ----
    expected : array-like
        基准分布（如训练集）。
    actual : array-like
        当前分布（如测试集/生产数据）。
    eps : float
        平滑常数，防止除零或 log(0)。

    返回
    ----
    psi_value : float
        总 PSI 值。
    detail_df : pd.DataFrame
        各类别的详细信息。
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("expected 或 actual 去除 NaN 后为空，无法计算 PSI。")

    # 合并所有类别
    all_categories = set(expected.unique()) | set(actual.unique())

    expected_counts = expected.value_counts()
    actual_counts = actual.value_counts()

    rows = []
    for cat in sorted(all_categories, key=str):
        e_cnt = expected_counts.get(cat, 0)
        a_cnt = actual_counts.get(cat, 0)
        rows.append({"category": cat, "expected_count": e_cnt, "actual_count": a_cnt})

    detail_df = pd.DataFrame(rows)

    e_total = len(expected)
    a_total = len(actual)

    detail_df["expected_pct"] = (detail_df["expected_count"] + eps) / (
        e_total + eps * len(detail_df)
    )
    detail_df["actual_pct"] = (detail_df["actual_count"] + eps) / (
        a_total + eps * len(detail_df)
    )
    detail_df["psi_bin"] = (
        (detail_df["actual_pct"] - detail_df["expected_pct"])
        * np.log(detail_df["actual_pct"] / detail_df["expected_pct"])
    )

    psi_value = float(detail_df["psi_bin"].sum())
    return psi_value, detail_df


# ──────────────────────────────────────────────────────────────────────────────
# 批量计算多特征 PSI
# ──────────────────────────────────────────────────────────────────────────────

def calc_psi_dataframe(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    bins: Union[int, dict] = 10,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    对 DataFrame 中的多个特征批量计算 PSI，并返回汇总表。

    参数
    ----
    expected_df : pd.DataFrame
        基准数据集（如训练集）。
    actual_df : pd.DataFrame
        当前数据集（如测试集/生产数据）。
    numeric_cols : list of str, optional
        数值型特征列名列表；为 None 时自动推断数值列。
    categorical_cols : list of str, optional
        类别型特征列名列表；为 None 时自动推断 object/category 列。
    bins : int 或 dict
        全局分箱数，或以列名为键的个性化分箱字典，仅对数值型特征生效。
    eps : float
        平滑常数。

    返回
    ----
    summary_df : pd.DataFrame
        包含每个特征的 PSI 值及稳定性评级（stable / warning / unstable）。
    """
    common_cols = set(expected_df.columns) & set(actual_df.columns)

    # 自动推断列类型
    if numeric_cols is None:
        numeric_cols = [
            c for c in common_cols
            if pd.api.types.is_numeric_dtype(expected_df[c])
        ]
    if categorical_cols is None:
        categorical_cols = [
            c for c in common_cols
            if (
                pd.api.types.is_string_dtype(expected_df[c])
                or pd.api.types.is_categorical_dtype(expected_df[c])
            )
            and not pd.api.types.is_numeric_dtype(expected_df[c])
        ]

    results = []

    for col in numeric_cols:
        col_bins = bins[col] if isinstance(bins, dict) else bins
        try:
            psi_val, _ = calc_psi_numeric(
                expected_df[col], actual_df[col], bins=col_bins, eps=eps
            )
            results.append({"feature": col, "type": "numeric", "psi": psi_val})
        except Exception as e:
            results.append({"feature": col, "type": "numeric", "psi": np.nan, "error": str(e)})

    for col in categorical_cols:
        try:
            psi_val, _ = calc_psi_categorical(expected_df[col], actual_df[col], eps=eps)
            results.append({"feature": col, "type": "categorical", "psi": psi_val})
        except Exception as e:
            results.append({"feature": col, "type": "categorical", "psi": np.nan, "error": str(e)})

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df["stability"] = summary_df["psi"].apply(_psi_label)
        summary_df = summary_df.sort_values("psi", ascending=False).reset_index(drop=True)

    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _psi_label(psi: float) -> str:
    """将 PSI 数值映射为稳定性标签。"""
    if pd.isna(psi):
        return "unknown"
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "warning"
    return "unstable"


def psi_report(summary_df: pd.DataFrame) -> str:
    """
    根据 calc_psi_dataframe 返回的汇总表生成文字报告。

    参数
    ----
    summary_df : pd.DataFrame
        calc_psi_dataframe 的返回值。

    返回
    ----
    report : str
        可直接打印的报告字符串。
    """
    lines = [
        "=" * 60,
        "  PSI 特征稳定性报告",
        "=" * 60,
        f"  共检测特征数：{len(summary_df)}",
    ]

    for label, threshold_desc in [
        ("unstable", "PSI ≥ 0.2  【不稳定，建议重训练】"),
        ("warning",  "0.1 ≤ PSI < 0.2  【轻微偏移，需关注】"),
        ("stable",   "PSI < 0.1  【稳定】"),
    ]:
        subset = summary_df[summary_df["stability"] == label]
        if not subset.empty:
            lines.append(f"\n  {threshold_desc}")
            for _, row in subset.iterrows():
                lines.append(f"    [{row['type']:>11}] {row['feature']:<30} PSI = {row['psi']:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)
