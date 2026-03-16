"""
PSI 计算示例脚本

演示三种场景：
  1. 数值型特征 PSI（单特征）
  2. 类别型特征 PSI（单特征）
  3. 多特征批量 PSI + 文字报告
"""

import numpy as np
import pandas as pd

from psi import (
    calc_psi_numeric,
    calc_psi_categorical,
    calc_psi_dataframe,
    psi_report,
)

np.random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# 场景 1：数值型特征 PSI
# ──────────────────────────────────────────────────────────────────────────────
print("\n【场景 1】数值型特征 PSI")
print("-" * 50)

# 基准分布（训练集）：均值=0
expected_scores = np.random.normal(loc=0, scale=1, size=5000)

# 稳定分布（测试集，均值与基准相同）
actual_stable = np.random.normal(loc=0, scale=1, size=2000)

# 偏移分布（生产数据，均值偏移至 0.6）
actual_shifted = np.random.normal(loc=0.6, scale=1.2, size=2000)

psi_stable, detail_stable = calc_psi_numeric(expected_scores, actual_stable, bins=10)
psi_shifted, detail_shifted = calc_psi_numeric(expected_scores, actual_shifted, bins=10)

print(f"稳定分布  PSI = {psi_stable:.4f}  （预期 < 0.1）")
print(f"偏移分布  PSI = {psi_shifted:.4f}  （预期 ≥ 0.1）")
print("\n偏移分布各箱明细：")
print(detail_shifted.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# 场景 2：类别型特征 PSI
# ──────────────────────────────────────────────────────────────────────────────
print("\n\n【场景 2】类别型特征 PSI")
print("-" * 50)

categories = ["A", "B", "C", "D"]

# 基准分布
expected_cat = np.random.choice(categories, size=5000, p=[0.5, 0.3, 0.15, 0.05])

# 稳定分布（类似比例）
actual_cat_stable = np.random.choice(categories, size=2000, p=[0.5, 0.3, 0.15, 0.05])

# 偏移分布（比例明显变化）
actual_cat_shifted = np.random.choice(categories, size=2000, p=[0.2, 0.2, 0.3, 0.3])

psi_cat_stable, detail_cat_stable = calc_psi_categorical(expected_cat, actual_cat_stable)
psi_cat_shifted, detail_cat_shifted = calc_psi_categorical(expected_cat, actual_cat_shifted)

print(f"稳定分布  PSI = {psi_cat_stable:.4f}  （预期 < 0.1）")
print(f"偏移分布  PSI = {psi_cat_shifted:.4f}  （预期 ≥ 0.2）")
print("\n偏移分布各类别明细：")
print(detail_cat_shifted.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# 场景 3：多特征批量 PSI + 报告
# ──────────────────────────────────────────────────────────────────────────────
print("\n\n【场景 3】多特征批量 PSI")
print("-" * 50)

N_TRAIN = 10_000
N_TEST = 3_000

train_df = pd.DataFrame({
    "age":       np.random.normal(35, 10, N_TRAIN).clip(18, 80),
    "income":    np.random.lognormal(10, 1, N_TRAIN),
    "score":     np.random.normal(600, 80, N_TRAIN).clip(300, 900),
    "gender":    np.random.choice(["M", "F"], N_TRAIN, p=[0.55, 0.45]),
    "city_tier": np.random.choice(["tier1", "tier2", "tier3"], N_TRAIN, p=[0.3, 0.4, 0.3]),
})

# 测试集：稳定（与训练集同分布）
test_stable_df = pd.DataFrame({
    "age":       np.random.normal(35, 10, N_TEST).clip(18, 80),
    "income":    np.random.lognormal(10, 1, N_TEST),
    "score":     np.random.normal(600, 80, N_TEST).clip(300, 900),
    "gender":    np.random.choice(["M", "F"], N_TEST, p=[0.55, 0.45]),
    "city_tier": np.random.choice(["tier1", "tier2", "tier3"], N_TEST, p=[0.3, 0.4, 0.3]),
})

# 生产数据：部分特征发生偏移
prod_shifted_df = pd.DataFrame({
    "age":       np.random.normal(45, 12, N_TEST).clip(18, 80),   # 年龄偏移
    "income":    np.random.lognormal(10, 1, N_TEST),               # 收入稳定
    "score":     np.random.normal(520, 100, N_TEST).clip(300, 900),# 分数下降
    "gender":    np.random.choice(["M", "F"], N_TEST, p=[0.55, 0.45]),
    "city_tier": np.random.choice(["tier1", "tier2", "tier3"], N_TEST, p=[0.1, 0.3, 0.6]),  # 城市分布变化
})

print("\n--- 测试集（稳定）vs 训练集 ---")
summary_stable = calc_psi_dataframe(train_df, test_stable_df)
print(summary_stable.to_string(index=False))
print(psi_report(summary_stable))

print("\n--- 生产数据（偏移）vs 训练集 ---")
summary_shifted = calc_psi_dataframe(train_df, prod_shifted_df)
print(summary_shifted.to_string(index=False))
print(psi_report(summary_shifted))
