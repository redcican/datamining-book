"""
图 13.7.2　特征相关性分析
PM2.5 与滞后特征、气象特征的 Pearson 相关系数热图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_pm25 import prepare_data

X_train, X_test, y_train, y_test, feature_names, df = prepare_data()

print(f"训练集: {len(X_train):,} 条 (2010-2013)")
print(f"测试集: {len(X_test):,} 条 (2014)")
print(f"特征数: {len(feature_names)}")

# ── 相关性分析 ────────────────────────────────────────────
# 选取关键特征（不含 wind dummies）
key_features = ["pm25_lag1", "pm25_lag24", "pm25_roll24_mean",
                "pm25_roll24_std", "DEWP", "TEMP", "PRES",
                "Iws", "hour", "month"]
corr_features = key_features + ["pm2.5"]

# 构建包含目标的 DataFrame
train_df = X_train[key_features].copy()
train_df["pm2.5"] = y_train.values

corr = train_df.corr()

# 与 PM2.5 的相关性排序
pm25_corr = corr["pm2.5"].drop("pm2.5").sort_values(ascending=False)
print("\n=== 与 PM2.5 的相关系数 ===")
for feat, val in pm25_corr.items():
    print(f"  {feat:<22s}  r = {val:+.3f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

# 重命名中文标签
label_map = {
    "pm25_lag1": "PM2.5(t-1)",
    "pm25_lag24": "PM2.5(t-24)",
    "pm25_roll24_mean": "24h滚动均值",
    "pm25_roll24_std": "24h滚动标准差",
    "DEWP": "露点温度",
    "TEMP": "气温",
    "PRES": "气压",
    "Iws": "风速",
    "hour": "小时",
    "month": "月份",
    "pm2.5": "PM2.5 (目标)",
}

corr_labeled = corr.rename(index=label_map, columns=label_map)

im = ax.imshow(corr_labeled.values, cmap="RdBu_r", vmin=-1, vmax=1,
               aspect="auto")
ax.set_xticks(range(len(corr_labeled)))
ax.set_xticklabels(corr_labeled.columns, fontsize=10, rotation=45,
                   ha="right")
ax.set_yticks(range(len(corr_labeled)))
ax.set_yticklabels(corr_labeled.index, fontsize=10)

# 数值标注
for i in range(len(corr_labeled)):
    for j in range(len(corr_labeled)):
        val = corr_labeled.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)

ax.set_title("特征相关性矩阵（PM2.5 预测）", fontweight="bold",
             fontsize=14)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Pearson 相关系数", fontsize=11)

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

plt.tight_layout()
save_fig(fig, __file__, "fig13_7_02_feature_engineering")
