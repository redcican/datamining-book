"""
fig7_3_01_knn_concept.py
KNN距离异常检测概念图
左：2D 散点图，展示 k 近邻连线与距离着色
右：距离排序曲线与异常阈值
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成数据：主簇 + 离群点 ──────────────────────────────────────
cluster = np.random.multivariate_normal([5, 5], [[0.8, 0.3], [0.3, 0.6]], 80)
outliers = np.array([
    [11, 10], [0, 11], [12, 3], [-1, 0], [10, 0.5],
])
data = np.vstack([cluster, outliers])
n = len(data)
k = 5

# ── 计算 k 近邻距离 ──────────────────────────────────────────────
dist_matrix = cdist(data, data)
np.fill_diagonal(dist_matrix, np.inf)
sorted_dists = np.sort(dist_matrix, axis=1)
knn_dist = sorted_dists[:, k - 1]  # 第 k 近邻距离

# 每个点的 k 个最近邻索引
knn_indices = np.argsort(dist_matrix, axis=1)[:, :k]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.3.1　k 近邻距离异常检测",
             fontsize=20, fontweight="bold", y=1.02)

# ── 左：k近邻距离示意 ─────────────────────────────────────────────
ax = axes[0]

# 画连线（每个点到其 k 近邻）
lines = []
line_colors = []
for i in range(n):
    for j in knn_indices[i]:
        lines.append([(data[i, 0], data[i, 1]),
                       (data[j, 0], data[j, 1])])
        line_colors.append(knn_dist[i])

lc = LineCollection(lines, cmap="YlOrRd", linewidths=0.4, alpha=0.3)
lc.set_array(np.array(line_colors))
ax.add_collection(lc)

# 散点着色
sc = ax.scatter(data[:, 0], data[:, 1], c=knn_dist, cmap="YlOrRd",
                s=60, edgecolors="k", linewidths=0.5, zorder=5)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("k近邻距离", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# 标注异常点
threshold = np.percentile(knn_dist, 90)
outlier_mask = knn_dist >= threshold
for idx in np.where(outlier_mask)[0]:
    ax.annotate("", xy=(data[idx, 0], data[idx, 1]),
                xytext=(data[idx, 0] + 0.6, data[idx, 1] + 0.6),
                arrowprops=dict(arrowstyle="->", color=COLORS["red"],
                                lw=1.5))

ax.set_title("(a) k近邻距离示意", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.tick_params(labelsize=13)

# ── 右：距离排序与阈值 ─────────────────────────────────────────────
ax = axes[1]

sorted_idx = np.argsort(knn_dist)
sorted_knn = knn_dist[sorted_idx]
ranks = np.arange(1, n + 1)

# 阈值
threshold_val = np.percentile(knn_dist, 90)
normal_mask = sorted_knn <= threshold_val
anomaly_mask = sorted_knn > threshold_val

ax.bar(ranks[normal_mask], sorted_knn[normal_mask],
       color=COLORS["blue"], alpha=0.7, label="正常点", width=1.0)
ax.bar(ranks[anomaly_mask], sorted_knn[anomaly_mask],
       color=COLORS["red"], alpha=0.8, label="异常点", width=1.0)

ax.axhline(threshold_val, color=COLORS["red"], ls="--", lw=2,
           label=f"阈值 = {threshold_val:.2f}")
ax.text(n * 0.35, threshold_val + 0.3,
        f"第 90 百分位阈值 = {threshold_val:.2f}",
        fontsize=14, color=COLORS["red"], fontweight="bold")

ax.set_title("(b) 距离排序与阈值", fontsize=17)
ax.set_xlabel("点（按 k-NN 距离升序排列）", fontsize=15)
ax.set_ylabel("第 k 近邻距离", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_3_01_knn_concept")
