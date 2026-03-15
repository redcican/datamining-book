"""
fig7_4_01_lof_concept.py
LOF vs KNN 在异质密度数据上的对比
左：KNN 距离评分（稀疏簇正常点被误判）
右：LOF 评分（正确识别真正异常点）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成异质密度数据 ──────────────────────────────────────────────
dense_cluster = np.random.multivariate_normal(
    [2, 2], [[0.09, 0], [0, 0.09]], 200)                    # 密集簇
sparse_cluster = np.random.multivariate_normal(
    [8, 8], [[1.44, 0], [0, 1.44]], 80)                     # 稀疏簇
anomalies = np.array([
    [12, 2], [0, 11], [13, 13], [-2, 8], [6, -1],
])                                                           # 5 个异常点

data = np.vstack([dense_cluster, sparse_cluster, anomalies])
n_dense = len(dense_cluster)
n_sparse = len(sparse_cluster)
n_anomaly = len(anomalies)
anomaly_idx = np.arange(n_dense + n_sparse, len(data))

k = 10

# ── KNN 距离评分 ──────────────────────────────────────────────────
dist_matrix = cdist(data, data)
np.fill_diagonal(dist_matrix, np.inf)
sorted_dists = np.sort(dist_matrix, axis=1)
knn_dist = sorted_dists[:, k - 1]

# ── LOF 评分 ─────────────────────────────────────────────────────
lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
lof.fit(data)
lof_scores = -lof.negative_outlier_factor_   # 转为正值，越大越异常

# ── 绘图 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.4.1　LOF 与 KNN 距离在异质密度数据上的对比",
             fontsize=20, fontweight="bold", y=1.02)

# ── 左：KNN 距离评分 ─────────────────────────────────────────────
ax = axes[0]
sc = ax.scatter(data[:, 0], data[:, 1], c=knn_dist, cmap="RdYlBu_r",
                s=50, edgecolors="k", linewidths=0.3, zorder=3)
# 标记真异常点
ax.scatter(data[anomaly_idx, 0], data[anomaly_idx, 1],
           marker="*", s=250, c=COLORS["red"], edgecolors="k",
           linewidths=0.8, zorder=5, label="真异常点")
cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("第 k 近邻距离", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# 标注稀疏簇误判
sparse_center = sparse_cluster.mean(axis=0)
ax.annotate("稀疏簇正常点\nKNN误判为异常",
            xy=(sparse_center[0], sparse_center[1]),
            xytext=(sparse_center[0] - 4.5, sparse_center[1] + 2.5),
            fontsize=13, fontweight="bold", color=COLORS["red"],
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["red"], alpha=0.9))

ax.set_title("(a) KNN 距离评分", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="lower right")
ax.tick_params(labelsize=13)

# ── 右：LOF 评分 ─────────────────────────────────────────────────
ax = axes[1]
sc = ax.scatter(data[:, 0], data[:, 1], c=lof_scores, cmap="RdYlBu_r",
                s=50, edgecolors="k", linewidths=0.3, zorder=3)
ax.scatter(data[anomaly_idx, 0], data[anomaly_idx, 1],
           marker="*", s=250, c=COLORS["red"], edgecolors="k",
           linewidths=0.8, zorder=5, label="真异常点")
cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("LOF 评分", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# 标注稀疏簇正确识别
ax.annotate("稀疏簇正常点\nLOF ≈ 1（正确）",
            xy=(sparse_center[0], sparse_center[1]),
            xytext=(sparse_center[0] - 4.5, sparse_center[1] + 2.5),
            fontsize=13, fontweight="bold", color=COLORS["green"],
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["green"], alpha=0.9))

ax.set_title("(b) LOF 评分", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="lower right")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_4_01_lof_concept")
