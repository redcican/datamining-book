"""
fig7_5_01_cluster_anomaly_types.py
三种基于聚类的异常模式
(a) 不属于任何簇  (b) 小簇异常  (c) 簇边缘异常
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 颜色映射 ──────────────────────────────────────────────────────
cluster_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]

# ══════════════════════════════════════════════════════════════════
# Panel (a): 不属于任何簇 — K-means 聚类，远离所有簇心的点为异常
# ══════════════════════════════════════════════════════════════════
X_a, _ = make_blobs(n_samples=[80, 60, 70], centers=[[-3, -3], [3, 3], [8, -2]],
                    cluster_std=[0.8, 0.7, 0.9], random_state=42)
# 添加远离所有簇的异常点
anomalies_a = np.array([[12, 8], [-7, 6], [0, -8], [14, -7]])
X_a_full = np.vstack([X_a, anomalies_a])
n_normal_a = len(X_a)

km_a = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_a = km_a.fit_predict(X_a_full)
centers_a = km_a.cluster_centers_

# 计算到所属簇心的距离
dists_a = np.array([np.linalg.norm(X_a_full[i] - centers_a[labels_a[i]])
                    for i in range(len(X_a_full))])

# ══════════════════════════════════════════════════════════════════
# Panel (b): 小簇异常 — DBSCAN 检测出一个极小簇
# ══════════════════════════════════════════════════════════════════
X_b, _ = make_blobs(n_samples=[120, 90], centers=[[-2, -1], [4, 4]],
                    cluster_std=[0.9, 0.8], random_state=42)
# 添加一个紧凑的小簇（3-4 个点），位置偏离主簇
small_cluster = np.array([[10, 10], [10.3, 10.5], [9.8, 10.3], [10.1, 9.8]])
X_b_full = np.vstack([X_b, small_cluster])

db_b = DBSCAN(eps=1.2, min_samples=5)
labels_b = db_b.fit_predict(X_b_full)

# ══════════════════════════════════════════════════════════════════
# Panel (c): 簇边缘异常 — 单簇内远离质心的点
# ══════════════════════════════════════════════════════════════════
X_c = np.random.multivariate_normal([0, 0], [[1.5, 0.5], [0.5, 1.5]], 200)
centroid_c = X_c.mean(axis=0)
dists_c = np.linalg.norm(X_c - centroid_c, axis=1)
threshold_c = np.percentile(dists_c, 90)

# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("图 7.5.1　三种基于聚类的异常模式",
             fontsize=20, fontweight="bold", y=1.02)

# ── (a) 不属于任何簇 ──────────────────────────────────────────────
ax = axes[0]
for c in range(3):
    mask = (labels_a == c) & (np.arange(len(X_a_full)) < n_normal_a)
    ax.scatter(X_a_full[mask, 0], X_a_full[mask, 1],
               c=cluster_colors[c], s=40, alpha=0.7, edgecolors="k",
               linewidths=0.3, zorder=3)

# 异常点（远离所有簇心的点）
ax.scatter(anomalies_a[:, 0], anomalies_a[:, 1],
           marker="*", s=250, c=COLORS["red"], edgecolors="k",
           linewidths=0.8, zorder=5, label="异常点")

# 画簇心
ax.scatter(centers_a[:, 0], centers_a[:, 1],
           marker="X", s=150, c="black", zorder=6, label="簇心")

ax.set_title("(a) 不属于任何簇", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)

# ── (b) 小簇异常 ─────────────────────────────────────────────────
ax = axes[1]
unique_labels_b = set(labels_b)
for lab in sorted(unique_labels_b):
    mask = labels_b == lab
    if lab == -1:
        ax.scatter(X_b_full[mask, 0], X_b_full[mask, 1],
                   c=COLORS["gray"], marker="x", s=50, alpha=0.7,
                   zorder=3, label="噪声点")
    else:
        # 检查是否是小簇
        cluster_size = mask.sum()
        if cluster_size <= 5:
            ax.scatter(X_b_full[mask, 0], X_b_full[mask, 1],
                       marker="*", s=250, c=COLORS["red"], edgecolors="k",
                       linewidths=0.8, zorder=5, label=f"小簇异常 (n={cluster_size})")
        else:
            color = cluster_colors[lab % len(cluster_colors)]
            ax.scatter(X_b_full[mask, 0], X_b_full[mask, 1],
                       c=color, s=40, alpha=0.7, edgecolors="k",
                       linewidths=0.3, zorder=3)

# 标注小簇
small_center = small_cluster.mean(axis=0)
ax.annotate("小簇 → 异常",
            xy=(small_center[0], small_center[1]),
            xytext=(small_center[0] - 5, small_center[1] + 1.5),
            fontsize=13, fontweight="bold", color=COLORS["red"],
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["red"], alpha=0.9))

ax.set_title("(b) 小簇异常", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)

# ── (c) 簇边缘异常 ───────────────────────────────────────────────
ax = axes[2]
normal_mask_c = dists_c <= threshold_c
anomaly_mask_c = dists_c > threshold_c

ax.scatter(X_c[normal_mask_c, 0], X_c[normal_mask_c, 1],
           c=COLORS["blue"], s=40, alpha=0.6, edgecolors="k",
           linewidths=0.3, zorder=3, label="正常点")
ax.scatter(X_c[anomaly_mask_c, 0], X_c[anomaly_mask_c, 1],
           marker="*", s=250, c=COLORS["red"], edgecolors="k",
           linewidths=0.8, zorder=5, label="边缘异常点")

# 画质心
ax.scatter(centroid_c[0], centroid_c[1],
           marker="X", s=150, c="black", zorder=6, label="质心")

# 画阈值圆
theta = np.linspace(0, 2 * np.pi, 200)
circle_x = centroid_c[0] + threshold_c * np.cos(theta)
circle_y = centroid_c[1] + threshold_c * np.sin(theta)
ax.plot(circle_x, circle_y, ls="--", lw=2, color=COLORS["red"],
        alpha=0.7, label=f"阈值半径 = {threshold_c:.1f}")

ax.set_title("(c) 簇边缘异常", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=13)
ax.set_aspect("equal")

fig.tight_layout()
save_fig(fig, __file__, "fig7_5_01_cluster_anomaly_types")
