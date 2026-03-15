"""
fig7_5_02_cluster_distance_score.py
聚类距离异常评分
(a) K-means 聚类与距离评分（颜色映射）
(b) 距离评分分布（直方图 + 阈值线）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成数据：3 个高斯簇 + 8 个异常点 ────────────────────────────
X_normal, y_normal = make_blobs(
    n_samples=[100, 80, 90], centers=[[-5, -3], [4, 5], [6, -4]],
    cluster_std=[0.8, 0.7, 0.9], random_state=42)

anomalies = np.array([
    [12, 10], [-10, 8], [0, -10], [14, -8],
    [-8, -9], [10, 0], [-3, 10], [0, 0],
])

X = np.vstack([X_normal, anomalies])
n_normal = len(X_normal)
n_anomaly = len(anomalies)
is_anomaly = np.zeros(len(X), dtype=bool)
is_anomaly[n_normal:] = True

# ── K-means 聚类 ─────────────────────────────────────────────────
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
centers = km.cluster_centers_

# 计算每个点到其所属簇心的距离
dists = np.array([np.linalg.norm(X[i] - centers[labels[i]])
                  for i in range(len(X))])

# 阈值：95 百分位
threshold = np.percentile(dists, 95)

# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.5.2　聚类距离异常评分",
             fontsize=20, fontweight="bold", y=1.02)

# ── (a) K-means 聚类与距离评分 ────────────────────────────────────
ax = axes[0]
sc = ax.scatter(X[:, 0], X[:, 1], c=dists, cmap="RdYlBu_r",
                s=50, edgecolors="k", linewidths=0.3, zorder=3)

# 簇心
ax.scatter(centers[:, 0], centers[:, 1],
           marker="X", s=200, c="black", zorder=6, label="簇心")

cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("到簇心距离", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# 标记超过阈值的点
detected = dists >= threshold
ax.scatter(X[detected, 0], X[detected, 1],
           facecolors="none", edgecolors=COLORS["red"], s=150,
           linewidths=2, zorder=4, label=f"异常 (≥ 95th)")

ax.set_title("(a) K-means 聚类与距离评分", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="lower right")
ax.tick_params(labelsize=13)

# ── (b) 距离评分分布 ─────────────────────────────────────────────
ax = axes[1]

dists_normal = dists[~is_anomaly]
dists_anomaly = dists[is_anomaly]

# 统一 bin 范围
bins = np.linspace(0, max(dists) * 1.05, 30)
ax.hist(dists_normal, bins=bins, color=COLORS["blue"], alpha=0.7,
        edgecolor="white", linewidth=0.5, label="正常点")
ax.hist(dists_anomaly, bins=bins, color=COLORS["red"], alpha=0.8,
        edgecolor="white", linewidth=0.5, label="异常点")

# 阈值线
ax.axvline(threshold, color=COLORS["red"], ls="--", lw=2.5,
           label=f"阈值 (95th) = {threshold:.1f}")

# 标注区域
y_max = ax.get_ylim()[1]
ax.text(threshold * 0.35, y_max * 0.85, "正常",
        fontsize=16, fontweight="bold", color=COLORS["blue"],
        ha="center")
ax.text(threshold + (max(dists) - threshold) * 0.5, y_max * 0.85, "异常",
        fontsize=16, fontweight="bold", color=COLORS["red"],
        ha="center")

ax.set_title("(b) 距离评分分布", fontsize=17)
ax.set_xlabel("到簇心距离", fontsize=15)
ax.set_ylabel("频数", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_5_02_cluster_distance_score")
