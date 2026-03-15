"""
fig7_5_03_method_comparison.py
聚类方法 vs 距离/密度方法对比
(a) K-means 异常检测 — 非凸形状下表现不佳
(b) DBSCAN 异常检测 — 自然处理非凸形状
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成数据：双月形 + 异常点 ─────────────────────────────────────
X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)
# 放大尺度方便可视化
X_moons *= 3

# 散布在周围的异常点
anomalies = np.array([
    [5.5, 4.0], [-3.5, 3.5], [7.0, -2.5], [-4.0, -3.0],
    [3.0, 5.0], [-2.0, -4.5], [8.0, 2.0], [6.0, -4.0],
])
X = np.vstack([X_moons, anomalies])
n_moons = len(X_moons)
n_anomaly = len(anomalies)

# ══════════════════════════════════════════════════════════════════
# K-means (k=2)
# ══════════════════════════════════════════════════════════════════
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km.fit_predict(X)
centers_km = km.cluster_centers_

# 到所属簇心的距离
dists_km = np.array([np.linalg.norm(X[i] - centers_km[labels_km[i]])
                     for i in range(len(X))])

# 距离阈值：95 百分位
threshold_km = np.percentile(dists_km, 95)
detected_km = dists_km >= threshold_km

# ══════════════════════════════════════════════════════════════════
# DBSCAN
# ══════════════════════════════════════════════════════════════════
db = DBSCAN(eps=0.6, min_samples=8)
labels_db = db.fit_predict(X)

# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.5.3　不同聚类方法的异常检测效果",
             fontsize=20, fontweight="bold", y=1.02)

cluster_colors = [COLORS["blue"], COLORS["green"]]

# ── (a) K-means 异常检测 ─────────────────────────────────────────
ax = axes[0]

# 按聚类标签着色，颜色深浅映射距离
for c in range(2):
    mask = (labels_km == c) & (~detected_km)
    ax.scatter(X[mask, 0], X[mask, 1], c=cluster_colors[c],
               s=40, alpha=0.6, edgecolors="k", linewidths=0.3, zorder=3)

# 检测到的异常（红圈标记）
ax.scatter(X[detected_km, 0], X[detected_km, 1],
           facecolors="none", edgecolors=COLORS["red"], s=160,
           linewidths=2.5, zorder=5, label="检测为异常 (top 5%)")

# 簇心
ax.scatter(centers_km[:, 0], centers_km[:, 1],
           marker="X", s=200, c="black", zorder=6, label="簇心")

# 标注问题
ax.annotate("非凸形状导致\n簇心偏移",
            xy=(centers_km[0, 0], centers_km[0, 1]),
            xytext=(centers_km[0, 0] + 2.5, centers_km[0, 1] + 2.5),
            fontsize=13, fontweight="bold", color=COLORS["orange"],
            arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["orange"], alpha=0.9))

ax.set_title("(a) K-means 异常检测", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=12, loc="lower left")
ax.tick_params(labelsize=13)

# ── (b) DBSCAN 异常检测 ──────────────────────────────────────────
ax = axes[1]

unique_labels = sorted(set(labels_db))
for lab in unique_labels:
    mask = labels_db == lab
    if lab == -1:
        ax.scatter(X[mask, 0], X[mask, 1],
                   marker="*", s=200, c=COLORS["red"], edgecolors="k",
                   linewidths=0.8, zorder=5, label="噪声/异常点")
    else:
        color = cluster_colors[lab % len(cluster_colors)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color,
                   s=40, alpha=0.6, edgecolors="k", linewidths=0.3,
                   zorder=3, label=f"簇 {lab}")

# 标注优势
ax.annotate("非凸形状\n正确聚类",
            xy=(1.5, -0.8),
            xytext=(4.0, -3.5),
            fontsize=13, fontweight="bold", color=COLORS["green"],
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["green"], alpha=0.9))

ax.set_title("(b) DBSCAN 异常检测", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=12, loc="lower left")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_5_03_method_comparison")
