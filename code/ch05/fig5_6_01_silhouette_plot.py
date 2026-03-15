"""
图 5.6.1　轮廓图分析
左：K-means 聚类散点图
右：按簇分组的轮廓系数条形图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成数据 ──────────────────────────────────────────
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=[0.8, 1.2, 0.6],
                       random_state=42)

# ── 2. K-means 聚类 ──────────────────────────────────────
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X)
sil_avg = silhouette_score(X, labels)
sil_vals = silhouette_samples(X, labels)

# ── 3. 绘图 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左：散点图
colors_list = PALETTE[:3]
for k in range(3):
    mask = labels == k
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors_list[k], s=25, alpha=0.7,
                edgecolors='white', linewidths=0.3, label=f'簇 {k+1}')
ax1.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c='black', marker='X', s=200, zorder=10)
ax1.set_title(f'K-means 聚类（K=3，轮廓系数={sil_avg:.3f}）', fontsize=13)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.legend()

# 右：轮廓图
y_lower = 10
for k in range(3):
    cluster_sils = sil_vals[labels == k]
    cluster_sils.sort()
    size = len(cluster_sils)
    y_upper = y_lower + size
    ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sils,
                      facecolor=colors_list[k], edgecolor=colors_list[k],
                      alpha=0.7)
    ax2.text(-0.05, y_lower + 0.5 * size, f'簇 {k+1}', fontsize=11,
             fontweight='bold')
    y_lower = y_upper + 10

ax2.axvline(x=sil_avg, color=COLORS['red'], linestyle='--', linewidth=2,
            label=f'平均轮廓系数={sil_avg:.3f}')
ax2.set_title('轮廓图（按簇分组）', fontsize=13)
ax2.set_xlabel('轮廓系数 $s(i)$')
ax2.set_ylabel('样本编号')
ax2.set_xlim(-0.2, 1.0)
ax2.legend(loc='lower right')

plt.tight_layout()
save_fig(fig, __file__, 'fig5_6_01_silhouette_plot')
