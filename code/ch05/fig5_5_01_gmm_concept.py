"""
图 5.5.1　GMM 软聚类与 K-means 硬聚类的对比
左：K-means 硬分割 + Voronoi 边界
右：GMM 等高线 + 概率着色
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成椭圆形簇数据 ──────────────────────────────────
n = 300
X1 = np.random.multivariate_normal([0, 0], [[1.5, 0.8], [0.8, 1.0]], 100)
X2 = np.random.multivariate_normal([4, 3], [[1.0, -0.5], [-0.5, 1.5]], 100)
X3 = np.random.multivariate_normal([1, 5], [[0.8, 0.3], [0.3, 0.6]], 100)
X = np.vstack([X1, X2, X3])
y_true = np.array([0]*100 + [1]*100 + [2]*100)

# ── 2. K-means vs GMM ────────────────────────────────────
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
gmm = GaussianMixture(n_components=3, n_init=10, random_state=42).fit(X)
labels_km = km.labels_
proba_gmm = gmm.predict_proba(X)
labels_gmm = gmm.predict(X)

# ── 3. 网格用于背景着色和等高线 ──────────────────────────
x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左：K-means
ax = axes[0]
km_pred = km.predict(grid).reshape(xx.shape)
cmap_bg = ListedColormap(['#dbeafe', '#fee2e2', '#dcfce7'])
ax.contourf(xx, yy, km_pred, alpha=0.3, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5, 2.5])
colors_km = [PALETTE[0], PALETTE[1], PALETTE[2]]
for k in range(3):
    mask = labels_km == k
    ax.scatter(X[mask, 0], X[mask, 1], c=colors_km[k], s=25, alpha=0.7,
               edgecolors='white', linewidths=0.3)
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c='black', marker='X', s=200, zorder=10, label='质心')
ax.set_title('K-means 硬聚类', fontsize=13)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(loc='upper left')

# 右：GMM
ax = axes[1]
# 绘制等高线
Z = -gmm.score_samples(grid).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=15, colors='gray', linewidths=0.5, alpha=0.6)
# 使用最高概率成分着色，但透明度反映确定性
max_proba = np.max(proba_gmm, axis=1)
for k in range(3):
    mask = labels_gmm == k
    sc = ax.scatter(X[mask, 0], X[mask, 1], c=colors_km[k], s=25,
                    alpha=max_proba[mask] * 0.8 + 0.1,
                    edgecolors='white', linewidths=0.3)
# 绘制椭圆
from matplotlib.patches import Ellipse
for k in range(3):
    mean = gmm.means_[k]
    cov = gmm.covariances_[k]
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for n_std in [1, 2]:
        ell = Ellipse(xy=mean, width=2*n_std*np.sqrt(vals[1]),
                      height=2*n_std*np.sqrt(vals[0]), angle=angle,
                      edgecolor=colors_km[k], facecolor='none',
                      linewidth=1.5 if n_std == 1 else 0.8,
                      linestyle='-' if n_std == 1 else '--')
        ax.add_patch(ell)
ax.set_title('GMM 软聚类（透明度∝确定性）', fontsize=13)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
save_fig(fig, __file__, 'fig5_5_01_gmm_concept')
