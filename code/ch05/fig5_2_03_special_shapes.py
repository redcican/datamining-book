"""
fig5_2_03_special_shapes.py
特殊形状数据集上各算法对比：
行: 月牙形 / 同心圆 / 各向异性分布
列: K-means / DBSCAN / OPTICS
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.datasets import make_moons, make_circles
from shared.plot_config import apply_style, save_fig, PALETTE

apply_style()
rng = np.random.default_rng(0)

# ── 生成三类数据 ──────────────────────────────────────────────
n = 250

# 月牙形
X_moons, _ = make_moons(n_samples=n, noise=0.07, random_state=0)

# 同心圆
X_circles, _ = make_circles(n_samples=n, factor=0.45, noise=0.05,
                             random_state=0)

# 各向异性（三个拉伸的高斯簇）
def anisotropic(n, seed):
    rng2 = np.random.default_rng(seed)
    centers = [[1, 2], [4, 1], [2.5, 4]]
    covs = [
        [[0.8, 0.6], [0.6, 0.2]],
        [[0.6, -0.4], [-0.4, 0.6]],
        [[0.2, 0], [0, 0.8]],
    ]
    return np.vstack([
        rng2.multivariate_normal(c, cov, n // 3)
        for c, cov in zip(centers, covs)
    ])

X_aniso = anisotropic(n, 0)

datasets = [
    (X_moons,   'make_moons',   2,  0.20, 5),
    (X_circles, 'make_circles', 2,  0.18, 5),
    (X_aniso,   'Anisotropic',  3,  0.45, 5),
]

algorithms = ['K-means', 'DBSCAN', 'OPTICS']

fig, axes = plt.subplots(3, 3, figsize=(11, 10))

for row, (X, dname, k, eps, minpts) in enumerate(datasets):
    # K-means
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    y_km = km.fit_predict(X)

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=minpts)
    y_db = db.fit_predict(X)

    # OPTICS
    op = OPTICS(min_samples=minpts, xi=0.05, min_cluster_size=0.1)
    y_op = op.fit_predict(X)

    for col, (labels, algo) in enumerate(zip([y_km, y_db, y_op], algorithms)):
        ax = axes[row][col]
        unique = sorted(set(labels))
        for lbl in unique:
            mask = labels == lbl
            if lbl == -1:
                ax.scatter(X[mask, 0], X[mask, 1],
                           c='#aaaaaa', s=12, marker='x',
                           linewidths=0.8, zorder=3)
            else:
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=PALETTE[lbl % len(PALETTE)], s=14,
                           alpha=0.8, linewidths=0, zorder=2)

        n_cl = len(unique) - (1 if -1 in unique else 0)
        n_ns = np.sum(np.array(labels) == -1)
        info = f'{n_cl} 簇' + (f', {n_ns} 噪声' if n_ns > 0 else '')
        ax.set_title(f'{info}', fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

# 行标签（数据集名称）
row_labels = ['月牙形\n(make_moons)', '同心圆\n(make_circles)', '各向异性\n(Anisotropic)']
for row, label in enumerate(row_labels):
    axes[row][0].set_ylabel(label, fontsize=9.5, rotation=90, labelpad=6)

# 列标签（算法名称）
for col, algo in enumerate(algorithms):
    axes[0][col].set_xlabel('')
    axes[0][col].set_title(f'{algo}\n' + axes[0][col].get_title(), fontsize=9.5)

fig.suptitle('特殊形状数据集：K-means vs DBSCAN vs OPTICS\n'
             '灰色×为噪声点', fontsize=10, y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, __file__, 'fig5_2_03_special_shapes')
