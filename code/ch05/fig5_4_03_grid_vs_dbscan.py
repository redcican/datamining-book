"""
图 5.4.3　网格聚类与 DBSCAN 在三类数据上的对比
行：高斯簇 / 月牙形 / 各向异性
列：网格聚类 / DBSCAN
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 网格聚类函数 ─────────────────────────────────────────
def grid_cluster(X, grid_size, tau):
    n = len(X)
    mins = X.min(axis=0) - 0.5
    maxs = X.max(axis=0) + 0.5
    deltas = (maxs - mins) / grid_size
    deltas[deltas == 0] = 1.0
    gc = np.floor((X - mins) / deltas).astype(int)
    gc = np.clip(gc, 0, grid_size - 1)
    cell_counts = {}
    cell_points = {}
    for i in range(n):
        key = tuple(gc[i])
        cell_counts[key] = cell_counts.get(key, 0) + 1
        cell_points.setdefault(key, []).append(i)
    dense = {k for k, v in cell_counts.items() if v >= tau}
    labels = np.full(n, -1)
    visited = set()
    cid = 0
    for cell in dense:
        if cell in visited:
            continue
        queue = deque([cell])
        visited.add(cell)
        comp = []
        while queue:
            curr = queue.popleft()
            comp.append(curr)
            for dim in range(2):
                for d in [-1, 1]:
                    nb = list(curr)
                    nb[dim] += d
                    nb = tuple(nb)
                    if nb in dense and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        for c in comp:
            for idx in cell_points.get(c, []):
                labels[idx] = cid
        cid += 1
    return labels

# ── 1. 生成三类数据集 ────────────────────────────────────
# 高斯簇
X1, y1 = make_blobs(n_samples=600, centers=3, cluster_std=0.7, random_state=42)
# 月牙形
X2, y2 = make_moons(n_samples=600, noise=0.07, random_state=42)
# 各向异性
from sklearn.datasets import make_blobs as _mb
X3_raw, y3 = _mb(n_samples=600, centers=3, cluster_std=0.5, random_state=42)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X3 = np.dot(X3_raw, transformation)

datasets = [
    (X1, y1, 3, '高斯簇', 25, 5, 0.8, 5),
    (X2, y2, 2, '月牙形', 40, 4, 0.15, 5),
    (X3, y3, 3, '各向异性', 25, 5, 0.6, 5),
]

# ── 2. 绘制 3×2 网格 ─────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 14))

for row, (X, y_true, n_c, title, g, tau, eps, min_s) in enumerate(datasets):
    # 网格聚类
    labels_g = grid_cluster(X, g, tau)
    ari_g = adjusted_rand_score(y_true, labels_g)
    ax = axes[row, 0]
    n_cl = len(set(labels_g)) - (1 if -1 in labels_g else 0)
    for k in range(min(n_cl, 10)):
        mask = labels_g == k
        ax.scatter(X[mask, 0], X[mask, 1], s=15, c=PALETTE[k % len(PALETTE)],
                   alpha=0.7)
    noise = labels_g == -1
    ax.scatter(X[noise, 0], X[noise, 1], s=8, c=COLORS['gray'],
               marker='x', alpha=0.4)
    ax.set_title(f'网格聚类  ARI={ari_g:.3f}', fontsize=12)
    if row == 2:
        ax.set_xlabel('$x_1$')
    ax.set_ylabel(title, fontsize=12)
    # DBSCAN
    labels_d = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
    ari_d = adjusted_rand_score(y_true, labels_d)
    ax = axes[row, 1]
    n_cl_d = len(set(labels_d)) - (1 if -1 in labels_d else 0)
    for k in range(min(n_cl_d, 10)):
        mask = labels_d == k
        ax.scatter(X[mask, 0], X[mask, 1], s=15, c=PALETTE[k % len(PALETTE)],
                   alpha=0.7)
    noise = labels_d == -1
    ax.scatter(X[noise, 0], X[noise, 1], s=8, c=COLORS['gray'],
               marker='x', alpha=0.4)
    ax.set_title(f'DBSCAN  ARI={ari_d:.3f}', fontsize=12)
    if row == 2:
        ax.set_xlabel('$x_1$')

plt.suptitle('网格聚类与 DBSCAN 在三类数据上的对比', fontsize=15, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_4_03_grid_vs_dbscan')
