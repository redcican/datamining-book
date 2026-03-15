"""
fig5_1_01_kmeans_iteration.py
K-means 迭代过程 4 步分解图：初始化 / 分配 / 更新 / 收敛
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
rng = np.random.default_rng(42)

# ── 生成三簇数据 ─────────────────────────────────────────────
centers_true = np.array([[1.5, 1.5], [5.0, 1.5], [3.2, 4.5]])
n_per = 30
X = np.vstack([
    rng.multivariate_normal(c, [[0.4, 0], [0, 0.4]], n_per)
    for c in centers_true
])
K = 3
CMAP = [COLORS["blue"], COLORS["red"], COLORS["green"]]
GRAY = "#aaaaaa"

def assign(X, centroids):
    dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
    return np.argmin(dists, axis=1)

def update(X, labels, K):
    return np.array([X[labels == k].mean(axis=0) for k in range(K)])

# ── 手动跑几步 ────────────────────────────────────────────────
# 初始化（故意选较差的起始点）
init_centroids = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 3.5]])
labels0 = assign(X, init_centroids)

cent1 = update(X, labels0, K)
labels1 = assign(X, cent1)

cent2 = update(X, labels1, K)
labels2 = assign(X, cent2)

cent3 = update(X, labels2, K)
labels3 = assign(X, cent3)

# 再跑几步直到收敛
cent_prev, labels_prev = cent3, labels3
for _ in range(20):
    c = update(X, labels_prev, K)
    l = assign(X, c)
    if np.all(l == labels_prev):
        break
    cent_prev, labels_prev = c, l
cent_final, labels_final = cent_prev, labels_prev

steps = [
    (init_centroids, labels0,   "（a）初始化：随机选择 $K=3$ 个质心"),
    (cent1,          labels1,   "（b）第 1 次迭代：分配 → 更新"),
    (cent2,          labels2,   "（c）第 2 次迭代：分配 → 更新"),
    (cent_final,     labels_final, "（d）收敛：质心不再移动"),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))

for ax, (centroids, labels, title) in zip(axes, steps):
    for k in range(K):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=CMAP[k], s=18,
                   alpha=0.65, linewidths=0)
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c=CMAP[:K], s=180, marker='*',
               edgecolors='black', linewidths=0.8, zorder=5)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.2, 7.0); ax.set_ylim(-0.5, 6.5)

# 图例
handles = [mpatches.Patch(color=CMAP[k], label=f'簇 {k+1}') for k in range(K)]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=False)
fig.suptitle('K-means 迭代过程：从初始化到收敛', fontsize=11, y=1.01)
fig.tight_layout()

save_fig(fig, __file__, 'fig5_1_01_kmeans_iteration')
