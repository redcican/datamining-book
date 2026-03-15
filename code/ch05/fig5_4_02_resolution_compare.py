"""
图 5.4.2　网格分辨率对聚类质量的影响
上排：g=5 / g=15 / g=50 的聚类结果
下排：ARI 随 g 变化的曲线
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.datasets import make_blobs
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

# ── 1. 生成数据 ──────────────────────────────────────────
X, y_true = make_blobs(n_samples=2000, centers=4, cluster_std=0.7, random_state=42)

# ── 2. 三种分辨率 ────────────────────────────────────────
g_values = [5, 15, 50]
tau = 5
fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                         gridspec_kw={'height_ratios': [2, 1.2]})

for col, g in enumerate(g_values):
    ax = axes[0, col]
    labels = grid_cluster(X, g, tau)
    ari = adjusted_rand_score(y_true, labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for k in range(min(n_clusters, 10)):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=8, c=PALETTE[k % len(PALETTE)],
                   alpha=0.6)
    noise = labels == -1
    ax.scatter(X[noise, 0], X[noise, 1], s=5, c=COLORS['gray'],
               marker='x', alpha=0.3)
    status = '过粗' if g == 5 else ('适中' if g == 15 else '过细')
    ax.set_title(f'$g={g}$（{status}）\nARI={ari:.3f}，{n_clusters}个簇', fontsize=12)
    ax.set_xlabel('$x_1$')
    if col == 0:
        ax.set_ylabel('$x_2$')

# ── 3. ARI 随 g 变化的曲线（跨越底部3列）────────────────
ax_bottom = fig.add_subplot(2, 1, 2)
# 隐藏原来的3个底部子图
for col in range(3):
    axes[1, col].set_visible(False)
g_range = range(3, 61)
aris = []
for g in g_range:
    labels = grid_cluster(X, g, tau)
    aris.append(adjusted_rand_score(y_true, labels))
ax_bottom.plot(list(g_range), aris, color=COLORS['blue'], linewidth=2)
ax_bottom.axhline(y=max(aris), color=COLORS['red'], linestyle='--', alpha=0.5,
                  label=f'最高 ARI={max(aris):.3f}')
best_g = list(g_range)[np.argmax(aris)]
ax_bottom.axvline(x=best_g, color=COLORS['green'], linestyle='--', alpha=0.5,
                  label=f'最优 g={best_g}')
# 标注三个示例点
for g_val in g_values:
    idx = g_val - 3
    if 0 <= idx < len(aris):
        ax_bottom.plot(g_val, aris[idx], 'o', markersize=10,
                       color=COLORS['red'], zorder=10)
ax_bottom.set_xlabel('网格分辨率 $g$')
ax_bottom.set_ylabel('ARI')
ax_bottom.set_title('ARI 随网格分辨率的变化')
ax_bottom.legend()
plt.tight_layout()
save_fig(fig, __file__, 'fig5_4_02_resolution_compare')
