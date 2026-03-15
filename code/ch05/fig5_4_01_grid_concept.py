"""
图 5.4.1　网格聚类的三步过程（二维示例）
左：原始数据点 + 网格线 + 密度热力图
中：密集单元标记
右：连通分量聚类结果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from sklearn.datasets import make_blobs
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成数据 ──────────────────────────────────────────
X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)

# ── 2. 网格参数 ──────────────────────────────────────────
g = 15
tau = 5
mins = X.min(axis=0) - 0.5
maxs = X.max(axis=0) + 0.5
deltas = (maxs - mins) / g

# ── 3. 网格量化 ──────────────────────────────────────────
grid_coords = np.floor((X - mins) / deltas).astype(int)
grid_coords = np.clip(grid_coords, 0, g - 1)
cell_counts = {}
cell_points = {}
for i in range(len(X)):
    key = tuple(grid_coords[i])
    cell_counts[key] = cell_counts.get(key, 0) + 1
    cell_points.setdefault(key, []).append(i)
dense_cells = {k for k, v in cell_counts.items() if v >= tau}

# ── 4. 连通分量（BFS）───────────────────────────────────
labels = np.full(len(X), -1)
visited = set()
cluster_id = 0
components = []
for cell in dense_cells:
    if cell in visited:
        continue
    queue = deque([cell])
    visited.add(cell)
    component = []
    while queue:
        curr = queue.popleft()
        component.append(curr)
        for dim in range(2):
            for delta in [-1, 1]:
                nb = list(curr)
                nb[dim] += delta
                nb = tuple(nb)
                if nb in dense_cells and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    for c in component:
        for idx in cell_points.get(c, []):
            labels[idx] = cluster_id
    components.append(component)
    cluster_id += 1

# ── 5. 绘图 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 辅助函数：绘制网格线
def draw_grid(ax):
    for i in range(g + 1):
        ax.axvline(mins[0] + i * deltas[0], color='gray', linewidth=0.3, alpha=0.5)
        ax.axhline(mins[1] + i * deltas[1], color='gray', linewidth=0.3, alpha=0.5)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])

# 左：原始数据 + 密度热力图
ax = axes[0]
draw_grid(ax)
# 绘制密度热力图背景
for key, count in cell_counts.items():
    rect = patches.Rectangle(
        (mins[0] + key[0] * deltas[0], mins[1] + key[1] * deltas[1]),
        deltas[0], deltas[1],
        facecolor=plt.cm.YlOrRd(min(count / 20, 1.0)),
        alpha=0.4, linewidth=0)
    ax.add_patch(rect)
ax.scatter(X[:, 0], X[:, 1], s=10, c='black', alpha=0.6, zorder=5)
ax.set_title('步骤1：数据扫描 + 网格量化', fontsize=12)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

# 中：密集单元标记
ax = axes[1]
draw_grid(ax)
for key, count in cell_counts.items():
    is_dense = key in dense_cells
    color = COLORS['blue'] if is_dense else 'white'
    alpha = 0.4 if is_dense else 0.05
    rect = patches.Rectangle(
        (mins[0] + key[0] * deltas[0], mins[1] + key[1] * deltas[1]),
        deltas[0], deltas[1],
        facecolor=color, alpha=alpha, edgecolor='gray', linewidth=0.3)
    ax.add_patch(rect)
ax.scatter(X[:, 0], X[:, 1], s=10, c='black', alpha=0.3, zorder=5)
ax.set_title(f'步骤2：密集单元识别（$\\tau={tau}$）', fontsize=12)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

# 右：聚类结果
ax = axes[2]
draw_grid(ax)
# 绘制密集单元连通分量颜色
for cid, comp in enumerate(components):
    color = PALETTE[cid % len(PALETTE)]
    for cell in comp:
        rect = patches.Rectangle(
            (mins[0] + cell[0] * deltas[0], mins[1] + cell[1] * deltas[1]),
            deltas[0], deltas[1],
            facecolor=color, alpha=0.25, linewidth=0)
        ax.add_patch(rect)
# 散点图
for k in range(cluster_id):
    mask = labels == k
    ax.scatter(X[mask, 0], X[mask, 1], s=15, c=PALETTE[k % len(PALETTE)],
               alpha=0.7, zorder=5)
noise_mask = labels == -1
ax.scatter(X[noise_mask, 0], X[noise_mask, 1], s=10, c=COLORS['gray'],
           marker='x', alpha=0.5, zorder=5)
ax.set_title('步骤3：连通分量聚类', fontsize=12)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

plt.tight_layout()
save_fig(fig, __file__, 'fig5_4_01_grid_concept')
