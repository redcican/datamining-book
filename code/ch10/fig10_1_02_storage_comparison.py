"""
fig10_1_02_storage_comparison.py
图存储结构对比：邻接矩阵 vs 邻接表
(a) 稀疏矩阵可视化 + 存储占比标注
(b) 存储空间随图密度变化的对比曲线
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle("图 10.1.2　图存储结构对比",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 邻接矩阵 vs 邻接表 — 稀疏矩阵可视化
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_title("(a) 邻接矩阵 vs 邻接表", fontsize=17)

# Generate a 20×20 sparse symmetric adjacency matrix (~10% density)
n_vis = 20
density = 0.10
upper = np.zeros((n_vis, n_vis), dtype=int)
n_edges = int(density * n_vis * (n_vis - 1) / 2)
edge_indices = set(np.random.choice(n_vis * (n_vis - 1) // 2, size=n_edges, replace=False))
idx = 0
for i in range(n_vis):
    for j in range(i + 1, n_vis):
        if idx in edge_indices:
            upper[i, j] = 1
            upper[j, i] = 1
        idx += 1

# Custom colormap: light gray for 0, blue for 1
cmap = ListedColormap([COLORS["light"], COLORS["blue"]])
ax.imshow(upper, cmap=cmap, aspect="equal", interpolation="nearest")

# Grid lines
for i in range(n_vis + 1):
    ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.8)
    ax.axvline(i - 0.5, color="white", linewidth=0.5, alpha=0.8)

# Calculate actual density
n_ones = int(upper.sum())
n_total = n_vis * n_vis
pct_zero = (n_total - n_ones) / n_total * 100

# Annotation: wasted space
ax.annotate(
    f"{pct_zero:.0f}% 零元素\n被强制存储",
    xy=(14, 14), xytext=(22, 8),
    fontsize=13, fontweight="bold", color=COLORS["gray"],
    ha="center",
    arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="white",
              ec=COLORS["gray"], alpha=0.95),
)

# Annotation: useful data
ax.annotate(
    f"仅 {n_ones} 条边\n邻接表只存这些",
    xy=(3, 8), xytext=(-6, 15),
    fontsize=13, fontweight="bold", color=COLORS["blue"],
    ha="center",
    arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="white",
              ec=COLORS["blue"], alpha=0.95),
)

ax.set_xticks([0, 4, 9, 14, 19])
ax.set_yticks([0, 4, 9, 14, 19])
ax.tick_params(labelsize=14)
# Storage comparison — two columns below xlabel
matrix_mem = n_total
list_mem = n_vis + n_ones  # node headers + edge entries
ax.text(0.25, -0.12, f"邻接矩阵: {matrix_mem} 单元",
        transform=ax.transAxes, fontsize=13, fontweight="bold",
        ha="center", va="top", color=COLORS["gray"],
        bbox=dict(boxstyle="round,pad=0.4", fc="#f1f5f9",
                  ec=COLORS["gray"], alpha=0.95))
ax.text(0.75, -0.12, f"邻接表: {list_mem} 单元",
        transform=ax.transAxes, fontsize=13, fontweight="bold",
        ha="center", va="top", color=COLORS["blue"],
        bbox=dict(boxstyle="round,pad=0.4", fc="#eff6ff",
                  ec=COLORS["blue"], alpha=0.95))
# ══════════════════════════════════════════════════════════════════
# (b) 存储空间对比 (n=1000)
# ══════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_title("(b) 存储空间对比 (n=1000)", fontsize=17)

n = 1000
rho = np.linspace(0.001, 1.0, 500)
matrix_space = np.full_like(rho, n * n, dtype=float)
list_space = n + rho * n * (n - 1)  # n node headers + 2m edge entries

# Find crossover point
crossover_idx = np.argmin(np.abs(matrix_space - list_space))
rho_cross = rho[crossover_idx]
space_cross = matrix_space[crossover_idx]

# Plot lines
ax.semilogy(rho, matrix_space, color=COLORS["red"], lw=2.5,
            label=f"邻接矩阵: $n^2 = {n*n:,}$", zorder=4)
ax.semilogy(rho, list_space, color=COLORS["blue"], lw=2.5,
            label=r"邻接表: $n + \rho \cdot n(n-1)$", zorder=4)

# Fill region where adjacency list is more efficient
mask_better = list_space < matrix_space
ax.fill_between(rho, list_space, matrix_space,
                where=mask_better, alpha=0.15, color=COLORS["blue"],
                label="邻接表更优区域")

# Fill region where matrix is better (or equal)
mask_worse = list_space >= matrix_space
ax.fill_between(rho, list_space, matrix_space,
                where=mask_worse, alpha=0.10, color=COLORS["red"],
                label="邻接矩阵更优区域")

# Mark crossover point
ax.plot(rho_cross, space_cross, "o", color=COLORS["orange"], markersize=10,
        zorder=5, markeredgecolor="white", markeredgewidth=2)
ax.annotate(
    f"交叉点\n$\\rho \\approx {rho_cross:.2f}$",
    xy=(rho_cross, space_cross),
    xytext=(rho_cross + 0.15, space_cross * 3),
    fontsize=14, fontweight="bold", color=COLORS["orange"],
    arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white",
              ec=COLORS["orange"], alpha=0.95),
)

# Annotations for sparse and dense regions
ax.annotate(
    "稀疏图\n邻接表节省空间",
    xy=(0.05, n + 0.05 * n * (n - 1)),
    xytext=(0.15, 8000),
    fontsize=13, color=COLORS["blue"],
    arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", fc="white",
              ec=COLORS["blue"], alpha=0.9),
)

ax.set_xlabel("图密度 $\\rho$", fontsize=16)
ax.set_ylabel("存储空间 (单元数)", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(fontsize=13, loc="lower right",
          framealpha=0.95, edgecolor=COLORS["light"])
ax.set_xlim(0, 1.0)
ax.set_ylim(500, 2e6)
ax.grid(True, which="both", alpha=0.25)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig10_1_02_storage_comparison")
