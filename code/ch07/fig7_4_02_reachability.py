"""
fig7_4_02_reachability.py
可达距离与局部可达密度示意
左：可达距离的 max 操作图解
右：密集 vs 稀疏区域的 LRD 对比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 绘图 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.4.2　可达距离与局部可达密度",
             fontsize=20, fontweight="bold", y=1.02)

# ======================================================================
# 左 panel：可达距离示意
# ======================================================================
ax = axes[0]
k = 3

# 生成一小组点
points = np.array([
    [3.0, 5.0],   # o — 参考点
    [3.5, 5.8],   # o 的近邻
    [2.5, 4.3],
    [3.8, 4.5],
    [2.2, 5.5],
    [4.2, 5.3],
    [1.5, 3.5],   # 较远点
    [5.0, 6.5],
    [1.0, 6.0],
    [4.5, 3.8],
    [2.0, 7.0],
    [5.5, 5.0],
    [0.5, 4.5],
    [6.0, 4.0],
    [3.0, 7.5],
])

# 定义目标点 p（离 o 较近）和 q（离 o 较远）
p_idx = 1   # 近邻点
o_idx = 0   # 参考点

# 计算 o 的 k-distance
dist_from_o = cdist([points[o_idx]], points)[0]
dist_from_o[o_idx] = np.inf
sorted_indices = np.argsort(dist_from_o)
k_dist_o = dist_from_o[sorted_indices[k - 1]]
k_neighbors_o = sorted_indices[:k]

# 画所有点
ax.scatter(points[:, 0], points[:, 1], s=80, c=COLORS["gray"],
           edgecolors="k", linewidths=0.5, zorder=4)

# 高亮 o 点
ax.scatter(*points[o_idx], s=160, c=COLORS["blue"], edgecolors="k",
           linewidths=1.0, zorder=6, marker="s")
ax.annotate("$o$", xy=points[o_idx], xytext=(points[o_idx][0] - 0.6,
            points[o_idx][1] + 0.4), fontsize=16, fontweight="bold",
            color=COLORS["blue"])

# 画 k-distance 圆
circle = plt.Circle(points[o_idx], k_dist_o, fill=False,
                     linestyle="--", linewidth=2.0, color=COLORS["blue"],
                     alpha=0.6, zorder=2)
ax.add_patch(circle)
ax.annotate(f"k-dist($o$) = {k_dist_o:.2f}",
            xy=(points[o_idx][0] + k_dist_o * 0.7,
                points[o_idx][1] + k_dist_o * 0.7),
            fontsize=13, color=COLORS["blue"], fontweight="bold")

# 高亮 k 近邻
ax.scatter(points[k_neighbors_o, 0], points[k_neighbors_o, 1],
           s=100, c=COLORS["teal"], edgecolors="k",
           linewidths=0.8, zorder=5)

# ── 情形 1：p 在 k-dist 圆内（reach-dist 被 clamp） ──
p_near = points[p_idx]
dist_p_o = np.linalg.norm(p_near - points[o_idx])
reach_p = max(k_dist_o, dist_p_o)

ax.scatter(*p_near, s=160, c=COLORS["orange"], edgecolors="k",
           linewidths=1.0, zorder=6, marker="D")
ax.annotate("$p_1$（近）", xy=p_near,
            xytext=(p_near[0] + 0.3, p_near[1] + 0.5),
            fontsize=14, fontweight="bold", color=COLORS["orange"])
# 画到 o 的距离线
ax.plot([p_near[0], points[o_idx][0]], [p_near[1], points[o_idx][1]],
        ls="-", lw=1.5, color=COLORS["orange"], alpha=0.7)

# ── 情形 2：q 在 k-dist 圆外 ──
q_idx = 7  # 较远点
p_far = points[q_idx]
dist_q_o = np.linalg.norm(p_far - points[o_idx])
reach_q = max(k_dist_o, dist_q_o)

ax.scatter(*p_far, s=160, c=COLORS["red"], edgecolors="k",
           linewidths=1.0, zorder=6, marker="D")
ax.annotate("$p_2$（远）", xy=p_far,
            xytext=(p_far[0] + 0.3, p_far[1] + 0.3),
            fontsize=14, fontweight="bold", color=COLORS["red"])
ax.plot([p_far[0], points[o_idx][0]], [p_far[1], points[o_idx][1]],
        ls="-", lw=1.5, color=COLORS["red"], alpha=0.7)

# 文字说明
textbox_props = dict(boxstyle="round,pad=0.4", fc="lightyellow",
                     ec=COLORS["gray"], alpha=0.95)
ax.text(0.02, 0.02,
        f"$p_1$: dist = {dist_p_o:.2f} < k-dist\n"
        f"  reach-dist = k-dist = {reach_p:.2f}（平滑）\n\n"
        f"$p_2$: dist = {dist_q_o:.2f} > k-dist\n"
        f"  reach-dist = dist = {reach_q:.2f}",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="bottom", bbox=textbox_props)

ax.set_title("(a) 可达距离示意", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.set_aspect("equal")
ax.tick_params(labelsize=13)

# ======================================================================
# 右 panel：局部可达密度（LRD）
# ======================================================================
ax = axes[1]
k = 4

# 密集区域点
dense_pts = np.random.multivariate_normal([3, 7], [[0.08, 0], [0, 0.08]], 20)
# 稀疏区域点
sparse_pts = np.random.multivariate_normal([8, 3], [[0.8, 0], [0, 0.8]], 12)
all_pts = np.vstack([dense_pts, sparse_pts])

# 计算每个点的 LRD
dist_mat = cdist(all_pts, all_pts)
np.fill_diagonal(dist_mat, np.inf)
sorted_d = np.sort(dist_mat, axis=1)
knn_idx = np.argsort(dist_mat, axis=1)[:, :k]
k_dists = sorted_d[:, k - 1]

# 可达距离
def compute_reach_dist(i, j):
    return max(k_dists[j], dist_mat[i, j])

# LRD
lrds = np.zeros(len(all_pts))
for i in range(len(all_pts)):
    avg_reach = np.mean([compute_reach_dist(i, j) for j in knn_idx[i]])
    lrds[i] = 1.0 / max(avg_reach, 1e-10)

# 选择代表点
dense_rep = 5   # 密集区域的一个点
sparse_rep = len(dense_pts) + 3  # 稀疏区域的一个点

# 画所有点
ax.scatter(dense_pts[:, 0], dense_pts[:, 1], s=70, c=COLORS["blue"],
           edgecolors="k", linewidths=0.4, zorder=4, label="密集区域")
ax.scatter(sparse_pts[:, 0], sparse_pts[:, 1], s=70, c=COLORS["orange"],
           edgecolors="k", linewidths=0.4, zorder=4, label="稀疏区域")

# 高亮密集区域代表点及其 k 近邻
ax.scatter(*all_pts[dense_rep], s=200, c=COLORS["blue"], edgecolors="k",
           linewidths=1.5, zorder=6, marker="s")
for j in knn_idx[dense_rep]:
    ax.plot([all_pts[dense_rep][0], all_pts[j][0]],
            [all_pts[dense_rep][1], all_pts[j][1]],
            ls="--", lw=1.5, color=COLORS["blue"], alpha=0.5)

# 高亮稀疏区域代表点及其 k 近邻
ax.scatter(*all_pts[sparse_rep], s=200, c=COLORS["orange"], edgecolors="k",
           linewidths=1.5, zorder=6, marker="s")
for j in knn_idx[sparse_rep]:
    ax.plot([all_pts[sparse_rep][0], all_pts[j][0]],
            [all_pts[sparse_rep][1], all_pts[j][1]],
            ls="--", lw=1.5, color=COLORS["orange"], alpha=0.5)

# 标注 LRD 值
ax.annotate(f"LRD = {lrds[dense_rep]:.1f}\n（高密度 → 高 LRD）",
            xy=all_pts[dense_rep],
            xytext=(all_pts[dense_rep][0] + 1.5, all_pts[dense_rep][1] + 1.0),
            fontsize=13, fontweight="bold", color=COLORS["blue"],
            arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["blue"], alpha=0.9))

ax.annotate(f"LRD = {lrds[sparse_rep]:.1f}\n（低密度 → 低 LRD）",
            xy=all_pts[sparse_rep],
            xytext=(all_pts[sparse_rep][0] - 3.5, all_pts[sparse_rep][1] - 2.0),
            fontsize=13, fontweight="bold", color=COLORS["orange"],
            arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["orange"], alpha=0.9))

# 公式文字
ax.text(0.50, 0.02,
        r"$\mathrm{LRD}_k(p) = \left(\frac{1}{|N_k(p)|}"
        r"\sum_{o \in N_k(p)} \mathrm{reach\text{-}dist}_k(p, o)\right)^{-1}$",
        transform=ax.transAxes, fontsize=13,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                  ec=COLORS["gray"], alpha=0.95))

ax.set_title("(b) 局部可达密度 (LRD)", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_4_02_reachability")
