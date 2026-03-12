"""
图 3.4.3  距离度量的几何差异：Minkowski 单位球与 Voronoi 图对比
对应节次：3.4 K 近邻（KNN）算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_4_03_distance_metrics.py
输出路径：public/figures/ch03/fig3_4_03_distance_metrics.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier

apply_style()

# ── 面板 (a): Minkowski 单位球 ─────────────────────────────────────────────────
theta = np.linspace(0, 2 * np.pi, 1000)
cos_t, sin_t = np.cos(theta), np.sin(theta)

def unit_ball_boundary(p, n=1000):
    """参数化 L_p 单位球边界：|x1|^p + |x2|^p = 1"""
    th = np.linspace(0, 2 * np.pi, n)
    x1 = np.sign(np.cos(th)) * (np.abs(np.cos(th)) ** (2/p) if p < np.inf else np.abs(np.cos(th)))
    x2 = np.sign(np.sin(th)) * (np.abs(np.sin(th)) ** (2/p) if p < np.inf else np.abs(np.sin(th)))
    return x1, x2

# L∞ 单位球 = 正方形 [-1,1]^2
def linf_ball():
    xs = [-1, 1, 1, -1, -1]
    ys = [-1, -1, 1, 1, -1]
    return xs, ys

# ── 合成 5 个中心点用于 Voronoi ────────────────────────────────────────────────
rng = np.random.default_rng(7)
centers = np.array([[-1.5, 0.5], [0.0, 1.2], [1.5, 0.3],
                    [-0.8, -1.2], [1.0, -1.0]])
labels  = np.array([0, 1, 2, 0, 1])
colors_voronoi = [COLORS["blue"], COLORS["red"], COLORS["teal"],
                  COLORS["blue"], COLORS["red"]]

def knn_regions(metric, ax, resolution=300):
    """绘制 1-NN 决策区域（等效 Voronoi）"""
    x1_min, x1_max = -2.8, 2.8
    x2_min, x2_max = -2.2, 2.2
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, resolution),
                         np.linspace(x2_min, x2_max, resolution))
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1, metric=metric)
    clf.fit(centers, labels)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap = plt.cm.Set2
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=cmap, alpha=0.28)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors=["#1e293b"], linewidths=1.4)
    for i, (c, col) in enumerate(zip(centers, colors_voronoi)):
        ax.scatter(c[0], c[1], s=100, color=col, edgecolors="white",
                   linewidths=1.2, zorder=5)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.38)

# ── 面板 (a): 单位球对比 ──────────────────────────────────────────────────────
ax = axes[0]
# L1
x1_l1, x2_l1 = unit_ball_boundary(1)
ax.fill(x1_l1, x2_l1, alpha=0.18, color=COLORS["blue"])
ax.plot(x1_l1, x2_l1, color=COLORS["blue"], lw=2.0,
        label="$L_1$（曼哈顿距离）")
# L2
x1_l2, x2_l2 = unit_ball_boundary(2)
ax.fill(x1_l2, x2_l2, alpha=0.18, color=COLORS["red"])
ax.plot(x1_l2, x2_l2, color=COLORS["red"], lw=2.0,
        label="$L_2$（欧氏距离）")
# L∞
xs_inf, ys_inf = linf_ball()
ax.fill(xs_inf, ys_inf, alpha=0.18, color=COLORS["teal"])
ax.plot(xs_inf, ys_inf, color=COLORS["teal"], lw=2.0,
        label="$L_\\infty$（切比雪夫距离）")

ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axhline(0, color="#94a3b8", lw=0.8, ls="--")
ax.axvline(0, color="#94a3b8", lw=0.8, ls="--")
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title("(a) $L_p$ 范数单位球\n"
             "$p$越小球越尖锐，$p\\to\\infty$ 趋向正方形",
             fontsize=12, pad=6)
ax.legend(fontsize=11, loc="upper right")

# ── 面板 (b): L2 距离的 Voronoi 区域 ─────────────────────────────────────────
ax = axes[1]
knn_regions("euclidean", ax)
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title("(b) $L_2$ 欧氏距离下的 1-NN 区域\n"
             "（等效于 Voronoi 图，边界为各类中心的垂直平分线）",
             fontsize=12, pad=6)

# ── 面板 (c): L1 距离的 Voronoi 区域 ─────────────────────────────────────────
ax = axes[2]
knn_regions("manhattan", ax)
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title("(c) $L_1$ 曼哈顿距离下的 1-NN 区域\n"
             "边界为 45° 折线，对角方向距离被等权衡量",
             fontsize=12, pad=6)

fig.suptitle(
    "距离度量的几何影响：$L_p$ 单位球形状决定了\"谁是邻居\"，进而决定决策边界",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_4_03_distance_metrics")
