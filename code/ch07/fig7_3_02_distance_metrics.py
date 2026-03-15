"""
fig7_3_02_distance_metrics.py
不同距离度量的等距线
三个面板分别展示欧氏距离、曼哈顿距离、切比雪夫距离的等距轮廓
以及在不同度量下异常点判定的差异
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成数据：主簇 + 不同方位离群点 ──────────────────────────────
cluster = np.random.multivariate_normal([0, 0], [[0.3, 0.05], [0.05, 0.3]], 60)
outliers = np.array([
    [2.5, 0.2],    # 沿 x 轴方向的离群点
    [0.3, 2.8],    # 沿 y 轴方向的离群点
    [1.8, 1.9],    # 对角方向的离群点
    [-2.2, 1.5],   # 左上方
    [1.5, -2.3],   # 右下方
])
data = np.vstack([cluster, outliers])

# ── 网格用于绘制等距线 ─────────────────────────────────────────────
grid_1d = np.linspace(-3.5, 3.5, 500)
xx, yy = np.meshgrid(grid_1d, grid_1d)

# 距离函数
def dist_l2(xx, yy):
    return np.sqrt(xx**2 + yy**2)

def dist_l1(xx, yy):
    return np.abs(xx) + np.abs(yy)

def dist_linf(xx, yy):
    return np.maximum(np.abs(xx), np.abs(yy))

metrics = [
    ("(a) 欧氏距离 (L₂)", dist_l2, lambda p: np.sqrt(p[0]**2 + p[1]**2)),
    ("(b) 曼哈顿距离 (L₁)", dist_l1, lambda p: np.abs(p[0]) + np.abs(p[1])),
    ("(c) 切比雪夫距离 (L∞)", dist_linf, lambda p: max(np.abs(p[0]), np.abs(p[1]))),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("图 7.3.2　Minkowski 距离族的等距线",
             fontsize=20, fontweight="bold", y=1.02)

threshold_r = 2.3  # 异常阈值半径

for ax, (title, dist_func, point_dist) in zip(axes, metrics):
    zz = dist_func(xx, yy)

    # 等距线
    levels = [0.5, 1.0, 1.5, 2.0, 2.3, 3.0]
    cs = ax.contour(xx, yy, zz, levels=levels,
                    colors=COLORS["gray"], linewidths=0.8, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=11, fmt="%.1f")

    # 阈值轮廓（加粗）
    ax.contour(xx, yy, zz, levels=[threshold_r],
               colors=COLORS["red"], linewidths=2.5, linestyles="--")

    # 分类数据点
    for i in range(len(data)):
        d = point_dist(data[i])
        if d > threshold_r:
            ax.scatter(data[i, 0], data[i, 1], c=COLORS["red"],
                       s=120, marker="*", edgecolors="k", linewidths=0.5,
                       zorder=6)
        else:
            ax.scatter(data[i, 0], data[i, 1], c=COLORS["blue"],
                       s=25, alpha=0.6, zorder=5)

    # 添加图例
    ax.scatter([], [], c=COLORS["blue"], s=25, label="正常点")
    ax.scatter([], [], c=COLORS["red"], s=120, marker="*",
               edgecolors="k", linewidths=0.5, label="异常点")
    ax.plot([], [], color=COLORS["red"], ls="--", lw=2.5,
            label=f"阈值 r={threshold_r}")

    # 原点标记
    ax.scatter(0, 0, c=COLORS["green"], s=80, marker="D",
               edgecolors="k", linewidths=0.8, zorder=7, label="中心")

    ax.set_title(title, fontsize=17)
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=11, loc="upper left")
    ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_3_02_distance_metrics")
