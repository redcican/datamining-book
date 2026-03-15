"""
fig7_2_03_kde_anomaly.py
核密度估计与带宽选择
左：1D KDE 不同带宽对比 + 异常阈值
右：2D KDE 等高线 + 异常点标记
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成 1D 数据：混合高斯 + 离群点 ──────────────────────────────
n1 = 150
data_1d = np.concatenate([
    np.random.normal(2.0, 0.6, n1),
    np.random.normal(5.0, 0.8, int(n1 * 0.6)),
])
outliers_1d = np.array([-1.5, 8.5, 9.0])
all_1d = np.concatenate([data_1d, outliers_1d])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.2.3　核密度估计与带宽选择", fontsize=20, fontweight="bold", y=1.02)

# ── 左：1D KDE 不同带宽 ──────────────────────────────────────────
ax = axes[0]
x_grid = np.linspace(-3, 11, 500)

bandwidths = [0.1, 0.5, 2.0]
bw_labels = ["h=0.1（过窄/噪声）", "h=0.5（合适）", "h=2.0（过平滑）"]
bw_colors = [COLORS["orange"], COLORS["blue"], COLORS["purple"]]
bw_styles = [":", "-", "--"]
bw_widths = [1.5, 2.5, 1.8]

kde_good = None
for bw, label, color, ls, lw in zip(bandwidths, bw_labels, bw_colors, bw_styles, bw_widths):
    kde = gaussian_kde(all_1d, bw_method=bw)
    density = kde(x_grid)
    ax.plot(x_grid, density, color=color, ls=ls, lw=lw, label=label, zorder=3)
    if bw == 0.5:
        kde_good = density

# 异常阈值 (在好的 KDE 上)
threshold = 0.02
ax.axhline(threshold, color=COLORS["red"], ls="--", lw=1.5, alpha=0.7,
           label=f"异常阈值 (ρ={threshold})")
ax.fill_between(x_grid, 0, kde_good,
                where=(kde_good < threshold),
                alpha=0.12, color=COLORS["red"])

# Rug plot
ax.scatter(all_1d[:-3], np.full(len(data_1d), -0.008), c=COLORS["blue"],
           s=8, alpha=0.5, marker="|", linewidths=1)
ax.scatter(outliers_1d, np.full(len(outliers_1d), -0.008), c=COLORS["red"],
           s=40, marker="|", linewidths=2, zorder=5)

# 标记异常点
for ox in outliers_1d:
    kde_val = gaussian_kde(all_1d, bw_method=0.5)(ox)[0]
    ax.scatter(ox, kde_val, c=COLORS["red"], s=60, marker="v", zorder=6,
               edgecolors="k", linewidths=0.5)

ax.set_ylim(bottom=-0.02)
ax.set_title("(a) 1D KDE 带宽对比", fontsize=17)
ax.set_xlabel("数据值", fontsize=15)
ax.set_ylabel("密度估计 $\\hat{f}(x)$", fontsize=15)
ax.legend(fontsize=12, loc="upper right")
ax.tick_params(labelsize=13)

# ── 右：2D KDE 等高线 ────────────────────────────────────────────
ax = axes[1]

# 生成 2D 混合高斯
n2a, n2b = 200, 120
cluster_a = np.random.multivariate_normal([2, 2], [[0.5, 0.2], [0.2, 0.5]], n2a)
cluster_b = np.random.multivariate_normal([5, 5], [[0.8, -0.3], [-0.3, 0.6]], n2b)
data_2d = np.vstack([cluster_a, cluster_b])

# 离群点
outliers_2d = np.array([
    [0, 6.5],
    [7, 1],
    [8, 7],
    [-1, 0],
    [6.5, -0.5],
])

all_2d = np.vstack([data_2d, outliers_2d])

# 2D KDE
kde_2d = gaussian_kde(all_2d.T, bw_method=0.3)
x_min, x_max = -2.5, 9.5
y_min, y_max = -2.5, 9.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))
positions = np.vstack([xx.ravel(), yy.ravel()])
zz = kde_2d(positions).reshape(xx.shape)

# 密度等高线
levels = np.percentile(zz[zz > 0], [5, 15, 30, 50, 70, 85, 95])
contour = ax.contourf(xx, yy, zz, levels=np.concatenate([[0], levels, [zz.max()]]),
                       cmap="Blues", alpha=0.5)
ax.contour(xx, yy, zz, levels=levels, colors=COLORS["blue"], linewidths=0.8, alpha=0.6)

# 低密度轮廓（异常边界）
low_level = np.percentile(zz[zz > 0], 5)
ax.contour(xx, yy, zz, levels=[low_level], colors=COLORS["red"],
           linewidths=2, linestyles="--")

# 数据点
ax.scatter(data_2d[:, 0], data_2d[:, 1], c=COLORS["blue"], s=12, alpha=0.4, label="正常数据")
ax.scatter(outliers_2d[:, 0], outliers_2d[:, 1], c=COLORS["red"], s=100, marker="*",
           edgecolors="k", linewidths=0.5, zorder=5, label="异常点")

# 标注低密度区域
ax.text(8.2, 8.5, "低密度区域\n→ 异常", fontsize=13, color=COLORS["red"],
        fontweight="bold", ha="center")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title("(b) 2D KDE 密度等高线", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_2_03_kde_anomaly")
