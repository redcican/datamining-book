"""
fig7_2_02_mahalanobis.py
欧氏距离 vs 马氏距离对比
左：欧氏距离同心圆 + 两个关键点
右：马氏距离椭圆等高线 + 正确识别异常
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成相关二维正态数据 ─────────────────────────────────────────
n = 300
mean = np.array([0, 0])
# 强正相关
cov = np.array([[3.0, 2.5],
                [2.5, 3.0]])
X = np.random.multivariate_normal(mean, cov, n)

# 两个关键点
# A: 沿相关轴方向远离中心 → 欧氏距离大，马氏距离小（正常）
point_A = np.array([3.5, 3.5])
# B: 垂直于相关轴 → 欧氏距离小，马氏距离大（异常）
point_B = np.array([2.0, -2.0])

# 计算距离
center = np.mean(X, axis=0)
cov_est = np.cov(X.T)
cov_inv = np.linalg.inv(cov_est)

def euclidean_dist(p, c):
    d = p - c
    return np.sqrt(d @ d)

def mahalanobis_dist(p, c, cov_inv):
    d = p - c
    return np.sqrt(d @ cov_inv @ d)

euc_A = euclidean_dist(point_A, center)
euc_B = euclidean_dist(point_B, center)
mah_A = mahalanobis_dist(point_A, center, cov_inv)
mah_B = mahalanobis_dist(point_B, center, cov_inv)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.2.2　欧氏距离 vs 马氏距离", fontsize=20, fontweight="bold", y=1.02)

# ── 左：欧氏距离 ────────────────────────────────────────────────
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=COLORS["blue"], s=15, alpha=0.4, label="正常数据")

# 同心圆
for r in [2, 3, 4, 5]:
    circle = plt.Circle(center, r, fill=False, color=COLORS["gray"],
                        ls="--", lw=1.2, alpha=0.6)
    ax.add_patch(circle)
    ax.text(center[0] + r * 0.707, center[1] - r * 0.707 - 0.3,
            f"d={r}", fontsize=11, color=COLORS["gray"], ha="center")

# 关键点
ax.scatter(*point_A, c=COLORS["green"], s=180, marker="D", edgecolors="k",
           linewidths=1, zorder=6, label=f"A (d$_E$={euc_A:.1f})")
ax.scatter(*point_B, c=COLORS["red"], s=180, marker="s", edgecolors="k",
           linewidths=1, zorder=6, label=f"B (d$_E$={euc_B:.1f})")

ax.annotate("A: 欧氏距离大\n但沿相关方向",
            xy=point_A, xytext=(point_A[0] - 3.5, point_A[1] + 1.5),
            fontsize=13, color=COLORS["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))
ax.annotate("B: 欧氏距离较小\n但偏离相关方向",
            xy=point_B, xytext=(point_B[0] - 4.0, point_B[1] - 1.8),
            fontsize=13, color=COLORS["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))

ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.set_aspect("equal")
ax.set_title("(a) 欧氏距离（同心圆）", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)

# ── 右：马氏距离 ────────────────────────────────────────────────
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=COLORS["blue"], s=15, alpha=0.4, label="正常数据")

# 马氏距离椭圆等高线
eigenvalues, eigenvectors = np.linalg.eigh(cov_est)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

for r, alpha_val in [(1, 0.7), (2, 0.5), (3, 0.4), (4, 0.3)]:
    w = 2 * r * np.sqrt(eigenvalues[0])
    h = 2 * r * np.sqrt(eigenvalues[1])
    ellipse = Ellipse(center, w, h, angle=angle, fill=False,
                      color=COLORS["teal"], ls="--", lw=1.5, alpha=alpha_val)
    ax.add_patch(ellipse)
    # 标注沿长轴方向
    label_angle_rad = np.radians(angle)
    lx = center[0] + r * np.sqrt(eigenvalues[1]) * np.cos(label_angle_rad)
    ly = center[1] + r * np.sqrt(eigenvalues[1]) * np.sin(label_angle_rad)
    ax.text(lx + 0.4, ly + 0.2, f"$d_M$={r}", fontsize=11, color=COLORS["teal"])

# 关键点
ax.scatter(*point_A, c=COLORS["green"], s=180, marker="D", edgecolors="k",
           linewidths=1, zorder=6, label=f"A (d$_M$={mah_A:.1f})")
ax.scatter(*point_B, c=COLORS["red"], s=180, marker="s", edgecolors="k",
           linewidths=1, zorder=6, label=f"B (d$_M$={mah_B:.1f})")

ax.annotate("A: 马氏距离小\n→ 正常",
            xy=point_A, xytext=(point_A[0] - 4.0, point_A[1] + 1.5),
            fontsize=13, color=COLORS["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))
ax.annotate("B: 马氏距离大\n→ 异常!",
            xy=point_B, xytext=(point_B[0] - 4.5, point_B[1] - 1.5),
            fontsize=13, color=COLORS["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))

ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.set_aspect("equal")
ax.set_title("(b) 马氏距离（等距椭圆）", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_2_02_mahalanobis")
