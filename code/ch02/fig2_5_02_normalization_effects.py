"""
图 2.5.2  标准化对算法的影响：KNN 距离等高线与梯度下降收敛
对应节次：2.5 数据标准化与归一化
运行方式：MPLBACKEND=Agg python code/ch02/fig2_5_02_normalization_effects.py
输出路径：public/figures/ch02/fig2_5_02_normalization_effects.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

apply_style()

C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_GREEN  = "#16a34a"
C_ORANGE = "#ea580c"
C_PURPLE = "#7c3aed"
C_GRAY   = "#94a3b8"
C_DARK   = "#1e293b"

rng = np.random.default_rng(2024)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.subplots_adjust(wspace=0.42)

# ── Panel (a): KNN distance iso-curves — before vs after normalization ──────
ax = axes[0]
# Feature 1: age [20,60], Feature 2: income [5,100] (thousand yuan)
# Query point
qx, qy = 35, 50

# Before: age scale dominates, distance circles become very elongated
# Draw iso-distance ellipses in the original scale
# distance = sqrt( (Δage)^2 + (Δincome)^2 )
# At r=10: circles in (age, income) space look nearly circular if same scale
# but income range [5,100] vs age range [20,60] means income has ~2.5x range

# We show the "true" iso-distance circles are stretched in original space
for r, alpha in [(8, 0.35), (16, 0.25), (24, 0.15)]:
    circle = plt.Circle((qx, qy), r, color=C_BLUE, fill=False,
                         lw=1.8, alpha=alpha, ls="--")
    ax.add_patch(circle)

ax.text(qx + 8, qy + 3, "等距离圆\n（归一化后）", fontsize=12,
        color=C_BLUE, ha="left")

# Before normalization: income scale ~10x age scale → ellipses very flat
# In original units: if we "zoom out" income axis to show the distortion:
# Simulate by drawing an ellipse where income has 2x the "size"
for r_age, alpha in [(8, 0.35), (16, 0.25), (24, 0.15)]:
    ell = Ellipse((qx, qy), width=r_age * 2, height=r_age * 0.35,
                  fill=False, color=C_RED, lw=1.8, alpha=alpha, linestyle="-")
    ax.add_patch(ell)

ax.text(qx + 8.5, qy - 6, "等距离椭圆\n（未归一化，\n收入尺度主导）",
        fontsize=12, color=C_RED, ha="left")

# Query point
ax.scatter([qx], [qy], s=120, color=C_ORANGE, zorder=6,
           marker="*", label="查询点 q")

# Annotate axes
ax.set_xlabel("年龄（岁）", fontsize=12)
ax.set_ylabel("月收入（千元）", fontsize=12)
ax.set_xlim(10, 60); ax.set_ylim(15, 80)
ax.set_title("(a) 未归一化 vs 归一化的\nKNN 等距离线", fontsize=12, pad=6)
ax.tick_params(labelsize=10)

p_before = mpatches.Patch(color=C_RED, label="未归一化（椭圆，偏置）")
p_after  = mpatches.Patch(color=C_BLUE, label="归一化后（圆形，均匀）")
ax.legend(handles=[p_before, p_after], fontsize=12, loc="upper left")

# ── Panel (b): Gradient descent convergence — elongated vs round contours ───
ax = axes[1]

# Loss contours: before normalization (elongated), after (round)
x1_grid = np.linspace(-3.5, 3.5, 300)
x2_grid = np.linspace(-3.5, 3.5, 300)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# Elongated loss (unnormalized: σ1=1, σ2=5 → loss elongated along x2)
Z_elong = X1**2 / 0.5 + X2**2 / 8.0

# Round loss (normalized)
Z_round = X1**2 / 2.0 + X2**2 / 2.0

levels = [0.5, 1, 2, 4, 8]

cs1 = ax.contour(X1, X2, Z_elong, levels=levels,
                 colors=[C_RED] * len(levels), alpha=0.6, linestyles="--")
cs2 = ax.contour(X1, X2, Z_round, levels=levels,
                 colors=[C_BLUE] * len(levels), alpha=0.6)

# Gradient descent paths
def gd_path(grad_fn, start, lr, steps):
    path = [np.array(start, dtype=float)]
    w = np.array(start, dtype=float)
    for _ in range(steps):
        g = grad_fn(w)
        w = w - lr * g
        path.append(w.copy())
        if np.linalg.norm(g) < 0.01:
            break
    return np.array(path)

# Elongated: slow convergence, zigzag
grad_elong = lambda w: np.array([2 * w[0] / 0.5, 2 * w[1] / 8.0])
path_e = gd_path(grad_elong, [-3.0, 3.2], lr=0.3, steps=60)

# Round: fast convergence
grad_round = lambda w: np.array([2 * w[0] / 2.0, 2 * w[1] / 2.0])
path_r = gd_path(grad_round, [-3.0, 3.2], lr=0.6, steps=30)

ax.plot(path_e[:, 0], path_e[:, 1], "o-", color=C_RED, ms=3, lw=1.5,
        alpha=0.8, label=f"未归一化（{len(path_e)-1} 步，锯齿状）")
ax.plot(path_r[:, 0], path_r[:, 1], "s-", color=C_BLUE, ms=3, lw=2.0,
        alpha=0.8, label=f"归一化后（{len(path_r)-1} 步，直线收敛）")

# Mark start and end
ax.scatter([path_e[0, 0]], [path_e[0, 1]], s=80, marker="^",
           color=C_ORANGE, zorder=6)
ax.scatter([0], [0], s=80, marker="*", color=C_GREEN, zorder=6)
ax.text(0.1, 0.15, "最优解", fontsize=12, color=C_GREEN)

ax.set_xlabel("参数 $w_1$", fontsize=12)
ax.set_ylabel("参数 $w_2$", fontsize=12)
ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.8, 3.8)
ax.set_title("(b) 梯度下降收敛：未归一化\n（锯齿）vs 归一化后（快速）", fontsize=12, pad=6)
ax.legend(fontsize=12.5, loc="lower right")
ax.tick_params(labelsize=10)

# ── Panel (c): Mahalanobis vs Euclidean distance ────────────────────────────
ax = axes[2]
rng2 = np.random.default_rng(7)

# Correlated bivariate normal data
mean = np.array([0, 0])
cov = np.array([[4.0, 2.8], [2.8, 2.5]])
pts = rng2.multivariate_normal(mean, cov, 200)
ax.scatter(pts[:, 0], pts[:, 1], s=18, color=C_GRAY, alpha=0.5, zorder=2)

# Query and test points
q = np.array([2.0, 0.0])
A = np.array([2.0, 2.5])   # close in Euclidean but far in Mahalanobis
B = np.array([3.5, 2.5])   # farther in Euclidean but closer in Mahal.

ax.scatter(*q, s=100, color=C_ORANGE, zorder=6, marker="*", label="查询点 q")
ax.scatter(*A, s=80, color=C_RED, zorder=6, marker="o", label=f"点 A")
ax.scatter(*B, s=80, color=C_BLUE, zorder=6, marker="^", label=f"点 B")

# Compute distances
S_inv = np.linalg.inv(cov)
d_euc_A = np.linalg.norm(A - q)
d_euc_B = np.linalg.norm(B - q)
d_mah_A = np.sqrt((A - q) @ S_inv @ (A - q))
d_mah_B = np.sqrt((B - q) @ S_inv @ (B - q))

ax.annotate(f"A\nd_E={d_euc_A:.1f}\nd_M={d_mah_A:.1f}",
            A, xytext=(A[0] + 0.4, A[1] + 0.4), fontsize=12,
            color=C_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
ax.annotate(f"B\nd_E={d_euc_B:.1f}\nd_M={d_mah_B:.1f}",
            B, xytext=(B[0] + 0.4, B[1] - 0.8), fontsize=12,
            color=C_BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2))

# Draw Mahalanobis iso-distance ellipses around q
for k, alpha in [(1.5, 0.4), (3.0, 0.25), (5.0, 0.15)]:
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * k * np.sqrt(vals)
    ell = Ellipse(q, width=w, height=h, angle=angle,
                  fill=False, color=C_GREEN, lw=1.5, alpha=alpha, ls="-")
    ax.add_patch(ell)
ax.text(q[0] + 0.1, q[1] - 1.8, "Mahal.\n等距离椭圆", fontsize=12,
        color=C_GREEN, ha="center")

ax.set_xlabel("特征 $x_1$", fontsize=12)
ax.set_ylabel("特征 $x_2$", fontsize=12)
ax.set_xlim(-6, 7); ax.set_ylim(-5, 6)
ax.set_title("(c) Mahalanobis 距离 vs 欧氏距离\n（椭圆等高线 vs 圆形等高线）", fontsize=12, pad=6)
ax.legend(fontsize=12.5, loc="upper left")
ax.tick_params(labelsize=10)
ax.text(0.03, 0.04,
        "A 欧氏距离更近，\n但在数据分布中属\"反常\"；\n"
        "Mahal. 距离 A > B，\n正确反映统计意义上的远近",
        transform=ax.transAxes, fontsize=12, va="bottom", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cbd5e1", alpha=0.9))

fig.suptitle("数据标准化的必要性：KNN 偏置 / 梯度下降加速 / Mahalanobis 距离",
             fontsize=14, y=1.02)
fig.text(
    0.5, -0.04,
    "(a) 尺度差异导致 KNN 中大尺度特征主导距离，归一化使等距离线回归圆形。"
    "(b) 归一化后损失曲面等高线趋近圆形，梯度下降无需锯齿，收敛步数大幅减少。"
    "(c) Mahalanobis 距离考虑特征相关性和方差，等高线为椭圆而非圆形，更符合数据分布。",
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_5_02_normalization_effects")
