"""
图 4.1.1  OLS 几何解释：正交投影、残差与梯度下降
对应节次：4.1 线性回归基础（OLS、正规方程、Gauss-Markov 定理）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_1_01_ols_geometry.py
输出路径：public/figures/ch04/fig4_1_01_ols_geometry.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

apply_style()

C_DATA  = COLORS["blue"]
C_FIT   = COLORS["red"]
C_RESID = COLORS["orange"]
C_PROJ  = COLORS["green"]
C_GRAY  = COLORS["gray"]
C_PURP  = COLORS["purple"]

np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# ── 面板(a): 正交投影几何示意（列空间） ───────────────────────────────────────
ax = axes[0]
ax.set_axis_off()
ax.set_xlim(-0.5, 8)
ax.set_ylim(-0.5, 7)

# 列空间：画为一条斜线（2D中的1D子空间示意）
col_dir = np.array([1.0, 0.55])
col_dir /= np.linalg.norm(col_dir)
t_range = np.linspace(-0.3, 6.5, 100)
col_line = np.outer(t_range, col_dir)
ax.plot(col_line[:, 0] + 0.5, col_line[:, 1] + 0.4,
        color=C_GRAY, lw=2.0, zorder=1)
ax.text(7.2, 4.2, r"$\mathrm{col}(\mathbf{X})$",
        fontsize=13, color=C_GRAY, fontweight="bold")

# y 向量（观测值，不在列空间内）
origin = np.array([0.5, 0.4])
y_vec  = np.array([2.5, 5.8])
y_abs  = origin + y_vec

# ŷ = 投影点（y在列空间的垂足）
t_proj = np.dot(y_vec, col_dir)
yhat_vec = t_proj * col_dir
yhat_abs = origin + yhat_vec

# 残差向量 e = y - ŷ
e_vec = y_vec - yhat_vec

def draw_vec(ax, start, end, color, label, label_offset=(0.1, 0.1),
             lw=2.0, style="-|>", fs=13):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    lx, ly = (start + end) / 2 + np.array(label_offset)
    ax.text(lx, ly, label, fontsize=fs, color=color, fontweight="bold",
            ha="center", va="center")

draw_vec(ax, origin, y_abs, C_DATA, r"$\mathbf{y}$",
         label_offset=np.array([-0.45, 0.15]))
draw_vec(ax, origin, yhat_abs, C_PROJ, r"$\hat{\mathbf{y}}$",
         label_offset=np.array([0.0, -0.35]))
draw_vec(ax, yhat_abs, y_abs, C_RESID, r"$\mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}$",
         label_offset=np.array([0.85, 0.0]))

# 直角标记（残差⊥列空间）
perp_size = 0.22
p1 = yhat_abs + perp_size * col_dir
p2 = p1 + perp_size * (e_vec / np.linalg.norm(e_vec))
p3 = yhat_abs + perp_size * (e_vec / np.linalg.norm(e_vec))
rect = plt.Polygon([yhat_abs, p1, p2, p3], fill=False,
                   edgecolor=C_RESID, lw=1.2, zorder=5)
ax.add_patch(rect)
ax.text(yhat_abs[0] + 0.42, yhat_abs[1] - 0.22, "⊥",
        fontsize=14, color=C_RESID, ha="center", va="center")

# 文字说明
ax.text(3.8, 0.9,
        r"$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y} = \mathbf{X}(\mathbf{X}^T\!\mathbf{X})^{-1}\!\mathbf{X}^T\mathbf{y}$",
        fontsize=12, color=C_PROJ, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec=C_PROJ, alpha=0.9))
ax.text(3.8, 0.15,
        r"$\mathbf{e} \perp \mathrm{col}(\mathbf{X})$  $\Rightarrow$  $\mathbf{X}^T\mathbf{e} = \mathbf{0}$",
        fontsize=12, color=C_RESID, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7ed", ec=C_RESID, alpha=0.9))
ax.set_title("(a) OLS 的正交投影几何解释\n"
             r"$\hat{\mathbf{y}}$ 是 $\mathbf{y}$ 在列空间 $\mathrm{col}(\mathbf{X})$ 上的正交投影",
             fontsize=13, pad=8)

# ── 面板(b): 一维 OLS 拟合与残差 ─────────────────────────────────────────────
ax = axes[1]
n = 20
x = np.linspace(0, 10, n)
y_true = 1.5 + 0.8 * x
y_obs  = y_true + np.random.normal(0, 1.2, n)

X_mat = np.column_stack([np.ones(n), x])
beta  = np.linalg.lstsq(X_mat, y_obs, rcond=None)[0]
y_hat = X_mat @ beta

# 散点
ax.scatter(x, y_obs, color=C_DATA, s=50, zorder=5, label="观测值 $y_i$")
# 拟合线
x_line = np.linspace(-0.3, 10.3, 200)
ax.plot(x_line, beta[0] + beta[1] * x_line, color=C_FIT, lw=2.2,
        label=fr"OLS 拟合线 $\hat{{y}}={beta[0]:.2f}+{beta[1]:.2f}x$")
# 残差竖线
for xi, yi, yhi in zip(x, y_obs, y_hat):
    ax.plot([xi, xi], [yi, yhi], color=C_RESID, lw=1.0, alpha=0.7, zorder=3)
# 残差标注（第一个）
idx = 8
ax.annotate(r"$e_i = y_i - \hat{y}_i$",
            xy=(x[idx], (y_obs[idx] + y_hat[idx]) / 2),
            xytext=(x[idx] + 2.0, (y_obs[idx] + y_hat[idx]) / 2),
            arrowprops=dict(arrowstyle="->", color=C_RESID, lw=1.3),
            fontsize=12, color=C_RESID,
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff7ed", alpha=0.9))
ax.set_xlabel("$x$（解释变量）", fontsize=13)
ax.set_ylabel("$y$（响应变量）", fontsize=13)
ax.legend(fontsize=11, loc="upper left")
ax.set_title("(b) 一维 OLS 拟合与残差\n"
             r"最小化 $\sum_i e_i^2 = \|\mathbf{y}-\mathbf{X}\hat{\boldsymbol{\beta}}\|^2$",
             fontsize=13, pad=8)

# ── 面板(c): 损失函数等高线与梯度下降路径 ────────────────────────────────────
ax = axes[2]
# 生成损失函数等高线（二维，β₀, β₁）
b0_grid = np.linspace(-1, 4, 200)
b1_grid = np.linspace(0, 1.8, 200)
B0, B1 = np.meshgrid(b0_grid, b1_grid)
# L(β₀, β₁) = Σ(yi - β₀ - β₁·xi)²
Loss = np.zeros_like(B0)
for xi, yi in zip(x, y_obs):
    Loss += (yi - B0 - B1 * xi) ** 2

cp = ax.contourf(B0, B1, Loss, levels=30, cmap="Blues_r", alpha=0.85)
ax.contour(B0, B1, Loss, levels=15, colors="white", linewidths=0.5, alpha=0.4)
plt.colorbar(cp, ax=ax, label="损失 $L(\\beta_0, \\beta_1)$", fraction=0.046, pad=0.04)

# 标注 OLS 最优解
ax.scatter([beta[0]], [beta[1]], color=C_FIT, s=120, zorder=6,
           marker="*", label=fr"OLS 最优解 $(\hat{{\beta}}_0, \hat{{\beta}}_1)$")

# 模拟梯度下降路径（学习率 0.0008，200 步保证收敛）
lr_gd = 0.0008
b_gd = np.array([-0.5, 1.6])
path = [b_gd.copy()]
for _ in range(200):
    grad = np.zeros(2)
    for xi, yi in zip(x, y_obs):
        err = b_gd[0] + b_gd[1] * xi - yi
        grad[0] += 2 * err
        grad[1] += 2 * err * xi
    b_gd = b_gd - lr_gd * grad
    path.append(b_gd.copy())
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], color=C_RESID, lw=1.8, zorder=5,
        label="梯度下降路径", marker="o", markersize=3, markevery=20)
ax.scatter([path[0, 0]], [path[0, 1]], color=C_RESID, s=80, zorder=7,
           marker="s", label="初始点")

ax.set_xlabel(r"$\beta_0$（截距）", fontsize=13)
ax.set_ylabel(r"$\beta_1$（斜率）", fontsize=13)
ax.legend(fontsize=11, loc="upper right")
ax.set_title("(c) 损失函数等高线与梯度下降路径\n"
             r"碗形曲面，OLS 解析解直达全局最优",
             fontsize=13, pad=8)

fig.suptitle("OLS 普通最小二乘：几何解释、残差可视化与优化路径",
             fontsize=14, fontweight="bold", y=1.01)
save_fig(fig, __file__, "fig4_1_01_ols_geometry")
