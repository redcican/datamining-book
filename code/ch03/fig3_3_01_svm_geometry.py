"""
图 3.3.1  SVM 几何直觉：间隔边界、支持向量与最优超平面
对应节次：3.3 支持向量机（SVM）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_3_01_svm_geometry.py
输出路径：public/figures/ch03/fig3_3_01_svm_geometry.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

apply_style()

rng = np.random.default_rng(0)

# ── 合成线性可分数据 ──────────────────────────────────────────────────────────
n = 18
X0 = rng.multivariate_normal([-1.8, 0.5], [[0.20, 0.05], [0.05, 0.25]], n)
X1 = rng.multivariate_normal([1.8, -0.3], [[0.22, 0.04], [0.04, 0.20]], n)
X = np.vstack([X0, X1])
y = np.array([-1] * n + [1] * n)

clf = SVC(kernel="linear", C=1e6)       # 硬间隔近似
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]
sv = clf.support_vectors_

# ── 绘图区域 ─────────────────────────────────────────────────────────────────
x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
xx_g = np.linspace(x_min, x_max, 400)

def hyperplane_y(x_vals, w, b, offset=0.0):
    """给定 w·x + b = offset 求 x2 的值。"""
    return (-w[0] * x_vals - b + offset) / w[1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.38)

# ── Panel (a): 三条候选超平面对比 ─────────────────────────────────────────────
ax = axes[0]
ax.scatter(X0[:, 0], X0[:, 1], s=55, color=COLORS["blue"], edgecolors="white",
           linewidths=0.6, zorder=4, label="负类（$y=-1$）")
ax.scatter(X1[:, 0], X1[:, 1], s=55, color=COLORS["red"], edgecolors="white",
           linewidths=0.6, zorder=4, label="正类（$y=+1$）")

# 最优超平面（蓝色实线）
ax.plot(xx_g, hyperplane_y(xx_g, w, b), color=COLORS["blue"], lw=2.4,
        label="最大间隔超平面（SVM）", zorder=3)
# 次优超平面 1（灰色）：平移最优超平面
ax.plot(xx_g, hyperplane_y(xx_g, w, b) + 0.9, color=COLORS["gray"],
        lw=1.6, ls="--", alpha=0.7, label="次优超平面 1")
# 次优超平面 2（灰色）：倾斜版
w2 = np.array([w[0] + 0.5, w[1] - 0.3])
b2 = b - 0.3
ax.plot(xx_g, hyperplane_y(xx_g, w2, b2), color=COLORS["gray"],
        lw=1.6, ls=":", alpha=0.7, label="次优超平面 2")

# 标记支持向量
ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolors="none",
           edgecolors=COLORS["orange"], linewidths=2.0, zorder=5,
           label="支持向量")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title("(a) 三条合法超平面\n蓝色最优，灰色次优（训练误差均为零）", fontsize=13, pad=6)
ax.legend(fontsize=11, loc="upper right", labelspacing=0.35)

# ── Panel (b): 间隔带可视化 ───────────────────────────────────────────────────
ax = axes[1]

# 决策边界与间隔线
ax.plot(xx_g, hyperplane_y(xx_g, w, b), color=COLORS["blue"], lw=2.4,
        label="决策边界 $\\mathbf{w}^\\top\\mathbf{x}+b=0$")
ax.plot(xx_g, hyperplane_y(xx_g, w, b, +1), color=COLORS["blue"],
        lw=1.6, ls="--", alpha=0.7,
        label="间隔线 $\\mathbf{w}^\\top\\mathbf{x}+b=+1$")
ax.plot(xx_g, hyperplane_y(xx_g, w, b, -1), color=COLORS["red"],
        lw=1.6, ls="--", alpha=0.7,
        label="间隔线 $\\mathbf{w}^\\top\\mathbf{x}+b=-1$")

# 填充间隔带
y_upper = hyperplane_y(xx_g, w, b, +1)
y_lower = hyperplane_y(xx_g, w, b, -1)
ax.fill_between(xx_g, y_lower, y_upper,
                where=(xx_g >= x_min) & (xx_g <= x_max),
                color=COLORS["blue"], alpha=0.10, label="间隔带（宽度 $\\gamma=2/\\|\\mathbf{w}\\|$）")

ax.scatter(X0[:, 0], X0[:, 1], s=55, color=COLORS["blue"], edgecolors="white",
           linewidths=0.6, zorder=4)
ax.scatter(X1[:, 0], X1[:, 1], s=55, color=COLORS["red"], edgecolors="white",
           linewidths=0.6, zorder=4)
ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolors="none",
           edgecolors=COLORS["orange"], linewidths=2.0, zorder=5,
           label="支持向量（决定 $\\mathbf{w}^*$）")

# 标注间隔宽度
gamma_val = 2.0 / np.linalg.norm(w)
mid_x = 0.0
mid_y_center = hyperplane_y(np.array([mid_x]), w, b)[0]
mid_y_top = hyperplane_y(np.array([mid_x]), w, b, +1)[0]
perp_dir = np.array([-w[1], w[0]]) / np.linalg.norm(w)
pt_top = np.array([mid_x, mid_y_center]) + perp_dir * gamma_val / 2
pt_bot = np.array([mid_x, mid_y_center]) - perp_dir * gamma_val / 2
ax.annotate("", xy=pt_top, xytext=pt_bot,
            arrowprops=dict(arrowstyle="<->", color=COLORS["teal"], lw=1.8))
ax.text(pt_top[0] + 0.18, (pt_top[1] + pt_bot[1]) / 2,
        f"$\\gamma={gamma_val:.2f}$", fontsize=12, color=COLORS["teal"])

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title("(b) 最优超平面与最大间隔带\n$\\gamma^* = 2/\\|\\mathbf{w}^*\\|$", fontsize=13, pad=6)
ax.legend(fontsize=11, loc="upper right", labelspacing=0.3)

fig.suptitle("支持向量机（SVM）：几何间隔最大化与最优超平面",
             fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_3_01_svm_geometry")
