"""
图 3.3.2  软间隔SVM：参数 C 对决策边界与间隔宽度的影响
对应节次：3.3 支持向量机（SVM）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_3_02_soft_margin.py
输出路径：public/figures/ch03/fig3_3_02_soft_margin.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

apply_style()

rng = np.random.default_rng(7)

# ── 含噪声的线性数据（类别有重叠） ────────────────────────────────────────────
n = 30
X0 = rng.multivariate_normal([-1.2, 0.0], [[0.5, 0.1], [0.1, 0.4]], n)
X1 = rng.multivariate_normal([1.2, 0.0], [[0.5, 0.1], [0.1, 0.4]], n)
X = np.vstack([X0, X1])
y = np.array([-1] * n + [1] * n)

C_vals = [0.01, 1.0, 100.0]
labels = ["$C=0.01$（大间隔，高偏差）", "$C=1.0$（平衡）", "$C=100$（窄间隔，高方差）"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.35)

x_min, x_max = -4.0, 4.0
y_min, y_max = -3.5, 3.5
xx_g = np.linspace(x_min, x_max, 400)

def get_hyperplane_y(clf, x_vals, offset=0.0):
    w = clf.coef_[0]
    b = clf.intercept_[0]
    if abs(w[1]) < 1e-10:
        return np.full_like(x_vals, np.nan)
    return (-w[0] * x_vals - b + offset) / w[1]

for ax, C, title in zip(axes, C_vals, labels):
    clf = SVC(kernel="linear", C=C)
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    gamma_val = 2.0 / np.linalg.norm(w)
    n_sv = clf.support_vectors_.shape[0]
    # 识别违约样本（$\xi_i > 0$）
    margins = y * (X @ w + b)
    violated = margins < 1.0
    correctly_classified = margins >= 0
    wrong = ~correctly_classified
    # 散点
    for cls_idx, (X_cls, col) in enumerate([(X0, COLORS["blue"]), (X1, COLORS["red"])]):
        ax.scatter(X_cls[:, 0], X_cls[:, 1], s=45, color=col,
                   edgecolors="white", linewidths=0.5, zorder=4, alpha=0.85)
    # 错误分类样本标记为叉号
    if wrong.any():
        for xi, col in zip(
            [wrong & (y == -1), wrong & (y == 1)],
            [COLORS["blue"], COLORS["red"]]
        ):
            if xi.any():
                ax.scatter(X[xi, 0], X[xi, 1], s=70, color=col,
                           marker="x", linewidths=2.0, zorder=6)
    # 超平面与间隔线
    ax.plot(xx_g, get_hyperplane_y(clf, xx_g), color="#1e293b", lw=2.2,
            label="决策边界", zorder=3)
    ax.plot(xx_g, get_hyperplane_y(clf, xx_g, +1), color=COLORS["red"],
            lw=1.5, ls="--", alpha=0.7, label="间隔线 $\\pm1$")
    ax.plot(xx_g, get_hyperplane_y(clf, xx_g, -1), color=COLORS["blue"],
            lw=1.5, ls="--", alpha=0.7)
    # 间隔带填充
    y_up = get_hyperplane_y(clf, xx_g, +1)
    y_dn = get_hyperplane_y(clf, xx_g, -1)
    valid = ~(np.isnan(y_up) | np.isnan(y_dn))
    ax.fill_between(xx_g[valid], y_dn[valid], y_up[valid],
                    color=COLORS["blue"], alpha=0.08)
    # 支持向量
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=120, facecolors="none", edgecolors=COLORS["orange"],
               linewidths=1.8, zorder=5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("$x_1$", fontsize=13)
    ax.set_ylabel("$x_2$", fontsize=13)
    ax.set_title(f"{title}\n间隔 $\\gamma={gamma_val:.2f}$，支持向量数={n_sv}",
                 fontsize=12, pad=6)

axes[0].legend(fontsize=11, loc="upper right")

fig.suptitle(
    "软间隔SVM：参数 $C$ 控制间隔宽度与违约代价的权衡\n"
    "叉号（×）为错误分类样本；圆圈（○）为支持向量；间隔带为蓝色填充区域",
    fontsize=13, y=1.02)

save_fig(fig, __file__, "fig3_3_02_soft_margin")
