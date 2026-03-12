"""
图 3.2.3  朴素贝叶斯的偏差-方差定位：三类数据集上的决策边界对比
对应节次：3.2 贝叶斯分类方法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_2_03_nb_comparison.py
输出路径：public/figures/ch03/fig3_2_03_nb_comparison.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs, make_moons, make_circles
from matplotlib.colors import ListedColormap

apply_style()

# ── 数据集 ────────────────────────────────────────────────────────────────────
datasets = [
    (make_blobs(n_samples=250, centers=2, cluster_std=1.2, random_state=42),
     "高斯团簇\n（NB 理想场景：各类近似高斯分布）"),
    (make_moons(n_samples=250, noise=0.22, random_state=42),
     "弯月形\n（非线性分布：NB 线性/二次边界受限）"),
    (make_circles(n_samples=250, noise=0.14, factor=0.45, random_state=42),
     "同心圆\n（径向结构：NB 高偏差，DT 可拟合）"),
]

classifiers = [
    (GaussianNB(),                          "高斯朴素贝叶斯（GaussianNB）"),
    (DecisionTreeClassifier(max_depth=5, random_state=42), "决策树（max_depth=5）"),
]

c0, c1 = COLORS["blue"], COLORS["red"]
cmap_bg  = ListedColormap(["#dbeafe", "#fee2e2"])
cmap_pts = ListedColormap([c0, c1])

h = 0.04
fig, axes = plt.subplots(2, 3, figsize=(21, 13))
fig.subplots_adjust(hspace=0.42, wspace=0.28)

for col, ((X, y), ds_title) in enumerate(datasets):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for row, (clf, clf_name) in enumerate(classifiers):
        ax = axes[row, col]
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        acc = clf.score(X, y)

        ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg)
        ax.contour(xx, yy, Z, colors=["#475569"], linewidths=1.4, alpha=0.75)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=26, color=c0,
                   edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3, label="类 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=26, color=c1,
                   edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3, label="类 1")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("$x_1$", fontsize=12)
        ax.set_ylabel("$x_2$", fontsize=12)

        row_prefix = "(a)" if row == 0 else "(b)"
        col_labels = ["i", "ii", "iii"]
        ax.set_title(f"{row_prefix}-{col_labels[col]} {clf_name}\n"
                     f"{ds_title}（训练准确率 {acc:.1%}）",
                     fontsize=11.5, pad=5)

        if col == 0 and row == 0:
            ax.legend(fontsize=10, loc="lower right", labelspacing=0.3)

fig.suptitle(
    "高斯朴素贝叶斯（上行）vs 决策树（下行）在三类数据上的决策边界对比\n"
    "NB 边界平滑（依赖高斯假设）；DT 边界轴对齐；高斯团簇上两者相当，非线性数据上 DT 更灵活",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_2_03_nb_comparison")
