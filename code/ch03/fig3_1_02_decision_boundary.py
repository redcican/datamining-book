"""
图 3.1.2  决策树深度与决策边界的关系：轴对齐矩形分区随深度增加的演变
对应节次：3.1 决策树算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_1_02_decision_boundary.py
输出路径：public/figures/ch03/fig3_1_02_decision_boundary.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

apply_style()

rng = np.random.default_rng(42)
X, y = make_moons(n_samples=300, noise=0.28, random_state=42)

configs = [
    (1,    "(a) max_depth = 1\n（决策桩，高偏差）"),
    (2,    "(b) max_depth = 2"),
    (3,    "(c) max_depth = 3"),
    (4,    "(d) max_depth = 4"),
    (6,    "(e) max_depth = 6"),
    (None, "(f) max_depth = 无限制\n（完全生长树，高方差）"),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.subplots_adjust(hspace=0.40, wspace=0.28)

# 网格分辨率
h = 0.03
x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

c0 = COLORS["blue"]
c1 = COLORS["red"]
cmap_bg  = ListedColormap(["#dbeafe", "#fee2e2"])  # 淡蓝 / 淡红
cmap_pts = ListedColormap([c0, c1])

for ax, (depth, title) in zip(axes.flat, configs):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg)
    ax.contour(xx, yy, Z, colors=["#475569"], linewidths=1.2, alpha=0.7)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=28, color=c0,
               edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3, label="类 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=28, color=c1,
               edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3, label="类 1")
    tr_acc = clf.score(X, y)
    n_leaves = clf.get_n_leaves()
    ax.set_title(f"{title}\n训练准确率 {tr_acc:.1%} | 叶节点数 {n_leaves}",
                 fontsize=12, pad=5)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

fig.suptitle(
    "CART 决策树深度（max_depth）对决策边界形状的影响\n"
    "深度增加 → 分区更细碎 → 训练准确率↑ → 但过深时在新数据上会过拟合（高方差）",
    fontsize=13, y=0.98)

save_fig(fig, __file__, "fig3_1_02_decision_boundary")
