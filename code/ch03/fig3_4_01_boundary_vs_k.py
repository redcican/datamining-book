"""
图 3.4.1  KNN 决策边界随 k 值变化：偏差–方差权衡的几何直观
对应节次：3.4 K 近邻（KNN）算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_4_01_boundary_vs_k.py
输出路径：public/figures/ch03/fig3_4_01_boundary_vs_k.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

apply_style()

rng = np.random.default_rng(42)

# ── 两类数据集 ────────────────────────────────────────────────────────────────
datasets = [
    make_moons(n_samples=200, noise=0.25, random_state=0),
    make_circles(n_samples=200, noise=0.18, factor=0.45, random_state=0),
]
dataset_labels = ["弯月形数据集", "同心圆数据集"]

# ── 三个 k 值 ─────────────────────────────────────────────────────────────────
k_vals   = [1, 5, 21]
k_labels = ["$k=1$（高方差，过拟合）", "$k=5$（偏差–方差均衡）", "$k=21$（高偏差，欠拟合）"]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.subplots_adjust(hspace=0.40, wspace=0.28)

for row, (X_raw, y) in enumerate(datasets):
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                         np.linspace(x2_min, x2_max, 300))
    for col, (k, klabel) in enumerate(zip(k_vals, k_labels)):
        ax = axes[row][col]
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        clf.fit(X, y)
        train_acc = clf.score(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        # 背景决策区域
        ax.contourf(xx, yy, Z, alpha=0.22,
                    cmap=plt.cm.RdBu_r, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors=["#1e293b"], linewidths=1.8)
        # 训练散点
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=30,
                   color=COLORS["blue"], edgecolors="white",
                   linewidths=0.4, zorder=4, alpha=0.90)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=30,
                   color=COLORS["red"], edgecolors="white",
                   linewidths=0.4, zorder=4, alpha=0.90)
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xlabel("$x_1$", fontsize=12)
        ax.set_ylabel("$x_2$", fontsize=12)
        # 标题：行标 + 列标
        col_title = klabel if row == 0 else ""
        row_title = dataset_labels[row] if col == 0 else ""
        sep = "\n" if col_title and row_title else ""
        ax.set_title(f"{row_title}{sep}{col_title}\n训练准确率={train_acc:.1%}",
                     fontsize=12, pad=4)

fig.suptitle(
    "KNN 决策边界随 $k$ 变化：$k=1$ 严重过拟合（锯齿边界），$k=21$ 过平滑（高偏差）\n"
    "蓝色/红色区域为预测类别；黑实线为决策边界",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_4_01_boundary_vs_k")
