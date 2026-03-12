"""
图 3.3.3  三种核函数（线性、多项式、RBF）在三类合成数据上的决策边界对比
对应节次：3.3 支持向量机（SVM）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_3_03_kernel_boundaries.py
输出路径：public/figures/ch03/fig3_3_03_kernel_boundaries.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

apply_style()

# ── 三类数据集 ────────────────────────────────────────────────────────────────
datasets = [
    make_blobs(n_samples=140, centers=2, cluster_std=0.85, random_state=1),
    make_moons(n_samples=140, noise=0.18, random_state=1),
    make_circles(n_samples=140, noise=0.12, factor=0.45, random_state=1),
]
dataset_titles = ["高斯团簇（线性可分）", "弯月形（非线性）", "同心圆（径向非线性）"]

# ── 三种核函数 ────────────────────────────────────────────────────────────────
kernels = [
    SVC(kernel="linear",  C=1.0),
    SVC(kernel="poly",    C=1.0, degree=3, coef0=1, gamma="auto"),
    SVC(kernel="rbf",     C=1.0, gamma=0.5),
]
kernel_labels = ["线性核", "多项式核（$p=3$）", "RBF 核（$\\gamma=0.5$）"]
kernel_colors = [COLORS["blue"], COLORS["purple"], COLORS["teal"]]

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.subplots_adjust(hspace=0.40, wspace=0.30)

for col, (X_raw, y) in enumerate(datasets):
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                         np.linspace(x2_min, x2_max, 300))
    for row, (clf, klabel, kcol) in enumerate(zip(kernels, kernel_labels, kernel_colors)):
        ax = axes[row][col]
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        # 背景色
        ax.contourf(xx, yy, Z, alpha=0.25,
                    cmap=plt.cm.RdBu_r, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors=["#1e293b"], linewidths=2.0)
        # 散点
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=35,
                   color=COLORS["blue"], edgecolors="white",
                   linewidths=0.5, zorder=4, alpha=0.90)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=35,
                   color=COLORS["red"], edgecolors="white",
                   linewidths=0.5, zorder=4, alpha=0.90)
        # 支持向量
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=100, facecolors="none", edgecolors=COLORS["orange"],
                   linewidths=1.6, zorder=5)
        train_acc = clf.score(X, y)
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xlabel("$x_1$", fontsize=12)
        ax.set_ylabel("$x_2$", fontsize=12)
        title_row = klabel if col == 0 else ""
        title_col = dataset_titles[col] if row == 0 else ""
        sep = "\n" if title_row and title_col else ""
        ax.set_title(f"{title_col}{sep}{title_row}\n训练准确率={train_acc:.1%}",
                     fontsize=12, pad=4)

fig.suptitle(
    "三种 SVM 核函数在三类数据上的决策边界（$C=1$）\n"
    "圆圈（○）为支持向量；黑实线为决策边界",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_3_03_kernel_boundaries")
