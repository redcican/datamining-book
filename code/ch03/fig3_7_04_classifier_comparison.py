"""
图 3.7.4  全章分类器决策边界九宫格对比
对应节次：3.7 分类算法评估与比较
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_7_04_classifier_comparison.py
输出路径：public/figures/ch03/fig3_7_04_classifier_comparison.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier)

apply_style()

# ── 数据集（3 个） ─────────────────────────────────────────────────────────────
datasets = [
    ("弯月形\n(make_moons)",
     make_moons(n_samples=300, noise=0.25, random_state=42)),
    ("同心圆\n(make_circles)",
     make_circles(n_samples=300, noise=0.18, factor=0.5, random_state=42)),
    ("线性可分\n(make_class.)",
     make_classification(n_samples=300, n_features=2, n_redundant=0,
                         n_informative=2, n_clusters_per_class=1,
                         random_state=42)),
]

# ── 分类器（3 个） ─────────────────────────────────────────────────────────────
classifiers = [
    ("决策树\n(depth=5)",
     DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("SVM\n(RBF 核)",
     SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ("随机森林\n($T=100$)",
     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 16))

C0, C1 = COLORS["blue"], COLORS["red"]
cmap_bg  = ListedColormap(["#dbeafe", "#fee2e2"])
cmap_pts = ListedColormap([C0, C1])

x_range = np.linspace(-3.2, 3.2, 300)
y_range = np.linspace(-3.2, 3.2, 300)
xx, yy = np.meshgrid(x_range, y_range)

for row, (ds_name, (X, y)) in enumerate(datasets):
    X_s = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.25, random_state=42, stratify=y)
    for col, (clf_name, clf) in enumerate(classifiers):
        ax = axes[row][col]
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
        ax.contour(xx, yy, Z, levels=[0.5], colors="gray",
                   linewidths=1.2, linestyles="--")
        ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1],
                   c=C0, s=28, edgecolors="white", lw=0.5,
                   alpha=0.9, zorder=4)
        ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1],
                   c=C1, marker="^", s=30, edgecolors="white", lw=0.5,
                   alpha=0.9, zorder=4)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"{clf_name}\nAcc={acc:.3f}"
        ax.set_title(title, fontsize=13, pad=4)
        if col == 0:
            ax.set_ylabel(ds_name, fontsize=13, labelpad=6)

fig.suptitle("全章分类器决策边界对比（三种数据集 × 三类核心算法）\n"
             "蓝色 = 类别 0，红色 = 类别 1；测试集样本以散点标注",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(h_pad=1.5, w_pad=1.0)
save_fig(fig, __file__, "fig3_7_04_classifier_comparison")
