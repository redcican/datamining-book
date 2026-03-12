"""
图 3.6.3  集成方法决策边界对比：单棵树 vs Bagging vs RF vs AdaBoost vs GBDT
对应节次：3.6 集成学习算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_6_03_decision_boundaries.py
输出路径：public/figures/ch03/fig3_6_03_decision_boundaries.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)

apply_style()

# ── 数据集 ────────────────────────────────────────────────────────────────────
datasets = [
    make_moons(n_samples=400,  noise=0.25, random_state=42),
    make_circles(n_samples=400, noise=0.18, factor=0.5, random_state=42),
]
ds_names = ["弯月形（make_moons）", "同心圆（make_circles）"]

# ── 分类器 ────────────────────────────────────────────────────────────────────
classifiers = [
    ("单棵决策树\n（无限深度）",  DecisionTreeClassifier(max_depth=None, random_state=42)),
    ("Bagging\n（$T=50$，决策树）", BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ("随机森林\n（$T=50$）",     RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ("AdaBoost\n（$T=50$）",     AdaBoostClassifier(n_estimators=50, algorithm="SAMME", random_state=42)),
    ("GBDT\n（$T=50$，depth=3）", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                               learning_rate=0.1, random_state=42)),
]
clf_names = [c[0] for c in classifiers]

COLORS_CLS = [COLORS["blue"], COLORS["red"]]

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.subplots_adjust(hspace=0.40, wspace=0.28)

for row, ((X, y), ds_name) in enumerate(zip(datasets, ds_names)):
    # 标准化
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    X_all_s = scaler.transform(X)

    x1_min, x1_max = X_all_s[:, 0].min() - 0.4, X_all_s[:, 0].max() + 0.4
    x2_min, x2_max = X_all_s[:, 1].min() - 0.4, X_all_s[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                         np.linspace(x2_min, x2_max, 300))

    for col, (clf_name, clf) in enumerate(classifiers):
        ax = axes[row, col]
        clf.fit(X_tr_s, y_tr)
        te_acc = clf.score(X_te_s, y_te)

        # 决策区域
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                    cmap=plt.cm.RdBu, alpha=0.25)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["#1e293b"], linewidths=1.8)

        # 测试散点
        for cls_label, c in zip([0, 1], COLORS_CLS):
            mask = y_te == cls_label
            X_te_plot = scaler.transform(X_te)
            ax.scatter(X_te_plot[mask, 0], X_te_plot[mask, 1],
                       c=c, s=18, alpha=0.75, edgecolors="none")

        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xticks([]); ax.set_yticks([])

        title = clf_name
        if row == 0:
            ax.set_title(f"{title}\n准确率={te_acc:.1%}", fontsize=11, pad=4)
        else:
            ax.set_title(f"准确率={te_acc:.1%}", fontsize=11, pad=4)

        if col == 0:
            ax.set_ylabel(ds_name, fontsize=12, fontweight="bold")

fig.suptitle(
    "集成方法决策边界对比：从单棵树到 GBDT 的边界演变\n"
    "单棵树：高方差、锯齿边界；Bagging/RF：平滑；Boosting：精细拟合",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_6_03_decision_boundaries")
