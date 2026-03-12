"""
图 3.1.3  决策树深度的偏差–方差权衡：max_depth vs 训练/测试准确率（乳腺癌数据集）
对应节次：3.1 决策树算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_1_03_depth_sensitivity.py
输出路径：public/figures/ch03/fig3_1_03_depth_sensitivity.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

apply_style()

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

depths = list(range(1, 26))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tr_accs, te_accs, cv_means, cv_stds = [], [], [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_tr, y_tr)
    tr_accs.append(clf.score(X_tr, y_tr))
    te_accs.append(clf.score(X_te, y_te))
    scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="accuracy")
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

tr_accs = np.array(tr_accs)
te_accs = np.array(te_accs)
cv_means = np.array(cv_means)
cv_stds  = np.array(cv_stds)

best_d_te = depths[np.argmax(te_accs)]
best_d_cv = depths[np.argmax(cv_means)]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.40)

# ── Panel (a): 训练 vs 测试准确率 ──────────────────────────────────────────────
ax = axes[0]
ax.plot(depths, tr_accs, color=COLORS["blue"], lw=2.4, marker="o", ms=5,
        label="训练准确率")
ax.plot(depths, te_accs, color=COLORS["red"],  lw=2.4, marker="s", ms=5,
        label="测试准确率（25% 留出）")
ax.axvline(best_d_te, color=COLORS["red"], lw=1.6, ls="--", alpha=0.7,
           label=f"测试最优深度 = {best_d_te}")
ax.fill_between(depths, tr_accs, te_accs,
                where=(tr_accs > te_accs), alpha=0.10, color=COLORS["orange"],
                label="过拟合差距（训练 − 测试）")
ax.set_xlabel("决策树最大深度（max_depth）", fontsize=13)
ax.set_ylabel("分类准确率", fontsize=13)
ax.set_title("(a) 训练 vs 测试准确率随深度的变化\n"
             "（数据集：乳腺癌 Wisconsin，569 样本，30 特征）", fontsize=13, pad=6)
ax.legend(fontsize=12, loc="lower right", labelspacing=0.3)
ax.set_xlim(1, 25); ax.set_ylim(0.80, 1.02)
ax.set_xticks(range(1, 26, 2))

# ── Panel (b): 5 折交叉验证（均值 ± 标准差）────────────────────────────────────
ax = axes[1]
ax.plot(depths, cv_means, color=COLORS["teal"], lw=2.4, marker="^", ms=5,
        label="CV 均值（5 折）")
ax.fill_between(depths,
                cv_means - cv_stds,
                cv_means + cv_stds,
                alpha=0.22, color=COLORS["teal"], label="±1σ 置信带")
ax.axvline(best_d_cv, color=COLORS["teal"], lw=1.8, ls="--", alpha=0.8,
           label=f"CV 最优深度 = {best_d_cv}")
# 标注最优点
ax.scatter([best_d_cv], [cv_means[best_d_cv - 1]], s=100, zorder=5,
           color=COLORS["teal"], edgecolors="white", linewidths=1.5)
ax.annotate(f"最优 CV 准确率\n{cv_means[best_d_cv-1]:.3f}",
            xy=(best_d_cv, cv_means[best_d_cv - 1]),
            xytext=(best_d_cv + 3, cv_means[best_d_cv - 1] - 0.012),
            fontsize=12, color=COLORS["teal"],
            arrowprops=dict(arrowstyle="-|>", color=COLORS["teal"], lw=1.2))
ax.set_xlabel("决策树最大深度（max_depth）", fontsize=13)
ax.set_ylabel("交叉验证准确率", fontsize=13)
ax.set_title("(b) 5 折交叉验证准确率随深度的变化\n"
             "（置信带：均值 ± 标准差；推荐用此曲线选择 max_depth）", fontsize=13, pad=6)
ax.legend(fontsize=12, loc="lower right", labelspacing=0.3)
ax.set_xlim(1, 25); ax.set_ylim(0.88, 0.98)
ax.set_xticks(range(1, 26, 2))

fig.suptitle(
    "决策树深度的偏差–方差权衡（乳腺癌数据集）\n"
    "深度过小 → 高偏差（欠拟合）；深度过大 → 高方差（过拟合）；交叉验证帮助找到最优深度",
    fontsize=13, y=1.02)

save_fig(fig, __file__, "fig3_1_03_depth_sensitivity")
