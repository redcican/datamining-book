"""
图 13.3.4　可解释决策树（深度 3）
展示临床可读的诊断决策路径
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_medical import load_cancer

X, y, feature_names = load_cancer()

# ── 训练深度 3 决策树 ────────────────────────────────────
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
f1_scores = cross_val_score(tree, X, y, cv=cv, scoring="f1")

print("=== 深度 3 决策树 ===")
print(f"  训练准确率: {tree.score(X, y):.3f}")
print(f"  5 折 CV 准确率: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
print(f"  5 折 CV F1:     {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
print(f"  叶节点数: {tree.get_n_leaves()}")
print(f"  树深度:   {tree.get_depth()}")

# 打印根节点分裂特征
root_feature = feature_names[tree.tree_.feature[0]]
root_threshold = tree.tree_.threshold[0]
print(f"\n  根节点分裂: {root_feature} <= {root_threshold:.2f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 8))

plot_tree(tree, ax=ax,
          feature_names=feature_names,
          class_names=["恶性 (Malignant)", "良性 (Benign)"],
          filled=True, rounded=True,
          fontsize=9, proportion=True,
          impurity=True)

ax.set_title("可解释决策树（深度 3）—— 乳腺癌诊断决策路径",
             fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
save_fig(fig, __file__, "fig13_3_04_decision_tree")
