"""
图 13.2.3　五种分类模型 ROC 曲线对比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_credit import prepare_data

X_train, X_test, y_train, y_test, preprocessor, *_ = prepare_data()

# ── 1. 训练模型并计算 ROC ────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
}

colors = [COLORS["blue"], COLORS["orange"], COLORS["green"],
          COLORS["red"], COLORS["purple"]]

fig, ax = plt.subplots(figsize=(9, 8))

print("=== ROC-AUC (测试集) ===")
for (name, model), color in zip(models.items(), colors):
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"  {name:<22s}  AUC = {roc_auc:.3f}")
    ax.plot(fpr, tpr, color=color, linewidth=2.0,
            label=f"{name} (AUC = {roc_auc:.3f})")

# ── 2. 绘制对角线和装饰 ─────────────────────────────────
ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--",
        linewidth=1, label="随机猜测 (AUC = 0.500)")
ax.set_xlabel("假正率 (FPR)", fontsize=12)
ax.set_ylabel("真正率 (TPR)", fontsize=12)
ax.set_title("五种分类模型 ROC 曲线对比", fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
save_fig(fig, __file__, "fig13_2_03_roc_curves")
