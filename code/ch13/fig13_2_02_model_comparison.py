"""
图 13.2.2　五种分类模型交叉验证性能对比
Logistic Regression / Decision Tree / Random Forest / Gradient Boosting / SVM
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
from sklearn.model_selection import cross_validate, StratifiedKFold
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_credit import prepare_data

X_train, X_test, y_train, y_test, preprocessor, *_ = prepare_data()

# ── 1. 定义模型 ──────────────────────────────────────────
models = {
    "Logistic\nRegression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision\nTree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random\nForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient\nBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM\n(RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
}

# ── 2. 交叉验证 ──────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ["accuracy", "precision", "recall", "f1"]
metric_zh = {"accuracy": "准确率", "precision": "精确率",
             "recall": "召回率", "f1": "F1"}

results = {}
print("=== 5 折交叉验证结果 ===")
print(f"{'模型':<22s} {'Accuracy':>10s} {'Precision':>10s} "
      f"{'Recall':>10s} {'F1':>10s}")
print("-" * 60)

# Combine train data for full CV
import pandas as pd
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

for name, model in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    scores = cross_validate(pipe, X_full, y_full, cv=cv,
                            scoring=metrics, return_train_score=False)
    results[name] = {m: scores[f"test_{m}"] for m in metrics}
    name_flat = name.replace("\n", " ")
    print(f"  {name_flat:<20s} "
          f"{scores['test_accuracy'].mean():>10.3f} "
          f"{scores['test_precision'].mean():>10.3f} "
          f"{scores['test_recall'].mean():>10.3f} "
          f"{scores['test_f1'].mean():>10.3f}")

# ── 3. 绘图 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(results.keys())
n_models = len(model_names)
n_metrics = len(metrics)
x = np.arange(n_models)
width = 0.18
colors = [COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["green"]]

for i, metric in enumerate(metrics):
    means = [results[m][metric].mean() for m in model_names]
    stds = [results[m][metric].std() for m in model_names]
    offset = (i - n_metrics / 2 + 0.5) * width
    bars = ax.bar(x + offset, means, width, yerr=stds,
                  label=f"{metric_zh[metric]} ({metric})",
                  color=colors[i], alpha=0.85,
                  edgecolor="white", linewidth=0.5,
                  capsize=3, error_kw={"linewidth": 1})
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{mean:.2f}", ha="center", va="bottom",
                fontsize=7, rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("分数")
ax.set_title("五种分类模型 5 折交叉验证性能对比", fontweight="bold")
ax.set_ylim(0, 1.12)
ax.legend(loc="upper left", ncol=4, fontsize=9)

plt.tight_layout()
save_fig(fig, __file__, "fig13_2_02_model_comparison")
