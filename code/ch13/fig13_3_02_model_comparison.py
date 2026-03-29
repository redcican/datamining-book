"""
图 13.3.2　五种分类模型交叉验证性能对比（乳腺癌诊断）
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_medical import load_cancer

X, y, _ = load_cancer()

models = {
    "Logistic\nRegression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision\nTree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random\nForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient\nBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM\n(RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metric_zh = {"accuracy": "准确率", "precision": "精确率",
             "recall": "召回率", "f1": "F1", "roc_auc": "AUC"}

results = {}
print("=== 5 折交叉验证结果（乳腺癌诊断）===")
print(f"{'模型':<22s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} "
      f"{'F1':>8s} {'AUC':>8s}")
print("-" * 60)

for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=metrics, return_train_score=False)
    results[name] = {m: scores[f"test_{m}"] for m in metrics}
    name_flat = name.replace("\n", " ")
    print(f"  {name_flat:<20s} "
          + " ".join(f"{scores[f'test_{m}'].mean():>8.3f}" for m in metrics))

# ── 绘图 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
plot_metrics = ["accuracy", "precision", "recall", "f1"]
model_names = list(results.keys())
n_models = len(model_names)
n_metrics = len(plot_metrics)
x = np.arange(n_models)
width = 0.18
colors = [COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["green"]]

for i, metric in enumerate(plot_metrics):
    means = [results[m][metric].mean() for m in model_names]
    stds = [results[m][metric].std() for m in model_names]
    offset = (i - n_metrics / 2 + 0.5) * width
    bars = ax.bar(x + offset, means, width, yerr=stds,
                  label=f"{metric_zh[metric]} ({metric})",
                  color=colors[i], alpha=0.85,
                  edgecolor="white", capsize=3,
                  error_kw={"linewidth": 1})
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{mean:.2f}", ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("分数")
ax.set_title("五种分类模型 5 折交叉验证性能对比（乳腺癌诊断）",
             fontweight="bold")
ax.set_ylim(0.8, 1.05)
ax.legend(loc="lower left", ncol=4, fontsize=9)

plt.tight_layout()
save_fig(fig, __file__, "fig13_3_02_model_comparison")
