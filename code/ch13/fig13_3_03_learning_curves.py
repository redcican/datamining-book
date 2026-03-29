"""
图 13.3.3　学习曲线：训练样本量 vs 模型性能
展示三种模型在不同训练集大小下的泛化表现
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, StratifiedKFold
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_medical import load_cancer

X, y, _ = load_cancer()

models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=42),
                            COLORS["blue"]),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42),
                            COLORS["green"]),
    "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=100, random_state=42),
                            COLORS["red"]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes = np.linspace(0.1, 1.0, 10)

fig, ax = plt.subplots(figsize=(10, 7))

print("=== 学习曲线 ===")
for name, (model, color) in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=cv, train_sizes=train_sizes,
        scoring="f1", n_jobs=-1, random_state=42)

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    print(f"  {name:<22s}  F1@10%={val_mean[0]:.3f}  "
          f"F1@100%={val_mean[-1]:.3f}")

    ax.plot(sizes, val_mean, "o-", color=color, linewidth=2,
            markersize=5, label=f"{name} (验证)")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=color)
    ax.plot(sizes, train_mean, "--", color=color, linewidth=1.2,
            alpha=0.5, label=f"{name} (训练)")

ax.set_xlabel("训练样本数", fontsize=12)
ax.set_ylabel("F1 分数", fontsize=12)
ax.set_title("学习曲线：训练集大小 vs F1 分数", fontweight="bold")
ax.legend(loc="lower right", fontsize=9, ncol=2)
ax.set_ylim(0.7, 1.02)

plt.tight_layout()
save_fig(fig, __file__, "fig13_3_03_learning_curves")
