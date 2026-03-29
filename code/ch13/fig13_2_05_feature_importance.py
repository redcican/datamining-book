"""
图 13.2.5　Gradient Boosting 特征重要性 Top-15
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_credit import prepare_data

X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols, df = \
    prepare_data()

# ── 1. 训练 Gradient Boosting 并提取特征重要性 ──────────
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
])
pipe.fit(X_train, y_train)

# Get feature names after preprocessing
feature_names = pipe.named_steps["pre"].get_feature_names_out()
# Clean up names: remove "num__" and "cat__" prefixes
feature_names = [n.replace("num__", "").replace("cat__", "")
                 for n in feature_names]

importances = pipe.named_steps["clf"].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("=== Gradient Boosting 特征重要性 Top-15 ===")
for name, imp in feat_imp.head(15).items():
    print(f"  {name:<40s}  {imp:.4f}")

# ── 2. 绘图 ──────────────────────────────────────────────
top15 = feat_imp.head(15).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))

bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(top15))]
bars = ax.barh(range(len(top15)), top15.values,
               color=bar_colors, edgecolor="white",
               height=0.65)

ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=10)
ax.set_xlabel("特征重要性 (Feature Importance)", fontsize=12)
ax.set_title("Gradient Boosting 特征重要性 Top-15", fontweight="bold")

for bar, val in zip(bars, top15.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color=COLORS["gray"])

ax.set_xlim(0, top15.max() * 1.18)

plt.tight_layout()
save_fig(fig, __file__, "fig13_2_05_feature_importance")
