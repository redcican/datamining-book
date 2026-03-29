"""
图 13.3.5　特征重要性对比
(a) Random Forest Gini 重要性  (b) 置换重要性（Permutation Importance）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_medical import prepare_data

X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()

# ── 1. 训练 Random Forest ────────────────────────────────
pipe = Pipeline([("scaler", scaler), ("clf", RandomForestClassifier(
    n_estimators=200, random_state=42))])
pipe.fit(X_train, y_train)

# (a) Gini importance
gini_imp = pd.Series(
    pipe.named_steps["clf"].feature_importances_,
    index=feature_names).sort_values(ascending=False)

# (b) Permutation importance
perm_result = permutation_importance(
    pipe, X_test, y_test, n_repeats=30, random_state=42, scoring="f1")
perm_imp = pd.Series(
    perm_result.importances_mean,
    index=feature_names).sort_values(ascending=False)

print("=== Gini 重要性 Top-10 ===")
for name, val in gini_imp.head(10).items():
    print(f"  {name:<28s}  {val:.4f}")
print("\n=== 置换重要性 Top-10 ===")
for name, val in perm_imp.head(10).items():
    print(f"  {name:<28s}  {val:.4f}")

# 检查两种方法的一致性
top10_gini = set(gini_imp.head(10).index)
top10_perm = set(perm_imp.head(10).index)
overlap = top10_gini & top10_perm
print(f"\nTop-10 重叠特征: {len(overlap)}/10")

# ── 2. 绘图 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
top_n = 10

# (a) Gini importance
top_gini = gini_imp.head(top_n).sort_values(ascending=True)
colors_a = [PALETTE[i % len(PALETTE)] for i in range(len(top_gini))]
bars1 = ax1.barh(range(len(top_gini)), top_gini.values,
                 color=colors_a, edgecolor="white", height=0.65)
ax1.set_yticks(range(len(top_gini)))
ax1.set_yticklabels(top_gini.index, fontsize=10)
ax1.set_xlabel("Gini 重要性")
ax1.set_title("(a) 基尼不纯度重要性 (Random Forest)", fontweight="bold")
for bar, val in zip(bars1, top_gini.values):
    ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color=COLORS["gray"])

# (b) Permutation importance
top_perm = perm_imp.head(top_n).sort_values(ascending=True)
perm_stds = pd.Series(perm_result.importances_std,
                       index=feature_names)[top_perm.index]
colors_b = [COLORS["orange"] if n in top10_gini else COLORS["gray"]
            for n in top_perm.index]
bars2 = ax2.barh(range(len(top_perm)), top_perm.values,
                 xerr=perm_stds.values, color=colors_b,
                 edgecolor="white", height=0.65,
                 capsize=3, error_kw={"linewidth": 1})
ax2.set_yticks(range(len(top_perm)))
ax2.set_yticklabels(top_perm.index, fontsize=10)
ax2.set_xlabel("置换重要性 (F1 下降)")
ax2.set_title("(b) 置换重要性 (Permutation Importance)",
              fontweight="bold")
for bar, val in zip(bars2, top_perm.values):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color=COLORS["gray"])

# 图例标注颜色含义
from matplotlib.patches import Patch
ax2.legend([Patch(facecolor=COLORS["orange"]), Patch(facecolor=COLORS["gray"])],
           ["Gini Top-10 重叠", "仅置换 Top-10"],
           loc="lower right", fontsize=9)

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_3_05_feature_importance")
