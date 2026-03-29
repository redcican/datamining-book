"""
图 13.6.4　特征重要性与故障模式分析
(a) Random Forest 特征重要性  (b) 故障模式分布
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
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_maintenance import prepare_data

(X_train, X_test, y_train, y_test, preprocessor,
 num_features, cat_features, df) = prepare_data()

# ── 训练 RF 并提取特征重要性 ─────────────────────────────
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                            random_state=42)
pipe = Pipeline([("pre", preprocessor), ("clf", rf)])
pipe.fit(X_train, y_train)

# 获取变换后的特征名
cat_encoder = preprocessor.named_transformers_["cat"]
cat_names = list(cat_encoder.get_feature_names_out(cat_features))
all_features = num_features + cat_names

importances = pd.Series(
    pipe.named_steps["clf"].feature_importances_,
    index=all_features).sort_values(ascending=False)

print("=== Random Forest 特征重要性 ===")
for name, val in importances.items():
    print(f"  {name:<30s}  {val:.4f}")

# ── 故障模式分析 ──────────────────────────────────────────
failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
mode_names = {
    "TWF": "工具磨损\n(TWF)",
    "HDF": "散热故障\n(HDF)",
    "PWF": "功率故障\n(PWF)",
    "OSF": "过载故障\n(OSF)",
    "RNF": "随机故障\n(RNF)",
}
mode_counts = {mode_names[m]: df[m].sum() for m in failure_modes}

print("\n=== 故障模式分布 ===")
for name, cnt in mode_counts.items():
    name_flat = name.replace("\n", " ")
    print(f"  {name_flat:<18s}  {cnt:>3d} 次")

# 各故障模式的特征均值差异
print("\n=== 各故障模式的关键特征均值 ===")
print(f"{'模式':<8s} {'Torque':>8s} {'Speed':>8s} {'Wear':>8s}")
for mode in failure_modes:
    subset = df[df[mode] == 1]
    print(f"  {mode:<6s} {subset['Torque [Nm]'].mean():>8.1f} "
          f"{subset['Rotational speed [rpm]'].mean():>8.0f} "
          f"{subset['Tool wear [min]'].mean():>8.0f}")
normal_sub = df[df["Machine failure"] == 0]
print(f"  {'正常':<6s} {normal_sub['Torque [Nm]'].mean():>8.1f} "
      f"{normal_sub['Rotational speed [rpm]'].mean():>8.0f} "
      f"{normal_sub['Tool wear [min]'].mean():>8.0f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                gridspec_kw={"width_ratios": [1.2, 1]})

# (a) 特征重要性
imp_sorted = importances.sort_values(ascending=True)
colors_a = [PALETTE[i % len(PALETTE)] for i in range(len(imp_sorted))]
bars = ax1.barh(range(len(imp_sorted)), imp_sorted.values,
                color=colors_a, edgecolor="white", height=0.65)
ax1.set_yticks(range(len(imp_sorted)))
ax1.set_yticklabels(imp_sorted.index, fontsize=10)
ax1.set_xlabel("重要性")
ax1.set_title("(a) Random Forest 特征重要性", fontweight="bold")
for bar, val in zip(bars, imp_sorted.values):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color=COLORS["gray"])

# (b) 故障模式分布
mode_labels = list(mode_counts.keys())
mode_vals = list(mode_counts.values())
bar_colors = [COLORS["red"], COLORS["orange"], COLORS["purple"],
              COLORS["blue"], COLORS["gray"]]
bars2 = ax2.bar(mode_labels, mode_vals, color=bar_colors,
                edgecolor="white", width=0.6)
for bar, cnt in zip(bars2, mode_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 2,
             f"{cnt}", ha="center", fontsize=11, fontweight="bold")
ax2.set_ylabel("故障次数")
ax2.set_title("(b) 五种故障模式分布", fontweight="bold")

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_6_04_feature_analysis")
