"""
图 13.7.3　四种回归模型性能对比
Linear Regression / Ridge / Random Forest / Gradient Boosting
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score)
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_pm25 import prepare_data

X_train, X_test, y_train, y_test, feature_names, df = prepare_data()

print(f"训练集: {len(X_train):,}, 测试集: {len(X_test):,}")
print(f"特征数: {len(feature_names)}")

# ── 训练四种模型 ──────────────────────────────────────────
models = {
    "Linear\nRegression": LinearRegression(),
    "Ridge\n(α=1.0)": Ridge(alpha=1.0),
    "Random\nForest": RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient\nBoosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42),
}

results = {}
print(f"\n{'模型':<22s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s}")
print("-" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
    name_flat = name.replace("\n", " ")
    print(f"  {name_flat:<20s} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.25
metrics = ["RMSE", "MAE", "R²"]
colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    offset = (i - len(metrics) / 2 + 0.5) * width

    if metric == "R²":
        # R² on secondary axis
        continue

    bars = ax.bar(x + offset, vals, width, label=metric,
                  color=colors[i], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}", ha="center", fontsize=9)

# R² on secondary axis
ax2 = ax.twinx()
r2_vals = [results[m]["R²"] for m in model_names]
offset = (2 - len(metrics) / 2 + 0.5) * width
bars_r2 = ax2.bar(x + offset, r2_vals, width, label="R²",
                   color=COLORS["green"], alpha=0.85, edgecolor="white")
for bar, val in zip(bars_r2, r2_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", fontsize=9)
ax2.set_ylabel("R²", fontsize=12)
ax2.set_ylim(0, 1.15)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("误差 (μg/m³)")
ax.set_title("四种回归模型性能对比（PM2.5 预测，测试集 2014 年）",
             fontweight="bold")

# 合并图例
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=COLORS["blue"], label="RMSE"),
    Patch(facecolor=COLORS["orange"], label="MAE"),
    Patch(facecolor=COLORS["green"], label="R²"),
], loc="upper left", fontsize=10)

plt.tight_layout()
save_fig(fig, __file__, "fig13_7_03_model_comparison")
