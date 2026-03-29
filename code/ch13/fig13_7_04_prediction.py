"""
图 13.7.4　预测效果可视化
(a) 实际 vs 预测（2 周测试窗口）  (b) 残差分布
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_pm25 import prepare_data

X_train, X_test, y_train, y_test, feature_names, df = prepare_data()

# ── 训练最佳模型（Gradient Boosting）──────────────────────
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
preds_all = gb.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds_all))
mae = mean_absolute_error(y_test, preds_all)
r2 = r2_score(y_test, preds_all)
residuals = y_test.values - preds_all

print("=== Gradient Boosting 预测结果 (2014 年测试集) ===")
print(f"  RMSE: {rmse:.2f} μg/m³")
print(f"  MAE:  {mae:.2f} μg/m³")
print(f"  R²:   {r2:.4f}")
print(f"  残差均值: {residuals.mean():.2f}")
print(f"  残差标准差: {residuals.std():.2f}")

# 特征重要性
importances = dict(zip(feature_names, gb.feature_importances_))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("\n  特征重要性 Top-5:")
for name, val in sorted_imp[:5]:
    print(f"    {name:<22s}  {val:.4f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# (a) 2 周预测窗口 (取 2014-01 前两周)
window_start = 0
window_len = 24 * 14  # 14 天
window_end = window_start + window_len

actual_window = y_test.values[window_start:window_end]
pred_window = preds_all[window_start:window_end]
time_index = y_test.index[window_start:window_end]

ax1.plot(time_index, actual_window, "-", color=COLORS["blue"],
         linewidth=1.2, alpha=0.8, label="实际值")
ax1.plot(time_index, pred_window, "-", color=COLORS["red"],
         linewidth=1.2, alpha=0.8, label="预测值")
ax1.fill_between(time_index,
                 actual_window, pred_window,
                 alpha=0.15, color=COLORS["orange"])

rmse_window = np.sqrt(np.mean((actual_window - pred_window) ** 2))
ax1.set_ylabel("PM2.5 (μg/m³)")
ax1.set_title(f"(a) 预测 vs 实际（2014年1月前两周, "
              f"RMSE={rmse_window:.1f}）", fontweight="bold")
ax1.legend(fontsize=10, loc="upper right")
ax1.tick_params(axis='x', rotation=20)

# (b) 残差分布
ax2.hist(residuals, bins=80, color=COLORS["blue"], edgecolor="white",
         alpha=0.85, density=True)
ax2.axvline(0, color="black", linestyle="-", linewidth=1)
ax2.axvline(residuals.mean(), color=COLORS["red"], linestyle="--",
            linewidth=1.5,
            label=f"均值 = {residuals.mean():.1f}")

# 标注标准差范围
for i, mult in enumerate([1, 2]):
    ax2.axvline(mult * residuals.std(), color=COLORS["orange"],
                linestyle=":", linewidth=1, alpha=0.7)
    ax2.axvline(-mult * residuals.std(), color=COLORS["orange"],
                linestyle=":", linewidth=1, alpha=0.7)

ax2.set_xlabel("残差 (实际 - 预测, μg/m³)")
ax2.set_ylabel("密度")
ax2.set_title(f"(b) 残差分布 (σ = {residuals.std():.1f})",
              fontweight="bold")
ax2.legend(fontsize=10)

plt.tight_layout(h_pad=3)
save_fig(fig, __file__, "fig13_7_04_prediction")
