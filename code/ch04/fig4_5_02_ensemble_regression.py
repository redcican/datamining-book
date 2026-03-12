"""
图 4.5.2  随机森林：Bagging 方差压缩、OOB 误差与特征重要性
对应节次：4.5 树模型回归
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_5_02_ensemble_regression.py
输出路径：public/figures/ch04/fig4_5_02_ensemble_regression.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
apply_style()
# --- 1. 生成 1D 数据 ---
np.random.seed(42)
x = np.linspace(0, 10, 80)
y = np.sin(x) + 0.5 * np.cos(2 * x) + np.random.normal(0, 0.3, len(x))
X = x.reshape(-1, 1)
x_plot = np.linspace(0, 10, 500)
X_plot = x_plot.reshape(-1, 1)
y_true = np.sin(x_plot) + 0.5 * np.cos(2 * x_plot)
# --- 2. 加载 California Housing ---
housing = fetch_california_housing()
X_cal, y_cal = housing.data, housing.target
feat_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
              'Population', 'AveOccup', 'Latitude', 'Longitude']
# --- 3. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax_a, ax_b, ax_c = axes
# --- 4. 面板 (a): Bagging 降低方差 ---
ax_a.scatter(x, y, color=COLORS['gray'], s=25, alpha=0.5, zorder=2, label="数据点")
ax_a.plot(x_plot, y_true, color=COLORS['green'], lw=2, ls='--', label="真实函数", zorder=5)
# Show B=5 individual bootstrap trees as thin gray lines
rng = np.random.RandomState(0)
for b in range(5):
    idx_boot = rng.choice(len(x), size=len(x), replace=True)
    clf_b = DecisionTreeRegressor(max_depth=4, random_state=b)
    clf_b.fit(X[idx_boot], y[idx_boot])
    ax_a.plot(x_plot, clf_b.predict(X_plot), color=COLORS['gray'], lw=0.8, alpha=0.3)
# Ensemble averages for B=1, 5, 20, 100
B_vals = [1, 5, 20, 100]
B_colors = [COLORS['orange'], COLORS['purple'], COLORS['red'], COLORS['blue']]
for B, color in zip(B_vals, B_colors):
    rf_tmp = RandomForestRegressor(n_estimators=B, max_depth=4, random_state=42)
    rf_tmp.fit(X, y)
    ax_a.plot(x_plot, rf_tmp.predict(X_plot), color=color, lw=2,
              label=f"集成均值 B={B}", zorder=4)
ax_a.set_xlabel("$x$", fontsize=13)
ax_a.set_ylabel("$y$", fontsize=13)
ax_a.set_title(r"(a) Bagging 降低方差" + "\n" +
               r"$\hat{f}_{\rm bag}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)$", fontsize=12)
ax_a.legend(fontsize=9, ncol=2)
# --- 5. 面板 (b): OOB 误差收敛曲线 ---
n_estimators_list = [1, 5, 10, 20, 50, 100, 200]
oob_rmse_list = []
for n_est in n_estimators_list:
    rf_oob = RandomForestRegressor(n_estimators=n_est, oob_score=True, random_state=42, n_jobs=-1)
    rf_oob.fit(X_cal, y_cal)
    oob_rmse = np.sqrt(1 - rf_oob.oob_score_) * np.std(y_cal)
    oob_rmse_list.append(oob_rmse)
oob_rmse_arr = np.array(oob_rmse_list)
min_oob = oob_rmse_arr.min()
ax_b.plot(n_estimators_list, oob_rmse_arr, color=COLORS['blue'], lw=2,
          marker='o', ms=6, label="OOB RMSE")
ax_b.axhline(min_oob, color=COLORS['red'], lw=1.5, ls='--',
             label=f"最小 OOB RMSE = {min_oob:.4f}")
ax_b.set_xlabel(r"树的数量 $B$", fontsize=13)
ax_b.set_ylabel("OOB RMSE", fontsize=13)
ax_b.set_title("(b) OOB 误差收敛曲线\n树数量增加后误差趋于稳定", fontsize=12)
ax_b.legend(fontsize=11)
# --- 6. 面板 (c): 特征重要性 ---
rf_full = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_full.fit(X_cal, y_cal)
importances = rf_full.feature_importances_
sort_idx = np.argsort(importances)
sorted_names = [feat_names[i] for i in sort_idx]
sorted_imp = importances[sort_idx]
# Top 3 (largest) in red, rest in blue
bar_colors = []
top3_indices = set(np.argsort(importances)[-3:])
for i in sort_idx:
    if i in top3_indices:
        bar_colors.append(COLORS['red'])
    else:
        bar_colors.append(COLORS['blue'])
bars = ax_c.barh(range(len(sorted_names)), sorted_imp, color=bar_colors,
                 alpha=0.7, height=0.6, edgecolor='white')
for i, val in enumerate(sorted_imp):
    ax_c.text(val + 0.002, i, f"{val:.3f}", va='center', ha='left', fontsize=10)
ax_c.set_yticks(range(len(sorted_names)))
ax_c.set_yticklabels(sorted_names, fontsize=11)
ax_c.set_xlabel("特征重要性（基于不纯度减少）", fontsize=13)
ax_c.set_title("(c) 随机森林特征重要性\n（基于不纯度减少，California Housing）", fontsize=12)
# --- 7. 总标题与保存 ---
fig.suptitle("随机森林：Bagging 方差压缩、OOB 误差与特征重要性",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_5_02_ensemble_regression")
