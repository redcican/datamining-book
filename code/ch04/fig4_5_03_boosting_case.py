"""
图 4.5.3  梯度提升树：逐步拟合机制、California Housing 方法对比与学习曲线
对应节次：4.5 树模型回归
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_5_03_boosting_case.py
输出路径：public/figures/ch04/fig4_5_03_boosting_case.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
apply_style()
# --- 1. 生成 1D 数据 ---
np.random.seed(42)
x = np.linspace(0, 10, 60)
y = np.sin(x) + 0.5 * np.cos(2 * x) + np.random.normal(0, 0.4, len(x))
X = x.reshape(-1, 1)
x_plot = np.linspace(0, 10, 500)
X_plot = x_plot.reshape(-1, 1)
y_true = np.sin(x_plot) + 0.5 * np.cos(2 * x_plot)
# --- 2. 加载 California Housing ---
housing = fetch_california_housing()
X_cal, y_cal = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)
# --- 3. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax_a, ax_b, ax_c = axes
# --- 4. 面板 (a): 梯度提升逐步拟合 ---
m_vals = [1, 5, 20, 100]
m_colors = [COLORS['orange'], COLORS['purple'], COLORS['red'], COLORS['blue']]
ax_a.scatter(x, y, color=COLORS['gray'], s=25, alpha=0.5, zorder=2, label="数据点")
ax_a.plot(x_plot, y_true, color=COLORS['green'], lw=2, ls='--', label="真实函数", zorder=5)
for m, color in zip(m_vals, m_colors):
    gbt = GradientBoostingRegressor(n_estimators=m, max_depth=2,
                                    learning_rate=0.1, random_state=42)
    gbt.fit(X, y)
    ax_a.plot(x_plot, gbt.predict(X_plot), color=color, lw=2,
              label=f"m={m} 步", zorder=4)
ax_a.set_xlabel("$x$", fontsize=13)
ax_a.set_ylabel("$y$", fontsize=13)
ax_a.set_title("(a) 梯度提升逐步拟合\n"
               r"每一步拟合前一步的伪残差 $r_{im} = y_i - \hat{f}_{m-1}(x_i)$",
               fontsize=12)
ax_a.legend(fontsize=10)
# --- 5. 面板 (b): California Housing 测试 RMSE 对比 ---
methods = []
rmse_vals = []
method_colors = []
# Linear Regression
lr = LinearRegression()
lr.fit(X_tr, y_tr)
rmse_lr = np.sqrt(mean_squared_error(y_te, lr.predict(X_te)))
methods.append("线性回归")
rmse_vals.append(rmse_lr)
method_colors.append(COLORS['blue'])
# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
rmse_ridge = np.sqrt(mean_squared_error(y_te, ridge.predict(X_te)))
methods.append("岭回归")
rmse_vals.append(rmse_ridge)
method_colors.append(COLORS['blue'])
# Decision Tree
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_tr, y_tr)
rmse_dt = np.sqrt(mean_squared_error(y_te, dt.predict(X_te)))
methods.append("决策树 (depth=5)")
rmse_vals.append(rmse_dt)
method_colors.append(COLORS['orange'])
# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
rmse_rf = np.sqrt(mean_squared_error(y_te, rf.predict(X_te)))
methods.append("随机森林 (B=100)")
rmse_vals.append(rmse_rf)
method_colors.append(COLORS['red'])
# Gradient Boosting
gbt_b = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbt_b.fit(X_tr, y_tr)
rmse_gbt = np.sqrt(mean_squared_error(y_te, gbt_b.predict(X_te)))
methods.append("梯度提升树 (M=100)")
rmse_vals.append(rmse_gbt)
method_colors.append(COLORS['red'])
# Optionally XGBoost
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                       random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    rmse_xgb = np.sqrt(mean_squared_error(y_te, xgb.predict(X_te)))
    methods.append("XGBoost (M=100)")
    rmse_vals.append(rmse_xgb)
    method_colors.append(COLORS['purple'])
except ImportError:
    pass
# Sort ascending by RMSE (best at top = ascending bar order means best at bottom;
# to have best at top in horizontal barh, sort descending so smallest is last/top)
sort_idx = np.argsort(rmse_vals)[::-1]  # descending: worst at top, best at bottom
methods_sorted = [methods[i] for i in sort_idx]
rmse_sorted = [rmse_vals[i] for i in sort_idx]
colors_sorted = [method_colors[i] for i in sort_idx]
y_pos = range(len(methods_sorted))
ax_b.barh(list(y_pos), rmse_sorted, color=colors_sorted, alpha=0.7,
          height=0.6, edgecolor='white')
for i, val in enumerate(rmse_sorted):
    ax_b.text(val + 0.002, i, f"{val:.4f}", va='center', ha='left', fontsize=10)
rmse_min_b = min(rmse_sorted) - 0.01
rmse_max_b = max(rmse_sorted) + 0.05
ax_b.set_xlim(rmse_min_b, rmse_max_b)
ax_b.set_yticks(list(y_pos))
ax_b.set_yticklabels(methods_sorted, fontsize=11)
ax_b.set_xlabel("测试集 RMSE（越低越好）", fontsize=13)
ax_b.set_title("(b) California Housing 测试 RMSE 对比\n树集成方法通常优于线性方法", fontsize=12)
# --- 6. 面板 (c): 梯度提升学习曲线 ---
n_estimators_range = np.linspace(1, 300, 30).astype(int)
train_rmse_list = []
test_rmse_list = []
for n_est in n_estimators_range:
    gbt_c = GradientBoostingRegressor(n_estimators=n_est, max_depth=3,
                                      learning_rate=0.1, random_state=42)
    gbt_c.fit(X_tr, y_tr)
    train_rmse_list.append(np.sqrt(mean_squared_error(y_tr, gbt_c.predict(X_tr))))
    test_rmse_list.append(np.sqrt(mean_squared_error(y_te, gbt_c.predict(X_te))))
train_rmse_arr = np.array(train_rmse_list)
test_rmse_arr = np.array(test_rmse_list)
opt_n = n_estimators_range[np.argmin(test_rmse_arr)]
ax_c.plot(n_estimators_range, train_rmse_arr, color=COLORS['blue'], lw=2, label="训练 RMSE")
ax_c.plot(n_estimators_range, test_rmse_arr, color=COLORS['red'], lw=2, label="测试 RMSE")
ax_c.axvline(opt_n, color=COLORS['orange'], lw=1.5, ls='--',
             label=f"最优 M={opt_n}")
ax_c.set_xlabel(r"梯度提升树数量 $M$", fontsize=13)
ax_c.set_ylabel("RMSE", fontsize=13)
ax_c.set_title(r"(c) 梯度提升学习曲线" + "\n" + r"学习率 $\eta=0.1$，max_depth=3",
               fontsize=12)
ax_c.legend(fontsize=11)
# --- 7. 总标题与保存 ---
fig.suptitle("梯度提升树：逐步拟合机制、California Housing 方法对比与学习曲线",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_5_03_boosting_case")
