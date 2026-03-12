"""
图 4.3.3  案例 4.3：California Housing 正则化回归综合分析
对应节次：4.3 正则化回归（Ridge, Lasso, 弹性网络）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_3_03_california_case.py
输出路径：public/figures/ch04/fig4_3_03_california_case.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, RidgeCV, LassoCV,
                                   ElasticNetCV)
from sklearn.metrics import r2_score, mean_squared_error
apply_style()
# --- 1. 数据准备 ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
feat_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
              "Population", "AveOccup", "Latitude", "Longitude"]
feat_names_cn = ["收入中位数", "房龄", "平均房间数", "平均卧室数",
                 "人口数量", "平均入住人数", "纬度", "经度"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
# --- 2. 模型训练 ---
ols = LinearRegression()
ols.fit(X_tr_s, y_tr)
alphas_cv = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas_cv, cv=5)
ridge_cv.fit(X_tr_s, y_tr)
lasso_cv = LassoCV(n_alphas=50, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_tr_s, y_tr)
en_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], n_alphas=50,
                     cv=5, max_iter=10000, random_state=42)
en_cv.fit(X_tr_s, y_tr)
# --- 3. 测试集指标 ---
def r2_rmse(model, Xte, yte):
    pred = model.predict(Xte)
    return r2_score(yte, pred), np.sqrt(mean_squared_error(yte, pred))
ols_r2, ols_rmse = r2_rmse(ols, X_te_s, y_te)
ridge_r2, ridge_rmse = r2_rmse(ridge_cv, X_te_s, y_te)
lasso_r2, lasso_rmse = r2_rmse(lasso_cv, X_te_s, y_te)
en_r2, en_rmse = r2_rmse(en_cv, X_te_s, y_te)
r2_vals = np.array([ols_r2, ridge_r2, lasso_r2, en_r2])
rmse_vals = np.array([ols_rmse, ridge_rmse, lasso_rmse, en_rmse])
method_labels = ["OLS", "岭回归", "Lasso", "弹性网络"]
# --- 4. 图形布局 ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
# --- 5. 面板(a)：Lasso CV 曲线 ---
ax = axes[0]
K = lasso_cv.cv
mse_path = lasso_cv.mse_path_
mean_mse = mse_path.mean(axis=1)
std_mse = mse_path.std(axis=1)
se_mse = std_mse / np.sqrt(K)
log_alphas = np.log10(lasso_cv.alphas_)
ax.errorbar(log_alphas, mean_mse, yerr=se_mse, fmt='-',
            color=COLORS['blue'], capsize=2, lw=1.5, elinewidth=0.8,
            label="CV均方误差（±1se 误差棒）")
min_idx = np.argmin(mean_mse)
alpha_min = lasso_cv.alphas_[min_idx]
ax.axvline(np.log10(alpha_min), color=COLORS['red'], lw=2, ls='--',
           label=f"$\\lambda_{{\\min}}$ (最小CV误差)")
threshold = mean_mse[min_idx] + se_mse[min_idx]
larger_mask = lasso_cv.alphas_ >= alpha_min
valid_1se = lasso_cv.alphas_[larger_mask][mean_mse[larger_mask] <= threshold]
if len(valid_1se) > 0:
    alpha_1se = valid_1se[-1]
    ax.axvline(np.log10(alpha_1se), color=COLORS['orange'], lw=2, ls='--',
               label=f"$\\lambda_{{1\\mathrm{{se}}}}$ (1倍SE规则)")
ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=13)
ax.set_ylabel("交叉验证 MSE", fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.set_title("(a) Lasso 交叉验证曲线（5折 CV）\n$\\lambda_{\\min}$ 最小化CV误差，$\\lambda_{1\\mathrm{se}}$ 选更稀疏模型",
             fontsize=13, pad=8)
# --- 6. 面板(b)：系数对比（水平柱状图）---
ax = axes[1]
ols_coef = ols.coef_
ridge_coef = ridge_cv.coef_
lasso_coef = lasso_cv.coef_
en_coef = en_cv.coef_
sort_idx = np.argsort(np.abs(ols_coef))[::-1]
coefs_sorted = [ols_coef[sort_idx], ridge_coef[sort_idx],
                lasso_coef[sort_idx], en_coef[sort_idx]]
labels_sorted = [feat_names_cn[i] for i in sort_idx]
n_feat = len(feat_names)
bar_width = 0.2
bar_colors_list = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red']]
bar_labels = ["OLS", "Ridge", "Lasso", "弹性网络"]
y_pos = np.arange(n_feat)
for k, (coef_arr, bc, bl) in enumerate(zip(coefs_sorted, bar_colors_list, bar_labels)):
    offset = (k - 1.5) * bar_width
    ax.barh(y_pos + offset, coef_arr, height=bar_width,
            color=bc, alpha=0.85, label=bl)
ax.axvline(0, color='k', lw=1, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_sorted, fontsize=12)
ax.set_xlabel("标准化系数值", fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.set_title("(b) 标准化回归系数对比（输入已标准化）\nLasso 将部分系数精确置零，Ridge 均匀收缩",
             fontsize=13, pad=8)
# --- 7. 面板(c)：测试集性能对比（双 Y 轴）---
ax1 = axes[2]
ax2 = ax1.twinx()
x = np.arange(len(method_labels))
bars = ax1.bar(x, r2_vals, color=COLORS['blue'], alpha=0.7,
               label=r"测试集 $R^2$", width=0.4)
ax2.plot(x, rmse_vals, 'o-', color=COLORS['red'], label="RMSE", lw=2, ms=8)
for i, (r2v, rmsev) in enumerate(zip(r2_vals, rmse_vals)):
    ax1.text(i, r2v + 0.005, f"{r2v:.4f}", ha='center', fontsize=11,
             color=COLORS['blue'], fontweight='bold')
    ax2.text(i, rmsev + 0.003, f"{rmsev:.4f}", ha='center', fontsize=11,
             color=COLORS['red'])
ax1.set_xticks(x)
ax1.set_xticklabels(method_labels, fontsize=12)
ax1.set_ylabel(r"$R^2$（越高越好）", fontsize=13, color=COLORS['blue'])
ax2.set_ylabel("RMSE（越低越好）", fontsize=13, color=COLORS['red'])
ax1.tick_params(axis='y', labelcolor=COLORS['blue'])
ax2.tick_params(axis='y', labelcolor=COLORS['red'])
r2_min = r2_vals.min()
ax1.set_ylim(r2_min * 0.998, r2_vals.max() * 1.015)
rmse_range = rmse_vals.max() - rmse_vals.min()
ax2.set_ylim(rmse_vals.min() - rmse_range * 2,
             rmse_vals.max() + rmse_range * 2)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='lower right')
n_test = len(y_te)
ax1.set_title(f"(c) 测试集性能对比（$n_{{\\text{{test}}}}={n_test}$）\n正则化改善泛化，最优 λ 选择至关重要",
              fontsize=13, pad=8)
# --- 8. 保存 ---
fig.suptitle("案例 4.3：California Housing 正则化回归综合分析",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_3_03_california_case")
