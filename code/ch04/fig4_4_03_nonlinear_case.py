"""
图 4.4.3  案例 4.4：California Housing 非线性效应分析
对应节次：4.4 非线性回归（多项式、样条、广义加法模型）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_4_03_nonlinear_case.py
输出路径：public/figures/ch04/fig4_4_03_nonlinear_case.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
apply_style()
# --- 1. 数据准备 ---
np.random.seed(42)
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
feat_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
              "Population", "AveOccup", "Latitude", "Longitude"]
# --- 2. 拟合 GAM 全特征样条（用于面板 (c) 偏依赖图）---
ct_full = ColumnTransformer([
    (f'sp_{i}', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [i])
    for i in range(8)
])
gam_full = Pipeline([
    ('transform', ct_full),
    ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5))
])
gam_full.fit(X_tr, y_tr)
# --- 3. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
ax_a, ax_b, ax_c = axes
# --- 4. 面板 (a): MedInc vs 房价（散点 + 平滑曲线）---
sample_idx = np.random.choice(len(X_tr), size=2000, replace=False)
ax_a.scatter(X_tr[sample_idx, 0], y_tr[sample_idx],
             alpha=0.15, s=8, color=COLORS['blue'], label="训练数据（抽样 2000）")
x_medinc = X_tr[:, 0].reshape(-1, 1)
spline_1d = make_pipeline(
    SplineTransformer(n_knots=8, degree=3, extrapolation='linear'),
    RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5)
)
spline_1d.fit(x_medinc, y_tr)
q2 = np.percentile(X_tr[:, 0], 2)
q98 = np.percentile(X_tr[:, 0], 98)
x_r = np.linspace(q2, q98, 200).reshape(-1, 1)
y_smooth = spline_1d.predict(x_r)
ols_1d = make_pipeline(StandardScaler(), LinearRegression())
ols_1d.fit(x_medinc, y_tr)
y_ols_1d = ols_1d.predict(x_r)
ax_a.plot(x_r, y_ols_1d, color=COLORS['gray'], ls='--', lw=2, label="线性回归", zorder=3)
ax_a.plot(x_r, y_smooth, color=COLORS['red'], lw=2.5, label="三次样条平滑", zorder=4)
flat_idx = np.argmax(np.abs(np.diff(y_smooth.ravel())) < 0.005) + len(y_smooth) * 2 // 3
x_flat = float(x_r[min(flat_idx + 20, len(x_r) - 1)])
y_flat = float(y_smooth[min(flat_idx + 20, len(x_r) - 1)])
ax_a.annotate("高收入边际效应递减",
               xy=(x_flat, y_flat), fontsize=12, color=COLORS['red'],
               xytext=(x_flat - 2.5, y_flat + 0.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.3))
ax_a.set_xlabel("街区收入中位数（万美元）", fontsize=13)
ax_a.set_ylabel("房价中位数（×10 万美元）", fontsize=13)
ax_a.set_title("(a) MedInc（收入中位数）vs 房价\n三次样条揭示高收入区域的边际效应递减", fontsize=13)
ax_a.legend(fontsize=12)
# --- 5. 面板 (b): 各方法测试集 RMSE ---
# OLS
ols_m = make_pipeline(StandardScaler(), LinearRegression())
ols_m.fit(X_tr, y_tr)
rmse_ols = np.sqrt(mean_squared_error(y_te, ols_m.predict(X_te)))
# Polynomial d=2
poly2_m = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())
poly2_m.fit(X_tr, y_tr)
rmse_poly2 = np.sqrt(mean_squared_error(y_te, poly2_m.predict(X_te)))
# Polynomial d=3
poly3_m = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression())
poly3_m.fit(X_tr, y_tr)
rmse_poly3 = np.sqrt(mean_squared_error(y_te, poly3_m.predict(X_te)))
# All-feature spline
ct_all = ColumnTransformer([
    (f'sp_{i}', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [i])
    for i in range(8)
])
spline_all_m = Pipeline([('transform', ct_all), ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5))])
spline_all_m.fit(X_tr, y_tr)
rmse_spline_all = np.sqrt(mean_squared_error(y_te, spline_all_m.predict(X_te)))
# GAM 3 features
linear_features = [2, 3, 4, 6, 7]
ct_gam3 = ColumnTransformer([
    ('spline_0', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [0]),
    ('spline_1', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [1]),
    ('spline_5', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [5]),
    ('linear', 'passthrough', linear_features)
])
gam3_m = Pipeline([('transform', ct_gam3), ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5))])
gam3_m.fit(X_tr, y_tr)
rmse_gam3 = np.sqrt(mean_squared_error(y_te, gam3_m.predict(X_te)))
# GAM all features
rmse_gam_full = np.sqrt(mean_squared_error(y_te, gam_full.predict(X_te)))
method_labels = ["线性回归 (OLS)", "多项式 d=2", "多项式 d=3",
                 "全特征样条", "GAM（3 特征平滑）", "GAM（全特征平滑）"]
rmse_vals = [rmse_ols, rmse_poly2, rmse_poly3, rmse_spline_all, rmse_gam3, rmse_gam_full]
# Sort by RMSE ascending (best = lowest at bottom for horizontal bar)
sort_idx = np.argsort(rmse_vals)[::-1]  # descending so lowest is at bottom
method_labels_sorted = [method_labels[i] for i in sort_idx]
rmse_sorted = [rmse_vals[i] for i in sort_idx]
bar_colors_b = [PALETTE[i % len(PALETTE)] for i in sort_idx]
y_pos_b = range(len(method_labels_sorted))
ax_b.barh(list(y_pos_b), rmse_sorted, color=bar_colors_b, height=0.55, edgecolor='white')
for i, val in enumerate(rmse_sorted):
    ax_b.text(val + 0.002, i, f"{val:.4f}", va='center', ha='left', fontsize=12)
ax_b.axvline(rmse_ols, color='gray', ls='--', lw=1.2, alpha=0.7, label=f"OLS 基准 ({rmse_ols:.3f})")
ax_b.set_yticks(list(y_pos_b))
ax_b.set_yticklabels(method_labels_sorted, fontsize=12)
rmse_min = min(rmse_sorted) - 0.01
rmse_max = max(rmse_sorted) + 0.05
ax_b.set_xlim(rmse_min, rmse_max)
ax_b.set_xlabel("测试集 RMSE（越低越好）", fontsize=13)
ax_b.set_title("(b) 各非线性方法测试集 RMSE\n适度非线性化（GAM）显著降低预测误差", fontsize=13)
ax_b.legend(fontsize=12)
# --- 6. 面板 (c): 偏依赖函数（使用 GAM 全特征）---
X_mean_pd = np.tile(X_tr.mean(axis=0), (100, 1))
pd_cfg = [
    (0, "MedInc（收入中位数）",  COLORS['blue']),
    (1, "HouseAge（房龄）",      COLORS['green']),
    (5, "AveOccup（入住人数）",  COLORS['orange']),
]
for feat_idx, label, color in pd_cfg:
    q5 = np.percentile(X_tr[:, feat_idx], 5)
    q95 = np.percentile(X_tr[:, feat_idx], 95)
    x_range = np.linspace(q5, q95, 100)
    x_norm = (x_range - q5) / (q95 - q5)
    X_eval = X_mean_pd.copy()
    X_eval[:, feat_idx] = x_range
    y_pd = gam_full.predict(X_eval)
    y_pd -= y_pd.mean()
    ax_c.plot(x_norm, y_pd, color=color, lw=2.2, label=label)
    # Rug plot
    feat_data = X_tr[:, feat_idx]
    feat_clip = feat_data[(feat_data >= q5) & (feat_data <= q95)]
    rug_norm = (feat_clip - q5) / (q95 - q5)
    ridx = np.random.choice(len(rug_norm), size=min(200, len(rug_norm)), replace=False)
    ax_c.plot(rug_norm[ridx], np.full(len(ridx), -0.22),
              '|', color=color, alpha=0.3, markersize=4)
ax_c.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax_c.set_xlabel("特征值分位数（0=5th 百分位，1=95th 百分位）", fontsize=13)
ax_c.set_ylabel("偏依赖效应（中心化）", fontsize=13)
ax_c.set_title("(c) 偏依赖函数（GAM 全特征，California Housing）\n各特征对房价的非线性独立效应形状", fontsize=13)
ax_c.legend(fontsize=12, loc='upper left')
# --- 7. 总标题与保存 ---
fig.suptitle("案例 4.4：California Housing 非线性效应分析与 GAM 建模",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_4_03_nonlinear_case")
