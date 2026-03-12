"""
图 4.4.2  B 样条基函数、GAM 偏反应函数与回填算法
对应节次：4.4 非线性回归（多项式、样条、广义加法模型）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_4_02_gam_basis.py
输出路径：public/figures/ch04/fig4_4_02_gam_basis.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
apply_style()
# --- 1. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
ax_a, ax_b, ax_c = axes
# --- 2. 面板 (a): B 样条基函数 ---
x_basis = np.linspace(0, 1, 300)
transformer = SplineTransformer(n_knots=6, degree=3, include_bias=True)
transformer.fit(x_basis.reshape(-1, 1))
B = transformer.transform(x_basis.reshape(-1, 1))
n_basis = B.shape[1]
for k in range(n_basis):
    color = PALETTE[k % len(PALETTE)]
    ax_a.plot(x_basis, B[:, k], color=color, lw=2, alpha=0.85)
knot_pos = np.linspace(0, 1, 6)
for i, xi in enumerate(knot_pos):
    ax_a.axvline(xi, color='gray', ls='--', alpha=0.45, lw=1.0)
ax_a.annotate("结点位置", xy=(knot_pos[2], 0.82), fontsize=12, color='gray',
               ha='center', va='bottom',
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.0),
               xytext=(knot_pos[2] + 0.15, 0.92))
ax_a.set_xlabel("$x$", fontsize=13)
ax_a.set_ylabel("基函数值 $B_k(x)$", fontsize=13)
ax_a.set_title("(a) 三次 B 样条基函数（6 个结点）\n每个基函数仅在局部区间非零（局部支撑性）", fontsize=13)
# --- 3. 加载 California Housing 数据 ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
feat_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
              "Population", "AveOccup", "Latitude", "Longitude"]
linear_features = [2, 3, 4, 6, 7]
ct = ColumnTransformer([
    ('spline_0', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [0]),
    ('spline_1', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [1]),
    ('spline_5', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [5]),
    ('linear', 'passthrough', linear_features)
])
gam_model = Pipeline([
    ('transform', ct),
    ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5))
])
gam_model.fit(X, y)
# --- 4. 面板 (b): GAM 偏反应函数 ---
X_mean = np.tile(X.mean(axis=0), (100, 1))
partial_cfg = [
    (0, "MedInc（收入中位数）",  COLORS['blue']),
    (1, "HouseAge（房龄）",      COLORS['green']),
    (5, "AveOccup（入住人数）",  COLORS['orange']),
]
y_base = gam_model.predict(X_mean[:1])
for feat_idx, label, color in partial_cfg:
    q5 = np.percentile(X[:, feat_idx], 5)
    q95 = np.percentile(X[:, feat_idx], 95)
    x_range = np.linspace(q5, q95, 100)
    x_norm = (x_range - q5) / (q95 - q5)
    X_eval = X_mean.copy()
    X_eval[:, feat_idx] = x_range
    y_partial = gam_model.predict(X_eval) - y_base
    ax_b.plot(x_norm, y_partial, color=color, lw=2.2, label=label)
    # Rug plot
    feat_data = X[:, feat_idx]
    feat_data_clip = feat_data[(feat_data >= q5) & (feat_data <= q95)]
    rug_norm = (feat_data_clip - q5) / (q95 - q5)
    sample_idx = np.random.choice(len(rug_norm), size=min(300, len(rug_norm)), replace=False)
    ax_b.plot(rug_norm[sample_idx], np.full(len(sample_idx), -0.35),
              '|', color=color, alpha=0.3, markersize=4)
ax_b.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax_b.set_xlabel("特征值分位数（0=5th 百分位，1=95th 百分位）", fontsize=13)
ax_b.set_ylabel(r"$\hat{f}_j(x_j)$（中心化）", fontsize=13)
ax_b.set_title(r"(b) GAM 偏反应函数 $\hat{f}_j(x_j)$" + "\n捕捉各特征对房价的非线性独立效应", fontsize=13)
ax_b.legend(fontsize=11)
# --- 5. 面板 (c): 模型性能对比（测试集 R²）---
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.2, random_state=42)
# OLS
ols = make_pipeline(StandardScaler(), LinearRegression())
ols.fit(X_tr2, y_tr2)
r2_ols = r2_score(y_te2, ols.predict(X_te2))
# Polynomial d=2
poly2 = make_pipeline(
    StandardScaler(),
    __import__('sklearn.preprocessing', fromlist=['PolynomialFeatures']).PolynomialFeatures(degree=2),
    LinearRegression()
)
poly2.fit(X_tr2, y_tr2)
r2_poly2 = r2_score(y_te2, poly2.predict(X_te2))
# Spline per feature
ct_all = ColumnTransformer([
    (f'sp_{i}', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [i])
    for i in range(8)
])
spline_all = Pipeline([('transform', ct_all), ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5))])
spline_all.fit(X_tr2, y_tr2)
r2_spline = r2_score(y_te2, spline_all.predict(X_te2))
# GAM (3 features)
ct_gam3 = ColumnTransformer([
    ('spline_0', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [0]),
    ('spline_1', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [1]),
    ('spline_5', SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), [5]),
    ('linear', 'passthrough', linear_features)
])
gam3 = Pipeline([('transform', ct_gam3), ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5))])
gam3.fit(X_tr2, y_tr2)
r2_gam3 = r2_score(y_te2, gam3.predict(X_te2))
methods = ["线性回归 (OLS)", "多项式 d=2", "样条变换（每特征）", "GAM（3 特征平滑）"]
r2_vals = [r2_ols, r2_poly2, r2_spline, r2_gam3]
bar_colors = [COLORS['gray'], COLORS['blue'], COLORS['green'], COLORS['teal']]
y_pos = range(len(methods))
bars = ax_c.barh(y_pos, r2_vals, color=bar_colors, height=0.55, edgecolor='white')
for i, (val, bar) in enumerate(zip(r2_vals, bars)):
    ax_c.text(val + 0.001, i, f"{val:.4f}", va='center', ha='left', fontsize=12)
ax_c.set_yticks(list(y_pos))
ax_c.set_yticklabels(methods, fontsize=12)
r2_min = min(r2_vals) - 0.02
r2_max = max(r2_vals) + 0.02
ax_c.set_xlim(r2_min, r2_max + 0.015)
ax_c.set_xlabel("测试集 $R^2$（越高越好）", fontsize=13)
ax_c.set_title("(c) 测试集 $R^2$ 对比（California Housing）\n非线性建模在适当配置下提升预测精度", fontsize=13)
# --- 6. 总标题与保存 ---
fig.suptitle("B 样条基函数、GAM 偏反应函数与预测性能", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_4_02_gam_basis")
