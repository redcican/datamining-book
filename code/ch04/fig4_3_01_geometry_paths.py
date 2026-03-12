"""
图 4.3.1  正则化几何直觉与系数路径（岭回归 vs Lasso）
对应节次：4.3 正则化回归（Ridge, Lasso, 弹性网络）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_3_01_geometry_paths.py
输出路径：public/figures/ch04/fig4_3_01_geometry_paths.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, lasso_path, LassoCV
apply_style()
# --- 1. 数据准备 ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
feat_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
              "Population", "AveOccup", "Latitude", "Longitude"]
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
# --- 2. 图形布局 ---
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
# --- 3. 面板(a)：Ridge L2 几何 ---
ax = axes[0, 0]
b0 = np.linspace(-1, 3, 300)
b1 = np.linspace(-1, 3, 300)
B0, B1 = np.meshgrid(b0, b1)
Loss = (B0 - 2.0)**2 * 2.5 + (B1 - 1.5)**2 * 1.5 + 0.8 * (B0 - 2.0) * (B1 - 1.5)
levels = np.linspace(Loss.min(), Loss.max() * 0.6, 30)
ax.contour(B0, B1, Loss, levels=levels, colors='#94a3b8', alpha=0.6)
circle = plt.Circle((0, 0), 1.0, fill=False, edgecolor=COLORS['blue'], lw=2)
ax.add_patch(circle)
t_vals = np.linspace(0, 2 * np.pi, 1000)
circle_pts = np.column_stack([np.cos(t_vals), np.sin(t_vals)])
circle_loss = np.array([
    (p[0] - 2.0)**2 * 2.5 + (p[1] - 1.5)**2 * 1.5 + 0.8 * (p[0] - 2.0) * (p[1] - 1.5)
    for p in circle_pts
])
ridge_idx = np.argmin(circle_loss)
ridge_sol = circle_pts[ridge_idx]
ax.scatter([2.0], [1.5], marker='*', color=COLORS['red'], s=200, zorder=5)
ax.scatter([ridge_sol[0]], [ridge_sol[1]], marker='o', color=COLORS['green'], s=100, zorder=5)
ax.plot([2.0, ridge_sol[0]], [1.5, ridge_sol[1]], '--', color=COLORS['gray'], lw=1.5)
ax.annotate(r"OLS解 $\hat{\beta}$", xy=(2.0, 1.5), xytext=(2.1, 1.7),
            fontsize=12, color=COLORS['red'],
            arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.2))
ax.annotate("Ridge解", xy=(ridge_sol[0], ridge_sol[1]),
            xytext=(ridge_sol[0] + 0.15, ridge_sol[1] - 0.3),
            fontsize=12, color=COLORS['green'],
            arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.2))
ax.text(0.05, 1.05, r"L2球 $\|\beta\|_2 \leq t$", fontsize=12, color=COLORS['blue'],
        transform=ax.transAxes)
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.axvline(0, color='k', lw=0.5, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.set_xlabel(r"$\beta_1$", fontsize=13)
ax.set_ylabel(r"$\beta_2$", fontsize=13)
ax.set_title("(a) 岭回归：L2 约束球\n等高线切圆→连续收缩，不置零", fontsize=13, pad=8)
# --- 4. 面板(b)：Lasso L1 几何 ---
ax = axes[0, 1]
ax.contour(B0, B1, Loss, levels=levels, colors='#94a3b8', alpha=0.6)
diamond_r = 1.2
diamond_verts = np.array([(diamond_r, 0), (0, diamond_r), (-diamond_r, 0), (0, -diamond_r), (diamond_r, 0)])
diamond_patch = mpatches.Polygon(diamond_verts[:4], closed=True,
                                  fill=False, edgecolor=COLORS['blue'], lw=2)
ax.add_patch(diamond_patch)
lasso_sol = np.array([diamond_r, 0.0])
ax.scatter([2.0], [1.5], marker='*', color=COLORS['red'], s=200, zorder=5)
ax.scatter([lasso_sol[0]], [lasso_sol[1]], marker='o', color=COLORS['orange'], s=150, zorder=5)
ax.plot([2.0, lasso_sol[0]], [1.5, lasso_sol[1]], '--', color=COLORS['gray'], lw=1.5)
ax.annotate(r"OLS解 $\hat{\beta}$", xy=(2.0, 1.5), xytext=(2.1, 1.7),
            fontsize=12, color=COLORS['red'],
            arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.2))
ax.annotate(r"Lasso解 $\beta_2=0$", xy=(lasso_sol[0], lasso_sol[1]),
            xytext=(lasso_sol[0] - 0.5, lasso_sol[1] + 0.5),
            fontsize=12, color=COLORS['orange'],
            arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.2))
ax.text(0.5, 0.08, "角点→系数精确置零", fontsize=12, color=COLORS['orange'],
        transform=ax.transAxes, ha='center')
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.axvline(0, color='k', lw=0.5, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.set_xlabel(r"$\beta_1$", fontsize=13)
ax.set_ylabel(r"$\beta_2$", fontsize=13)
ax.set_title("(b) Lasso 回归：L1 约束菱形\n等高线切菱形角点→精确稀疏", fontsize=13, pad=8)
# --- 5. 面板(c)：Ridge 正则化路径 ---
ax = axes[1, 0]
alphas_ridge = np.logspace(-2, 4, 100)
coefs_ridge = []
from sklearn.linear_model import Ridge
for a in alphas_ridge:
    ridge = Ridge(alpha=a)
    ridge.fit(X_sc, y)
    coefs_ridge.append(ridge.coef_)
coefs_ridge = np.array(coefs_ridge).T
ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5)
ridge_cv.fit(X_sc, y)
best_alpha_ridge = ridge_cv.alpha_
colors_path = PALETTE[:8]
log_alphas_ridge = np.log10(alphas_ridge)
for j in range(8):
    ax.plot(log_alphas_ridge, coefs_ridge[j], color=colors_path[j], lw=1.8, label=feat_names[j])
ax.axvline(np.log10(best_alpha_ridge), color=COLORS['gray'], lw=2, ls='--',
           label=f"最优λ (CV)")
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.set_xlabel(r"$\log_{10}(\lambda)$（正则化强度）", fontsize=13)
ax.set_ylabel(r"标准化系数 $\hat{\beta}_j$", fontsize=13)
ax.legend(loc='upper right', fontsize=10, ncol=1)
ax.set_title("(c) 岭回归正则化路径（California Housing）\n系数随 λ 增大连续收缩趋于零", fontsize=13, pad=8)
# --- 6. 面板(d)：Lasso 正则化路径 ---
ax = axes[1, 1]
alphas_lasso, coefs_path_lasso, _ = lasso_path(X_sc, y, n_alphas=100)
x_axis_lasso = -np.log10(alphas_lasso)
for j in range(8):
    ax.plot(x_axis_lasso, coefs_path_lasso[j], color=colors_path[j], lw=1.8, label=feat_names[j])
lasso_cv_obj = LassoCV(n_alphas=50, cv=5, max_iter=10000)
lasso_cv_obj.fit(X_sc, y)
best_alpha_lasso = lasso_cv_obj.alpha_
ax.axvline(-np.log10(best_alpha_lasso), color=COLORS['gray'], lw=2, ls='--',
           label=f"最优λ (CV)")
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
x_right = x_axis_lasso[-1]
for j in range(8):
    coef_end = coefs_path_lasso[j, -1]
    if abs(coef_end) > 0.05:
        ax.text(x_right + 0.05, coef_end, feat_names[j], fontsize=10,
                va='center', color=colors_path[j])
ax.set_xlabel(r"$-\log_{10}(\lambda)$（←稀疏  稠密→）", fontsize=13)
ax.set_ylabel(r"标准化系数 $\hat{\beta}_j$", fontsize=13)
ax.legend(loc='upper left', fontsize=10, ncol=1)
ax.set_title("(d) Lasso 正则化路径（California Housing）\n系数随 λ 增大逐步精确置零（变量选择）", fontsize=13, pad=8)
# --- 7. 保存 ---
fig.suptitle("正则化几何直觉与系数路径（岭回归 vs Lasso）",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_3_01_geometry_paths")
