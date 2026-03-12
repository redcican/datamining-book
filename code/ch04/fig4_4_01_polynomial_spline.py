"""
图 4.4.1  多项式回归与样条：过拟合、结点与边界行为
对应节次：4.4 非线性回归（多项式、样条、广义加法模型）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_4_01_polynomial_spline.py
输出路径：public/figures/ch04/fig4_4_01_polynomial_spline.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
apply_style()
# --- 1. 共享数据 ---
np.random.seed(42)
n = 30
x_tr = np.linspace(0, 1, n)
y_tr = np.sin(2 * np.pi * x_tr) + np.random.normal(0, 0.25, n)
x_plot = np.linspace(-0.05, 1.05, 300)
y_true = np.sin(2 * np.pi * x_plot)
# --- 2. 截断幂基函数 ---
def truncated_power_basis(x, knots, degree=3):
    cols = [x**k for k in range(degree + 1)]
    for xi in knots:
        cols.append(np.maximum(0, x - xi)**degree)
    return np.column_stack(cols)
# --- 3. 创建图形 ---
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
ax_a, ax_b = axes[0, 0], axes[0, 1]
ax_c, ax_d = axes[1, 0], axes[1, 1]
# --- 4. 面板 (a): 多项式回归阶数 ---
ax_a.plot(x_plot, y_true, 'k--', lw=2, label=r"真实函数 $\sin(2\pi x)$")
ax_a.scatter(x_tr, y_tr, color=COLORS['gray'], s=40, alpha=0.7, zorder=5, label="训练数据 ($n=30$)")
degree_cfg = [
    (1,  COLORS['blue'],   "阶数 1（欠拟合）"),
    (3,  COLORS['green'],  "阶数 3（近优）"),
    (7,  COLORS['orange'], "阶数 7（过拟合）"),
    (15, COLORS['red'],    "阶数 15（严重过拟合）"),
]
for deg, color, label in degree_cfg:
    coeffs = np.polyfit(x_tr, y_tr, deg)
    y_pred = np.polyval(coeffs, x_plot)
    ax_a.plot(x_plot, np.clip(y_pred, -2.5, 2.5), color=color, lw=2, label=label)
ax_a.set_ylim(-2.5, 2.5)
ax_a.set_xlabel("$x$", fontsize=13)
ax_a.set_ylabel("$y$", fontsize=13)
ax_a.set_title("(a) 多项式回归：阶数与过拟合\n阶数 7 和 15 在边界处剧烈振荡（Runge 现象）", fontsize=13)
ax_a.legend(fontsize=12)
# --- 5. 面板 (b): 三次回归样条（不同结点数） ---
ax_b.plot(x_plot, y_true, 'k--', lw=2, label=r"真实函数 $\sin(2\pi x)$")
ax_b.scatter(x_tr, y_tr, color=COLORS['gray'], s=40, alpha=0.7, zorder=5, label="训练数据 ($n=30$)")
knot_cfg = [
    ([0.5],                          COLORS['blue'],   "1 个结点"),
    ([0.25, 0.5, 0.75],              COLORS['green'],  "3 个结点"),
    (list(np.linspace(0.1, 0.9, 6)), COLORS['orange'], "6 个结点（最优）"),
]
for knots, color, label in knot_cfg:
    X_des = truncated_power_basis(x_tr, knots)
    X_plot_des = truncated_power_basis(x_plot, knots)
    coeffs, _, _, _ = np.linalg.lstsq(X_des, y_tr, rcond=None)
    y_pred = X_plot_des @ coeffs
    ax_b.plot(x_plot, y_pred, color=color, lw=2, label=label)
knots6 = list(np.linspace(0.1, 0.9, 6))
for xi in knots6:
    ax_b.axvline(xi, color='gray', ls='--', alpha=0.4, lw=0.8)
ax_b.set_xlabel("$x$", fontsize=13)
ax_b.set_ylabel("$y$", fontsize=13)
ax_b.set_title("(b) 三次回归样条：结点数的影响\n适量结点（6 个）灵活捕捉非线性，无振荡", fontsize=13)
ax_b.legend(fontsize=12)
# --- 6. 面板 (c): 自然样条 vs 回归样条边界行为 ---
x_plot_ext = np.linspace(-0.2, 1.2, 400)
y_true_ext = np.sin(2 * np.pi * x_plot_ext)
ax_c.fill_betweenx([-2.5, 2.5], 0, 1, alpha=0.05, color=COLORS['blue'])
ax_c.scatter(x_tr, y_tr, color=COLORS['gray'], s=40, alpha=0.7, zorder=5, label="训练数据")
ax_c.plot(x_plot_ext, y_true_ext, 'k--', lw=2, label="真实函数")
reg_spline = make_pipeline(
    SplineTransformer(n_knots=6, degree=3, extrapolation='continue'),
    LinearRegression()
)
reg_spline.fit(x_tr.reshape(-1, 1), y_tr)
y_reg = reg_spline.predict(x_plot_ext.reshape(-1, 1))
ax_c.plot(x_plot_ext, np.clip(y_reg, -3, 3), color=COLORS['orange'], lw=2, label="回归样条（边界可振荡）")
nat_spline = make_pipeline(
    SplineTransformer(n_knots=6, degree=3, extrapolation='linear'),
    LinearRegression()
)
nat_spline.fit(x_tr.reshape(-1, 1), y_tr)
y_nat = nat_spline.predict(x_plot_ext.reshape(-1, 1))
ax_c.plot(x_plot_ext, y_nat, color=COLORS['blue'], lw=2, label="自然三次样条（边界线性）")
ax_c.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
ax_c.axvline(1, color='gray', ls='--', lw=1.2, alpha=0.7)
ax_c.annotate("外推区域", xy=(-0.1, 1.5), fontsize=12, color='gray',
               ha='center', va='center')
ax_c.annotate("外推区域", xy=(1.1, 1.5), fontsize=12, color='gray',
               ha='center', va='center')
ax_c.set_ylim(-2.5, 2.5)
ax_c.set_xlabel("$x$", fontsize=13)
ax_c.set_ylabel("$y$", fontsize=13)
ax_c.set_title("(c) 自然三次样条 vs 回归样条：边界行为\n自然样条在数据范围外退化为线性（更稳健的外推）", fontsize=13)
ax_c.legend(fontsize=12)
# --- 7. 面板 (d): 各方法比较（含测试 RMSE）---
x_test_d = np.linspace(0, 1, 200)
y_test_d = np.sin(2 * np.pi * x_test_d)
ax_d.plot(x_plot, y_true, 'k--', lw=2, label="真实函数")
ax_d.scatter(x_tr, y_tr, color=COLORS['gray'], s=40, alpha=0.7, zorder=5)
# OLS degree 1
coeffs1 = np.polyfit(x_tr, y_tr, 1)
y_ols = np.polyval(coeffs1, x_plot)
y_ols_te = np.polyval(coeffs1, x_test_d)
rmse_ols = np.sqrt(mean_squared_error(y_test_d, y_ols_te))
ax_d.plot(x_plot, y_ols, color=COLORS['gray'], ls='--', lw=1.5,
          label=f"线性回归 (RMSE={rmse_ols:.3f})")
# Polynomial degree 7
coeffs7 = np.polyfit(x_tr, y_tr, 7)
y_poly7 = np.polyval(coeffs7, x_plot)
y_poly7_te = np.polyval(coeffs7, x_test_d)
rmse_poly7 = np.sqrt(mean_squared_error(y_test_d, np.clip(y_poly7_te, -5, 5)))
ax_d.plot(x_plot, np.clip(y_poly7, -2.5, 2.5), color=COLORS['red'], lw=1.5,
          label=f"多项式 d=7 (RMSE={rmse_poly7:.3f})")
# Regression spline K=6
X_des6 = truncated_power_basis(x_tr, knots6)
X_plot_des6 = truncated_power_basis(x_plot, knots6)
X_test_des6 = truncated_power_basis(x_test_d, knots6)
coeffs6, _, _, _ = np.linalg.lstsq(X_des6, y_tr, rcond=None)
y_rs = X_plot_des6 @ coeffs6
y_rs_te = X_test_des6 @ coeffs6
rmse_rs = np.sqrt(mean_squared_error(y_test_d, y_rs_te))
ax_d.plot(x_plot, y_rs, color=COLORS['blue'], lw=2,
          label=f"回归样条 K=6 (RMSE={rmse_rs:.3f})")
# Natural spline
nat_spline_d = make_pipeline(
    SplineTransformer(n_knots=6, degree=3, extrapolation='linear'),
    LinearRegression()
)
nat_spline_d.fit(x_tr.reshape(-1, 1), y_tr)
y_nat_d = nat_spline_d.predict(x_plot.reshape(-1, 1))
y_nat_te = nat_spline_d.predict(x_test_d.reshape(-1, 1))
rmse_nat = np.sqrt(mean_squared_error(y_test_d, y_nat_te))
ax_d.plot(x_plot, y_nat_d, color=COLORS['green'], lw=2,
          label=f"自然三次样条 (RMSE={rmse_nat:.3f})")
ax_d.set_ylim(-2.5, 2.5)
ax_d.set_xlabel("$x$", fontsize=13)
ax_d.set_ylabel("$y$", fontsize=13)
ax_d.set_title("(d) 各非线性方法比较（同一数据集）\n样条在灵活性与稳定性间取得最优平衡", fontsize=13)
ax_d.legend(fontsize=11)
# --- 8. 总标题与保存 ---
fig.suptitle("多项式回归与样条：阶数选择、结点数与边界行为", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_4_01_polynomial_spline")
