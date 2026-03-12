"""
图 4.7.1  全章回归方法拟合曲线叠加对比与偏差–方差谱分解
对应节次：4.7 回归算法系统比较与案例分析
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_7_01_method_comparison.py
输出路径：public/figures/ch04/fig4_7_01_method_comparison.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

apply_style()

np.random.seed(0)

# ── 真实函数与数据生成 ────────────────────────────────────────────────────────
def true_f(x):
    return np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)

x_plot = np.linspace(0, 1, 300)
y_true = true_f(x_plot)

# 单个演示数据集（n=30）
n_demo = 30
x_demo = np.sort(np.random.uniform(0, 1, n_demo))
y_demo = true_f(x_demo) + np.random.normal(0, 0.3, n_demo)
X_demo = x_demo.reshape(-1, 1)
X_plot = x_plot.reshape(-1, 1)

# ── 定义七类方法 ──────────────────────────────────────────────────────────────
methods = {
    "OLS 线性":          make_pipeline(PolynomialFeatures(1, include_bias=False), Ridge(alpha=1e-6)),
    "Ridge (λ=0.1)":    make_pipeline(PolynomialFeatures(1, include_bias=False), Ridge(alpha=0.1)),
    "多项式 deg=7":      make_pipeline(PolynomialFeatures(7, include_bias=False), Ridge(alpha=1e-4)),
    "自然样条 k=8":      make_pipeline(SplineTransformer(n_knots=8, degree=3, knots='quantile'), Ridge(alpha=1e-3)),
    "CART depth=3":      DecisionTreeRegressor(max_depth=3, random_state=0),
    "随机森林 B=200":    RandomForestRegressor(n_estimators=200, max_features=0.8, random_state=0, n_jobs=-1),
    "GBDT M=200":        GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.1, random_state=0),
}

method_colors = [
    COLORS['gray'], COLORS['blue'], COLORS['orange'],
    COLORS['teal'], COLORS['red'], COLORS['green'], COLORS['purple'],
]
method_ls = ['--', '-.', '--', ':', '--', '-', '-']

# ── 偏差–方差分解（100次蒙特卡洛重复） ───────────────────────────────────────
n_repeat = 100
n_train  = 30
sigma    = 0.3
noise_var = sigma ** 2

preds_all = {name: [] for name in methods}
for _ in range(n_repeat):
    xi = np.sort(np.random.uniform(0, 1, n_train))
    yi = true_f(xi) + np.random.normal(0, sigma, n_train)
    Xi = xi.reshape(-1, 1)
    for name, mdl in methods.items():
        mdl.fit(Xi, yi)
        preds_all[name].append(mdl.predict(X_plot))

bias2_all  = {}
var_all    = {}
mse_all    = {}
for name in methods:
    preds = np.array(preds_all[name])          # (n_repeat, n_plot)
    mean_pred = preds.mean(axis=0)
    bias2  = ((mean_pred - y_true) ** 2).mean()
    var    = preds.var(axis=0).mean()
    mse    = ((preds - y_true[None, :]) ** 2).mean()
    bias2_all[name] = bias2
    var_all[name]   = var
    mse_all[name]   = mse

# ── 作图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
ax_a, ax_b, ax_c = axes

# ── 面板 (a): 拟合曲线叠加 ────────────────────────────────────────────────────
ax_a.plot(x_plot, y_true, 'k-', lw=2.5, label='真实函数 $f(x)$', zorder=5)
ax_a.scatter(x_demo, y_demo, color=COLORS['gray'], s=25, alpha=0.6, zorder=3, label='训练数据')

for (name, mdl), col, ls in zip(methods.items(), method_colors, method_ls):
    mdl.fit(X_demo, y_demo)
    y_pred = mdl.predict(X_plot)
    ax_a.plot(x_plot, y_pred, color=col, ls=ls, lw=1.8, label=name, alpha=0.9)

ax_a.set_xlim(0, 1)
ax_a.set_xlabel('$x$', fontsize=13)
ax_a.set_ylabel('$y$', fontsize=13)
ax_a.set_title('(a) 七种方法拟合曲线叠加对比（$n=30$）', fontsize=13, fontweight='bold')
ax_a.legend(fontsize=9, loc='upper right', framealpha=0.9)

# ── 面板 (b): 偏差²–方差–MSE 条形图（堆叠） ───────────────────────────────────
names_short = ['OLS\n线性', 'Ridge\nλ=0.1', '多项式\ndeg=7', '样条\nk=8',
               'CART\ndep=3', 'RF\nB=200', 'GBDT\nM=200']
bias2_vals = [bias2_all[k] for k in methods]
var_vals   = [var_all[k]   for k in methods]
noise_vals = [noise_var] * len(methods)

x_idx = np.arange(len(methods))
bar_w = 0.55
p1 = ax_b.bar(x_idx, bias2_vals, bar_w, label='偏差²', color=COLORS['red'], alpha=0.85)
p2 = ax_b.bar(x_idx, var_vals, bar_w, bottom=bias2_vals, label='方差', color=COLORS['blue'], alpha=0.85)
noise_bottom = [b + v for b, v in zip(bias2_vals, var_vals)]
p3 = ax_b.bar(x_idx, noise_vals, bar_w, bottom=noise_bottom, label='不可约噪声 $\\sigma^2$',
              color=COLORS['gray'], alpha=0.5)

# 在条形顶部标注MSE
for i, mse in enumerate(mse_all.values()):
    ax_b.text(i, noise_bottom[i] + noise_var + 0.005, f'{mse:.3f}',
              ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax_b.set_xticks(x_idx)
ax_b.set_xticklabels(names_short, fontsize=9)
ax_b.set_ylabel('误差分解（均值）', fontsize=12)
ax_b.set_title('(b) 偏差²–方差–噪声分解\n（100次蒙特卡洛，$n=30$，$\\sigma=0.3$）', fontsize=13, fontweight='bold')
ax_b.legend(fontsize=10, loc='upper center')

# ── 面板 (c): 偏差–方差权衡散点图 ─────────────────────────────────────────────
for i, (name, col) in enumerate(zip(methods, method_colors)):
    b2 = bias2_all[name]
    v  = var_all[name]
    ax_c.scatter(b2, v, color=col, s=120, zorder=4)
    ax_c.annotate(names_short[i].replace('\n', ' '),
                  xy=(b2, v), xytext=(5, 3), textcoords='offset points', fontsize=9)

# 等MSE线
b2_grid = np.linspace(0, 0.15, 200)
for mse_lv in [0.10, 0.15, 0.20, 0.25]:
    v_iso = mse_lv - noise_var - b2_grid
    mask  = v_iso >= 0
    ax_c.plot(b2_grid[mask], v_iso[mask], 'k--', lw=0.8, alpha=0.35)
    ax_c.text(b2_grid[mask][-1], v_iso[mask][-1], f'MSE={mse_lv:.2f}',
              fontsize=7.5, color='gray', ha='left', va='center')

ax_c.set_xlabel('偏差² $\\mathrm{Bias}^2$', fontsize=12)
ax_c.set_ylabel('方差 $\\mathrm{Var}$', fontsize=12)
ax_c.set_title('(c) 偏差–方差权衡定位图\n（虚线：等 MSE 轮廓，已扣除 $\\sigma^2$）', fontsize=13, fontweight='bold')
ax_c.set_xlim(left=0)
ax_c.set_ylim(bottom=0)

plt.tight_layout()
save_fig(fig, __file__, 'fig4_7_01_method_comparison')
