"""
图 4.6.1  多项式回归的偏差–方差分解：degree=1,3,7,15 的经典过拟合演示
对应节次：4.6 偏差–方差分解与超参数选择
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_6_01_polynomial_bv.py
输出路径：public/figures/ch04/fig4_6_01_polynomial_bv.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

apply_style()
np.random.seed(0)

# ── 真实函数 ──────────────────────────────────────────────────────────────────
def true_f(x):
    return np.sin(2 * np.pi * x)

sigma   = 0.3
n_train = 30
n_rep   = 200          # 蒙特卡洛重复次数
degrees = [1, 3, 7, 15]
x_plot  = np.linspace(0, 1, 300)
y_true  = true_f(x_plot)

# ── 每个 degree 的偏差–方差分解 ───────────────────────────────────────────────
preds_all = {d: [] for d in degrees}
for _ in range(n_rep):
    x_tr = np.sort(np.random.uniform(0, 1, n_train))
    y_tr = true_f(x_tr) + np.random.normal(0, sigma, n_train)
    for d in degrees:
        mdl = make_pipeline(PolynomialFeatures(d, include_bias=True), Ridge(alpha=1e-9))
        mdl.fit(x_tr.reshape(-1, 1), y_tr)
        preds_all[d].append(mdl.predict(x_plot.reshape(-1, 1)))

bias2_by_deg = {}
var_by_deg   = {}
mse_by_deg   = {}
for d in degrees:
    P = np.array(preds_all[d])          # (n_rep, n_plot)
    mean_p = P.mean(axis=0)
    bias2_by_deg[d] = ((mean_p - y_true) ** 2).mean()
    var_by_deg[d]   = P.var(axis=0).mean()
    mse_by_deg[d]   = ((P - y_true[None, :]) ** 2).mean()

# ── 作图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
ax_a, ax_b, ax_c = axes
deg_colors = [COLORS['gray'], COLORS['teal'], COLORS['orange'], COLORS['red']]
deg_ls     = ['-', '--', ':', '-.']

# ── 面板 (a): 单次训练下的拟合曲线（n_rep=200 之第一次） ─────────────────────
np.random.seed(5)
x_demo = np.sort(np.random.uniform(0, 1, n_train))
y_demo = true_f(x_demo) + np.random.normal(0, sigma, n_train)
ax_a.plot(x_plot, y_true, 'k-', lw=2.5, label='真实函数 $f(x)=\\sin(2\\pi x)$', zorder=5)
ax_a.scatter(x_demo, y_demo, color=COLORS['gray'], s=25, alpha=0.6, zorder=3, label='训练数据 ($n=30$)')
for d, col, ls in zip(degrees, deg_colors, deg_ls):
    mdl = make_pipeline(PolynomialFeatures(d, include_bias=True), Ridge(alpha=1e-9))
    mdl.fit(x_demo.reshape(-1, 1), y_demo)
    y_pred = mdl.predict(x_plot.reshape(-1, 1))
    ax_a.plot(x_plot, y_pred, color=col, ls=ls, lw=1.8, label=f'degree={d}', alpha=0.9)
ax_a.set_xlim(0, 1); ax_a.set_ylim(-3.5, 3.5)
ax_a.set_xlabel('$x$', fontsize=13); ax_a.set_ylabel('$y$', fontsize=13)
ax_a.set_title('(a) 多项式回归拟合曲线（单次训练集）\n低阶欠拟合，高阶振荡过拟合', fontsize=13, fontweight='bold')
ax_a.legend(fontsize=9.5)

# ── 面板 (b): 偏差²–方差–MSE 堆叠条形图 ─────────────────────────────────────
noise_val = sigma ** 2
x_idx = np.arange(len(degrees))
bar_w = 0.5
bias2_vals = [bias2_by_deg[d] for d in degrees]
var_vals   = [var_by_deg[d]   for d in degrees]
noise_vals = [noise_val] * len(degrees)

p1 = ax_b.bar(x_idx, bias2_vals, bar_w, label='偏差²', color=COLORS['red'], alpha=0.85)
p2 = ax_b.bar(x_idx, var_vals, bar_w, bottom=bias2_vals, label='方差', color=COLORS['blue'], alpha=0.85)
nb = [b + v for b, v in zip(bias2_vals, var_vals)]
p3 = ax_b.bar(x_idx, noise_vals, bar_w, bottom=nb,
              label=f'不可约噪声 $\\sigma^2={noise_val:.2f}$', color=COLORS['gray'], alpha=0.45)

for i, d in enumerate(degrees):
    mse = mse_by_deg[d]
    ax_b.text(i, nb[i] + noise_val + 0.005, f'MSE={mse:.3f}',
              ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_b.set_xticks(x_idx)
ax_b.set_xticklabels([f'degree={d}' for d in degrees], fontsize=11)
ax_b.set_ylabel('误差分解（均值）', fontsize=12)
ax_b.set_title(f'(b) 偏差²–方差–噪声分解\n（{n_rep}次蒙特卡洛，$n={n_train}$，$\\sigma={sigma}$）',
               fontsize=13, fontweight='bold')
ax_b.legend(fontsize=10)

# ── 面板 (c): Bias²、方差、总误差 vs degree 连线图（U形曲线） ─────────────────
d_fine = list(range(1, 16))
bias2_fine, var_fine, mse_fine = [], [], []
for d in d_fine:
    preds = []
    for _ in range(n_rep):
        x_tr = np.sort(np.random.uniform(0, 1, n_train))
        y_tr  = true_f(x_tr) + np.random.normal(0, sigma, n_train)
        mdl   = make_pipeline(PolynomialFeatures(d, include_bias=True), Ridge(alpha=1e-9))
        mdl.fit(x_tr.reshape(-1, 1), y_tr)
        preds.append(mdl.predict(x_plot.reshape(-1, 1)))
    P     = np.array(preds)
    mean_p = P.mean(axis=0)
    bias2_fine.append(((mean_p - y_true) ** 2).mean())
    var_fine.append(P.var(axis=0).mean())
    mse_fine.append(((P - y_true[None, :]) ** 2).mean())

ax_c.plot(d_fine, bias2_fine, 'o-', color=COLORS['red'],   lw=2, ms=6, label='偏差²')
ax_c.plot(d_fine, var_fine,   's-', color=COLORS['blue'],  lw=2, ms=6, label='方差')
ax_c.plot(d_fine, mse_fine,   '^-', color=COLORS['purple'],lw=2.5, ms=7, label='总 MSE（含 $\\sigma^2$）')
ax_c.axhline(y=noise_val, color=COLORS['gray'], ls='--', lw=1.2, alpha=0.7,
             label=f'不可约噪声 $\\sigma^2={noise_val:.2f}$')

# 标记最优degree
best_d = d_fine[int(np.argmin(mse_fine))]
best_mse = min(mse_fine)
ax_c.axvline(x=best_d, color='green', ls=':', lw=1.5, alpha=0.8)
ax_c.annotate(f'最优 degree={best_d}', xy=(best_d, best_mse),
              xytext=(best_d + 1, best_mse + 0.04),
              arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')

ax_c.set_xlabel('多项式阶数 degree', fontsize=12)
ax_c.set_ylabel('误差（均值）', fontsize=12)
ax_c.set_title('(c) 偏差²–方差–总误差 vs 多项式阶数\n经典 U 形曲线：最优复杂度在中间', fontsize=13, fontweight='bold')
ax_c.legend(fontsize=10, loc='upper left')
ax_c.set_ylim(bottom=0)

plt.tight_layout()
save_fig(fig, __file__, 'fig4_6_01_polynomial_bv')
