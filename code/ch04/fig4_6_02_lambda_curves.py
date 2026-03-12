"""
图 4.6.2  Ridge 正则化强度 λ 对偏差²–方差–总误差三曲线的影响
对应节次：4.6 偏差–方差分解与超参数选择
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_6_02_lambda_curves.py
输出路径：public/figures/ch04/fig4_6_02_lambda_curves.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

apply_style()
np.random.seed(42)

# ── 真实函数与数据生成 ────────────────────────────────────────────────────────
def true_f(x):
    return 2 * np.sin(2 * np.pi * x) - np.cos(4 * np.pi * x)

sigma    = 0.5
n_train  = 40
n_rep    = 150
degree   = 8                           # 高次多项式，λ 起关键作用
lambdas  = np.logspace(-5, 2, 40)     # 40 个 λ 值，对数均匀

x_plot   = np.linspace(0, 1, 200)
y_true   = true_f(x_plot)

# ── 蒙特卡洛偏差–方差分解（对每个 λ） ──────────────────────────────────────────
bias2_all, var_all, mse_all = [], [], []
for lam in lambdas:
    preds = []
    for _ in range(n_rep):
        x_tr = np.sort(np.random.uniform(0, 1, n_train))
        y_tr  = true_f(x_tr) + np.random.normal(0, sigma, n_train)
        mdl   = make_pipeline(PolynomialFeatures(degree, include_bias=True), Ridge(alpha=lam))
        mdl.fit(x_tr.reshape(-1, 1), y_tr)
        preds.append(mdl.predict(x_plot.reshape(-1, 1)))
    P      = np.array(preds)
    mean_p = P.mean(axis=0)
    bias2_all.append(((mean_p - y_true) ** 2).mean())
    var_all.append(P.var(axis=0).mean())
    mse_all.append(((P - y_true[None, :]) ** 2).mean())

bias2_all = np.array(bias2_all)
var_all   = np.array(var_all)
mse_all   = np.array(mse_all)
noise_val = sigma ** 2

# 最优 λ
opt_idx  = int(np.argmin(mse_all))
opt_lam  = lambdas[opt_idx]
opt_mse  = mse_all[opt_idx]

# ── 过拟合/欠拟合区间 ────────────────────────────────────────────────────────
underfit_thresh = lambdas[np.argmin(np.abs(bias2_all - mse_all.max() * 0.4))]
overfit_thresh  = lambdas[np.argmin(np.abs(var_all - mse_all.max() * 0.3))]

# ── 交叉验证误差曲线（5折CV代理） ─────────────────────────────────────────────
np.random.seed(7)
cv_errors = []
n_cv = 5
for lam in lambdas:
    fold_errors = []
    x_all = np.sort(np.random.uniform(0, 1, n_train * 2))
    y_all = true_f(x_all) + np.random.normal(0, sigma, len(x_all))
    fold_size = len(x_all) // n_cv
    for k in range(n_cv):
        val_idx  = np.arange(k * fold_size, (k + 1) * fold_size)
        tr_idx   = np.setdiff1d(np.arange(len(x_all)), val_idx)
        mdl = make_pipeline(PolynomialFeatures(degree, include_bias=True), Ridge(alpha=lam))
        mdl.fit(x_all[tr_idx].reshape(-1, 1), y_all[tr_idx])
        pred_val = mdl.predict(x_all[val_idx].reshape(-1, 1))
        fold_errors.append(np.mean((y_all[val_idx] - pred_val) ** 2))
    cv_errors.append(np.mean(fold_errors))

cv_errors = np.array(cv_errors)
cv_opt_idx = int(np.argmin(cv_errors))
cv_opt_lam = lambdas[cv_opt_idx]

# ── 作图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
ax_a, ax_b, ax_c = axes

# ── 面板 (a): 三曲线图（主图） ────────────────────────────────────────────────
ax_a.semilogx(lambdas, bias2_all, 'o-', color=COLORS['red'],    lw=2, ms=4, label='偏差²', alpha=0.85)
ax_a.semilogx(lambdas, var_all,   's-', color=COLORS['blue'],   lw=2, ms=4, label='方差', alpha=0.85)
ax_a.semilogx(lambdas, mse_all,   '^-', color=COLORS['purple'], lw=2.5, ms=5, label='总 MSE（含 $\\sigma^2$）', alpha=0.9)
ax_a.axhline(y=noise_val, color=COLORS['gray'], ls='--', lw=1.2, alpha=0.7,
             label=f'$\\sigma^2={noise_val:.2f}$（不可约下界）')

# 标注最优 λ
ax_a.axvline(x=opt_lam, color='green', ls=':', lw=1.8, alpha=0.8)
ax_a.scatter([opt_lam], [opt_mse], color='green', s=120, zorder=5,
             label=f'最优 $\\lambda^*={opt_lam:.3f}$')

# 标注过拟合/欠拟合区
ax_a.fill_betweenx([0, ax_a.get_ylim()[1] if ax_a.get_ylim()[1] > 0 else 3],
                    lambdas[0], opt_lam, color=COLORS['blue'], alpha=0.06)
ax_a.fill_betweenx([0, 3], opt_lam, lambdas[-1], color=COLORS['red'], alpha=0.06)
ax_a.text(lambdas[3], mse_all.max() * 0.8, '高方差\n（过拟合）', color=COLORS['blue'],
          fontsize=10, ha='center', style='italic')
ax_a.text(lambdas[-5], mse_all.max() * 0.8, '高偏差\n（欠拟合）', color=COLORS['red'],
          fontsize=10, ha='center', style='italic')

ax_a.set_xlabel('正则化强度 $\\lambda$（对数刻度）', fontsize=12)
ax_a.set_ylabel('误差（均值）', fontsize=12)
ax_a.set_title(f'(a) Ridge 偏差²–方差–总误差 vs $\\lambda$\n（度={degree}多项式，$n={n_train}$，$\\sigma={sigma}$）',
               fontsize=13, fontweight='bold')
ax_a.legend(fontsize=9.5, loc='upper left')
ax_a.set_ylim(bottom=0)

# ── 面板 (b): 交叉验证误差曲线（与真实 MSE 对比） ─────────────────────────────
ax_b.semilogx(lambdas, mse_all,   '-',  color=COLORS['purple'], lw=2, alpha=0.7, label='真实总 MSE')
ax_b.semilogx(lambdas, cv_errors, '--', color=COLORS['orange'],  lw=2, label='5折 CV 误差', alpha=0.9)
ax_b.axvline(x=opt_lam,    color=COLORS['purple'], ls=':',  lw=1.5, alpha=0.6, label=f'真实最优 $\\lambda^*={opt_lam:.4f}$')
ax_b.axvline(x=cv_opt_lam, color=COLORS['orange'],  ls='-.', lw=1.5, alpha=0.8, label=f'CV 选 $\\hat{{\\lambda}}={cv_opt_lam:.4f}$')
ax_b.scatter([opt_lam], [min(mse_all)], color=COLORS['purple'], s=100, zorder=5)
ax_b.scatter([cv_opt_lam], [cv_errors[cv_opt_idx]], color=COLORS['orange'],  s=100, zorder=5)

ax_b.set_xlabel('正则化强度 $\\lambda$（对数刻度）', fontsize=12)
ax_b.set_ylabel('MSE / CV 误差', fontsize=12)
ax_b.set_title('(b) 交叉验证误差 vs 真实总 MSE\nCV 选出的 $\\hat{\\lambda}$ 与真实最优 $\\lambda^*$ 接近',
               fontsize=13, fontweight='bold')
ax_b.legend(fontsize=9.5)
ax_b.set_ylim(bottom=0)

# ── 面板 (c): 不同 λ 下的拟合曲线对比 ───────────────────────────────────────────
np.random.seed(3)
x_demo = np.sort(np.random.uniform(0, 1, n_train))
y_demo = true_f(x_demo) + np.random.normal(0, sigma, n_train)

ax_c.plot(x_plot, y_true, 'k-', lw=2.5, label='真实函数', zorder=5)
ax_c.scatter(x_demo, y_demo, color=COLORS['gray'], s=25, alpha=0.6, zorder=3, label='训练数据')

showcase_lambdas = [1e-5, opt_lam, 10.0]
showcase_labels  = [f'$\\lambda=10^{{-5}}$（过拟合）', f'$\\lambda={opt_lam:.3f}$（最优）', '$\\lambda=10$（欠拟合）']
showcase_colors  = [COLORS['red'], COLORS['green'], COLORS['blue']]
for lam, lbl, col in zip(showcase_lambdas, showcase_labels, showcase_colors):
    mdl = make_pipeline(PolynomialFeatures(degree, include_bias=True), Ridge(alpha=lam))
    mdl.fit(x_demo.reshape(-1, 1), y_demo)
    ax_c.plot(x_plot, mdl.predict(x_plot.reshape(-1, 1)),
              color=col, lw=2, label=lbl, alpha=0.85)

ax_c.set_xlim(0, 1); ax_c.set_ylim(-5, 5)
ax_c.set_xlabel('$x$', fontsize=12)
ax_c.set_ylabel('$y$', fontsize=12)
ax_c.set_title(f'(c) 不同 $\\lambda$ 下的拟合曲线（degree={degree}）\n正则化控制过/欠拟合的连续调节',
               fontsize=13, fontweight='bold')
ax_c.legend(fontsize=10)

plt.tight_layout()
save_fig(fig, __file__, 'fig4_6_02_lambda_curves')
