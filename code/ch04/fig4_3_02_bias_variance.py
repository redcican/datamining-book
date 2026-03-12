"""
图 4.3.2  过拟合演示、偏差–方差权衡与软阈值算子
对应节次：4.3 正则化回归（Ridge, Lasso, 弹性网络）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_3_02_bias_variance.py
输出路径：public/figures/ch04/fig4_3_02_bias_variance.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
import numpy as np
import matplotlib.pyplot as plt
apply_style()
np.random.seed(42)
# --- 1. 图形布局 ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
# --- 2. 面板(a)：多项式过拟合演示 ---
ax = axes[0]
x_true = np.linspace(0, 1, 300)
y_true = np.sin(2 * np.pi * x_true)
x_train = np.linspace(0, 1, 30) + np.random.randn(30) * 0.005
x_train = np.clip(x_train, 0, 1)
y_train = np.sin(2 * np.pi * x_train) + np.random.randn(30) * 0.2
ax.scatter(x_train, y_train, color='#94a3b8', s=40, zorder=3, label="训练数据")
ax.plot(x_true, y_true, 'k--', lw=2, label="真实函数 sin(2πx)")
degree_colors = {1: COLORS['blue'], 3: COLORS['green'], 7: COLORS['red']}
for deg, col in degree_colors.items():
    coeffs = np.polyfit(x_train, y_train, deg)
    y_fit = np.polyval(coeffs, x_true)
    ax.plot(x_true, y_fit, color=col, lw=2, label=f"{deg} 次多项式")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("$x$", fontsize=13)
ax.set_ylabel("$y$", fontsize=13)
ax.legend(fontsize=11, loc='upper right')
ax.set_title("(a) 多项式阶数与过拟合\n阶数越高，拟合噪声越严重（训练误差↓，泛化误差↑）",
             fontsize=13, pad=8)
# --- 3. 面板(b)：偏差²–方差权衡（Ridge 模拟）---
ax = axes[1]
n_tr, n_te, p = 100, 50, 10
beta_true = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
X_tr_sim = np.random.randn(n_tr, p)
X_te_sim = np.random.randn(n_te, p)
lambda_range = np.logspace(-2, 3, 80)
M = 500
preds = np.zeros((M, len(lambda_range), n_te))
for m in range(M):
    y_m = X_tr_sim @ beta_true + np.random.randn(n_tr)
    for li, lam in enumerate(lambda_range):
        A = X_tr_sim.T @ X_tr_sim + lam * np.eye(p)
        beta_hat = np.linalg.solve(A, X_tr_sim.T @ y_m)
        preds[m, li, :] = X_te_sim @ beta_hat
y_true_te = X_te_sim @ beta_true
bias2 = np.mean((preds.mean(axis=0) - y_true_te[None, :])**2, axis=1)
variance = np.mean(preds.var(axis=0), axis=1)
mse = bias2 + variance
log_lam = np.log10(lambda_range)
opt_idx = np.argmin(mse)
ax.plot(log_lam, bias2, color=COLORS['blue'], lw=2, label=r"偏差² Bias²")
ax.plot(log_lam, variance, color=COLORS['orange'], lw=2, label="方差 Variance")
ax.plot(log_lam, mse, color=COLORS['red'], lw=2, ls='--', label="总MSE")
ax.axvline(log_lam[opt_idx], color=COLORS['gray'], lw=1.8, ls=':',
           label=f"最优 λ*={lambda_range[opt_idx]:.2f}")
ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=13)
ax.set_ylabel("误差", fontsize=13)
ax.legend(fontsize=11)
ax.set_title(r"(b) 偏差²–方差权衡（Ridge 回归，模拟数据）" + "\n最优 λ* 在偏差² ≈ 方差处（非零！）",
             fontsize=13, pad=8)
# --- 4. 面板(c)：软阈值算子 ---
ax = axes[2]
z = np.linspace(-3, 3, 300)
ax.plot(z, z, color='#94a3b8', lw=1.5, ls='--', label="OLS（恒等）")
ax.axhline(0, color='k', lw=0.5, alpha=0.4)
ax.axvline(0, color='k', lw=0.5, alpha=0.4)
ax.axvspan(-1.0, 1.0, alpha=0.1, color='yellow', label="零区间 (γ=1.0)")
gamma_colors = {0.5: COLORS['blue'], 1.0: COLORS['orange'], 1.5: COLORS['red']}
for gamma, col in gamma_colors.items():
    soft = np.sign(z) * np.maximum(np.abs(z) - gamma, 0)
    ax.plot(z, soft, color=col, lw=2, label=f"S(z, γ={gamma})")
ax.set_xlabel("$z$（偏残差内积）", fontsize=13)
ax.set_ylabel(r"$\hat{\beta}_j$（坐标更新值）", fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.set_title(r"(c) 软阈值算子 $\mathcal{S}(z,\gamma) = \mathrm{sign}(z)(|z|-\gamma)_+$"
             + "\n绝对值 ≤ γ 的系数精确置零（Lasso 稀疏性的根源）",
             fontsize=13, pad=8)
# --- 5. 保存 ---
fig.suptitle("过拟合演示、偏差–方差权衡与软阈值算子",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_3_02_bias_variance")
