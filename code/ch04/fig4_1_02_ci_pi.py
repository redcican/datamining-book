"""
图 4.1.2  置信区间 vs 预测区间：宽度对比与数学解释
对应节次：4.1 线性回归基础（OLS、正规方程、Gauss-Markov 定理）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_1_02_ci_pi.py
输出路径：public/figures/ch04/fig4_1_02_ci_pi.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

apply_style()

C_DATA = COLORS["blue"]
C_FIT  = COLORS["red"]
C_CI   = COLORS["green"]
C_PI   = COLORS["orange"]
C_GRAY = COLORS["gray"]

np.random.seed(0)
n = 40
x = np.sort(np.random.uniform(0, 10, n))
beta0_true, beta1_true, sigma = 2.0, 0.7, 1.5
y = beta0_true + beta1_true * x + np.random.normal(0, sigma, n)

X = np.column_stack([np.ones(n), x])
beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
y_hat    = X @ beta_hat
residuals = y - y_hat
s2 = np.sum(residuals**2) / (n - 2)   # MSE (unbiased)
s  = np.sqrt(s2)

x_new  = np.linspace(-0.5, 10.5, 300)
X_new  = np.column_stack([np.ones(300), x_new])
y_new  = X_new @ beta_hat

# lever = x0ᵀ (XᵀX)⁻¹ x0
XtX_inv = np.linalg.inv(X.T @ X)
h_new   = np.array([X_new[i] @ XtX_inv @ X_new[i] for i in range(300)])

t_crit = stats.t.ppf(0.975, df=n - 2)   # 95% 双侧

# 95% CI for E[y|x₀]: ŷ ± t * s * sqrt(h)
ci_half = t_crit * s * np.sqrt(h_new)
# 95% PI for y_new: ŷ ± t * s * sqrt(1 + h)
pi_half = t_crit * s * np.sqrt(1 + h_new)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# ── 面板(a): CI vs PI 带状区间 ────────────────────────────────────────────────
ax = axes[0]
ax.scatter(x, y, color=C_DATA, s=40, zorder=5, alpha=0.85, label="观测数据")
ax.plot(x_new, y_new, color=C_FIT, lw=2.2, label=r"OLS 拟合线 $\hat{y}$")
ax.fill_between(x_new, y_new - ci_half, y_new + ci_half,
                alpha=0.35, color=C_CI,
                label=r"95% 置信区间（均值响应 $E[y|\mathbf{x}_0]$）")
ax.fill_between(x_new, y_new - pi_half, y_new + pi_half,
                alpha=0.18, color=C_PI,
                label=r"95% 预测区间（单个新观测 $y_{\mathrm{new}}$）")
# 标注宽度差异
x_ann = 8.5
ci_w = 2 * t_crit * s * np.sqrt(XtX_inv[0,0] + 2*XtX_inv[0,1]*x_ann + XtX_inv[1,1]*x_ann**2)
pi_w = 2 * t_crit * s * np.sqrt(1 + XtX_inv[0,0] + 2*XtX_inv[0,1]*x_ann + XtX_inv[1,1]*x_ann**2)
y_ann = beta_hat[0] + beta_hat[1] * x_ann
ax.annotate("", xy=(x_ann + 0.15, y_ann + ci_w/2),
            xytext=(x_ann + 0.15, y_ann - ci_w/2),
            arrowprops=dict(arrowstyle="<->", color=C_CI, lw=1.5))
ax.text(x_ann + 0.55, y_ann, f"CI\n={ci_w:.2f}",
        fontsize=11, color=C_CI, va="center", fontweight="bold")
ax.annotate("", xy=(x_ann - 0.3, y_ann + pi_w/2),
            xytext=(x_ann - 0.3, y_ann - pi_w/2),
            arrowprops=dict(arrowstyle="<->", color=C_PI, lw=1.5))
ax.text(x_ann - 1.1, y_ann, f"PI\n={pi_w:.2f}",
        fontsize=11, color=C_PI, va="center", fontweight="bold")
ax.set_xlabel("$x$", fontsize=13)
ax.set_ylabel("$y$", fontsize=13)
ax.legend(fontsize=11, loc="upper left")
ax.set_title("(a) 95% 置信区间 vs 预测区间\n"
             "CI 捕获均值响应，PI 捕获单个新观测（更宽）",
             fontsize=13, pad=8)

# ── 面板(b): CI/PI 半宽随 x 的变化（喇叭口效应） ────────────────────────────
ax = axes[1]
ax.plot(x_new, ci_half, color=C_CI, lw=2.2, label=r"CI 半宽：$t\cdot s\sqrt{h(\mathbf{x}_0)}$")
ax.plot(x_new, pi_half, color=C_PI, lw=2.2, label=r"PI 半宽：$t\cdot s\sqrt{1+h(\mathbf{x}_0)}$")
ax.fill_between(x_new, ci_half, pi_half, alpha=0.15, color=C_GRAY,
                label=r"额外不确定性 $t\cdot s$（个体随机噪声 $\sigma^2$）")
x_bar = x.mean()
ax.axvline(x_bar, color=C_GRAY, ls="--", lw=1.5, label=f"$\\bar{{x}}={x_bar:.2f}$（最窄处）")
ax.set_xlabel("$x_0$（新预测点）", fontsize=13)
ax.set_ylabel("区间半宽", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(b) 区间半宽随预测点 $x_0$ 的变化\n"
             r"距 $\bar{x}$ 越远，两种区间均变宽（喇叭口效应）",
             fontsize=13, pad=8)
ax.text(x_bar + 0.2, ci_half.min() + 0.05,
        r"最窄点位于 $x_0 = \bar{x}$",
        fontsize=11, color=C_GRAY,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_GRAY, alpha=0.9))

fig.suptitle("置信区间与预测区间：CI 仅含参数不确定性，PI 额外含个体随机噪声",
             fontsize=14, fontweight="bold", y=1.01)
save_fig(fig, __file__, "fig4_1_02_ci_pi")
