"""
图 3.2.1  高斯朴素贝叶斯的概率视角：类条件分布与后验决策边界
对应节次：3.2 贝叶斯分类方法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_2_01_gaussian_nb.py
输出路径：public/figures/ch03/fig3_2_01_gaussian_nb.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from matplotlib.patches import Ellipse

apply_style()

rng = np.random.default_rng(42)

# ── Panel (a) data: 1D class-conditional Gaussians + posterior ────────────────
mu0, sigma0 = 36.8, 0.35   # 健康体温 ~ N(36.8, 0.35)
mu1, sigma1 = 38.7, 0.55   # 感染体温 ~ N(38.7, 0.55)
pi0, pi1 = 0.6, 0.4        # 类先验

x_range = np.linspace(35.5, 40.5, 800)
p_x_given_0 = norm.pdf(x_range, mu0, sigma0)
p_x_given_1 = norm.pdf(x_range, mu1, sigma1)
p_x = pi0 * p_x_given_0 + pi1 * p_x_given_1
posterior_1 = (pi1 * p_x_given_1) / p_x

# ── Panel (b) data: 2D two-class Gaussian dataset ─────────────────────────────
n_per_class = 80
X0 = rng.multivariate_normal([37.0, 6.5], [[0.12, 0.05], [0.05, 0.50]], n_per_class)
X1 = rng.multivariate_normal([38.8, 11.5], [[0.25, 0.08], [0.08, 0.80]], n_per_class)
X = np.vstack([X0, X1])
y = np.array([0] * n_per_class + [1] * n_per_class)

clf = GaussianNB()
clf.fit(X, y)

xx, yy = np.meshgrid(np.linspace(35.8, 40.5, 300), np.linspace(3.5, 15.0, 300))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.40)

# ── Panel (a): 1D 类条件密度 + 后验概率 ──────────────────────────────────────
ax = axes[0]
ax.fill_between(x_range, p_x_given_0 * pi0, alpha=0.25, color=COLORS["blue"])
ax.fill_between(x_range, p_x_given_1 * pi1, alpha=0.25, color=COLORS["red"])
ax.plot(x_range, p_x_given_0 * pi0, color=COLORS["blue"], lw=2.4,
        label=r"$\pi_0\,p(x_1\mid y=0)$（健康，$\pi_0=0.6$）")
ax.plot(x_range, p_x_given_1 * pi1, color=COLORS["red"], lw=2.4,
        label=r"$\pi_1\,p(x_1\mid y=1)$（感染，$\pi_1=0.4$）")

ax2 = ax.twinx()
ax2.plot(x_range, posterior_1, color=COLORS["teal"], lw=2.2, ls="--",
         label=r"后验 $P(y=1 \mid x_1)$（右轴）")
ax2.set_ylabel(r"后验概率 $P(y=1 \mid x_1)$", fontsize=12, color=COLORS["teal"])
ax2.tick_params(axis="y", labelcolor=COLORS["teal"])
ax2.set_ylim(-0.05, 1.15)
ax2.axhline(0.5, color=COLORS["teal"], lw=1.2, ls=":", alpha=0.6)

# 决策边界（posterior = 0.5）
db = x_range[np.argmin(np.abs(posterior_1 - 0.5))]
ax.axvline(db, color="#64748b", lw=1.8, ls="--", alpha=0.8,
           label=f"决策边界 $x_1^*≈{db:.2f}$°C")

ax.set_xlabel("体温 $x_1$（°C）", fontsize=13)
ax.set_ylabel("加权类条件密度 $\\pi_k \\cdot p(x_1 \\mid y=k)$", fontsize=12)
ax.set_title("(a) 1D 类条件密度与后验概率\n"
             "贝叶斯决策：选择后验最大的类", fontsize=13, pad=6)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper left",
          labelspacing=0.35)
ax.set_xlim(35.5, 40.5)

# ── Panel (b): 2D 高斯 NB 决策边界 ────────────────────────────────────────────
ax = axes[1]
cf = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.55)
ax.contour(xx, yy, Z, levels=[0.5], colors=["#1e293b"], linewidths=2.0)
plt.colorbar(cf, ax=ax, shrink=0.85, label="$P(y=1 \\mid \\mathbf{x})$")

ax.scatter(X0[:, 0], X0[:, 1], s=30, color=COLORS["blue"],
           edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3, label="类 0（健康）")
ax.scatter(X1[:, 0], X1[:, 1], s=30, color=COLORS["red"],
           edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3, label="类 1（感染）")

# 标注各类的均值 ± 1σ 椭圆（轴对齐，体现独立假设）
for k, (mean, std_arr, color) in enumerate([
    (clf.theta_[0], np.sqrt(clf.var_[0]), COLORS["blue"]),
    (clf.theta_[1], np.sqrt(clf.var_[1]), COLORS["red"]),
]):
    for n_sigma in [1, 2]:
        ell = Ellipse(xy=mean, width=2 * n_sigma * std_arr[0],
                      height=2 * n_sigma * std_arr[1],
                      angle=0, edgecolor=color, facecolor="none",
                      linewidth=1.6, linestyle="--", alpha=0.7, zorder=4)
        ax.add_patch(ell)
    ax.scatter(*mean, s=80, color=color, marker="*", zorder=5, edgecolors="white", linewidths=1)

ax.set_xlabel("体温 $x_1$（°C）", fontsize=13)
ax.set_ylabel(r"白细胞计数 $x_2$（$\times 10^9$/L）", fontsize=13)
ax.set_title("(b) 高斯朴素贝叶斯的 2D 决策边界\n"
             "虚线椭圆：轴对齐（独立假设），黑线：决策边界", fontsize=13, pad=6)
ax.legend(fontsize=11, loc="upper left", labelspacing=0.35)
ax.set_xlim(35.8, 40.5)
ax.set_ylim(3.5, 15.0)

fig.suptitle(
    "高斯朴素贝叶斯：从类条件密度到后验决策\n"
    r"$h_{MAP}(\mathbf{x}) = \arg\max_k\; P(y=k)\prod_j P(x_j\mid y=k)$",
    fontsize=13, y=1.02)

save_fig(fig, __file__, "fig3_2_01_gaussian_nb")
