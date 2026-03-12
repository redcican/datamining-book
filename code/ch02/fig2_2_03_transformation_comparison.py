"""
图 2.2.3  数值变量分布变换对比（原始 / Log / Sqrt / Box-Cox）
对应节次：2.2 数据集成与数据变换
运行方式：python code/ch02/fig2_2_03_transformation_comparison.py
输出路径：public/figures/ch02/fig2_2_03_transformation_comparison.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

apply_style()

rng = np.random.default_rng(2024)

# ── Generate right-skewed income data (log-normal like) ───────────────────
n = 600
# Base: log-normal with strong right skew (like household income in CNY/month)
raw = rng.lognormal(mean=8.5, sigma=0.85, size=n)   # median ~ 4919, mean >> median
raw = raw[(raw > 0) & (raw < 1e6)]                   # clip extremes

# ── Apply transformations ─────────────────────────────────────────────────
log_t  = np.log1p(raw)                               # log(x+1) — handles x≥0
sqrt_t = np.sqrt(raw)

# Box-Cox (requires x > 0, already satisfied)
bc_t, lam = stats.boxcox(raw)

def skewness(x):
    m = x.mean(); s = x.std()
    return np.mean(((x - m) / s) ** 3)

transforms = [
    ("(a) 原始分布（偏态）",  raw,   "#dc2626", "upper right"),
    ("(b) Log(1+x) 变换",    log_t, "#2563eb", "upper left"),
    ("(c) Sqrt(x) 变换",     sqrt_t,"#16a34a", "upper right"),
    (f"(d) Box-Cox ($\\lambda$={lam:.2f})", bc_t, "#7c3aed", "upper left"),
]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.48, wspace=0.32)

for ax, (title, data, color, leg_loc) in zip(axes.flat, transforms):
    # Histogram
    ax.hist(data, bins=45, color=color, alpha=0.55, density=True,
            edgecolor="white", linewidth=0.4, label="经验分布")

    # KDE curve
    xmin, xmax = data.min(), data.max()
    xs = np.linspace(xmin, xmax, 300)
    try:
        kde = stats.gaussian_kde(data, bw_method=0.25)
        ax.plot(xs, kde(xs), color=color, lw=2.2, label="KDE")
    except Exception:
        pass

    # Fitted normal overlay (red dashed)
    mu_, s_ = data.mean(), data.std()
    norm_y = stats.norm.pdf(xs, mu_, s_)
    ax.plot(xs, norm_y, color="#dc2626", lw=1.6, ls="--",
            alpha=0.85, label=f"正态拟合\nN({mu_:.1f},{s_:.1f})")

    ax.set_title(title, fontsize=12, pad=6)
    ax.set_ylabel("概率密度", fontsize=12)

    ax.legend(fontsize=12.5, loc=leg_loc, framealpha=0.85)

# Annotate mean/median gap on panel (a)
ax0 = axes[0, 0]
mu0, med0 = raw.mean(), np.median(raw)
ylim0 = ax0.get_ylim()
ax0.axvline(mu0,  color="#0f172a", lw=1.3, ls=":", label=f"均值={mu0:.0f}")
ax0.axvline(med0, color="#64748b", lw=1.3, ls="--", label=f"中位数={med0:.0f}")
ax0.legend(fontsize=12, loc="upper right")

fig.suptitle("数值变量分布变换对比：偏态消除效果（以收入/租金类变量为例）",
             fontsize=14.5, y=1.01)
fig.text(0.5, -0.02,
         "数据：合成的对数正态分布样本（n=600，模拟月收入，单位：元/月）。"
         "红色虚线为正态拟合曲线；偏度越接近 0，分布越接近正态。"
         f"Box-Cox（最优 λ={lam:.2f}）偏度最小，效果最优。",
         ha="center", fontsize=12.5, color="#64748b", style="italic")

save_fig(fig, __file__, "fig2_2_03_transformation_comparison")
