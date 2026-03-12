"""
图 2.5.1  四种数值归一化/标准化方法对比
对应节次：2.5 数据标准化与归一化
运行方式：MPLBACKEND=Agg python code/ch02/fig2_5_01_normalization_methods.py
输出路径：public/figures/ch02/fig2_5_01_normalization_methods.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

apply_style()

C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_GREEN  = "#16a34a"
C_ORANGE = "#ea580c"
C_PURPLE = "#7c3aed"
C_GRAY   = "#94a3b8"
C_DARK   = "#1e293b"

rng = np.random.default_rng(42)

# ── Generate skewed data with outliers ─────────────────────────────────────
n = 300
# Base: right-skewed salary data (unit: thousand yuan/month)
base = rng.exponential(scale=8, size=n) + 5
# Add a few high outliers
outlier_idx = rng.choice(n, 8, replace=False)
base[outlier_idx] += rng.uniform(60, 100, 8)
x_raw = base

# ── Compute transformations ─────────────────────────────────────────────────
# Min-Max
x_min, x_max = x_raw.min(), x_raw.max()
x_minmax = (x_raw - x_min) / (x_max - x_min)

# Z-score (Standard)
mu, sigma = x_raw.mean(), x_raw.std()
x_zscore = (x_raw - mu) / sigma

# Robust (median / IQR)
med = np.median(x_raw)
q25, q75 = np.percentile(x_raw, 25), np.percentile(x_raw, 75)
iqr = q75 - q25
x_robust = (x_raw - med) / iqr

# Max-Abs
x_maxabs = x_raw / np.abs(x_raw).max()

# ── Layout 2×4: raw + 3 methods, each with box and histogram ────────────────
fig = plt.figure(figsize=(17, 8))
fig.subplots_adjust(hspace=0.52, wspace=0.38)

datasets = [
    (x_raw,    "原始数据（月薪）",           "千元",    C_GRAY,   None),
    (x_minmax, "Min-Max 归一化",             "[0, 1]",  C_BLUE,   (0, 1)),
    (x_zscore, "Z-score 标准化",             "标准差单位", C_GREEN, None),
    (x_robust, "鲁棒标准化（Robust Scaler）","IQR 单位",C_ORANGE, None),
]

for col, (x, title, unit, color, xlim) in enumerate(datasets):
    # Top row: histogram + KDE
    ax_top = fig.add_subplot(2, 4, col + 1)
    counts, bins, _ = ax_top.hist(x, bins=35, color=color, alpha=0.65,
                                  edgecolor="white", lw=0.5, density=True)
    # KDE
    kde_x = np.linspace(x.min() - 0.05 * (x.max() - x.min()),
                        x.max() + 0.05 * (x.max() - x.min()), 300)
    try:
        kde = stats.gaussian_kde(x, bw_method=0.25)
        ax_top.plot(kde_x, kde(kde_x), color=color, lw=2.2)
    except Exception:
        pass
    ax_top.set_title(title, fontsize=12.5, pad=5)
    ax_top.set_xlabel(unit, fontsize=12)
    ax_top.set_ylabel("密度", fontsize=12)
    ax_top.tick_params(labelsize=9)
    if xlim:
        ax_top.set_xlim(xlim)

    # Annotate stats
    stats_text = (f"均值={x.mean():.2f}\n"
                  f"标准差={x.std():.2f}\n"
                  f"最大值={x.max():.1f}")
    ax_top.text(0.97, 0.97, stats_text, transform=ax_top.transAxes,
                fontsize=12.5, ha="right", va="top", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cbd5e1", alpha=0.9))

    # Bottom row: boxplot
    ax_bot = fig.add_subplot(2, 4, col + 5)
    bp = ax_bot.boxplot(x, vert=False, patch_artist=True,
                        flierprops=dict(marker=".", color=C_RED,
                                        markersize=4, alpha=0.5),
                        medianprops=dict(color="white", lw=2.5),
                        boxprops=dict(facecolor=color, alpha=0.7),
                        whiskerprops=dict(color=C_DARK),
                        capprops=dict(color=C_DARK))
    ax_bot.set_xlabel(unit, fontsize=12)
    ax_bot.set_yticks([])
    ax_bot.tick_params(labelsize=9)
    if xlim:
        ax_bot.set_xlim(-0.05, 1.05)

    # Mark outlier threshold for Z-score panel
    if col == 2:
        for v in [-2, 2]:
            ax_top.axvline(v, color=C_RED, lw=1.2, ls="--", alpha=0.7)
            ax_bot.axvline(v, color=C_RED, lw=1.2, ls="--", alpha=0.7)
        ax_top.text(2.05, ax_top.get_ylim()[1] * 0.9, "|z|=2",
                    fontsize=12, color=C_RED)

fig.suptitle("四种数值标准化方法对比：以月薪分布为例（含高薪离群点）",
             fontsize=14, y=1.01)
fig.text(
    0.5, -0.03,
    "原始月薪数据呈右偏分布，含少量高薪离群点。"
    "Min-Max 将值域压缩至 [0,1]，但对离群点敏感；"
    "Z-score 以均值为中心、标准差为尺度；"
    "Robust 用中位数和 IQR 替代均值和标准差，对离群点具有鲁棒性。",
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_5_01_normalization_methods")
