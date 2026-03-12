"""
图 2.1.3  Z-score 与 IQR 异常值检测方法对比
对应节次：2.1 数据清洗技术
运行方式：python code/ch02/fig2_1_03_outlier_methods.py
输出路径：public/figures/ch02/fig2_1_03_outlier_methods.png
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

rng = np.random.default_rng(2024)

# ── Dataset: N(50, 6²) + injected outliers ────────────────────────────────
normal_pts = rng.normal(50, 6, 120)
outliers   = np.array([18.0, 21.5, 78.0, 82.5, 85.0])
data = np.concatenate([normal_pts, outliers])

mu, sigma = data.mean(), data.std()
Q1, Q3 = np.percentile(data, 25), np.percentile(data, 75)
IQR = Q3 - Q1
lo_z, hi_z = mu - 3 * sigma, mu + 3 * sigma
lo_iqr, hi_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

z_outliers   = data[np.abs((data - mu) / sigma) > 3]
iqr_outliers = data[(data < lo_iqr) | (data > hi_iqr)]

C_NORM = "#2563eb"
C_OUT  = "#dc2626"
C_LINE = "#dc2626"
C_BOX  = "#0891b2"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(wspace=0.38)

# ── Panel (a): Z-score detection ──────────────────────────────────────────
x_range = np.linspace(0, 100, 400)
kde = stats.gaussian_kde(normal_pts, bw_method=0.4)
ax1.fill_between(x_range, kde(x_range), alpha=0.25, color=C_NORM, label="正常数据分布（KDE）")
ax1.plot(x_range, kde(x_range), color=C_NORM, lw=2)

# Rug plot of data points (strip at y = -0.003)
norm_in   = data[(data >= lo_z) & (data <= hi_z)]
norm_out  = data[(data < lo_z) | (data > hi_z)]
ax1.scatter(norm_in,  np.full_like(norm_in,  -0.003), s=22, color=C_NORM,
            alpha=0.6, zorder=4, clip_on=False)
ax1.scatter(norm_out, np.full_like(norm_out, -0.003), s=80, marker="*",
            color=C_OUT, zorder=5, clip_on=False, label="Z-score 异常点")

# Boundary lines
ax1.axvline(lo_z, color=C_LINE, lw=1.6, ls="--", label=f"$\\mu \\pm 3\\sigma$")
ax1.axvline(hi_z, color=C_LINE, lw=1.6, ls="--")
ax1.axvline(mu,   color="#64748b", lw=1.2, ls=":", label=f"$\\mu={mu:.1f}$")

# Shade ±3σ region
ax1.axvspan(lo_z, hi_z, alpha=0.08, color=C_NORM)

# Annotate outliers — stagger y to avoid overlap
left_out  = sorted([v for v in norm_out if v < mu])
right_out = sorted([v for v in norm_out if v > mu])
left_ys   = [0.026, 0.015]
right_ys  = [0.026, 0.016, 0.007]

for v, ty in zip(left_out, left_ys):
    z = (v - mu) / sigma
    ax1.annotate(f"z={z:.1f}", xy=(v, -0.003),
                 xytext=(v + 4, ty),
                 fontsize=12, color=C_OUT,
                 arrowprops=dict(arrowstyle="-", color=C_OUT, lw=0.8))

for v, ty in zip(right_out, right_ys):
    z = (v - mu) / sigma
    ax1.annotate(f"z={z:.1f}", xy=(v, -0.003),
                 xytext=(v - 9, ty),
                 fontsize=12, color=C_OUT,
                 arrowprops=dict(arrowstyle="-", color=C_OUT, lw=0.8))

ax1.set_title("(a) Z-score 检测法", fontsize=13)
ax1.set_xlabel("特征值 $x$", fontsize=12)
ax1.set_ylabel("概率密度", fontsize=12)
ax1.set_xlim(0, 100)
ax1.set_ylim(-0.01, None)
ax1.legend(fontsize=11, loc="upper left",
           labelspacing=0.3, handlelength=1.5,
           handletextpad=0.5, borderpad=0.4)

# Formula inside
ax1.text(0.97, 0.96,
         "$z_i = \\dfrac{x_i - \\mu}{\\sigma}$\n异常：$|z_i| > 3$",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=12, color="#1e293b",
         bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                   edgecolor="#cbd5e1", alpha=0.95))

# ── Panel (b): IQR detection ──────────────────────────────────────────────
bp = ax2.boxplot(data, vert=True, widths=0.5, patch_artist=True,
                 boxprops=dict(facecolor=C_BOX + "33", edgecolor=C_BOX, linewidth=2),
                 medianprops=dict(color=C_BOX, linewidth=2.5),
                 whiskerprops=dict(color=C_BOX, linewidth=1.8, linestyle="--"),
                 capprops=dict(color=C_BOX, linewidth=2),
                 flierprops=dict(marker="o", color=C_OUT, markersize=9,
                                 markerfacecolor=C_OUT, markeredgecolor=C_OUT),
                 showfliers=True)

# Annotate IQR components
ax2.annotate("", xy=(1.38, Q3), xytext=(1.38, Q1),
             arrowprops=dict(arrowstyle="<->", color=C_BOX, lw=1.6))
ax2.text(1.42, (Q1 + Q3) / 2, f"IQR={IQR:.1f}", va="center",
         fontsize=12.5, color=C_BOX, fontweight="bold")

ax2.axhline(lo_iqr, color=C_LINE, lw=1.2, ls=":", alpha=0.6)
ax2.axhline(hi_iqr, color=C_LINE, lw=1.2, ls=":", alpha=0.6)

ax2.text(1.54, Q1,   f"Q1={Q1:.1f}", va="center", fontsize=12, color=C_BOX)
ax2.text(1.54, Q3,   f"Q3={Q3:.1f}", va="center", fontsize=12, color=C_BOX)
ax2.text(1.54, lo_iqr + 0.5, f"下限={lo_iqr:.1f}", va="bottom",
         fontsize=12.5, color=C_LINE)
ax2.text(1.54, hi_iqr - 0.5, f"上限={hi_iqr:.1f}", va="top",
         fontsize=12.5, color=C_LINE)

ax2.set_title("(b) IQR 四分位距法", fontsize=13)
ax2.set_ylabel("特征值 $x$", fontsize=12)
ax2.set_xticks([])
ax2.set_xlim(0.4, 2.0)

ax2.text(0.97, 0.96,
         "$\\mathrm{IQR} = Q_3 - Q_1$\n"
         "异常：$x < Q_1 - 1.5\\,\\mathrm{IQR}$\n"
         "$\\quad\\quad\\;$ 或 $x > Q_3 + 1.5\\,\\mathrm{IQR}$",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=12, color="#1e293b",
         bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                   edgecolor="#cbd5e1", alpha=0.95))

# Legend for outliers
out_patch = mpatches.Patch(color=C_OUT, label=f"IQR 异常点 (n={len(iqr_outliers)})")
ax2.legend(handles=[out_patch], fontsize=12, loc="upper left")

fig.suptitle("异常值检测：Z-score 法 vs IQR 法对比", fontsize=15, y=1.02)
fig.text(0.5, -0.04,
         f"数据集：N(50, 6²) 正常样本 120 个 + 人工注入异常点 5 个。"
         f"Z-score 检测到 {len(z_outliers)} 个异常（|z|>3），"
         f"IQR 检测到 {len(iqr_outliers)} 个异常（Tukey 围栏）。"
         f"两者结果存在差异，说明所选阈值与分布假设对检测结果有显著影响。",
         ha="center", fontsize=12.5, color="#64748b", style="italic")

save_fig(fig, __file__, "fig2_1_03_outlier_methods")
