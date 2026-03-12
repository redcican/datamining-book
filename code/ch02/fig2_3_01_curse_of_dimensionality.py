"""
图 2.3.1  维度灾难：单位超球体积收缩与高维空间距离集中效应
对应节次：2.3 数据规约方法
运行方式：python code/ch02/fig2_3_01_curse_of_dimensionality.py
输出路径：public/figures/ch02/fig2_3_01_curse_of_dimensionality.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.spatial.distance import pdist
from scipy import stats

apply_style()

C_BLUE   = "#2563eb"
C_GREEN  = "#16a34a"
C_RED    = "#dc2626"
C_PURPLE = "#7c3aed"
C_GRAY   = "#64748b"

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(wspace=0.36)

# ── Panel (a): Volume of d-dimensional unit hypersphere ───────────────────
ax = axes[0]
dims = np.arange(1, 21)
vols = np.pi ** (dims / 2) / gamma(dims / 2 + 1)

ax.plot(dims, vols, color=C_BLUE, lw=2.5, zorder=5,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgewidth=2)
ax.fill_between(dims, vols, alpha=0.12, color=C_BLUE)

# Annotate peak
peak_idx = int(np.argmax(vols))
ax.annotate(
    f"峰值：d = {dims[peak_idx]}，V ≈ {vols[peak_idx]:.2f}",
    xy=(dims[peak_idx], vols[peak_idx]),
    xytext=(dims[peak_idx] + 3.5, vols[peak_idx] - 0.9),
    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.5),
    fontsize=12, color=C_GRAY,
)

# d=20 annotation
ax.annotate(
    f"d = 20：V ≈ {vols[-1]:.4f}",
    xy=(20, vols[-1]),
    xytext=(15.5, 1.2),
    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.5),
    fontsize=12.5, color=C_GRAY,
)

ax.text(16.5, 1.8,
        r"$V_d = \dfrac{\pi^{d/2}}{\Gamma(d/2 + 1)}$",
        fontsize=13, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff",
                  edgecolor=C_BLUE, alpha=0.95))

ax.set_xlabel("维度 $d$", fontsize=13)
ax.set_ylabel("单位超球体积 $V_d$", fontsize=13)
ax.set_title("(a) 单位超球体积随维度衰减", fontsize=13, pad=8)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(-0.2, 6.2)
ax.tick_params(labelsize=11)

# ── Panel (b): Distance concentration (pairwise distances KDE) ─────────────
ax = axes[1]
rng = np.random.default_rng(2024)
n_pts = 400

cases = [
    (2,   C_BLUE,   "d = 2"),
    (10,  C_GREEN,  "d = 10"),
    (50,  C_RED,    "d = 50"),
]

cv_info = []
for d, color, label in cases:
    pts = rng.uniform(0, 1, (n_pts, d))
    # Use a fixed subset for reproducibility
    sub = pts[:150]
    dists = pdist(sub)  # all pairwise distances

    kde = stats.gaussian_kde(dists, bw_method=0.18)
    xs = np.linspace(dists.min() * 0.92, dists.max() * 1.04, 300)
    ax.plot(xs, kde(xs), color=color, lw=2.2, label=label, zorder=4)
    ax.axvline(dists.mean(), color=color, lw=1.2, ls=":", alpha=0.65, zorder=3)

    cv = dists.std() / dists.mean()
    cv_info.append((label, cv))

# CV annotation box (no LaTeX in label, just text)
cv_lines = "变异系数 (std/mean):\n"
for lbl, cv in cv_info:
    cv_lines += f"  {lbl}:  {cv:.4f}\n"
cv_lines = cv_lines.rstrip()

ax.text(0.03, 0.97, cv_lines,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cbd5e1", alpha=0.95))

ax.set_xlabel("欧氏距离", fontsize=13)
ax.set_ylabel("概率密度", fontsize=13)
ax.set_title("(b) 高维空间距离集中效应（n = 400 均匀随机点）", fontsize=13, pad=8)
ax.legend(fontsize=12, framealpha=0.88)
ax.tick_params(labelsize=11)

# ── Global title & caption ────────────────────────────────────────────────
fig.suptitle("维度灾难：超球体积收缩与距离集中效应",
             fontsize=16, y=1.01)
fig.text(
    0.5, -0.03,
    '左图：d 维单位超球体积在 d ≈ 5 达峰后迅速趋近于零，高维空间近乎空无一物。'
    '右图：均匀分布点对间欧氏距离的核密度估计（n=400），'
    '维度升高时分布高度集中（变异系数趋近于 0），最近邻与最远邻距离趋于相等。',
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_3_01_curse_of_dimensionality")
