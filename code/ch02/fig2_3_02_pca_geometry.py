"""
图 2.3.2  主成分分析几何直觉：PC 方向、投影压缩、碎石图、方差解释率
对应节次：2.3 数据规约方法
运行方式：python code/ch02/fig2_3_02_pca_geometry.py
输出路径：public/figures/ch02/fig2_3_02_pca_geometry.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

apply_style()

C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_GREEN  = "#16a34a"
C_PURPLE = "#7c3aed"
C_GRAY   = "#94a3b8"
C_DARK   = "#1e293b"

rng = np.random.default_rng(2024)

# ── Generate correlated 2D data ────────────────────────────────────────────
n = 250
cov_mat = [[4.0, 2.8], [2.8, 2.2]]
X = rng.multivariate_normal([0, 0], cov_mat, n)

# Manual PCA
C = X.T @ X / n
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]   # columns are PC directions
pc1, pc2 = eigvecs[:, 0], eigvecs[:, 1]

# PCA-projected coordinates
X_pca = X @ eigvecs   # shape (n, 2): PC1 scores, PC2 scores

# ── High-dimensional dataset for scree/EVR panels ─────────────────────────
d_hd = 20
# Data with decaying eigenspectrum (realistic)
true_comps = rng.normal(0, 1, (500, d_hd))
scales = np.array([12 / (k + 1) ** 1.2 for k in range(d_hd)])
X_hd = true_comps * scales
X_hd -= X_hd.mean(axis=0)
C_hd = X_hd.T @ X_hd / len(X_hd)
eigvals_hd, _ = np.linalg.eigh(C_hd)
eigvals_hd = eigvals_hd[::-1]  # descending
evr_hd = eigvals_hd / eigvals_hd.sum()
cumevr_hd = np.cumsum(evr_hd)

# ── Layout ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.subplots_adjust(hspace=0.42, wspace=0.34)

# ── Panel (a): Original data + PC1/PC2 arrows ─────────────────────────────
ax = axes[0, 0]
ax.scatter(X[:, 0], X[:, 1], alpha=0.35, s=18, color=C_GRAY, zorder=2)

# PC arrows (scaled by 2*sqrt(eigenvalue) for visibility)
scale1 = 2.2 * np.sqrt(eigvals[0])
scale2 = 2.2 * np.sqrt(eigvals[1])

ax.annotate("", xy=(pc1[0] * scale1, pc1[1] * scale1),
            xytext=(-pc1[0] * scale1, -pc1[1] * scale1),
            arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.5,
                            mutation_scale=20), zorder=5)
ax.annotate("", xy=(pc2[0] * scale2, pc2[1] * scale2),
            xytext=(-pc2[0] * scale2, -pc2[1] * scale2),
            arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.0,
                            mutation_scale=16), zorder=5)

# Labels for arrows
ax.text(pc1[0] * scale1 + 0.15, pc1[1] * scale1 + 0.15,
        "PC1", fontsize=12, color=C_RED, fontweight="bold")
ax.text(pc2[0] * scale2 * 1.2, pc2[1] * scale2 * 1.2,
        "PC2", fontsize=12, color=C_BLUE, fontweight="bold")

ax.text(0.05, 0.96,
        f"$\\lambda_1$ = {eigvals[0]:.2f} （方差 {eigvals[0]/(eigvals[0]+eigvals[1])*100:.1f}%）\n"
        f"$\\lambda_2$ = {eigvals[1]:.2f} （方差 {eigvals[1]/(eigvals[0]+eigvals[1])*100:.1f}%）",
        transform=ax.transAxes, ha="left", va="top", fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cbd5e1", alpha=0.9))

ax.set_title("(a) 原始数据与主成分方向", fontsize=13, pad=6)
ax.set_xlabel("特征 $x_1$", fontsize=12)
ax.set_ylabel("特征 $x_2$", fontsize=12)
ax.set_aspect("equal")
ax.tick_params(labelsize=10)

# ── Panel (b): Projection onto PC1 (compression illustration) ─────────────
ax = axes[0, 1]

# Show a subset for clarity
n_show = 60
idx_show = rng.choice(n, size=n_show, replace=False)
X_show = X[idx_show]

# Project onto PC1: reconstruction in original space
proj_scalar = X_show @ pc1           # 1D scores
X_reconstructed = np.outer(proj_scalar, pc1)  # back to 2D

# Original points (gray)
ax.scatter(X_show[:, 0], X_show[:, 1], s=22, color=C_GRAY,
           alpha=0.5, zorder=3, label="原始点")

# Reconstructed points (on PC1 line, red)
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
           s=30, color=C_RED, alpha=0.85, zorder=5, label="投影点（PC1）")

# Residual lines
for i in range(n_show):
    ax.plot([X_show[i, 0], X_reconstructed[i, 0]],
            [X_show[i, 1], X_reconstructed[i, 1]],
            color="#94a3b8", lw=0.6, alpha=0.5, zorder=2)

# PC1 axis line
t = np.linspace(-5.5, 5.5, 100)
ax.plot(t * pc1[0], t * pc1[1], color=C_RED, lw=1.5, ls="--",
        alpha=0.5, zorder=1)

ax.set_title("(b) 投影到 PC1：1 维数据压缩", fontsize=13, pad=6)
ax.set_xlabel("特征 $x_1$", fontsize=12)
ax.set_ylabel("特征 $x_2$", fontsize=12)
ax.set_aspect("equal")
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=10)

# Residual annotation
ax.text(0.97, 0.04,
        "灰线为重建误差\n（丢弃 PC2 信息）",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#94a3b8", alpha=0.9))

# ── Panel (c): Scree plot for high-D dataset ──────────────────────────────
ax = axes[1, 0]
k_show = 15
ks = np.arange(1, k_show + 1)

# Bar chart (eigenvalues)
colors_bar = [C_BLUE if i < 5 else "#cbd5e1" for i in range(k_show)]
bars = ax.bar(ks, eigvals_hd[:k_show], color=colors_bar,
              edgecolor="white", linewidth=0.6, zorder=3)

# Cumulative variance line (second y-axis)
ax2 = ax.twinx()
ax2.plot(ks, cumevr_hd[:k_show] * 100, color=C_RED, lw=2.2,
         marker="o", markersize=5, markerfacecolor="white",
         markeredgewidth=2, zorder=5)
ax2.axhline(95, color=C_RED, lw=1.0, ls=":", alpha=0.6)
ax2.text(k_show + 0.1, 94, "95%", fontsize=12, color=C_RED, va="top")

# Mark 95% threshold
idx_95 = int(np.searchsorted(cumevr_hd, 0.95))
ax2.axvline(idx_95 + 1, color="#ea580c", lw=1.2, ls="--", alpha=0.7)
ax2.text(idx_95 + 1.2, 30, f"前 {idx_95+1} 个\n成分达 95%",
         fontsize=12.5, color="#ea580c")

ax.set_xlabel("主成分序号", fontsize=12)
ax.set_ylabel("特征值（方差量）", fontsize=12)
ax2.set_ylabel("累计方差解释率 (%)", fontsize=12, color=C_RED)
ax2.tick_params(axis="y", colors=C_RED, labelsize=10)
ax2.set_ylim(0, 108)
ax.set_title("(c) 碎石图（20 维合成数据集）", fontsize=13, pad=6)
ax.tick_params(labelsize=10)

# Legend patches
retained = mpatches.Patch(color=C_BLUE, label="保留成分（前 5 个）")
discarded = mpatches.Patch(color="#cbd5e1", label="丢弃成分")
ax.legend(handles=[retained, discarded], fontsize=12.5, loc="lower right")

# ── Panel (d): Cumulative EVR (full 20 components) ─────────────────────────
ax = axes[1, 1]
ks_all = np.arange(1, d_hd + 1)
ax.plot(ks_all, cumevr_hd * 100, color=C_PURPLE, lw=2.5,
        marker="o", markersize=5, markerfacecolor="white",
        markeredgewidth=2, zorder=5)
ax.fill_between(ks_all, cumevr_hd * 100, alpha=0.12, color=C_PURPLE)

# Horizontal threshold lines
for pct, color in [(80, C_GREEN), (90, "#ea580c"), (95, C_RED)]:
    ax.axhline(pct, color=color, lw=1.2, ls="--", alpha=0.7)
    ax.text(d_hd + 0.2, pct, f"{pct}%", fontsize=12, color=color, va="center")
    k_pct = int(np.searchsorted(cumevr_hd, pct / 100)) + 1
    ax.axvline(k_pct, color=color, lw=1.0, ls=":", alpha=0.5)

ax.set_xlabel("保留主成分数 $K$", fontsize=12)
ax.set_ylabel("累计方差解释率 (%)", fontsize=12)
ax.set_title("(d) 累计方差解释率曲线", fontsize=13, pad=6)
ax.set_xlim(0.5, d_hd + 0.5)
ax.set_ylim(0, 105)
ax.tick_params(labelsize=10)

fig.suptitle("主成分分析（PCA）几何直觉与方差分析", fontsize=16, y=1.01)
fig.text(
    0.5, -0.02,
    "数据：(a)(b) 采用二维合成相关数据（n=250），展示 PC 方向与 1 维投影压缩；"
    "(c)(d) 采用 20 维合成数据（n=500），展示碎石图与累计方差解释率，"
    "通常以 95% 阈值确定保留成分数。",
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_3_02_pca_geometry")
