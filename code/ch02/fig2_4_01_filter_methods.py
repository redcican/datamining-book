"""
图 2.4.1  过滤式特征选择：相关性矩阵、互信息排名与线性/非线性检测
对应节次：2.4 特征选择与特征工程
运行方式：python code/ch02/fig2_4_01_filter_methods.py
输出路径：public/figures/ch02/fig2_4_01_filter_methods.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler

apply_style()

C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_GREEN  = "#16a34a"
C_ORANGE = "#ea580c"
C_PURPLE = "#7c3aed"
C_GRAY   = "#94a3b8"
C_DARK   = "#1e293b"

rng = np.random.default_rng(42)
n = 500

# ── Generate dataset: y is QUADRATIC in z1, linear in z2 ──────────────────
# This means x1/x2 (measuring z1) have HIGH MI but near-ZERO |r| with y
# x3/x4 (measuring z2) have HIGH MI AND HIGH |r|
# x5 (measuring z1^2) has HIGH MI AND HIGH |r|
z1 = rng.normal(0, 1, n)
z2 = rng.normal(0, 1, n)

X = np.zeros((n, 10))
X[:, 0] = z1 + rng.normal(0, 0.15, n)            # x1: z1 (nonlinear rel to y)
X[:, 1] = z1 * 0.85 + rng.normal(0, 0.28, n)     # x2: correlated with x1
X[:, 2] = z2 + rng.normal(0, 0.15, n)            # x3: z2 (linear rel to y)
X[:, 3] = z2 * 0.72 + rng.normal(0, 0.38, n)     # x4: correlated with x3
X[:, 4] = z1**2 - 1 + rng.normal(0, 0.28, n)     # x5: z1^2 (direct signal)
X[:, 5] = z2 * 0.38 + rng.normal(0, 0.82, n)     # x6: weak z2 signal
X[:, 6] = rng.normal(0, 1, n)                     # x7: pure noise
X[:, 7] = rng.normal(0, 1, n)                     # x8: pure noise
X[:, 8] = rng.normal(0, 1, n)                     # x9: pure noise
X[:, 9] = z1 * 0.12 + rng.normal(0, 0.98, n)     # x10: very weak z1 signal

# Target: quadratic in z1, linear in z2
y = 2.0 * z1**2 + 1.5 * z2 + rng.normal(0, 0.4, n)

X_sc = StandardScaler().fit_transform(X)
feat_labels = [f"x{i+1}" for i in range(10)]

# ── Filter statistics ──────────────────────────────────────────────────────
corr_mat  = np.corrcoef(X_sc.T)
mi        = mutual_info_regression(X_sc, y, random_state=42, n_neighbors=7)
f_stat, _ = f_regression(X_sc, y)
pearson_r = np.array([abs(np.corrcoef(X_sc[:, i], y)[0, 1]) for i in range(10)])

mi_norm = mi / mi.max()
f_norm  = f_stat / f_stat.max()

# ── Layout 2×2 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.48, wspace=0.40)

# ── (a) Correlation heatmap ─────────────────────────────────────────────────
ax = axes[0, 0]
im = ax.imshow(np.abs(corr_mat), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(feat_labels, fontsize=12.5)
ax.set_yticklabels(feat_labels, fontsize=12.5)

for i in range(10):
    for j in range(10):
        v = corr_mat[i, j]
        c = "white" if abs(v) > 0.65 else C_DARK
        ax.text(j, i, f"{abs(v):.2f}", ha="center", va="center",
                fontsize=12.5, color=c)

# Highlight correlated pairs
for (r1, c1, r2, c2), ec in [((0, 1, 1, 0), C_RED), ((2, 3, 3, 2), C_ORANGE)]:
    for (row, col) in [(min(r1,c1), min(r1,c1)), (min(r2,c2), min(r2,c2))]:
        pass
for ((i, j), ec) in [((0, 1), C_RED), ((2, 3), C_ORANGE)]:
    for ii, jj in [(i, j), (j, i)]:
        rect = plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False,
                              edgecolor=ec, lw=2.5, zorder=5)
        ax.add_patch(rect)

plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
ax.set_title("(a) 特征相关性矩阵（|Pearson r|）", fontsize=13, pad=6)
ax.text(0.5, -0.13, "红框 (x1,x2)：高冗余对；橙框 (x3,x4)：中等冗余对",
        transform=ax.transAxes, ha="center", fontsize=12, color="#475569")
ax.tick_params(labelsize=9.5)

# ── (b) MI vs |Pearson r| scatter ──────────────────────────────────────────
ax = axes[0, 1]
scatter_colors = [C_BLUE, C_BLUE, C_GREEN, C_GREEN,
                  C_PURPLE, C_ORANGE, C_GRAY, C_GRAY, C_GRAY, C_GRAY]
ax.scatter(pearson_r, mi_norm, s=130, c=scatter_colors, zorder=5,
           edgecolors=C_DARK, linewidths=1.3, alpha=0.92)

offsets = [(6, 5), (-22, 5), (6, 5), (6, -14),
           (6, 5), (6, 5), (6, 5), (6, -14), (-22, 5), (6, 5)]
for i in range(10):
    dx, dy = offsets[i]
    ax.annotate(f"x{i+1}", (pearson_r[i], mi_norm[i]), fontsize=12,
                xytext=(dx, dy), textcoords="offset points", color=C_DARK)

ax.axhline(0.40, color=C_GRAY, lw=1, ls=":", alpha=0.65)
ax.axvline(0.25, color=C_GRAY, lw=1, ls=":", alpha=0.65)

box_kw = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88)
ax.text(0.02, 0.98, "高MI、低|r|\n（非线性特征）",
        transform=ax.transAxes, fontsize=12.5, va="top", color=C_BLUE,
        bbox=dict(**box_kw, edgecolor=C_BLUE))
ax.text(0.56, 0.98, "高MI、高|r|\n（线性/混合特征）",
        transform=ax.transAxes, fontsize=12.5, va="top", color=C_GREEN,
        bbox=dict(**box_kw, edgecolor=C_GREEN))
ax.text(0.56, 0.08, "低MI、低|r|\n（噪声特征）",
        transform=ax.transAxes, fontsize=12.5, va="bottom", color=C_GRAY,
        bbox=dict(**box_kw, edgecolor=C_GRAY))

ax.set_xlabel("|Pearson r|（与目标变量的线性相关系数）", fontsize=12)
ax.set_ylabel("归一化互信息得分", fontsize=12)
ax.set_title("(b) 互信息 vs |Pearson r|：检测非线性特征", fontsize=13, pad=6)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.08, 1.12)
ax.tick_params(labelsize=10)

# ── (c) MI score bar chart (sorted) ────────────────────────────────────────
ax = axes[1, 0]
thresh = 0.30
sort_idx = np.argsort(mi_norm)[::-1]
bar_colors = [C_BLUE if mi_norm[i] >= thresh else C_GRAY for i in sort_idx]
ax.bar(range(10), mi_norm[sort_idx], color=bar_colors,
       edgecolor="white", lw=0.8, zorder=3)
ax.set_xticks(range(10))
ax.set_xticklabels([feat_labels[i] for i in sort_idx], fontsize=12)
ax.axhline(thresh, color=C_RED, lw=1.8, ls="--", alpha=0.85)
ax.text(9.5, thresh + 0.025, f"阈值 {thresh:.2f}",
        fontsize=12.5, color=C_RED, va="bottom", ha="right")

retained = mpatches.Patch(color=C_BLUE, label="保留（MI >= 0.30）")
removed  = mpatches.Patch(color=C_GRAY, label="过滤（MI < 0.30）")
ax.legend(handles=[retained, removed], fontsize=12, loc="upper right")
ax.set_xlabel("特征（按互信息得分降序排列）", fontsize=12)
ax.set_ylabel("归一化互信息得分", fontsize=12)
ax.set_title("(c) 互信息过滤排名", fontsize=13, pad=6)
ax.tick_params(labelsize=10)

# ── (d) MI vs F-statistic normalized score comparison ──────────────────────
ax = axes[1, 1]
w = 0.38
x_pos = np.arange(10)
ax.bar(x_pos - w / 2, mi_norm, w, color=C_BLUE, alpha=0.85,
       label="互信息（MI）", edgecolor="white")
ax.bar(x_pos + w / 2, f_norm,  w, color=C_ORANGE, alpha=0.85,
       label="F 统计量", edgecolor="white")
ax.set_xticks(x_pos)
ax.set_xticklabels(feat_labels, fontsize=12.5)
ax.set_ylabel("归一化得分", fontsize=12)
ax.set_title("(d) 互信息与 F 统计量得分对比", fontsize=13, pad=6)
ax.legend(fontsize=12.5, loc="upper right")
ax.tick_params(labelsize=10)

# Annotate x1/x2 disagreement
ax.annotate("x1/x2: MI 高\nF 统计量低\n（非线性）",
            xy=(0, mi_norm[0]),
            xytext=(3, 0.55),
            arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.5),
            fontsize=12, color=C_PURPLE,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=C_PURPLE, alpha=0.85))

fig.suptitle("过滤式特征选择：相关性、互信息与 F 统计量", fontsize=15, y=1.01)
fig.text(
    0.5, -0.02,
    "合成数据集（n=500，10 个特征）。目标 y = 2z1² + 1.5z2 + ε（二次 + 线性）。"
    "x1、x2 测量 z1（高 MI、低 |r|，因为 y 与 z1 为非线性关系），"
    "x3、x4 测量 z2（高 MI 且高 |r|），x5 = z1²（高 MI 且高 |r|），x7-x9 为噪声。",
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_4_01_filter_methods")
