"""
图 2.3.3  三种采样策略对比：简单随机采样 / 分层采样 / 系统采样
对应节次：2.3 数据规约方法
运行方式：python code/ch02/fig2_3_03_sampling_methods.py
输出路径：public/figures/ch02/fig2_3_03_sampling_methods.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

C_A = "#2563eb"    # 高收入层（少数群体）
C_B = "#16a34a"    # 中收入层（主体）
C_C = "#ea580c"    # 低收入层（较多）
C_SEL = "#1e293b"  # selected border

rng = np.random.default_rng(42)

# ── Population: 3 strata ────────────────────────────────────────────────────
# Stratum A: high-value users, n=20 (minority group)
nA, nB, nC = 20, 70, 30   # total = 120
N = nA + nB + nC

xA = rng.uniform(7, 10, nA)
yA = rng.uniform(7, 10, nA)
xB = rng.uniform(2, 8, nB)
yB = rng.uniform(2, 8, nB)
xC = rng.uniform(0, 5, nC)
yC = rng.uniform(0, 5, nC)

X_all = np.vstack([np.c_[xA, yA], np.c_[xB, yB], np.c_[xC, yC]])
strata_labels = np.array(["A"] * nA + ["B"] * nB + ["C"] * nC)
colors_all = (
    [C_A] * nA + [C_B] * nB + [C_C] * nC
)

# ── Sampling ─────────────────────────────────────────────────────────────
n_sample = 24  # 20% of 120

# (1) Simple Random Sampling
srs_idx = rng.choice(N, size=n_sample, replace=False)
srs_selected = np.zeros(N, dtype=bool)
srs_selected[srs_idx] = True

# (2) Proportional Stratified Sampling
nA_s = round(n_sample * nA / N)   # 4
nB_s = round(n_sample * nB / N)   # 14
nC_s = n_sample - nA_s - nB_s     # 6
idxA = rng.choice(nA, size=nA_s, replace=False)
idxB = rng.choice(nB, size=nB_s, replace=False)
idxC = rng.choice(nC, size=nC_s, replace=False)
strat_selected = np.zeros(N, dtype=bool)
strat_selected[idxA] = True
strat_selected[nA + idxB] = True
strat_selected[nA + nB + idxC] = True

# (3) Systematic Sampling (sort by x-coordinate, step = N//n_sample)
sort_idx = np.argsort(X_all[:, 0])
step = N // n_sample
sys_idx = sort_idx[::step][:n_sample]
sys_selected = np.zeros(N, dtype=bool)
sys_selected[sys_idx] = True

# ── Shared helper to count stratum representation in sample ────────────────
def stratum_counts(selected):
    sa = selected[:nA].sum()
    sb = selected[nA:nA+nB].sum()
    sc = selected[nA+nB:].sum()
    return sa, sb, sc


def draw_panel(ax, selected, title, subtitle=""):
    """Draw population scatter with selected points highlighted."""
    # Unselected: small, translucent
    mask_not = ~selected
    ax.scatter(X_all[mask_not, 0], X_all[mask_not, 1],
               c=[colors_all[i] for i in range(N) if mask_not[i]],
               s=18, alpha=0.25, zorder=2, edgecolors="none")

    # Selected: larger, solid, dark edge
    for i in range(N):
        if selected[i]:
            ax.scatter(X_all[i, 0], X_all[i, 1],
                       c=[colors_all[i]], s=90, alpha=0.92,
                       edgecolors=C_SEL, linewidths=1.4, zorder=5)

    ax.set_xlim(-0.3, 10.8)
    ax.set_ylim(-0.3, 10.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, pad=6)
    if subtitle:
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=12, color="#475569",
                style="italic")
    ax.set_xlabel("特征 $x_1$", fontsize=12)
    ax.set_ylabel("特征 $x_2$", fontsize=12)
    ax.tick_params(labelsize=10)

    # Stratum count inset
    sa, sb, sc = stratum_counts(selected)
    info = (
        f"样本构成（n={n_sample}）\n"
        f"  A 层：{sa}/{nA}（{sa/nA*100:.0f}%）\n"
        f"  B 层：{sb}/{nB}（{sb/nB*100:.0f}%）\n"
        f"  C 层：{sc}/{nC}（{sc/nC*100:.0f}%）"
    )
    ax.text(0.03, 0.97, info,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cbd5e1", alpha=0.95))


# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.subplots_adjust(wspace=0.38)

draw_panel(axes[0], srs_selected,
           "(a) 简单随机采样",
           "等概率随机抽取，少数层可能被低估")
draw_panel(axes[1], strat_selected,
           "(b) 分层采样（按比例）",
           "各层按人口比例抽取，代表性最强")
draw_panel(axes[2], sys_selected,
           "(c) 系统采样",
           "按 x 排序后等间距抽取（步长 = 5）")

# ── Shared legend ─────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_A, label=f"层 A（高价值，N={nA}）"),
    mpatches.Patch(color=C_B, label=f"层 B（中等，N={nB}）"),
    mpatches.Patch(color=C_C, label=f"层 C（低频，N={nC}）"),
]
fig.legend(handles=legend_patches, fontsize=12.5,
           loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.08),
           framealpha=0.9)

fig.suptitle(
    f"三种采样策略对比（总体 N={N}，采样率 {n_sample}/{N}={n_sample/N*100:.0f}%）",
    fontsize=15, y=1.03,
)
fig.text(
    0.5, -0.17,
    f"总体共 {N} 个样本，含三个层次（A/B/C）。分层采样严格按比例抽取（A:{nA_s}，B:{nB_s}，C:{nC_s}），"
    "确保少数层 A 的代表性；简单随机采样可能导致 A 层被严重低估。",
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_3_03_sampling_methods")
