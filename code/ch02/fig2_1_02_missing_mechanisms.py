"""
图 2.1.2  三种缺失机制对比（MCAR / MAR / MNAR）
对应节次：2.1 数据清洗技术
运行方式：python code/ch02/fig2_1_02_missing_mechanisms.py
输出路径：public/figures/ch02/fig2_1_02_missing_mechanisms.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

apply_style()

rng = np.random.default_rng(42)
n = 120

# ── Generate base data ────────────────────────────────────────────────────
age = rng.normal(50, 15, n).clip(18, 85)           # X1: 年龄
bp_true = 70 + 0.6 * age + rng.normal(0, 8, n)    # X2: 血压（真值）

# ── Three missing mechanisms ──────────────────────────────────────────────
# MCAR: missing completely at random (30% random)
mcar_mask = rng.random(n) < 0.30

# MAR: missing depends on observed X (age < 35 → 70% chance missing)
mar_mask = np.where(age < 35, rng.random(n) < 0.75, rng.random(n) < 0.05)

# MNAR: missing depends on the unobserved Y itself (high BP patients avoid)
mnar_mask = np.where(bp_true > 110, rng.random(n) < 0.75, rng.random(n) < 0.05)

# ── Layout ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.subplots_adjust(wspace=0.38)

C_OBS  = "#2563eb"   # observed  — blue
C_MIS  = "#dc2626"   # missing   — red
C_ZONE = "#fef2f2"   # shading   — light red

panel_data = [
    ("(a) MCAR：完全随机缺失", mcar_mask,
     None, None,
     "$P(\\mathbf{R}\\,|\\,\\mathbf{X}_{obs},\\,\\mathbf{X}_{mis}) = P(\\mathbf{R})$\n"
     "缺失与任何变量无关", 0.50),
    ("(b) MAR：随机缺失", mar_mask,
     (18, 35), "x",
     "$P(\\mathbf{R}\\,|\\,\\mathbf{X}_{obs},\\,\\mathbf{X}_{mis}) = P(\\mathbf{R}\\,|\\,\\mathbf{X}_{obs})$\n"
     "缺失仅取决于已观测变量（年龄 < 35）", 0.50),
    ("(c) MNAR：非随机缺失", mnar_mask,
     (110, 160), "y",
     "$P(\\mathbf{R}\\,|\\,\\mathbf{X}_{obs},\\,\\mathbf{X}_{mis}) \\neq P(\\mathbf{R}\\,|\\,\\mathbf{X}_{obs})$\n"
     "缺失取决于缺失值本身（高血压者拒绝测量）", 0.58),
]

for ax, (title, mask, zone_range, zone_axis, formula, txt_x) in zip(axes, panel_data):
    obs_age = age[~mask]
    obs_bp  = bp_true[~mask]
    mis_age = age[mask]
    mis_bp  = bp_true[mask]

    # Shaded zone
    if zone_range is not None:
        if zone_axis == "x":
            ax.axvspan(zone_range[0], zone_range[1],
                       color=C_ZONE, alpha=0.7, zorder=0,
                       label="缺失高概率区")
        else:  # y
            ax.axhspan(zone_range[0], zone_range[1],
                       color=C_ZONE, alpha=0.7, zorder=0,
                       label="缺失高概率区")

    # Observed points
    ax.scatter(obs_age, obs_bp, s=28, color=C_OBS, alpha=0.75,
               label=f"已观测 (n={len(obs_age)})", zorder=3)

    # Missing positions (shown as ×)
    ax.scatter(mis_age, mis_bp, s=60, marker="x",
               color=C_MIS, linewidths=1.6, alpha=0.80,
               label=f"已缺失 (n={len(mis_age)})", zorder=4)

    ax.set_title(title, fontsize=12.5, pad=8)
    ax.set_xlabel("年龄（岁）", fontsize=12)
    ax.set_ylabel("血压（mmHg）", fontsize=12)
    ax.set_xlim(10, 90)
    ax.set_ylim(50, 160)

    # Formula annotation inside the plot
    ax.text(txt_x, 0.03, formula,
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12.5,
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cbd5e1", alpha=0.92))

    ax.legend(fontsize=12, loc="upper left", framealpha=0.85)

# ── Global title & annotation ─────────────────────────────────────────────
fig.suptitle("缺失值产生机制：MCAR / MAR / MNAR 可视化对比",
             fontsize=15, y=1.02)
fig.text(0.5, -0.04,
         "注：红色 × 标记表示\"已缺失\"记录在假想完整数据中的位置（现实中不可直接观测）。"
         "  MNAR 情形下，高血压患者倾向于不测量，导致数据中的血压值普遍偏低，"
         "对均值等统计量产生系统性低估偏差。",
         ha="center", fontsize=12.5, color="#64748b", style="italic")

save_fig(fig, __file__, "fig2_1_02_missing_mechanisms")
