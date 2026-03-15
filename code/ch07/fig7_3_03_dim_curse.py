"""
fig7_3_03_dim_curse.py
高维距离集中现象
左：不同维度下成对距离的分布
右：相对对比度 RC 随维度的衰减
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 参数设置 ──────────────────────────────────────────────────────
dims_hist = [2, 5, 10, 50, 100, 500]
dims_rc = np.unique(np.logspace(np.log10(2), np.log10(500), 80).astype(int))
n_points = 200

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.3.3　高维距离集中现象",
             fontsize=20, fontweight="bold", y=1.02)

# ── 左：距离分布随维度变化 ─────────────────────────────────────────
ax = axes[0]

color_list = [COLORS["blue"], COLORS["green"], COLORS["orange"],
              COLORS["red"], COLORS["purple"], COLORS["teal"]]

for d, color in zip(dims_hist, color_list):
    points = np.random.randn(n_points, d)
    dists = pdist(points, metric="euclidean")
    # 归一化到 [0, 1]，便于比较不同维度
    dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-10)
    ax.hist(dists_norm, bins=50, density=True, alpha=0.45,
            color=color, histtype="stepfilled", linewidth=0)
    ax.hist(dists_norm, bins=50, density=True,
            color=color, histtype="step", linewidth=1.8,
            label=f"d = {d}")

ax.set_title("(a) 归一化距离分布", fontsize=17)
ax.set_xlabel("归一化距离  $(d - d_{min}) / (d_{max} - d_{min})$", fontsize=14)
ax.set_ylabel("密度", fontsize=15)
ax.legend(fontsize=13, loc="upper left", title="维度", title_fontsize=14)
ax.tick_params(labelsize=13)
ax.set_xlim(-0.05, 1.05)

# 添加趋势说明
ax.annotate("维度越高，\n距离越集中",
            xy=(0.5, 8), xytext=(0.7, 10),
            fontsize=14, color=COLORS["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
            ha="center")

# ── 右：相对对比度 RC vs 维度 ──────────────────────────────────────
ax = axes[1]

rc_values = []
for d in dims_rc:
    points = np.random.randn(n_points, d)
    dists = pdist(points, metric="euclidean")
    d_max, d_min = dists.max(), dists.min()
    rc = (d_max - d_min) / (d_min + 1e-10)
    rc_values.append(rc)

rc_values = np.array(rc_values)

ax.plot(dims_rc, rc_values, color=COLORS["blue"], lw=2.5,
        marker="o", markersize=4, markerfacecolor=COLORS["blue"],
        markeredgecolor="white", markeredgewidth=0.5, label="RC 实测值")

# 参考线 RC = 1
ax.axhline(1.0, color=COLORS["red"], ls="--", lw=2, alpha=0.7,
           label="RC = 1（参考线）")

# 标注关键区域
ax.fill_between(dims_rc, 0, 1, alpha=0.08, color=COLORS["red"])
ax.text(100, 0.5, "RC < 1\n距离难以区分\n异常检测失效",
        fontsize=14, color=COLORS["red"], fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["red"], alpha=0.9))

ax.set_xscale("log")
ax.set_title("(b) 相对对比度随维度变化", fontsize=17)
ax.set_xlabel("维度 d（对数尺度）", fontsize=15)
ax.set_ylabel("相对对比度  RC = $(d_{max} - d_{min}) / d_{min}$", fontsize=14)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_3_03_dim_curse")
