"""
图 10.5.3  频繁子图挖掘案例结果
Frequent subgraph mining case study results.

(a) 频繁子图模式 — top-6 frequent subgraph patterns ordered by support.
(b) 支持度分布 — long-tail distribution of subgraph pattern supports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

np.random.seed(42)

# ── Define 6 frequent subgraph patterns (ordered by decreasing support) ──
patterns = [
    ("单边", nx.path_graph(2), 0.98),
    ("路径-2", nx.path_graph(3), 0.89),
    ("星-3", nx.star_graph(3), 0.76),
    ("三角形", nx.cycle_graph(3), 0.68),
    ("路径-3", nx.path_graph(4), 0.54),
    ("四边形", nx.cycle_graph(4), 0.35),
]

# ── Generate synthetic support distribution (long-tail) ──────────────────
n_patterns = 25
# Use a power-law-like distribution for supports
supports = np.sort(np.random.beta(1.2, 3.0, size=n_patterns))[::-1]
# Scale to realistic range [0.05, 0.98]
supports = supports * 0.93 + 0.05
# Ensure top patterns match our defined ones
for i, (_, _, sup) in enumerate(patterns):
    if i < len(supports):
        supports[i] = sup

sigma_min = 0.30

# ── Figure layout ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
fig.suptitle("图 10.5.3　频繁子图挖掘案例结果", fontsize=22, fontweight="bold", y=0.98)

outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1.2])

# ── Panel (a): 频繁子图模式 (2×3 grid) ──────────────────────────────────
inner_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[0],
                                            wspace=0.35, hspace=0.55)

for idx, (name, G_pat, sup) in enumerate(patterns):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(inner_gs[row, col])
    ax.axis("off")
    ax.grid(False)

    pos = nx.spring_layout(G_pat, seed=42 + idx, k=2.0)
    nx.draw_networkx_edges(G_pat, pos, ax=ax, edge_color=COLORS["blue"],
                           width=2.5, alpha=0.8)
    nx.draw_networkx_nodes(G_pat, pos, ax=ax, node_size=300,
                           node_color=COLORS["blue"], edgecolors="white",
                           linewidths=2)
    ax.set_title(f"{name}\nsup = {sup:.0%}", fontsize=12, fontweight="bold",
                 color=COLORS["blue"])

fig.text(0.27, 0.03, "(a) 频繁子图模式（按支持度排序）",
         ha="center", fontsize=17, fontweight="bold")

# ── Panel (b): 支持度分布 ────────────────────────────────────────────────
ax_b = fig.add_subplot(outer_gs[1])

bar_colors = [COLORS["blue"] if s >= sigma_min else COLORS["gray"] for s in supports]

bars = ax_b.bar(range(1, n_patterns + 1), supports * 100,
                color=bar_colors, edgecolor="white", linewidth=0.8, width=0.7)

# Minimum support threshold line
ax_b.axhline(y=sigma_min * 100, color=COLORS["red"], linewidth=2.0,
             linestyle="--", zorder=5)
ax_b.text(n_patterns + 0.5, sigma_min * 100 + 1.5,
          f"$\\sigma_{{min}}$ = {sigma_min:.0%}", fontsize=13,
          color=COLORS["red"], fontweight="bold", ha="right")

# Count frequent vs infrequent
n_freq = sum(1 for s in supports if s >= sigma_min)
ax_b.text(n_patterns * 0.5, 90, f"频繁: {n_freq} 个", fontsize=13,
          color=COLORS["blue"], fontweight="bold", ha="center")
ax_b.text(n_patterns * 0.75, 15, f"非频繁: {n_patterns - n_freq} 个", fontsize=13,
          color=COLORS["gray"], fontweight="bold", ha="center")

ax_b.set_xlabel("子图模式（按支持度降序）", fontsize=16)
ax_b.set_ylabel("支持度 (%)", fontsize=16)
ax_b.set_title("(b) 支持度分布", fontsize=17, fontweight="bold")
ax_b.tick_params(axis="both", labelsize=14)
ax_b.set_ylim(0, 105)
ax_b.set_xlim(0, n_patterns + 1)

fig.tight_layout(rect=[0, 0.06, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_5_03_case_result")
