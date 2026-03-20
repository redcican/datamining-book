"""
图 10.5.2  gSpan 搜索过程
DFS code and gSpan search tree illustration.

(a) DFS 编码 — a 4-node labeled graph with DFS tree and corresponding DFS code.
(b) 搜索树 — gSpan search tree with rightmost extension and pruning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Figure ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                gridspec_kw={"width_ratios": [1, 1.3]})
fig.suptitle("图 10.5.2　gSpan 搜索过程", fontsize=22, fontweight="bold", y=0.98)

# ── Panel (a): DFS 编码 ─────────────────────────────────────────────────
ax1.set_xlim(-1.5, 4.5)
ax1.set_ylim(-4.5, 3.0)
ax1.axis("off")
ax1.grid(False)
ax1.set_title("(a) DFS 编码", fontsize=17, fontweight="bold")

# 4-node labeled graph: nodes 0(A), 1(B), 2(A), 3(C)
node_labels = ["A", "B", "A", "C"]
node_pos = {0: (0.0, 2.0), 1: (2.0, 2.0), 2: (2.0, 0.5), 3: (0.0, 0.5)}

# Edges: 0-1, 1-2, 2-3, 0-3, 0-2
all_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]

# DFS tree edges (one possible DFS: 0→1→2→3)
tree_edges = [(0, 1), (1, 2), (2, 3)]
back_edges = [(0, 3), (0, 2)]

# Draw back edges (dashed)
for u, v in back_edges:
    ax1.plot([node_pos[u][0], node_pos[v][0]],
             [node_pos[u][1], node_pos[v][1]],
             color=COLORS["orange"], linewidth=2.5, linestyle="--", alpha=0.7, zorder=1)

# Draw tree edges (solid, thick)
for u, v in tree_edges:
    ax1.plot([node_pos[u][0], node_pos[v][0]],
             [node_pos[u][1], node_pos[v][1]],
             color=COLORS["blue"], linewidth=3.5, zorder=1)

# Draw nodes
for node in range(4):
    x, y = node_pos[node]
    ax1.scatter(x, y, s=700, c=COLORS["blue"], edgecolors="white",
                linewidth=2.5, zorder=2)
    ax1.text(x, y, f"{node}:{node_labels[node]}", ha="center", va="center",
             fontsize=13, fontweight="bold", color="white", zorder=3)

# DFS discovery order arrows
ax1.annotate("DFS 起点", xy=(0, 2.0), xytext=(-1.3, 2.5),
             fontsize=11, color=COLORS["gray"],
             arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))

# Legend for edge types
legend_handles = [
    mpatches.Patch(facecolor=COLORS["blue"], label="前向边 (Tree Edge)"),
    mpatches.Patch(facecolor=COLORS["orange"], label="后向边 (Back Edge)"),
]
ax1.legend(handles=legend_handles, loc="upper right", fontsize=12,
           framealpha=0.9, edgecolor=COLORS["light"])

# DFS code table below the graph
table_y_start = -1.0
ax1.text(1.0, table_y_start, "DFS 编码序列:", ha="center", fontsize=14,
         fontweight="bold", color=COLORS["blue"])

headers = ["边", "i", "j", "$\\ell_V(v_i)$", "$\\ell_E$", "$\\ell_V(v_j)$", "类型"]
dfs_code = [
    ["$e_1$", "0", "1", "A", "—", "B", "前向"],
    ["$e_2$", "1", "2", "B", "—", "A", "前向"],
    ["$e_3$", "2", "0", "A", "—", "A", "后向"],
    ["$e_4$", "2", "3", "A", "—", "C", "前向"],
    ["$e_5$", "3", "0", "C", "—", "A", "后向"],
]

row_height = 0.45
col_widths = [0.7, 0.35, 0.35, 0.6, 0.4, 0.6, 0.6]
x_start = -0.8

# Draw header
y = table_y_start - 0.6
for c, (header, w) in enumerate(zip(headers, col_widths)):
    x = x_start + sum(col_widths[:c]) + w / 2
    ax1.text(x, y, header, ha="center", va="center", fontsize=11,
             fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["blue"],
                       edgecolor="none"))

# Draw rows
for r, row in enumerate(dfs_code):
    y = table_y_start - 0.6 - (r + 1) * row_height
    row_color = COLORS["blue"] if row[-1] == "前向" else COLORS["orange"]
    for c, (val, w) in enumerate(zip(row, col_widths)):
        x = x_start + sum(col_widths[:c]) + w / 2
        bg_color = "#f0f4ff" if r % 2 == 0 else "white"
        ax1.text(x, y, val, ha="center", va="center", fontsize=11,
                 color=row_color if c == 6 else COLORS["gray"],
                 fontweight="bold" if c == 6 else "normal",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor=bg_color,
                           edgecolor=COLORS["light"], linewidth=0.5))

# ── Panel (b): 搜索树 ───────────────────────────────────────────────────
ax2.set_xlim(-1, 11)
ax2.set_ylim(-1, 7)
ax2.axis("off")
ax2.grid(False)
ax2.set_title("(b) gSpan 搜索树", fontsize=17, fontweight="bold")

def draw_node_box(ax, x, y, text, color=COLORS["blue"], alpha=1.0, fontsize=11):
    """Draw a rounded box with text."""
    bbox = FancyBboxPatch((x - 0.55, y - 0.3), 1.1, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor="white", edgecolor=color,
                          linewidth=2, alpha=alpha, zorder=2)
    ax.add_patch(bbox)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=color, alpha=alpha, zorder=3)

def draw_edge(ax, x1, y1, x2, y2, color=COLORS["blue"], style="-", alpha=1.0):
    """Draw a line between two nodes."""
    ax.plot([x1, x2], [y1 - 0.3, y2 + 0.3], color=color, linewidth=2,
            linestyle=style, alpha=alpha, zorder=1)

# Level 0: root
draw_node_box(ax2, 5, 6.2, "∅", fontsize=14)

# Level 1: single-edge patterns
l1_nodes = [(1.5, 4.5, "A—A"), (5, 4.5, "A—B"), (8.5, 4.5, "A—C")]
for x, y, text in l1_nodes:
    draw_edge(ax2, 5, 6.2, x, y)
    draw_node_box(ax2, x, y, text)

# Level 2: extensions from A—A
l2_from_aa = [(0.3, 2.5, "A—A—B", True), (2.7, 2.5, "A—A—A", True)]
for x, y, text, active in l2_from_aa:
    draw_edge(ax2, 1.5, 4.5, x, y, color=COLORS["blue"])
    draw_node_box(ax2, x, y, text, fontsize=10)

# Level 2: extensions from A—B
l2_from_ab = [(4.0, 2.5, "A—B—A", True), (6.0, 2.5, "A—B—C", True)]
for x, y, text, active in l2_from_ab:
    draw_edge(ax2, 5, 4.5, x, y, color=COLORS["blue"])
    draw_node_box(ax2, x, y, text, fontsize=10)

# Level 2: extensions from A—C (pruned)
l2_from_ac = [(7.5, 2.5, "A—C—A", False), (9.5, 2.5, "A—C—B", False)]
for x, y, text, active in l2_from_ac:
    draw_edge(ax2, 8.5, 4.5, x, y, color=COLORS["gray"], style="--", alpha=0.5)
    draw_node_box(ax2, x, y, text, color=COLORS["gray"], alpha=0.5, fontsize=10)
    ax2.text(x, y - 0.65, "✗ 剪枝", ha="center", fontsize=10,
             color=COLORS["red"], alpha=0.7, fontweight="bold")

# Level 3: some extensions from A—A—B
l3 = [(0.3, 0.7, "A—A—B\n   +A", True)]
for x, y, text, active in l3:
    draw_edge(ax2, 0.3, 2.5, x, y, color=COLORS["blue"])
    draw_node_box(ax2, x, y, text, fontsize=9)

# Pruned extension from A—A—A
draw_edge(ax2, 2.7, 2.5, 2.7, 0.7, color=COLORS["gray"], style="--", alpha=0.5)
draw_node_box(ax2, 2.7, 0.7, "非规范", color=COLORS["gray"], alpha=0.5, fontsize=9)
ax2.text(2.7, 0.05, "✗ 剪枝", ha="center", fontsize=10,
         color=COLORS["red"], alpha=0.7, fontweight="bold")

# Annotation: rightmost extension
ax2.annotate("右最扩展", xy=(4.0, 2.5), xytext=(4.0, 1.3),
             fontsize=12, color=COLORS["green"], fontweight="bold",
             ha="center",
             arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=2))

# Level labels
for level, y, label in [(0, 6.2, "0-边"), (1, 4.5, "1-边"), (2, 2.5, "2-边")]:
    ax2.text(-0.5, y, label, ha="center", va="center", fontsize=11,
             color=COLORS["gray"], fontstyle="italic")

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_5_02_gspan_search")
