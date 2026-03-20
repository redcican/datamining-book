"""
图 10.7.2  GCN 与 GAT 聚合机制对比
Comparison of GCN (fixed weights) and GAT (attention weights) aggregation.

(a) GCN: 固定权重 — All edges have equal normalized weight 1/sqrt(d_u * d_v).
(b) GAT: 注意力权重 — Edges have different attention weights (alpha).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Graph layout ─────────────────────────────────────────────────────────
# Central node at origin, 4 neighbors evenly spaced around it
center = np.array([0.0, 0.0])
angles = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])  # top, right, bottom, left
radius = 1.3
neighbor_pos = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
neighbor_labels = ["$u_1$", "$u_2$", "$u_3$", "$u_4$"]

# Node sizes
node_radius_center = 0.18
node_radius_neighbor = 0.15


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def draw_node(ax, xy, r, color, label, label_color="white", fontsize=14):
    """Draw a filled circle node with a label."""
    circle = mpatches.Circle(xy, r, facecolor=color, edgecolor="white",
                             linewidth=2.0, zorder=5)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=label_color, zorder=6)


def draw_edge(ax, p1, p2, r1, r2, linewidth=2.0, color=COLORS["gray"],
              alpha=0.8):
    """Draw an edge between two nodes, trimmed to node radii."""
    direction = p2 - p1
    dist = np.linalg.norm(direction)
    unit = direction / dist
    start = p1 + unit * r1
    end = p2 - unit * r2
    ax.plot([start[0], end[0]], [start[1], end[1]],
            linewidth=linewidth, color=color, alpha=alpha, solid_capstyle="round",
            zorder=2)


def edge_midpoint(p1, p2, offset=0.12):
    """Compute a label position slightly offset from the edge midpoint."""
    mid = (p1 + p2) / 2
    direction = p2 - p1
    # Perpendicular vector for offset
    perp = np.array([-direction[1], direction[0]])
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 0:
        perp = perp / perp_norm
    return mid + perp * offset


# ── Figure ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.7.2　GCN 与 GAT 聚合机制对比",
             fontsize=22, fontweight="bold", y=1.02)

# ── Color definitions ────────────────────────────────────────────────────
blue_rgb = hex_to_rgb(COLORS["blue"])
light_blue = tuple(min(1.0, c * 1.15 + 0.35) for c in blue_rgb)

# ── Panel (a): GCN — Fixed weights ──────────────────────────────────────
ax1.set_title("(a) GCN: 固定权重", fontsize=17, fontweight="bold")

# GCN: all edges have equal weight, uniform appearance
gcn_lw = 2.5
gcn_alpha = 0.7

for i, npos in enumerate(neighbor_pos):
    draw_edge(ax1, center, npos, node_radius_center, node_radius_neighbor,
              linewidth=gcn_lw, color=COLORS["gray"], alpha=gcn_alpha)

    # Weight label at midpoint
    label_pos = edge_midpoint(center, npos, offset=0.18)
    ax1.text(label_pos[0], label_pos[1],
             r"$\frac{1}{\sqrt{d_u d_v}}$",
             ha="center", va="center", fontsize=10,
             color=COLORS["gray"], fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor=COLORS["light"], alpha=0.9))

# Draw neighbor nodes (lighter blue)
for i, npos in enumerate(neighbor_pos):
    draw_node(ax1, npos, node_radius_neighbor, light_blue,
              neighbor_labels[i], label_color=COLORS["blue"], fontsize=13)

# Draw central node (blue)
draw_node(ax1, center, node_radius_center, COLORS["blue"],
          "$v$", label_color="white", fontsize=15)

# Formula below the graph
ax1.text(0.0, -2.05,
         r"$h_v = \sigma\!\left(\sum \frac{1}{\sqrt{d_u d_v}}\, W h_u\right)$",
         ha="center", va="center", fontsize=15,
         color=COLORS["blue"],
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef2ff",
                   edgecolor=COLORS["blue"], alpha=0.8))

ax1.set_xlim(-2.0, 2.0)
ax1.set_ylim(-2.5, 2.0)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.grid(False)

# ── Panel (b): GAT — Attention weights ──────────────────────────────────
ax2.set_title("(b) GAT: 注意力权重", fontsize=17, fontweight="bold")

# GAT: different attention weights per edge
attn_weights = [0.4, 0.3, 0.2, 0.1]
# Map attention weights to line widths: range [1.0, 6.0]
max_lw, min_lw = 6.0, 1.0
lw_values = [min_lw + (max_lw - min_lw) * (w / max(attn_weights))
             for w in attn_weights]
# Map to alpha: range [0.4, 1.0]
alpha_values = [0.4 + 0.6 * (w / max(attn_weights)) for w in attn_weights]

for i, npos in enumerate(neighbor_pos):
    draw_edge(ax2, center, npos, node_radius_center, node_radius_neighbor,
              linewidth=lw_values[i], color=COLORS["orange"],
              alpha=alpha_values[i])

    # Attention weight label at midpoint
    label_pos = edge_midpoint(center, npos, offset=0.18)
    ax2.text(label_pos[0], label_pos[1],
             rf"$\alpha={attn_weights[i]}$",
             ha="center", va="center", fontsize=11,
             color=COLORS["orange"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor=COLORS["light"], alpha=0.9))

# Draw neighbor nodes (lighter blue)
for i, npos in enumerate(neighbor_pos):
    draw_node(ax2, npos, node_radius_neighbor, light_blue,
              neighbor_labels[i], label_color=COLORS["blue"], fontsize=13)

# Draw central node (blue)
draw_node(ax2, center, node_radius_center, COLORS["blue"],
          "$v$", label_color="white", fontsize=15)

# Formula below the graph
ax2.text(0.0, -2.05,
         r"$h_v = \sigma\!\left(\sum \alpha_{vu}\, W h_u\right)$",
         ha="center", va="center", fontsize=15,
         color=COLORS["orange"],
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ed",
                   edgecolor=COLORS["orange"], alpha=0.8))

ax2.set_xlim(-2.0, 2.0)
ax2.set_ylim(-2.5, 2.0)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.grid(False)

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_7_02_gcn_gat")
