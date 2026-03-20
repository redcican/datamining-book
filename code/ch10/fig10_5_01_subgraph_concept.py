"""
图 10.5.1  子图同构与频繁子图
Subgraph isomorphism concept and frequent subgraph support counting.

(a) 子图同构 — pattern graph g (triangle) mapped into a larger data graph G.
(b) 频繁子图支持度 — 6 small graphs; 4 contain the triangle pattern (support 67%).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

np.random.seed(42)

# ── Figure layout ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
fig.suptitle("图 10.5.1　子图同构与频繁子图", fontsize=22, fontweight="bold", y=0.98)

outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05, width_ratios=[1, 1.2])

# ── Panel (a): 子图同构 ──────────────────────────────────────────────────
ax_a = fig.add_subplot(outer_gs[0])
ax_a.set_xlim(-0.3, 3.3)
ax_a.set_ylim(-0.5, 2.0)
ax_a.axis("off")
ax_a.grid(False)
ax_a.set_title("(a) 子图同构", fontsize=17, fontweight="bold")

# Pattern graph g (triangle)
g_pos = {0: (0.0, 1.5), 1: (0.5, 0.5), 2: (-0.5, 0.5)}
g_edges = [(0, 1), (1, 2), (0, 2)]
G_pattern = nx.Graph()
G_pattern.add_nodes_from([0, 1, 2])
G_pattern.add_edges_from(g_edges)

# Draw pattern
for u, v in g_edges:
    ax_a.plot([g_pos[u][0], g_pos[v][0]], [g_pos[u][1], g_pos[v][1]],
              color=COLORS["red"], linewidth=3, zorder=1)
for node, (x, y) in g_pos.items():
    ax_a.scatter(x, y, s=400, c=COLORS["red"], edgecolors="white",
                 linewidth=2, zorder=2)
    ax_a.text(x, y, str(node), ha="center", va="center",
              fontsize=12, fontweight="bold", color="white", zorder=3)

ax_a.text(0.0, -0.2, "模式 $g$", ha="center", fontsize=14, fontweight="bold")

# Arrow
ax_a.annotate("", xy=(1.7, 1.0), xytext=(0.9, 1.0),
              arrowprops=dict(arrowstyle="->,head_width=0.3", color=COLORS["gray"],
                              lw=2.5))
ax_a.text(1.3, 1.2, "同构映射", ha="center", fontsize=13, color=COLORS["gray"],
          fontstyle="italic")

# Data graph G (10 nodes with a triangle highlighted)
G_data = nx.Graph()
G_data.add_nodes_from(range(10))
data_edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6),
              (4, 5), (4, 7), (5, 8), (6, 7), (7, 8), (7, 9), (8, 9)]
G_data.add_edges_from(data_edges)

# Position data graph on the right side
d_pos = nx.spring_layout(G_data, seed=15, k=0.7)
# Shift to right side of panel
for k in d_pos:
    d_pos[k] = (d_pos[k][0] * 0.7 + 2.5, d_pos[k][1] * 0.7 + 1.0)

# Triangle in data graph: nodes 7, 8, 9
tri_nodes = {7, 8, 9}
tri_edges = [(7, 8), (7, 9), (8, 9)]

# Draw non-highlighted edges
for u, v in data_edges:
    if (u, v) not in tri_edges and (v, u) not in tri_edges:
        ax_a.plot([d_pos[u][0], d_pos[v][0]], [d_pos[u][1], d_pos[v][1]],
                  color=COLORS["gray"], linewidth=1.5, alpha=0.4, zorder=1)

# Draw highlighted edges (triangle)
for u, v in tri_edges:
    ax_a.plot([d_pos[u][0], d_pos[v][0]], [d_pos[u][1], d_pos[v][1]],
              color=COLORS["red"], linewidth=3.5, alpha=0.8, zorder=1)

# Draw nodes
for node in G_data.nodes():
    x, y = d_pos[node]
    color = COLORS["red"] if node in tri_nodes else COLORS["blue"]
    alpha = 1.0 if node in tri_nodes else 0.6
    ax_a.scatter(x, y, s=350, c=color, edgecolors="white",
                 linewidth=2, zorder=2, alpha=alpha)
    ax_a.text(x, y, str(node), ha="center", va="center",
              fontsize=10, fontweight="bold", color="white", zorder=3)

ax_a.text(2.5, -0.3, "数据图 $G$", ha="center", fontsize=14, fontweight="bold")

# ── Panel (b): 频繁子图支持度 ────────────────────────────────────────────
inner_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[1],
                                            wspace=0.3, hspace=0.45)

# Create 6 small graphs: 4 with triangles, 2 without
small_graphs = []
has_tri = [True, True, False, True, True, False]

# G1: triangle + extra node
G1 = nx.Graph([(0,1),(1,2),(0,2),(2,3)])
small_graphs.append(G1)

# G2: triangle embedded in 5-node graph
G2 = nx.Graph([(0,1),(1,2),(0,2),(1,3),(3,4)])
small_graphs.append(G2)

# G3: path graph, no triangle
G3 = nx.path_graph(5)
small_graphs.append(G3)

# G4: two triangles sharing an edge
G4 = nx.Graph([(0,1),(1,2),(0,2),(2,3),(1,3)])
small_graphs.append(G4)

# G5: triangle + star
G5 = nx.Graph([(0,1),(1,2),(0,2),(0,3),(0,4)])
small_graphs.append(G5)

# G6: star graph, no triangle
G6 = nx.star_graph(4)
small_graphs.append(G6)

for idx, (G_small, contains) in enumerate(zip(small_graphs, has_tri)):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(inner_gs[row, col])
    ax.axis("off")
    ax.grid(False)

    pos = nx.spring_layout(G_small, seed=42 + idx, k=1.5)

    # Find a triangle if exists (for highlighting)
    tri_highlight = set()
    tri_edge_highlight = set()
    if contains:
        for n1 in G_small.nodes():
            for n2 in G_small.neighbors(n1):
                for n3 in G_small.neighbors(n2):
                    if n3 != n1 and G_small.has_edge(n1, n3):
                        tri_highlight = {n1, n2, n3}
                        tri_edge_highlight = {(min(n1,n2),max(n1,n2)),
                                              (min(n2,n3),max(n2,n3)),
                                              (min(n1,n3),max(n1,n3))}
                        break
                if tri_highlight:
                    break
            if tri_highlight:
                break

    # Draw edges
    for u, v in G_small.edges():
        edge_key = (min(u,v), max(u,v))
        if edge_key in tri_edge_highlight:
            nx.draw_networkx_edges(G_small, pos, edgelist=[(u,v)], ax=ax,
                                   edge_color=COLORS["red"], width=3.0, alpha=0.8)
        else:
            nx.draw_networkx_edges(G_small, pos, edgelist=[(u,v)], ax=ax,
                                   edge_color=COLORS["gray"], width=1.5, alpha=0.4)

    # Draw nodes
    node_colors = [COLORS["red"] if n in tri_highlight else COLORS["blue"]
                   for n in G_small.nodes()]
    node_alphas = [1.0 if n in tri_highlight else 0.6
                   for n in G_small.nodes()]

    nx.draw_networkx_nodes(G_small, pos, ax=ax, node_size=250,
                           node_color=node_colors, edgecolors="white",
                           linewidths=1.5, alpha=0.8)

    # Border color indicates match
    border_color = COLORS["green"] if contains else COLORS["gray"]
    mark = "(Y)" if contains else "(N)"
    mark_color = COLORS["green"] if contains else COLORS["red"]

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3 if contains else 1.5)

    ax.set_title(f"$G_{idx+1}$  {mark}", fontsize=14, fontweight="bold",
                 color=mark_color)

# Panel (b) title
fig.text(0.73, 0.04, f"支持度 = 4/6 ≈ 67%",
         ha="center", fontsize=15, fontweight="bold", color=COLORS["blue"],
         bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"],
                   edgecolor=COLORS["blue"], alpha=0.8))

fig.text(0.73, 0.92, "(b) 频繁子图支持度", ha="center",
         fontsize=17, fontweight="bold")

fig.tight_layout(rect=[0, 0.08, 1, 0.92])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_5_01_subgraph_concept")
