"""
Figure 10.2.1 - Comparison of four centrality measures on the Karate Club graph.

Generates a 2x2 panel figure showing the same network with nodes colored
by degree, betweenness, closeness, and eigenvector centrality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

apply_style()

# ── 1. Build graph and compute layout ─────────────────────────────────────
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42, k=0.5)

# ── 2. Compute centrality measures ────────────────────────────────────────
centralities = {
    "(a) 度中心性 (Degree)": nx.degree_centrality(G),
    "(b) 介数中心性 (Betweenness)": nx.betweenness_centrality(G),
    "(c) 接近中心性 (Closeness)": nx.closeness_centrality(G),
    "(d) 特征向量中心性 (Eigenvector)": nx.eigenvector_centrality(G, max_iter=1000),
}

# ── 3. Create figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 13))
fig.suptitle("图 10.2.1　四种中心性度量对比", fontsize=22, fontweight="bold", y=0.97)

cmap = plt.cm.RdYlBu_r

for ax, (title, cent_dict) in zip(axes.flat, centralities.items()):
    # Extract centrality values in node order
    nodes = list(G.nodes())
    cent_values = np.array([cent_dict[n] for n in nodes])

    # Node sizes proportional to centrality
    sizes = 100 + cent_values * 2000

    # Normalize for colormap
    vmin, vmax = cent_values.min(), cent_values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=COLORS["light"],
        alpha=0.4,
        width=0.8,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=cent_values,
        cmap=cmap,
        node_size=sizes,
        edgecolors="white",
        linewidths=1,
        vmin=vmin,
        vmax=vmax,
    )

    # Annotate top-3 highest centrality nodes
    top3_indices = np.argsort(cent_values)[-3:][::-1]
    for idx in top3_indices:
        node_id = nodes[idx]
        x, y = pos[node_id]
        ax.annotate(
            str(node_id),
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=COLORS["gray"],
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=COLORS["gray"],
                alpha=0.8,
            ),
        )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("中心性值", fontsize=12)

    # Panel title and cleanup
    ax.set_title(title, fontsize=17, fontweight="bold", pad=12)
    ax.axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.94])

# ── 4. Save ───────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_2_01_centrality_comparison")
