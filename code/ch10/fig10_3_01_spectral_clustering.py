"""
图 10.3.1  谱聚类方法
Spectral clustering on Zachary's Karate Club using the Fiedler vector.

(a) Fiedler vector bar chart — bars colored by sign to show natural community split.
(b) Network visualization with spectral clustering result and misclassification highlights.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────
G = nx.karate_club_graph()

# True community labels: 0 = Mr. Hi, 1 = Officer
true_labels = np.array(
    [0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in G.nodes()]
)

# Normalized Laplacian → eigendecomposition
L = nx.normalized_laplacian_matrix(G).toarray()
eigvals, eigvecs = np.linalg.eigh(L)
fiedler = eigvecs[:, 1]  # second-smallest eigenvalue → Fiedler vector

# Spectral labels from sign of Fiedler vector
spectral_labels = (fiedler > 0).astype(int)

# Identify misclassified nodes
misclassified = spectral_labels != true_labels

# ── Figure ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.3.1　谱聚类方法", fontsize=22, fontweight="bold", y=0.98)

# ── Panel (a): Fiedler vector bar chart ──────────────────────────────────
# Sort nodes: Mr. Hi community first, then Officer community
nodes = np.array(list(G.nodes()))
sort_idx = np.argsort(true_labels * 1000 + nodes)  # group by community, then node ID
sorted_nodes = nodes[sort_idx]
sorted_fiedler = fiedler[sort_idx]

bar_colors = [COLORS["blue"] if v < 0 else COLORS["red"] for v in sorted_fiedler]

ax1.bar(range(len(sorted_nodes)), sorted_fiedler, color=bar_colors, edgecolor="white",
        linewidth=0.5, width=0.8)
ax1.axhline(y=0, color=COLORS["gray"], linestyle="--", linewidth=1.2, alpha=0.7)
ax1.set_xticks(range(len(sorted_nodes)))
ax1.set_xticklabels(sorted_nodes, fontsize=9, rotation=0)
ax1.set_xlabel("节点", fontsize=16)
ax1.set_ylabel("Fiedler 向量分量", fontsize=16)
ax1.set_title("(a) Fiedler 向量", fontsize=17, fontweight="bold")
ax1.tick_params(axis="both", labelsize=14)

# ── Panel (b): Network visualization ────────────────────────────────────
pos = nx.spring_layout(G, seed=42)

# Node colors by spectral prediction
node_colors = [COLORS["blue"] if spectral_labels[v] == 0 else COLORS["red"]
               for v in G.nodes()]

# Edge colors for misclassified nodes
edge_colors = ["#f59e0b" if misclassified[v] else "white" for v in G.nodes()]
edge_widths = [3.0 if misclassified[v] else 1.5 for v in G.nodes()]

# Draw edges first
nx.draw_networkx_edges(G, pos, ax=ax2, edge_color=COLORS["gray"],
                       alpha=0.3, width=0.8)

# Draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax2,
                       node_color=node_colors,
                       edgecolors=edge_colors,
                       linewidths=edge_widths,
                       node_size=420)

# Draw labels
nx.draw_networkx_labels(G, pos, ax=ax2,
                        font_size=8, font_color="white", font_weight="bold")

ax2.set_title("(b) 谱聚类结果", fontsize=17, fontweight="bold")
ax2.axis("off")
ax2.grid(False)

# Legend
legend_handles = [
    mpatches.Patch(facecolor=COLORS["blue"], edgecolor="white", label="社区 1 (预测)"),
    mpatches.Patch(facecolor=COLORS["red"], edgecolor="white", label="社区 2 (预测)"),
    mpatches.Patch(facecolor=COLORS["gray"], edgecolor="#f59e0b",
                   linewidth=2.5, label="误分节点"),
]
ax2.legend(handles=legend_handles, loc="lower left", fontsize=12,
           framealpha=0.9, edgecolor=COLORS["light"])

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_3_01_spectral_clustering")
