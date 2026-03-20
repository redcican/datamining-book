"""
图 10.7.3  GCN 节点分类结果
GCN node classification results on the Karate Club graph.

(a) GCN 分类结果 — Karate Club graph colored by true community labels,
    with labeled training nodes (0 and 33) highlighted with thick black borders.
(b) 嵌入空间 — 2-D PCA projection of GCN hidden embeddings (ReLU(A_hat @ X @ W1))
    colored by true community, showing cluster separation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

# ── Build Karate Club graph ──────────────────────────────────────────────────
G = nx.karate_club_graph()
n_nodes = G.number_of_nodes()

# True community labels: Mr. Hi (0) vs Officer (1)
true_labels = np.array(
    [0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n_nodes)]
)

# ── Compute normalized adjacency A_hat ───────────────────────────────────────
A = nx.to_numpy_array(G)
A_tilde = A + np.eye(n_nodes)                     # A + I
D_tilde = np.diag(A_tilde.sum(axis=1))
D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

# ── One-hot features ─────────────────────────────────────────────────────────
X = np.eye(n_nodes)

# ── GCN hidden embeddings: H = ReLU(A_hat @ X @ W1) ─────────────────────────
np.random.seed(42)
W1 = np.random.randn(n_nodes, 16) * 0.1
H = A_hat @ X @ W1
H = np.maximum(H, 0)  # ReLU

# ── PCA to 2-D for visualization ────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
H_2d = pca.fit_transform(H)

# ── Color mapping ────────────────────────────────────────────────────────────
color_map = [COLORS["blue"] if lbl == 0 else COLORS["red"] for lbl in true_labels]
labeled_nodes = {0, 33}  # training nodes

# ── Figure ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "图 10.7.3　GCN 节点分类结果",
    fontsize=22, fontweight="bold", y=0.98,
)

# ── Panel (a): GCN classification on graph ───────────────────────────────────
pos = nx.spring_layout(G, seed=42, k=0.5)

# Edge widths
edge_widths = [0.8] * G.number_of_edges()

# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, width=edge_widths)

# Draw non-labeled nodes
non_labeled = [i for i in range(n_nodes) if i not in labeled_nodes]
nx.draw_networkx_nodes(
    G, pos, nodelist=non_labeled, ax=ax1,
    node_color=[color_map[i] for i in non_labeled],
    node_size=300, alpha=0.85,
    edgecolors="white", linewidths=1.0,
)

# Draw labeled nodes with thick black borders
labeled_list = sorted(labeled_nodes)
nx.draw_networkx_nodes(
    G, pos, nodelist=labeled_list, ax=ax1,
    node_color=[color_map[i] for i in labeled_list],
    node_size=500, alpha=1.0,
    edgecolors="black", linewidths=3.0,
)

# Node labels
nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_color="white",
                        font_weight="bold")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["blue"],
           markersize=10, label="Mr. Hi"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["red"],
           markersize=10, label="Officer"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gray"],
           markersize=12, markeredgecolor="black", markeredgewidth=2.5,
           label="标记节点"),
]
ax1.legend(handles=legend_elements, fontsize=14, loc="upper left",
           framealpha=0.9, edgecolor=COLORS["light"])

ax1.set_title("(a) GCN 分类结果", fontsize=17, fontweight="bold")
ax1.axis("off")

# ── Panel (b): Embedding space ──────────────────────────────────────────────
for cls_id, color, label in [(0, COLORS["blue"], "Mr. Hi"),
                              (1, COLORS["red"], "Officer")]:
    mask = true_labels == cls_id
    ax2.scatter(
        H_2d[mask, 0], H_2d[mask, 1],
        c=color, label=label, s=80, alpha=0.8,
        edgecolors="white", linewidths=0.5,
    )

# Highlight labeled nodes in embedding space
for node_id in labeled_nodes:
    ax2.scatter(
        H_2d[node_id, 0], H_2d[node_id, 1],
        c=color_map[node_id], s=160, alpha=1.0,
        edgecolors="black", linewidths=2.5, zorder=5,
    )
    ax2.annotate(
        str(node_id),
        (H_2d[node_id, 0], H_2d[node_id, 1]),
        textcoords="offset points", xytext=(8, 8),
        fontsize=12, fontweight="bold",
    )

ax2.set_xlabel("PC1", fontsize=16)
ax2.set_ylabel("PC2", fontsize=16)
ax2.set_title("(b) 嵌入空间", fontsize=17, fontweight="bold")
ax2.tick_params(axis="both", labelsize=14)
ax2.legend(fontsize=14, loc="best", framealpha=0.9, edgecolor=COLORS["light"])

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_7_03_case_result")
