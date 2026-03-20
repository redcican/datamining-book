"""
图 10.4.2  图分类流程
Demonstrate graph classification: generate cycle, tree, and random graphs,
extract statistical features, and visualise with PCA.

(a) 三种图示例 — one cycle graph, one tree graph, one Erdos-Renyi random graph.
(b) PCA 特征空间 — scatter plot of 6-dimensional graph features projected to 2D.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

np.random.seed(42)

# ── Graph generators ─────────────────────────────────────────────────────
def make_cycle(n):
    return nx.cycle_graph(n)

def make_tree(n):
    return nx.random_labeled_tree(n, seed=np.random.randint(0, 2**31))

def make_random(n):
    return nx.erdos_renyi_graph(n, p=0.3, seed=np.random.randint(0, 2**31))

# ── Feature extraction ───────────────────────────────────────────────────
def extract_features(G):
    """Return 6-dimensional feature vector for a graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    cc = nx.average_clustering(G)
    density = nx.density(G)
    return [n, m, mean_deg, std_deg, cc, density]

# ── Generate datasets ────────────────────────────────────────────────────
n_per_class = 50
class_names = ["环图 (Cycle)", "树图 (Tree)", "随机图 (Random)"]
generators = [make_cycle, make_tree, make_random]
class_colors = [COLORS["blue"], COLORS["green"], COLORS["red"]]

features = []
labels = []
for cls_idx, gen in enumerate(generators):
    for _ in range(n_per_class):
        n_nodes = np.random.randint(10, 21)  # 10-20 nodes
        G = gen(n_nodes)
        features.append(extract_features(G))
        labels.append(cls_idx)

features = np.array(features)
labels = np.array(labels)

# ── PCA ──────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(features)

# ── Example graphs for Panel (a) ─────────────────────────────────────────
np.random.seed(42)
example_graphs = [
    nx.cycle_graph(12),
    nx.random_labeled_tree(12, seed=42),
    nx.erdos_renyi_graph(12, 0.3, seed=42),
]
example_titles = ["环图 (Cycle)", "树图 (Tree)", "随机图 (Random)"]

# ── Figure layout ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6))
fig.suptitle(
    "图 10.4.2　图分类流程",
    fontsize=22, fontweight="bold", y=0.98,
)

# Outer grid: left half for panel (a), right half for panel (b)
outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30, width_ratios=[1, 1])

# ── Panel (a): 三种图示例 (3 small subplots) ─────────────────────────────
inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.35)

for i, (G_ex, title, color) in enumerate(
    zip(example_graphs, example_titles, class_colors)
):
    ax = fig.add_subplot(inner_gs[i])
    pos = nx.spring_layout(G_ex, seed=42)
    nx.draw_networkx(
        G_ex, pos, ax=ax,
        node_color=color, node_size=200, edge_color=COLORS["gray"],
        width=1.5, font_size=0, with_labels=False, alpha=0.9,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    # Remove grid for graph drawing axes
    ax.grid(False)

# Add a shared panel title for (a)
fig.text(
    0.27, 0.03, "(a) 三种图示例",
    ha="center", fontsize=17, fontweight="bold",
)

# ── Panel (b): PCA 特征空间 ──────────────────────────────────────────────
ax_pca = fig.add_subplot(outer_gs[1])

for cls_idx in range(3):
    mask = labels == cls_idx
    ax_pca.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=class_colors[cls_idx], label=class_names[cls_idx],
        s=60, alpha=0.75, edgecolors="white", linewidths=0.5,
    )

ax_pca.set_xlabel("PC 1", fontsize=16)
ax_pca.set_ylabel("PC 2", fontsize=16)
ax_pca.set_title("(b) PCA 特征空间", fontsize=17, fontweight="bold")
ax_pca.tick_params(axis="both", labelsize=14)
ax_pca.legend(fontsize=14, loc="best", framealpha=0.9, edgecolor=COLORS["light"])

fig.tight_layout(rect=[0, 0.06, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_4_02_graph_classification")
