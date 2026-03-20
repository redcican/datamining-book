"""
Figure 10.3.2 - Louvain community detection on Zachary's Karate Club.

Panel (a): Network graph with nodes colored by detected community.
Panel (b): Horizontal bar chart comparing modularity Q across different
           community partitions (singleton, random, true labels, Louvain).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

apply_style()

# ── 1. Build graph and detect communities ────────────────────────────────
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42, k=0.5)

# Primary: networkx greedy_modularity_communities (Clauset-Newman-Moore,
# similar quality to Louvain on small graphs).
# Optional fallback to python-louvain if available.
try:
    import community as community_louvain

    partition = community_louvain.best_partition(G, random_state=42)
    communities_list = []
    for cid in sorted(set(partition.values())):
        communities_list.append({n for n, c in partition.items() if c == cid})
    Q_louvain = community_louvain.modularity(partition, G)
except ImportError:
    communities_sorted = list(greedy_modularity_communities(G))
    partition = {}
    for i, comm in enumerate(communities_sorted):
        for node in comm:
            partition[node] = i
    communities_list = [set(c) for c in communities_sorted]
    Q_louvain = nx.community.modularity(G, communities_list)

num_communities = len(communities_list)

# ── 2. Compute modularity for different partitions ───────────────────────
nodes = list(G.nodes())

# Singleton: each node is its own community
singleton_communities = [{n} for n in nodes]
Q_singleton = nx.community.modularity(G, singleton_communities)

# True partition: based on the club attribute (Mr. Hi vs Officer)
club_map = nx.get_node_attributes(G, "club")
true_groups = {}
for n, club in club_map.items():
    true_groups.setdefault(club, set()).add(n)
true_communities = list(true_groups.values())
Q_true = nx.community.modularity(G, true_communities)

# Random partition: average over 100 random 2-splits
rng = np.random.RandomState(42)
Q_random_list = []
for _ in range(100):
    labels = rng.randint(0, 2, size=len(nodes))
    rand_comms = [
        {nodes[j] for j in range(len(nodes)) if labels[j] == k}
        for k in range(2)
    ]
    # Remove empty sets
    rand_comms = [c for c in rand_comms if len(c) > 0]
    Q_random_list.append(nx.community.modularity(G, rand_comms))
Q_random = np.mean(Q_random_list)

# ── 3. Create figure ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "图 10.3.2　Louvain 算法",
    fontsize=22,
    fontweight="bold",
    y=0.98,
)

# ── Panel (a): Community detection result ────────────────────────────────
# Assign colors from a qualitative palette
community_colors = [
    COLORS["blue"],
    COLORS["red"],
    COLORS["green"],
    COLORS["orange"],
    COLORS["purple"],
    COLORS["teal"],
]
node_colors = [community_colors[partition[n] % len(community_colors)] for n in G.nodes()]

# Draw edges
nx.draw_networkx_edges(
    G, pos, ax=ax1,
    edge_color=COLORS["light"],
    alpha=0.5,
    width=0.8,
)

# Draw nodes
nx.draw_networkx_nodes(
    G, pos, ax=ax1,
    node_color=node_colors,
    node_size=300,
    edgecolors="white",
    linewidths=1.2,
)

# Draw node labels
nx.draw_networkx_labels(
    G, pos, ax=ax1,
    font_size=8,
    font_color="white",
    font_weight="bold",
)

# Legend for communities
for i in range(num_communities):
    ax1.scatter(
        [], [],
        c=community_colors[i % len(community_colors)],
        s=80,
        label=f"社区 {i + 1}",
        edgecolors="white",
        linewidths=0.5,
    )
ax1.legend(
    loc="upper left",
    fontsize=12,
    framealpha=0.9,
    title="社区划分",
    title_fontsize=13,
)

ax1.set_title(
    f"(a) 社区检测结果 (Q = {Q_louvain:.4f})",
    fontsize=17,
    fontweight="bold",
    pad=12,
)
ax1.axis("off")

# ── Panel (b): Modularity comparison ────────────────────────────────────
methods = [
    "单点社区\n(Singleton)",
    "随机划分\n(Random)",
    "真实标签\n(Ground Truth)",
    "Louvain",
]
Q_values = [Q_singleton, Q_random, Q_true, Q_louvain]
bar_colors = [COLORS["gray"], COLORS["orange"], COLORS["teal"], COLORS["blue"]]

y_positions = np.arange(len(methods))
bars = ax2.barh(
    y_positions, Q_values,
    color=bar_colors,
    height=0.55,
    edgecolor="white",
    linewidth=1.2,
)

# Annotate Q values on bars
for bar, q_val in zip(bars, Q_values):
    # Place label to the right of the bar (or to the left if negative)
    if q_val >= 0:
        x_text = q_val + 0.01
        ha = "left"
    else:
        x_text = q_val - 0.01
        ha = "right"
    ax2.text(
        x_text,
        bar.get_y() + bar.get_height() / 2,
        f"Q = {q_val:.4f}",
        va="center",
        ha=ha,
        fontsize=14,
        fontweight="bold",
        color=COLORS["gray"],
    )

ax2.set_yticks(y_positions)
ax2.set_yticklabels(methods, fontsize=14)
ax2.set_xlabel("模块度 Q", fontsize=16)
ax2.tick_params(axis="x", labelsize=14)
ax2.set_title(
    "(b) 不同划分方式的模块度对比",
    fontsize=17,
    fontweight="bold",
    pad=12,
)

# Add vertical line at Q = 0
ax2.axvline(x=0, color=COLORS["gray"], linewidth=0.8, linestyle="--", alpha=0.5)

# Extend x-axis slightly for annotation space
x_min = min(Q_values) - 0.08
x_max = max(Q_values) + 0.12
ax2.set_xlim(x_min, x_max)

# Invert y-axis so Louvain appears at bottom (best result)
ax2.invert_yaxis()

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── 4. Save ──────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_3_02_louvain")
