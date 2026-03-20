"""
Figure 10.2.3 - Node similarity comparison on Zachary's Karate Club.

Panel (a): Jaccard similarity matrix heatmap with nodes reordered by community.
Panel (b): Box plots comparing CN, Jaccard, and Adamic-Adar distributions
           for intra-community vs inter-community node pairs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import seaborn as sns

apply_style()

# ── 1. Load graph and extract community labels ──────────────────────────
G = nx.karate_club_graph()
n = G.number_of_nodes()

# Community labels: 'Mr. Hi' or 'Officer'
community = {node: G.nodes[node]["club"] for node in G.nodes()}

# Reorder nodes: Mr. Hi first, then Officer (for heatmap clarity)
hi_nodes = sorted([v for v in G.nodes() if community[v] == "Mr. Hi"])
officer_nodes = sorted([v for v in G.nodes() if community[v] == "Officer"])
ordered_nodes = hi_nodes + officer_nodes
split_idx = len(hi_nodes)  # boundary between two communities

# Mapping from original node id to reordered index
node_to_idx = {node: i for i, node in enumerate(ordered_nodes)}

# ── 2. Compute Jaccard similarity matrix (reordered) ────────────────────
jaccard_matrix = np.zeros((n, n))
for i, u in enumerate(ordered_nodes):
    for j, v in enumerate(ordered_nodes):
        if u == v:
            jaccard_matrix[i][j] = 1.0
        else:
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            cn = len(neighbors_u & neighbors_v)
            union = len(neighbors_u | neighbors_v)
            jaccard_matrix[i][j] = cn / union if union > 0 else 0


# ── 3. Similarity metrics for all node pairs ────────────────────────────
def common_neighbors_count(G, u, v):
    """Common Neighbors count."""
    return len(set(G.neighbors(u)) & set(G.neighbors(v)))


def jaccard_coefficient(G, u, v):
    """Jaccard similarity coefficient."""
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    union = len(neighbors_u | neighbors_v)
    if union == 0:
        return 0.0
    return len(neighbors_u & neighbors_v) / union


def adamic_adar(G, u, v):
    """Adamic-Adar index."""
    cn = set(G.neighbors(u)) & set(G.neighbors(v))
    return sum(1 / np.log(G.degree(w)) for w in cn if G.degree(w) > 1)


# Compute for all distinct pairs
cn_intra, cn_inter = [], []
jac_intra, jac_inter = [], []
aa_intra, aa_inter = [], []

nodes_list = list(G.nodes())
for i in range(len(nodes_list)):
    for j in range(i + 1, len(nodes_list)):
        u, v = nodes_list[i], nodes_list[j]
        same_community = community[u] == community[v]

        cn_val = common_neighbors_count(G, u, v)
        jac_val = jaccard_coefficient(G, u, v)
        aa_val = adamic_adar(G, u, v)

        if same_community:
            cn_intra.append(cn_val)
            jac_intra.append(jac_val)
            aa_intra.append(aa_val)
        else:
            cn_inter.append(cn_val)
            jac_inter.append(jac_val)
            aa_inter.append(aa_val)

# ── 4. Create figure ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "图 10.2.3　节点相似度对比",
    fontsize=22,
    fontweight="bold",
    y=1.02,
)

# ── Panel (a): Jaccard similarity heatmap ────────────────────────────────
sns.heatmap(
    jaccard_matrix,
    ax=ax1,
    cmap="YlOrRd",
    square=True,
    xticklabels=ordered_nodes,
    yticklabels=ordered_nodes,
    cbar_kws={"shrink": 0.8, "label": "Jaccard 相似度"},
    linewidths=0,
    vmin=0,
    vmax=1,
)

# Draw red dashed lines separating the two communities
ax1.axhline(y=split_idx, color=COLORS["red"], linestyle="--", linewidth=2)
ax1.axvline(x=split_idx, color=COLORS["red"], linestyle="--", linewidth=2)

ax1.set_title("(a) Jaccard 相似度矩阵", fontsize=17, fontweight="bold")
ax1.tick_params(labelsize=7)

# Turn off the default grid for the heatmap (it interferes with the matrix)
ax1.grid(False)

# ── Panel (b): Box plots of similarity distributions ────────────────────
# Prepare data for grouped box plots
import pandas as pd

records = []
for val in cn_intra:
    records.append({"指标": "CN", "类型": "社区内", "值": val})
for val in cn_inter:
    records.append({"指标": "CN", "类型": "社区间", "值": val})
for val in jac_intra:
    records.append({"指标": "Jaccard", "类型": "社区内", "值": val})
for val in jac_inter:
    records.append({"指标": "Jaccard", "类型": "社区间", "值": val})
for val in aa_intra:
    records.append({"指标": "Adamic-Adar", "类型": "社区内", "值": val})
for val in aa_inter:
    records.append({"指标": "Adamic-Adar", "类型": "社区间", "值": val})

df = pd.DataFrame(records)

color_map = {"社区内": COLORS["blue"], "社区间": COLORS["red"]}

sns.boxplot(
    data=df,
    x="指标",
    y="值",
    hue="类型",
    ax=ax2,
    palette=color_map,
    width=0.6,
    linewidth=1.5,
    fliersize=3,
)

ax2.set_title("(b) 社区内 vs 社区间相似度", fontsize=17, fontweight="bold", pad=12)
ax2.set_xlabel("相似度指标", fontsize=16)
ax2.set_ylabel("相似度值", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=14, title=None, loc="upper right")

fig.tight_layout(rect=[0, 0, 1, 0.95])

# ── 5. Save ─────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_2_03_node_similarity")
