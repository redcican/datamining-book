"""
图 10.6.2 链接预测方法对比
Left panel:  Adamic-Adar score heatmap on Karate Club (non-edges only)
Right panel: AUC comparison of CN / Jaccard / AA / PA / Katz on held-out edges
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

np.random.seed(42)

# ── Build Karate Club graph ──────────────────────────────────────────────
G_full = nx.karate_club_graph()
nodes = sorted(G_full.nodes())
n = len(nodes)
node_idx = {v: i for i, v in enumerate(nodes)}

# ── Panel (a): Adamic-Adar score matrix (non-edges only) ────────────────
edges_set = set(G_full.edges())
aa_matrix = np.zeros((n, n))
for u, v, score in nx.adamic_adar_index(G_full):
    i, j = node_idx[u], node_idx[v]
    # Only fill non-edges
    if (u, v) not in edges_set and (v, u) not in edges_set:
        aa_matrix[i, j] = score
        aa_matrix[j, i] = score

# ── Panel (b): AUC evaluation with train/test split ─────────────────────
edges = list(G_full.edges())
np.random.shuffle(edges)
n_test = max(1, int(len(edges) * 0.2))
test_edges = edges[:n_test]
train_edges = edges[n_test:]

G_train = nx.Graph()
G_train.add_nodes_from(G_full.nodes())
G_train.add_edges_from(train_edges)

# Generate negative samples (non-edges in full graph)
non_edges_full = list(nx.non_edges(G_full))
np.random.shuffle(non_edges_full)
neg_edges = non_edges_full[:n_test]

# All test pairs: positive (removed edges) + negative (non-edges)
test_pairs = test_edges + neg_edges
y_true = np.array([1] * len(test_edges) + [0] * len(neg_edges))

# --- Scoring functions on training graph ---

# Common Neighbors
cn_preds = nx.common_neighbor_centrality(G_train, test_pairs)
# nx.common_neighbor_centrality may not exist; use resource_allocation or manual
# Use manual computation for robustness
def common_neighbors_score(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))

def jaccard_score_pair(G, u, v):
    cn = set(G.neighbors(u)) & set(G.neighbors(v))
    union = set(G.neighbors(u)) | set(G.neighbors(v))
    if len(union) == 0:
        return 0.0
    return len(cn) / len(union)

def adamic_adar_score(G, u, v):
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        deg = G.degree(w)
        if deg > 1:
            score += 1.0 / np.log(deg)
    return score

def preferential_attachment_score(G, u, v):
    return G.degree(u) * G.degree(v)

# Katz (truncated): score = beta*A + beta^2*A^2 + beta^3*A^3
A_train = nx.to_numpy_array(G_train, nodelist=nodes)
beta = 0.01
A2 = A_train @ A_train
A3 = A2 @ A_train
katz_matrix = beta * A_train + beta**2 * A2 + beta**3 * A3

methods = {
    "Common Neighbors": lambda u, v: common_neighbors_score(G_train, u, v),
    "Jaccard": lambda u, v: jaccard_score_pair(G_train, u, v),
    "Adamic-Adar": lambda u, v: adamic_adar_score(G_train, u, v),
    "Pref. Attach.": lambda u, v: preferential_attachment_score(G_train, u, v),
    "Katz (truncated)": lambda u, v: katz_matrix[node_idx[u], node_idx[v]],
}

auc_results = {}
for name, scorer in methods.items():
    scores = np.array([scorer(u, v) for u, v in test_pairs])
    try:
        auc_val = roc_auc_score(y_true, scores)
    except ValueError:
        auc_val = 0.5
    auc_results[name] = auc_val

# ── Plotting ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.6.2  链接预测方法对比", fontsize=22)

# Panel (a): Heatmap
sns.heatmap(
    aa_matrix,
    ax=ax1,
    cmap="Blues",
    square=True,
    cbar_kws={"shrink": 0.75, "label": "AA 评分"},
    linewidths=0,
    xticklabels=5,
    yticklabels=5,
)
ax1.set_title("(a) Adamic-Adar 评分矩阵", fontsize=17)
ax1.set_xlabel("节点编号", fontsize=16)
ax1.set_ylabel("节点编号", fontsize=16)
ax1.tick_params(labelsize=14)
# Restore spines for heatmap
for spine in ax1.spines.values():
    spine.set_visible(True)
ax1.set_aspect("equal")

# Panel (b): Horizontal bar chart
method_names = list(auc_results.keys())
auc_values = [auc_results[m] for m in method_names]

bars = ax2.barh(
    method_names,
    auc_values,
    color=COLORS["blue"],
    edgecolor="white",
    height=0.55,
)
# Value labels
for bar, val in zip(bars, auc_values):
    ax2.text(
        val + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

ax2.set_title("(b) 链接预测 AUC 对比", fontsize=17)
ax2.set_xlabel("AUC", fontsize=16)
ax2.set_ylabel("预测方法", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.set_xlim(0, 1.05)
ax2.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, __file__, "fig10_6_02_similarity_comparison")
