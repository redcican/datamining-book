"""
图 10.3.3  社区发现方法对比
Compare community detection methods on synthetic planted-partition graphs and
on Zachary's Karate Club.

(a) NMI vs mixing parameter μ — spectral clustering vs greedy modularity on
    planted partition graphs with increasing inter-community edge probability.
(b) Grouped bar chart — NMI and modularity Q for three methods on Karate Club.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.generators.community import planted_partition_graph
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    modularity,
)
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

# ── Panel (a): NMI vs mixing parameter μ ─────────────────────────────────
n_communities = 4
n_per = 30
p_in = 0.3
mu_values = np.linspace(0.05, 0.50, 10)

# True labels: nodes 0..29 → community 0, 30..59 → 1, etc.
true_labels_synth = []
for i in range(n_communities):
    true_labels_synth.extend([i] * n_per)
true_labels_synth = np.array(true_labels_synth)

nmi_spectral = []
nmi_greedy = []

for mu in mu_values:
    p_out = mu * p_in  # inter-community edge probability
    G_synth = planted_partition_graph(
        n_communities, n_per, p_in, p_out, seed=42
    )

    # ── Spectral clustering (k = n_communities) ──────────────────────────
    L_sym = nx.normalized_laplacian_matrix(G_synth).toarray()
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # Use the first k eigenvectors (smallest eigenvalues) for embedding
    X_embed = eigvecs[:, :n_communities]
    km = KMeans(n_clusters=n_communities, n_init=20, random_state=42)
    spectral_pred = km.fit_predict(X_embed)
    nmi_spectral.append(
        normalized_mutual_info_score(true_labels_synth, spectral_pred)
    )

    # ── Greedy modularity ─────────────────────────────────────────────────
    greedy_comms = list(greedy_modularity_communities(G_synth))
    greedy_pred = np.zeros(G_synth.number_of_nodes(), dtype=int)
    for idx, comm in enumerate(greedy_comms):
        for node in comm:
            greedy_pred[node] = idx
    nmi_greedy.append(
        normalized_mutual_info_score(true_labels_synth, greedy_pred)
    )

# ── Panel (b): Methods on Karate Club ────────────────────────────────────
G_karate = nx.karate_club_graph()
n_karate = G_karate.number_of_nodes()

# True labels: Mr. Hi = 0, Officer = 1
true_labels_karate = np.array(
    [0 if G_karate.nodes[v]["club"] == "Mr. Hi" else 1 for v in G_karate.nodes()]
)

# ── Spectral clustering (k = 2) ──────────────────────────────────────────
L_k = nx.normalized_laplacian_matrix(G_karate).toarray()
eigvals_k, eigvecs_k = np.linalg.eigh(L_k)
spectral_labels_k = (eigvecs_k[:, 1] > 0).astype(int)

# ── Greedy modularity ────────────────────────────────────────────────────
greedy_comms_k = list(greedy_modularity_communities(G_karate))
greedy_labels_k = np.zeros(n_karate, dtype=int)
for idx, comm in enumerate(greedy_comms_k):
    for node in comm:
        greedy_labels_k[node] = idx

# ── Label propagation ────────────────────────────────────────────────────
# Run multiple times and pick the best (LP is non-deterministic)
best_lp_nmi = -1.0
best_lp_labels = None
for _ in range(50):
    lp_comms_k = list(label_propagation_communities(G_karate))
    lp_labels_k = np.zeros(n_karate, dtype=int)
    for idx, comm in enumerate(lp_comms_k):
        for node in comm:
            lp_labels_k[node] = idx
    cur_nmi = normalized_mutual_info_score(true_labels_karate, lp_labels_k)
    if cur_nmi > best_lp_nmi:
        best_lp_nmi = cur_nmi
        best_lp_labels = lp_labels_k.copy()
lp_labels_k = best_lp_labels

# Helper: convert label array to list-of-sets for modularity()
def labels_to_communities(labels, G):
    comms = {}
    for node in G.nodes():
        c = labels[node]
        comms.setdefault(c, set()).add(node)
    return list(comms.values())

# Compute NMI and Q for each method
methods = ["谱聚类", "贪心模块度", "标签传播"]
all_labels = [spectral_labels_k, greedy_labels_k, lp_labels_k]

nmis = []
qs = []
for lbl in all_labels:
    nmis.append(normalized_mutual_info_score(true_labels_karate, lbl))
    comm_sets = labels_to_communities(lbl, G_karate)
    qs.append(modularity(G_karate, comm_sets))

# ── Figure ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "图 10.3.3　社区发现方法对比",
    fontsize=22, fontweight="bold", y=0.98,
)

# ── Panel (a) ─────────────────────────────────────────────────────────────
ax1.plot(
    mu_values, nmi_spectral,
    "o-", color=COLORS["blue"], label="谱聚类", markersize=7, linewidth=2.2,
)
ax1.plot(
    mu_values, nmi_greedy,
    "s--", color=COLORS["red"], label="贪心模块度", markersize=7, linewidth=2.2,
)
ax1.set_xlabel("混合参数 μ (p_out / p_in)", fontsize=16)
ax1.set_ylabel("NMI", fontsize=16)
ax1.set_title("(a) NMI vs 混合参数", fontsize=17, fontweight="bold")
ax1.set_ylim(-0.05, 1.05)
ax1.tick_params(axis="both", labelsize=14)
ax1.legend(fontsize=13, loc="lower left", framealpha=0.9, edgecolor=COLORS["light"])

# ── Panel (b) ─────────────────────────────────────────────────────────────
x_pos = np.arange(len(methods))
bar_width = 0.32

bars_nmi = ax2.bar(
    x_pos - bar_width / 2, nmis,
    width=bar_width, color=COLORS["blue"], label="NMI", edgecolor="white",
    linewidth=0.8,
)
bars_q = ax2.bar(
    x_pos + bar_width / 2, qs,
    width=bar_width, color=COLORS["green"], label="模块度 Q", edgecolor="white",
    linewidth=0.8,
)

# Value labels on bars
for bar in bars_nmi:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2, h + 0.02,
        f"{h:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        color=COLORS["blue"],
    )
for bar in bars_q:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2, h + 0.02,
        f"{h:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        color=COLORS["green"],
    )

ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods, fontsize=14)
ax2.set_ylabel("分数", fontsize=16)
ax2.set_title("(b) 空手道俱乐部结果对比", fontsize=17, fontweight="bold")
ax2.set_ylim(0, 1.15)
ax2.tick_params(axis="both", labelsize=14)
ax2.legend(fontsize=13, loc="upper right", framealpha=0.9, edgecolor=COLORS["light"])

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_3_03_method_comparison")
