"""
图 13.4.4　算法评估与中心性关联
(a) 四种算法 × 三种指标对比  (b) 中心性度量相关性热图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import (
    louvain_communities, girvan_newman,
    greedy_modularity_communities, label_propagation_communities,
    modularity)
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_social import load_graph, get_community_labels

G, ground_truth, pos = load_graph()
n = G.number_of_nodes()
gt_labels = np.array([ground_truth[i] for i in range(n)])

# ── 1. 四种算法 ──────────────────────────────────────────
algorithms = {}

# Louvain
comms = louvain_communities(G, seed=42, resolution=1.0)
algorithms["Louvain"] = comms

# Girvan-Newman (2 communities)
gn_gen = girvan_newman(G)
algorithms["Girvan-\nNewman"] = next(gn_gen)

# Label Propagation
algorithms["Label\nPropagation"] = label_propagation_communities(G)

# Greedy Modularity
algorithms["Greedy\nModularity"] = greedy_modularity_communities(G)

# 评估
results = {}
print("=== 四种算法评估 ===")
print(f"{'算法':<22s} {'社区数':>6s} {'Q':>8s} {'NMI':>8s} {'ARI':>8s}")
print("-" * 50)

for name, comms in algorithms.items():
    labels = get_community_labels(comms, n)
    q = modularity(G, comms)
    nmi = normalized_mutual_info_score(gt_labels, labels)
    ari = adjusted_rand_score(gt_labels, labels)
    results[name] = {"Q": q, "NMI": nmi, "ARI": ari,
                     "n_comms": len(set(labels))}
    name_flat = name.replace("\n", " ")
    print(f"  {name_flat:<20s} {len(set(labels)):>6d} "
          f"{q:>8.4f} {nmi:>8.4f} {ari:>8.4f}")

# ── 2. 中心性相关性 ──────────────────────────────────────
cent_data = pd.DataFrame({
    "Degree": pd.Series(nx.degree_centrality(G)),
    "Betweenness": pd.Series(nx.betweenness_centrality(G)),
    "Closeness": pd.Series(nx.closeness_centrality(G)),
    "Eigenvector": pd.Series(nx.eigenvector_centrality(G, max_iter=1000)),
})
corr = cent_data.corr()

print("\n=== 中心性相关性矩阵 ===")
print(corr.round(3).to_string())

# ── 3. 绘图 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                gridspec_kw={"width_ratios": [1.2, 1]})

# (a) 算法评估对比
algo_names = list(results.keys())
metrics = ["Q", "NMI", "ARI"]
metric_labels = ["模块度 (Q)", "NMI", "ARI"]
n_algos = len(algo_names)
n_metrics = len(metrics)
x = np.arange(n_algos)
width = 0.22
colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    vals = [results[a][metric] for a in algo_names]
    offset = (i - n_metrics / 2 + 0.5) * width
    bars = ax1.bar(x + offset, vals, width, label=label,
                   color=colors[i], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(algo_names, fontsize=10)
ax1.set_ylabel("分数")
ax1.set_title("(a) 社区发现算法评估", fontweight="bold")
ax1.set_ylim(0, 1.1)
ax1.legend(loc="upper right", fontsize=9)

# (b) 中心性相关性热图
im = ax2.imshow(corr.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(corr.columns)))
ax2.set_xticklabels(corr.columns, fontsize=10, rotation=30, ha="right")
ax2.set_yticks(range(len(corr.index)))
ax2.set_yticklabels(corr.index, fontsize=10)

# 数值标注
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        color = "white" if val > 0.7 else "black"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=11, fontweight="bold", color=color)

ax2.set_title("(b) 中心性度量相关性", fontweight="bold")
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label("Pearson 相关系数", fontsize=10)

# 恢复 spines 给热图
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_4_04_evaluation")
