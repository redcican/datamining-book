"""
图 13.4.3　社区发现结果对比
(a) 真实派系  (b) Louvain 算法  (c) Girvan-Newman 算法
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import networkx as nx
from networkx.algorithms.community import (
    louvain_communities, girvan_newman, modularity)
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_social import load_graph, get_community_labels

G, ground_truth, pos = load_graph()
n = G.number_of_nodes()

# ── 真实标签 ──────────────────────────────────────────────
gt_labels = np.array([ground_truth[i] for i in range(n)])
gt_communities = [
    frozenset(i for i in range(n) if ground_truth[i] == 0),
    frozenset(i for i in range(n) if ground_truth[i] == 1),
]
gt_mod = modularity(G, gt_communities)

# ── Louvain ───────────────────────────────────────────────
louvain_comms = louvain_communities(G, seed=42, resolution=1.0)
louvain_labels = get_community_labels(louvain_comms, n)
louvain_mod = modularity(G, louvain_comms)
louvain_nmi = normalized_mutual_info_score(gt_labels, louvain_labels)
louvain_ari = adjusted_rand_score(gt_labels, louvain_labels)

# ── Girvan-Newman ─────────────────────────────────────────
gn_gen = girvan_newman(G)
gn_comms_2 = next(gn_gen)  # First split → 2 communities
gn_labels = get_community_labels(gn_comms_2, n)
gn_mod = modularity(G, gn_comms_2)
gn_nmi = normalized_mutual_info_score(gt_labels, gn_labels)
gn_ari = adjusted_rand_score(gt_labels, gn_labels)

print("=== 社区发现结果 ===")
print(f"  真实派系:    社区数={len(gt_communities)}, "
      f"Q={gt_mod:.4f}")
print(f"  Louvain:     社区数={len(louvain_comms)}, "
      f"Q={louvain_mod:.4f}, NMI={louvain_nmi:.4f}, ARI={louvain_ari:.4f}")
print(f"  Girvan-Newman: 社区数={len(gn_comms_2)}, "
      f"Q={gn_mod:.4f}, NMI={gn_nmi:.4f}, ARI={gn_ari:.4f}")

# 错误分配的节点
for name, labels in [("Louvain", louvain_labels), ("Girvan-Newman", gn_labels)]:
    mismatches = [i for i in range(n) if labels[i] != gt_labels[i]]
    # 简单对齐：如果多数不匹配则翻转
    if len(mismatches) > n // 2:
        labels_flipped = 1 - labels
        mismatches = [i for i in range(n) if labels_flipped[i] != gt_labels[i]]
    print(f"  {name} 错误节点: {mismatches} ({len(mismatches)} 个)")

# ── 绘图 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
color_maps = [
    {0: COLORS["blue"], 1: COLORS["red"]},
    {i: PALETTE[i % len(PALETTE)] for i in range(10)},
    {0: COLORS["green"], 1: COLORS["orange"]},
]

datasets = [
    ("(a) 真实派系 (Ground Truth)", gt_labels, color_maps[0]),
    ("(b) Louvain 算法", louvain_labels, color_maps[1]),
    ("(c) Girvan-Newman 算法", gn_labels, color_maps[2]),
]

for ax, (title, labels, cmap_dict) in zip(axes, datasets):
    node_colors = [cmap_dict.get(labels[n_id], COLORS["gray"])
                   for n_id in G.nodes()]
    degrees = dict(G.degree())
    node_sizes = [degrees[n_id] * 60 + 100 for n_id in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.7,
                           edge_color=COLORS["gray"])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="white",
                           linewidths=1.2, alpha=0.9)

    # 节点编号
    for n_id in G.nodes():
        x, y = pos[n_id]
        ax.text(x, y, str(n_id), fontsize=7, ha="center", va="center",
                color="white", fontweight="bold")

    # 社区数 + 模块度标注
    n_comms = len(set(labels))
    if title.startswith("(a)"):
        info = f"Q = {gt_mod:.3f}"
    elif title.startswith("(b)"):
        info = f"Q = {louvain_mod:.3f}, NMI = {louvain_nmi:.3f}"
    else:
        info = f"Q = {gn_mod:.3f}, NMI = {gn_nmi:.3f}"

    ax.text(0.5, -0.02, f"社区数: {n_comms} | {info}",
            transform=ax.transAxes, ha="center", fontsize=10,
            color=COLORS["gray"])

    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.axis("off")
    ax.grid(False)

plt.tight_layout(w_pad=1.5)
save_fig(fig, __file__, "fig13_4_03_community_detection")
