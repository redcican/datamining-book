"""
图 13.4.1　网络概览与度分布
(a) Zachary Karate Club 网络图（节点大小 ∝ 度，颜色 = 真实派系）
(b) 度分布直方图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS, add_panel_label

apply_style()

from _load_social import load_graph

G, ground_truth, pos = load_graph()

# ── 基本统计 ──────────────────────────────────────────────
print("=== Zachary Karate Club 基本统计 ===")
print(f"  节点数: {G.number_of_nodes()}")
print(f"  边数:   {G.number_of_edges()}")
print(f"  密度:   {nx.density(G):.4f}")
print(f"  平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
print(f"  聚类系数: {nx.average_clustering(G):.4f}")
print(f"  直径:   {nx.diameter(G)}")
print(f"  派系 0 (Mr. Hi): {sum(1 for v in ground_truth.values() if v == 0)} 人")
print(f"  派系 1 (Officer): {sum(1 for v in ground_truth.values() if v == 1)} 人")

degrees = dict(G.degree())
top5 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n  度最高的 5 个节点:")
for node, deg in top5:
    club = "Mr. Hi" if ground_truth[node] == 0 else "Officer"
    print(f"    节点 {node:>2d}: 度={deg:>2d}  ({club})")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                gridspec_kw={"width_ratios": [1.3, 1]})

# (a) 网络图
node_colors = [COLORS["blue"] if ground_truth[n] == 0 else COLORS["red"]
               for n in G.nodes()]
node_sizes = [degrees[n] * 80 + 100 for n in G.nodes()]

nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, width=0.8,
                       edge_color=COLORS["gray"])
nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                       node_size=node_sizes, edgecolors="white",
                       linewidths=1.5, alpha=0.9)

# 标注关键节点
key_nodes = {0: "0\n(Mr. Hi)", 33: "33\n(Officer)"}
for node, label in key_nodes.items():
    x, y = pos[node]
    ax1.annotate(label, (x, y), fontsize=9, fontweight="bold",
                 ha="center", va="bottom",
                 xytext=(0, 12), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                           alpha=0.8))

# 其他节点编号
for n in G.nodes():
    if n not in key_nodes:
        x, y = pos[n]
        ax1.text(x, y, str(n), fontsize=7, ha="center", va="center",
                 color="white", fontweight="bold")

from matplotlib.patches import Patch
ax1.legend([Patch(facecolor=COLORS["blue"]),
            Patch(facecolor=COLORS["red"])],
           ["Mr. Hi 派系", "Officer 派系"],
           loc="upper left", fontsize=10)
ax1.set_title("(a) 空手道俱乐部社交网络", fontweight="bold")
ax1.axis("off")
ax1.grid(False)

# (b) 度分布
deg_values = list(degrees.values())
bins = range(min(deg_values), max(deg_values) + 2)
ax2.hist(deg_values, bins=bins, color=COLORS["blue"], edgecolor="white",
         alpha=0.85, align="left", rwidth=0.8)
ax2.set_xlabel("节点度 (Degree)")
ax2.set_ylabel("节点数量")
ax2.set_title("(b) 度分布", fontweight="bold")
ax2.axvline(np.mean(deg_values), color=COLORS["red"], linestyle="--",
            linewidth=1.5, label=f"平均度 = {np.mean(deg_values):.1f}")
ax2.legend(fontsize=10)

plt.tight_layout(w_pad=2)
save_fig(fig, __file__, "fig13_4_01_network_overview")
