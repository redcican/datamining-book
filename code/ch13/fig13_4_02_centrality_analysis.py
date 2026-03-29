"""
图 13.4.2　节点中心性分析（四种中心性度量对比）
(a) 度中心性  (b) 介数中心性  (c) 接近中心性  (d) 特征向量中心性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_social import load_graph

G, ground_truth, pos = load_graph()

# ── 计算四种中心性 ──────────────────────────────────────
centralities = {
    "度中心性\n(Degree)": nx.degree_centrality(G),
    "介数中心性\n(Betweenness)": nx.betweenness_centrality(G),
    "接近中心性\n(Closeness)": nx.closeness_centrality(G),
    "特征向量中心性\n(Eigenvector)": nx.eigenvector_centrality(G, max_iter=1000),
}

# 打印 Top-5
print("=== 四种中心性度量 Top-5 ===")
for name, cent in centralities.items():
    name_flat = name.replace("\n", " ")
    top5 = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  {name_flat}:")
    for node, val in top5:
        print(f"    节点 {node:>2d}: {val:.4f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
cmap = cm.YlOrRd

titles = list(centralities.keys())
panels = ["(a)", "(b)", "(c)", "(d)"]

for ax, (title, cent), panel in zip(axes.flat, centralities.items(), panels):
    values = np.array([cent[n] for n in G.nodes()])
    vmin, vmax = values.min(), values.max()

    # 归一化用于颜色映射
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    node_colors = cmap(norm(values))
    node_sizes = values * 1500 + 100

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.6,
                           edge_color=COLORS["gray"])
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=node_sizes, edgecolors="white",
                                   linewidths=1.2)

    # 标注 Top-3 节点
    top3 = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:3]
    for node, val in top3:
        x, y = pos[node]
        ax.annotate(f"{node}\n({val:.3f})", (x, y),
                    fontsize=8, fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat",
                              alpha=0.85))

    ax.set_title(f"{panel} {title}", fontweight="bold", fontsize=12)
    ax.axis("off")
    ax.grid(False)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

fig.suptitle("节点中心性分析 —— Zachary Karate Club",
             fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, __file__, "fig13_4_02_centrality_analysis")
