"""
fig9_6_01_textrank.py
TextRank 摘要算法：networkx 句子相似度图 + 迭代收敛过程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 数据准备 ──────────────────────────────────────────────────────
labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
n_nodes = len(labels)
top2_idx = {0, 2}  # S1, S3
sim_matrix = np.array([
    [0.00, 0.18, 0.30, 0.05, 0.22, 0.03],
    [0.18, 0.00, 0.15, 0.12, 0.08, 0.04],
    [0.30, 0.15, 0.00, 0.07, 0.25, 0.06],
    [0.05, 0.12, 0.07, 0.00, 0.14, 0.11],
    [0.22, 0.08, 0.25, 0.14, 0.00, 0.09],
    [0.03, 0.04, 0.06, 0.11, 0.09, 0.00],
])
# ── 构建 networkx 图 ─────────────────────────────────────────────
G = nx.Graph()
for i in range(n_nodes):
    G.add_node(labels[i])
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        w = sim_matrix[i, j]
        if w > 0.06:
            G.add_edge(labels[i], labels[j], weight=w)
# TextRank 得分（用 networkx pagerank）
pr = nx.pagerank(G, alpha=0.85, weight="weight")
scores = np.array([pr[l] for l in labels])
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.6.1　TextRank 摘要算法",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：networkx 图可视化 ──────────────────────────────────
ax = axes[0]
pos = nx.circular_layout(G)
node_colors = [COLORS["red"] if i in top2_idx else COLORS["blue"]
               for i in range(n_nodes)]
node_sizes = [800 + scores[i] * 5000 for i in range(n_nodes)]
edge_widths = [G[u][v]["weight"] * 12 for u, v in G.edges()]
# 画边
nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                       edge_color=COLORS["gray"], alpha=0.4)
# 画边权标签
edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=10,
                             font_color=COLORS["gray"],
                             bbox=dict(boxstyle="round,pad=0.1",
                                       fc="white", ec="none", alpha=0.8))
# 画节点
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                       node_size=node_sizes, alpha=0.9,
                       edgecolors="white", linewidths=2)
# 画标签
node_labels = {l: f"{l}\n({scores[i]:.2f})" for i, l in enumerate(labels)}
nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=12,
                        font_weight="bold", font_color="white")
# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["red"],
           markersize=14, label="Top-2 句子"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["blue"],
           markersize=14, label="其他句子"),
]
ax.legend(handles=legend_elements, fontsize=14, loc="lower right")
ax.set_title("(a) 句子相似度图", fontsize=17)
ax.axis("off")
# ── 右面板：TextRank 迭代收敛 ─────────────────────────────────
ax = axes[1]
damping = 0.85
n_iter = 21
col_sums = sim_matrix.sum(axis=0)
col_sums[col_sums == 0] = 1.0
trans_matrix = sim_matrix / col_sums
score_history = np.zeros((n_iter, n_nodes))
score_history[0] = 1.0 / n_nodes
for t in range(1, n_iter):
    score_history[t] = ((1 - damping) / n_nodes
                        + damping * trans_matrix @ score_history[t - 1])
for i in range(n_nodes):
    ax.plot(range(n_iter), score_history[:, i], color=PALETTE[i],
            lw=2.0, marker="o", markersize=3, label=labels[i])
ax.set_xlabel("迭代次数", fontsize=16)
ax.set_ylabel("TextRank 得分", fontsize=16)
ax.legend(fontsize=14, loc="right", ncol=1)
ax.tick_params(labelsize=14)
ax.set_title("(b) TextRank 迭代收敛", fontsize=17)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_6_01_textrank")
