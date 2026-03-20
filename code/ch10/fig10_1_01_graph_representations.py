"""
fig10_1_01_graph_representations.py
图的矩阵表示：示例图 + 邻接矩阵、度矩阵、拉普拉斯矩阵、归一化拉普拉斯矩阵
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import seaborn as sns
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

# ── 构建 6 节点示例图 ─────────────────────────────────────────────
G = nx.Graph()
node_labels = [f"$v_{i}$" for i in range(1, 7)]
G.add_nodes_from(range(6))
edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 4),
    (2, 3), (2, 5),
    (3, 5),
    (4, 5),
]
G.add_edges_from(edges)

n = G.number_of_nodes()

# ── 计算矩阵 ──────────────────────────────────────────────────────
A = nx.adjacency_matrix(G).toarray().astype(float)
degrees = np.array([G.degree(i) for i in range(n)], dtype=float)
D = np.diag(degrees)
L = D - A

# 归一化拉普拉斯: L_sym = D^{-1/2} L D^{-1/2}
D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
L_sym = D_inv_sqrt @ L @ D_inv_sqrt

# ── 布局 (2x3 gridspec, 左列合并为图面板) ──────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("图 10.1.1　图的矩阵表示", fontsize=22, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.40)

ax_graph = fig.add_subplot(gs[:, 0])  # 左列跨两行
ax_A = fig.add_subplot(gs[0, 1])
ax_D = fig.add_subplot(gs[0, 2])
ax_L = fig.add_subplot(gs[1, 1])
ax_Lsym = fig.add_subplot(gs[1, 2])

# ── (a) 示例图 ────────────────────────────────────────────────────
pos = nx.spring_layout(G, seed=42, k=1.8)

# 节点颜色：按度映射 (蓝→红)
cmap = plt.cm.RdYlBu_r
norm = plt.Normalize(vmin=min(degrees), vmax=max(degrees))
node_colors = [cmap(norm(d)) for d in degrees]

# 节点大小：按度缩放
node_sizes = [400 + 200 * G.degree(i) for i in range(n)]

# 边宽度
edge_widths = [1.5] * len(edges)

# 关闭网格和边框（图可视化不需要）
ax_graph.set_axis_off()

nx.draw_networkx_edges(G, pos, ax=ax_graph, width=edge_widths,
                       edge_color=COLORS["gray"], alpha=0.6)
nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=node_sizes,
                       node_color=node_colors, edgecolors="white", linewidths=2)
nx.draw_networkx_labels(G, pos, ax=ax_graph,
                        labels={i: node_labels[i] for i in range(n)},
                        font_size=16, font_weight="bold")

# 添加度标注
for i in range(n):
    x, y = pos[i]
    ax_graph.annotate(f"d={int(degrees[i])}", xy=(x, y),
                      xytext=(0, -22), textcoords="offset points",
                      fontsize=12, ha="center", color=COLORS["gray"])

ax_graph.set_title("(a) 示例图", fontsize=17, fontweight="bold", pad=15)

# ── 热力图辅助函数 ──────────────────────────────────────────────────
tick_labels_short = [f"$v_{i}$" for i in range(1, 7)]


def plot_heatmap(ax, matrix, title, fmt="d", cmap="YlOrRd", vmin=None, vmax=None):
    """在给定 axes 上绘制矩阵热力图。"""
    # 关闭默认网格（热力图不需要）
    ax.grid(False)
    # 对整数矩阵转 int
    if fmt == "d":
        matrix = matrix.astype(int)

    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=fmt,
        cmap=cmap, cbar=False,
        linewidths=1.0, linecolor="white",
        xticklabels=tick_labels_short,
        yticklabels=tick_labels_short,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        vmin=vmin, vmax=vmax,
        square=True,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=14, length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


# ── (b) 邻接矩阵 A ────────────────────────────────────────────────
plot_heatmap(ax_A, A, "邻接矩阵 $\\mathbf{A}$", fmt="d", cmap="Blues")

# ── (c) 度矩阵 D ──────────────────────────────────────────────────
plot_heatmap(ax_D, D, "度矩阵 $\\mathbf{D}$", fmt="d", cmap="Oranges")

# ── (d) 拉普拉斯矩阵 L = D - A ────────────────────────────────────
plot_heatmap(ax_L, L, "拉普拉斯矩阵 $\\mathbf{L}=\\mathbf{D}-\\mathbf{A}$",
             fmt="d", cmap="coolwarm", vmin=-4, vmax=4)

# ── (e) 归一化拉普拉斯 L_sym ──────────────────────────────────────
plot_heatmap(ax_Lsym, L_sym,
             "归一化拉普拉斯 $\\mathbf{L}_{sym}$",
             fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0)

# ── 保存 ──────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_1_01_graph_representations")
