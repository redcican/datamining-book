"""
fig10_2_02_pagerank.py
PageRank 算法：有向图 PageRank 分布 + 幂迭代收敛过程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

# ── 构建有向图 (~15 节点, ~26 条边) ──────────────────────────────────
G = nx.DiGraph()
edges = [
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (2, 1), (3, 1), (6, 1), (7, 1),
    (6, 2), (7, 2), (8, 3), (9, 3),
    (9, 0), (10, 5), (10, 6), (11, 9),
    (11, 10), (12, 9), (13, 11), (14, 0),
    (14, 1), (14, 9), (8, 4), (12, 7),
    (13, 12), (5, 3),
]
G.add_edges_from(edges)

# ── 计算 PageRank ──────────────────────────────────────────────────
pr = nx.pagerank(G, alpha=0.85)

# ── 幂迭代 (手动实现, 记录收敛历史) ────────────────────────────────
nodes = sorted(G.nodes())
n = len(nodes)
node_idx = {v: i for i, v in enumerate(nodes)}
d = 0.85
n_iter = 40

# 构建列随机转移矩阵 M: M[j, i] = 1/out_degree(i) if edge (i->j)
M = np.zeros((n, n))
for u, v in G.edges():
    out_deg = G.out_degree(u)
    if out_deg > 0:
        M[node_idx[v], node_idx[u]] = 1.0 / out_deg

# 处理 dangling nodes (出度为 0 的节点): 均匀分配
dangling = np.array([1.0 if G.out_degree(v) == 0 else 0.0 for v in nodes])

pr_vec = np.ones(n) / n
pr_history = np.zeros((n_iter, n))

for it in range(n_iter):
    pr_history[it] = pr_vec
    # PageRank iteration: pr = d * (M @ pr + dangling_contrib) + (1-d)/n
    dangling_contrib = (dangling @ pr_vec) * np.ones(n) / n
    pr_vec = d * (M @ pr_vec + dangling_contrib) + (1.0 - d) / n

# ── 找出 PageRank 最高的 5 个节点 ──────────────────────────────────
pr_values = np.array([pr[v] for v in nodes])
top5_indices = np.argsort(pr_values)[-5:][::-1]
top5_set = set(top5_indices)

# ── 绘图 ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.2.2　PageRank 算法", fontsize=22, fontweight="bold", y=1.02)

# ── (a) 有向图 PageRank 分布 ─────────────────────────────────────
ax1.set_axis_off()
ax1.grid(False)

pos = nx.spring_layout(G, seed=42, k=1.5)

# 节点大小与颜色按 PageRank 值映射
node_sizes = [200 + pr[v] * 8000 for v in nodes]
cmap = cm.RdYlBu_r
vmin = min(pr.values())
vmax = max(pr.values())
norm = plt.Normalize(vmin=vmin, vmax=vmax)
node_colors = [cmap(norm(pr[v])) for v in nodes]

# 绘制边 (带箭头)
nx.draw_networkx_edges(
    G, pos, ax=ax1,
    arrows=True, arrowsize=15,
    edge_color=COLORS["gray"], alpha=0.4,
    connectionstyle="arc3,rad=0.1",
)

# 绘制节点
nx.draw_networkx_nodes(
    G, pos, ax=ax1,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="white", linewidths=1.5,
)

# 节点标签
nx.draw_networkx_labels(
    G, pos, ax=ax1,
    labels={v: str(v) for v in nodes},
    font_size=10, font_weight="bold",
)

# 添加颜色条
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, shrink=0.8, pad=0.02)
cbar.set_label("PageRank", fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax1.set_title("(a) PageRank 分布", fontsize=17, fontweight="bold", pad=15)

# ── (b) 幂迭代收敛过程 ──────────────────────────────────────────
iterations = np.arange(n_iter)

# 先画非 top-5 节点 (灰色)
for i in range(n):
    if i not in top5_set:
        ax2.plot(iterations, pr_history[:, i],
                 color=COLORS["gray"], alpha=0.3, linewidth=1.0)

# 再画 top-5 节点 (彩色, 带标签)
for rank, idx in enumerate(top5_indices):
    color = PALETTE[rank % len(PALETTE)]
    ax2.plot(iterations, pr_history[:, idx],
             color=color, linewidth=2.2, label=f"节点 {nodes[idx]}")

ax2.set_xlabel("迭代次数", fontsize=16)
ax2.set_ylabel("PageRank 值", fontsize=16)
ax2.set_title("(b) 幂迭代收敛过程", fontsize=17, fontweight="bold", pad=15)
ax2.tick_params(axis="both", labelsize=14)
ax2.legend(fontsize=14, loc="center right")

plt.tight_layout()

# ── 保存 ────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_2_02_pagerank")
