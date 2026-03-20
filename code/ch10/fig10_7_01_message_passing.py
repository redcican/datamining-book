"""
fig10_7_01_message_passing.py
消息传递与 GCN 聚合：(a) 消息传递示意图  (b) GCN 感受野
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.7.1　消息传递与 GCN 聚合", fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════════
# (a) 消息传递示意图
# ══════════════════════════════════════════════════════════════════════
ax1.set_xlim(-1.8, 1.8)
ax1.set_ylim(-1.6, 1.6)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.grid(False)

# 节点位置
v_pos = np.array([0.0, -0.2])
u_positions = {
    "u1": np.array([-1.2, 1.0]),
    "u2": np.array([0.0, 1.2]),
    "u3": np.array([1.2, 1.0]),
}
neighbor_colors = [PALETTE[2], PALETTE[3], PALETTE[4]]  # green, orange, purple
msg_labels = ["$m_1$", "$m_2$", "$m_3$"]

# 绘制中心节点 v
v_circle = plt.Circle(v_pos, 0.28, fc=COLORS["blue"], ec="white",
                       linewidth=3, zorder=5)
ax1.add_patch(v_circle)
ax1.text(v_pos[0], v_pos[1], "$v$", ha="center", va="center",
         fontsize=20, fontweight="bold", color="white", zorder=6)

# 绘制邻居节点 u1, u2, u3 并画箭头
for i, (name, pos) in enumerate(u_positions.items()):
    # 邻居节点
    u_circle = plt.Circle(pos, 0.22, fc=neighbor_colors[i], ec="white",
                           linewidth=2.5, zorder=5)
    ax1.add_patch(u_circle)
    label = f"${name[0]}_{name[1]}$"
    ax1.text(pos[0], pos[1], label, ha="center", va="center",
             fontsize=16, fontweight="bold", color="white", zorder=6)

    # 箭头：从邻居到中心节点
    direction = v_pos - pos
    dist = np.linalg.norm(direction)
    unit = direction / dist

    # 起点：邻居圆边缘外侧
    arrow_start = pos + unit * 0.26
    # 终点：中心节点圆边缘外侧
    arrow_end = v_pos - unit * 0.32

    arrow = FancyArrowPatch(
        arrow_start, arrow_end,
        arrowstyle="->,head_width=8,head_length=6",
        color=neighbor_colors[i], linewidth=2.5,
        connectionstyle="arc3,rad=0.0",
        zorder=4,
    )
    ax1.add_patch(arrow)

    # 消息标签：放在箭头中点偏外侧
    mid = (arrow_start + arrow_end) / 2
    # 偏移方向（垂直于箭头方向）
    perp = np.array([-unit[1], unit[0]])
    offset = perp * 0.22
    ax1.text(mid[0] + offset[0], mid[1] + offset[1], msg_labels[i],
             ha="center", va="center", fontsize=15, fontweight="bold",
             color=neighbor_colors[i], zorder=6)

# AGG + UPD 标签
ax1.text(v_pos[0], v_pos[1] - 0.55, "AGG + UPD", ha="center", va="center",
         fontsize=14, fontweight="bold", color=COLORS["gray"],
         bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["light"],
                   ec=COLORS["gray"], linewidth=1.5))

ax1.set_title("(a) 消息传递", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════════
# (b) GCN 感受野
# ══════════════════════════════════════════════════════════════════════
ax2.axis("off")
ax2.grid(False)

# 构建一个 8 节点的小图
G = nx.Graph()
G.add_nodes_from(range(8))
edges = [
    (0, 1), (0, 2), (0, 3),   # v=0 的直接邻居
    (1, 4), (1, 5),            # 2-hop 邻居
    (2, 5), (2, 6),
    (3, 6), (3, 7),
]
G.add_edges_from(edges)

# 固定布局
pos = nx.spring_layout(G, seed=88, k=2.0, iterations=100)

# 分类节点
target = 0
hop1 = {1, 2, 3}
hop2 = {4, 5, 6, 7}

# 节点颜色
color_target = COLORS["blue"]
color_hop1 = "#93c5fd"   # light blue
color_hop2 = "#dbeafe"   # lighter blue

node_colors = []
for n in G.nodes():
    if n == target:
        node_colors.append(color_target)
    elif n in hop1:
        node_colors.append(color_hop1)
    else:
        node_colors.append(color_hop2)

node_sizes = []
for n in G.nodes():
    if n == target:
        node_sizes.append(800)
    elif n in hop1:
        node_sizes.append(500)
    else:
        node_sizes.append(400)

# 绘制边
nx.draw_networkx_edges(G, pos, ax=ax2, width=1.8,
                       edge_color=COLORS["gray"], alpha=0.5)

# 绘制节点
nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=node_sizes,
                       node_color=node_colors, edgecolors="white",
                       linewidths=2.5)

# 节点标签
labels = {0: "$v$"}
for i in range(1, 8):
    labels[i] = f"$u_{i}$"
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax2,
                        font_size=13, font_weight="bold")

# ── 感受野环形标注 ──────────────────────────────────────────────────
# 计算各 hop 节点的中心，用于放置标注
center = np.array(pos[target])

# 1-hop 环
hop1_positions = np.array([pos[n] for n in hop1])
r1 = np.max(np.linalg.norm(hop1_positions - center, axis=1)) + 0.15

circle1 = plt.Circle(center, r1, fc="none", ec=COLORS["blue"],
                      linewidth=2.0, linestyle="--", alpha=0.7, zorder=1)
ax2.add_patch(circle1)

# 2-hop 环
hop2_positions = np.array([pos[n] for n in hop2])
r2 = np.max(np.linalg.norm(hop2_positions - center, axis=1)) + 0.15

circle2 = plt.Circle(center, r2, fc="none", ec=COLORS["purple"],
                      linewidth=2.0, linestyle="--", alpha=0.7, zorder=1)
ax2.add_patch(circle2)

# 标注箭头
# "1层 GCN" 指向 1-hop 环
label1_pos = center + np.array([r1 + 0.45, r1 * 0.4])
ax2.annotate("1层 GCN", xy=(center[0] + r1 * 0.7, center[1] + r1 * 0.7),
             xytext=label1_pos,
             fontsize=13, fontweight="bold", color=COLORS["blue"],
             arrowprops=dict(arrowstyle="->", color=COLORS["blue"],
                             linewidth=1.8),
             ha="left", va="center", zorder=7)

# "2层 GCN" 指向 2-hop 环
label2_pos = center + np.array([r2 + 0.35, -r2 * 0.5])
ax2.annotate("2层 GCN", xy=(center[0] + r2 * 0.7, center[1] - r2 * 0.7),
             xytext=label2_pos,
             fontsize=13, fontweight="bold", color=COLORS["purple"],
             arrowprops=dict(arrowstyle="->", color=COLORS["purple"],
                             linewidth=1.8),
             ha="left", va="center", zorder=7)

# 图例
legend_elements = [
    mpatches.Patch(fc=color_target, ec="white", label="目标节点 $v$"),
    mpatches.Patch(fc=color_hop1, ec="white", label="1-hop 邻居"),
    mpatches.Patch(fc=color_hop2, ec="white", label="2-hop 邻居"),
]
ax2.legend(handles=legend_elements, loc="lower left", fontsize=11,
           framealpha=0.9)

ax2.set_title("(b) GCN 感受野", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, __file__, "fig10_7_01_message_passing")
