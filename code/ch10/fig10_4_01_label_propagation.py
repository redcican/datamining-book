"""
图 10.4.1  标签传播节点分类
Label propagation on Zachary's Karate Club graph.

(a) 初始状态 — Only ~10% of nodes are labeled (solid, colored); the rest are unlabeled (hollow gray).
(b) 传播后 — After convergence all nodes are colored by predicted label, alpha reflects confidence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────
G = nx.karate_club_graph()
n = G.number_of_nodes()

# True community: 0 = Mr. Hi, 1 = Officer
true_labels = np.array(
    [0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in G.nodes()]
)

# Labeled seed nodes (~10%): node 0 → Mr. Hi (class 0), node 33 → Officer (class 1)
labeled_nodes = {0: 0, 33: 1}
num_classes = 2

# Fixed layout for both panels
pos = nx.spring_layout(G, seed=42)

# ── Label propagation: f^(t+1) = beta * P * f^(t) + (1 - beta) * y_0 ──
beta = 0.8
max_iter = 50
tol = 1e-6

# Row-normalized adjacency (transition matrix P)
A = nx.to_numpy_array(G)
deg = A.sum(axis=1, keepdims=True)
deg[deg == 0] = 1  # avoid division by zero
P = A / deg

# Initial soft label matrix y_0 (n x num_classes)
y0 = np.zeros((n, num_classes))
for node, cls in labeled_nodes.items():
    y0[node, cls] = 1.0

# Iterative propagation
f = y0.copy()
for _ in range(max_iter):
    f_new = beta * P @ f + (1 - beta) * y0
    if np.max(np.abs(f_new - f)) < tol:
        break
    f = f_new

# Predicted labels and confidence
predicted_labels = np.argmax(f, axis=1)
confidence = np.max(f, axis=1)
# Normalize confidence to [0.3, 1.0] range for visual clarity
conf_min, conf_max = confidence.min(), confidence.max()
if conf_max > conf_min:
    alpha_vals = 0.3 + 0.7 * (confidence - conf_min) / (conf_max - conf_min)
else:
    alpha_vals = np.ones(n)

# ── Color helpers ────────────────────────────────────────────────────────
class_colors = [COLORS["blue"], COLORS["red"]]


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color string to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


# ── Figure ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.4.1　标签传播节点分类", fontsize=22, fontweight="bold", y=1.02)

# ── Panel (a): Initial state ─────────────────────────────────────────────
# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=COLORS["gray"],
                       alpha=0.3, width=0.8)

# Separate labeled and unlabeled nodes
unlabeled = [v for v in G.nodes() if v not in labeled_nodes]
labeled_list = list(labeled_nodes.keys())

# Draw unlabeled nodes: hollow gray circles
nx.draw_networkx_nodes(G, pos, nodelist=unlabeled, ax=ax1,
                       node_color="white",
                       edgecolors=COLORS["gray"],
                       linewidths=1.8,
                       node_size=420)

# Draw labeled nodes: solid colored circles
for node, cls in labeled_nodes.items():
    nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax1,
                           node_color=class_colors[cls],
                           edgecolors="white",
                           linewidths=1.5,
                           node_size=500)

# Draw node labels
nx.draw_networkx_labels(G, pos, ax=ax1,
                        font_size=8, font_color=COLORS["gray"],
                        font_weight="bold")

ax1.set_title("(a) 初始状态", fontsize=17, fontweight="bold")
ax1.axis("off")
ax1.grid(False)

# Legend for panel (a)
legend_a = [
    mpatches.Patch(facecolor=COLORS["blue"], edgecolor="white",
                   label='已标注 (Mr. Hi)'),
    mpatches.Patch(facecolor=COLORS["red"], edgecolor="white",
                   label='已标注 (Officer)'),
    mpatches.Patch(facecolor="white", edgecolor=COLORS["gray"],
                   linewidth=1.5, label='未标注节点'),
]
ax1.legend(handles=legend_a, loc="lower left", fontsize=12,
           framealpha=0.9, edgecolor=COLORS["light"])

# ── Panel (b): After propagation ─────────────────────────────────────────
# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax2, edge_color=COLORS["gray"],
                       alpha=0.3, width=0.8)

# Draw each node with color = predicted class and alpha = confidence
for v in G.nodes():
    rgba = hex_to_rgba(class_colors[predicted_labels[v]], alpha=float(alpha_vals[v]))
    nx.draw_networkx_nodes(G, pos, nodelist=[v], ax=ax2,
                           node_color=[rgba],
                           edgecolors="white",
                           linewidths=1.5,
                           node_size=420)

# Draw node labels
nx.draw_networkx_labels(G, pos, ax=ax2,
                        font_size=8, font_color="white",
                        font_weight="bold")

ax2.set_title("(b) 传播后", fontsize=17, fontweight="bold")
ax2.axis("off")
ax2.grid(False)

# Legend for panel (b)
legend_b = [
    mpatches.Patch(facecolor=COLORS["blue"], edgecolor="white",
                   label='预测: Mr. Hi'),
    mpatches.Patch(facecolor=COLORS["red"], edgecolor="white",
                   label='预测: Officer'),
    mpatches.Patch(facecolor=COLORS["gray"], edgecolor="white",
                   alpha=0.4, label='透明度 = 置信度'),
]
ax2.legend(handles=legend_b, loc="lower left", fontsize=12,
           framealpha=0.9, edgecolor=COLORS["light"])

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_4_01_label_propagation")
