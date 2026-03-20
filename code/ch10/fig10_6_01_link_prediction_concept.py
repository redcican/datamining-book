"""
图 10.6.1  链接预测概念与局部相似度
Link prediction concept and local similarity metrics comparison.

(a) 链接预测示意 — Karate Club graph with existing edges (gray) and
    top-10 Adamic-Adar predicted edges (dashed red, intensity by score).
(b) 局部指标对比 — Box plot of normalized CN, Jaccard, AA, PA scores
    for the top-50 non-edges.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import numpy as np

np.random.seed(42)

# ── Data: Karate Club graph ──────────────────────────────────────────────
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42)
existing_edges = set(G.edges())

# Compute all non-edges
all_nodes = list(G.nodes())
non_edges = [(u, v) for i, u in enumerate(all_nodes)
             for v in all_nodes[i + 1:]
             if not G.has_edge(u, v)]

# ── Compute similarity scores for all non-edges ─────────────────────────
cn_scores = {}
jaccard_scores = {}
aa_scores = {}
pa_scores = {}

for u, v in non_edges:
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common = neighbors_u & neighbors_v

    # Common Neighbors
    cn_scores[(u, v)] = len(common)

    # Jaccard Coefficient
    union = neighbors_u | neighbors_v
    jaccard_scores[(u, v)] = len(common) / len(union) if len(union) > 0 else 0.0

    # Adamic-Adar
    aa = sum(1.0 / np.log(G.degree(w)) for w in common if G.degree(w) > 1)
    aa_scores[(u, v)] = aa

    # Preferential Attachment
    pa_scores[(u, v)] = G.degree(u) * G.degree(v)

# ── Figure layout ────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.6.1　链接预测概念与局部相似度",
             fontsize=22, fontweight="bold", y=0.98)

# ── Panel (a): 链接预测示意 ──────────────────────────────────────────────
ax_a.set_title("(a) 链接预测示意", fontsize=17, fontweight="bold")
ax_a.axis("off")
ax_a.grid(False)

# Draw existing edges (light gray)
nx.draw_networkx_edges(G, pos, ax=ax_a, edge_color=COLORS["light"],
                       width=1.0, alpha=0.7)

# Draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax_a, node_size=200,
                       node_color=COLORS["blue"], edgecolors="white",
                       linewidths=1.2, alpha=0.85)
nx.draw_networkx_labels(G, pos, ax=ax_a, font_size=8, font_color="white",
                        font_weight="bold")

# Top-10 predicted edges by Adamic-Adar
sorted_aa = sorted(aa_scores.items(), key=lambda x: x[1], reverse=True)
top10 = sorted_aa[:10]
top10_scores = [s for _, s in top10]
score_min, score_max = min(top10_scores), max(top10_scores)

# Colormap: higher score = darker red
cmap = cm.Reds
norm = mcolors.Normalize(vmin=score_min * 0.8, vmax=score_max * 1.05)

for (u, v), score in top10:
    x_coords = [pos[u][0], pos[v][0]]
    y_coords = [pos[u][1], pos[v][1]]
    color = cmap(norm(score))
    ax_a.plot(x_coords, y_coords, linestyle="--", linewidth=2.2,
              color=color, alpha=0.85, zorder=0)

# Legend
existing_line = mlines.Line2D([], [], color=COLORS["gray"], linewidth=1.5,
                               label="已知边")
predicted_line = mlines.Line2D([], [], color=COLORS["red"], linewidth=2.2,
                                linestyle="--", label="预测边")
ax_a.legend(handles=[existing_line, predicted_line], loc="lower left",
            fontsize=14, framealpha=0.9)

# ── Panel (b): 局部指标对比 ──────────────────────────────────────────────
ax_b.set_title("(b) 局部指标对比", fontsize=17, fontweight="bold")

# Select top-50 non-edges by AA score for consistent comparison
top50_pairs = [pair for pair, _ in sorted_aa[:50]]

# Gather raw scores for each indicator
raw_cn = np.array([cn_scores[p] for p in top50_pairs], dtype=float)
raw_jaccard = np.array([jaccard_scores[p] for p in top50_pairs], dtype=float)
raw_aa = np.array([aa_scores[p] for p in top50_pairs], dtype=float)
raw_pa = np.array([pa_scores[p] for p in top50_pairs], dtype=float)


# Normalize each to [0, 1]
def normalize(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


norm_cn = normalize(raw_cn)
norm_jaccard = normalize(raw_jaccard)
norm_aa = normalize(raw_aa)
norm_pa = normalize(raw_pa)

# Box plot
indicator_names = ["CN", "Jaccard", "AA", "PA"]
data = [norm_cn, norm_jaccard, norm_aa, norm_pa]
box_colors = [COLORS["blue"], COLORS["green"], COLORS["red"], COLORS["orange"]]

bp = ax_b.boxplot(data, labels=indicator_names, patch_artist=True,
                  widths=0.5, showfliers=True,
                  flierprops=dict(marker="o", markersize=4, alpha=0.5),
                  medianprops=dict(color="white", linewidth=2))

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
    patch.set_edgecolor("white")
    patch.set_linewidth(1.5)

# Whisker / cap styling
for element in ["whiskers", "caps"]:
    for line in bp[element]:
        line.set_color(COLORS["gray"])
        line.set_linewidth(1.5)

ax_b.set_ylabel("归一化分数", fontsize=16)
ax_b.set_xlabel("相似度指标", fontsize=16)
ax_b.tick_params(axis="both", labelsize=14)
ax_b.set_ylim(-0.05, 1.08)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# ── Save ─────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_6_01_link_prediction_concept")
