"""
图 10.4.3  图分类案例结果
Graph classification case study results.

(a) PCA 特征空间 — PCA scatter plot of graph-level features for three classes
    (cycle, tree, random) showing separable clusters.
(b) 特征重要性 — Horizontal bar chart of RandomForest feature importances.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# ── Generate graphs ─────────────────────────────────────────────────────────
n_per_class = 50
node_counts = np.random.randint(10, 21, size=n_per_class * 3)

graphs = []
labels = []

for i in range(n_per_class):
    n = node_counts[i]
    G = nx.cycle_graph(n)
    graphs.append(G)
    labels.append(0)  # cycle

for i in range(n_per_class):
    n = node_counts[n_per_class + i]
    # Random tree: use random Prufer sequence
    T = nx.random_labeled_tree(n, seed=42 + i)
    graphs.append(T)
    labels.append(1)  # tree

for i in range(n_per_class):
    n = node_counts[2 * n_per_class + i]
    G = nx.erdos_renyi_graph(n, 0.3, seed=42 + i)
    graphs.append(G)
    labels.append(2)  # random

labels = np.array(labels)

# ── Extract features ────────────────────────────────────────────────────────
feature_names = ["节点数", "边数", "平均度", "度标准差", "聚集系数", "密度"]


def extract_features(G):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()])
    mean_deg = degrees.mean()
    std_deg = degrees.std()
    avg_clust = nx.average_clustering(G)
    density = nx.density(G)
    return [n_nodes, n_edges, mean_deg, std_deg, avg_clust, density]


X = np.array([extract_features(G) for G in graphs])

# ── Train classifier ────────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, labels)
importances = clf.feature_importances_

# ── PCA ─────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# ── Figure ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "图 10.4.3　图分类案例结果",
    fontsize=22, fontweight="bold", y=0.98,
)

# ── Panel (a): PCA scatter ──────────────────────────────────────────────────
class_info = [
    (0, COLORS["blue"], "环形图 (cycle)"),
    (1, COLORS["green"], "树状图 (tree)"),
    (2, COLORS["red"], "随机图 (random)"),
]

for cls_id, color, label in class_info:
    mask = labels == cls_id
    ax1.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=color, label=label, s=60, alpha=0.75, edgecolors="white", linewidths=0.5,
    )

# Draw 95 % confidence ellipses for each class
from matplotlib.patches import Ellipse

for cls_id, color, _ in class_info:
    mask = labels == cls_id
    pts = X_pca[mask]
    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues descending
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    # 95 % confidence: chi-squared with 2 dof, p=0.05 => 5.991
    scale = np.sqrt(5.991)
    width = 2 * scale * np.sqrt(eigvals[0])
    height = 2 * scale * np.sqrt(eigvals[1])
    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5,
        linestyle="--",
    )
    ax1.add_patch(ellipse)

ax1.set_xlabel("PC1", fontsize=16)
ax1.set_ylabel("PC2", fontsize=16)
ax1.set_title("(a) PCA 特征空间", fontsize=17, fontweight="bold")
ax1.tick_params(axis="both", labelsize=14)
ax1.legend(fontsize=14, loc="best", framealpha=0.9, edgecolor=COLORS["light"])

# ── Panel (b): Feature importance ───────────────────────────────────────────
# Sort by importance descending (most important at top)
sorted_idx = np.argsort(importances)  # ascending order for barh (bottom to top)
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

bars = ax2.barh(
    range(len(sorted_names)), sorted_importances,
    color=COLORS["blue"], edgecolor="white", linewidth=0.8, height=0.6,
)
ax2.set_yticks(range(len(sorted_names)))
ax2.set_yticklabels(sorted_names, fontsize=14)
ax2.set_xlabel("重要性", fontsize=16)
ax2.set_title("(b) 特征重要性", fontsize=17, fontweight="bold")
ax2.tick_params(axis="both", labelsize=14)

# Value labels on bars
for bar in bars:
    w = bar.get_width()
    ax2.text(
        w + 0.005, bar.get_y() + bar.get_height() / 2,
        f"{w:.3f}", ha="left", va="center", fontsize=12, fontweight="bold",
        color=COLORS["blue"],
    )

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ────────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_4_03_case_result")
