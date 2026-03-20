"""
图 10.6.3  链接预测案例结果
Link prediction case study results.

(a) ROC 曲线 — ROC curves for CN, AA, PA, Katz on the Karate Club graph
    with 20% edges removed.
(b) 准确率 vs 训练比例 — AUC of CN and AA across different training ratios.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import roc_curve, auc

np.random.seed(42)


# ── Helper functions ────────────────────────────────────────────────────────

def _split_edges(G, test_fraction):
    """Remove a fraction of edges for testing; return train graph and test sets."""
    edges = list(G.edges())
    np.random.shuffle(edges)
    n_test = max(1, int(len(edges) * test_fraction))
    test_edges = edges[:n_test]
    train_edges = edges[n_test:]

    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    return G_train, test_edges


def _negative_samples(G, n_samples):
    """Sample non-edges as negative examples."""
    non_edges = list(nx.non_edges(G))
    np.random.shuffle(non_edges)
    return non_edges[:n_samples]


def _katz_scores(G, beta=0.01, max_power=3):
    """Compute truncated Katz similarity: sum_{l=1}^{max_power} beta^l * A^l."""
    A = nx.adjacency_matrix(G).toarray().astype(float)
    n = A.shape[0]
    S = np.zeros((n, n))
    A_power = np.eye(n)
    for l in range(1, max_power + 1):
        A_power = A_power @ A
        S += (beta ** l) * A_power
    return S


def _compute_scores(G_train, pairs, method):
    """Compute link prediction scores for a list of node pairs."""
    nodes = list(G_train.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    if method == "CN":
        scores = []
        for u, v in pairs:
            scores.append(len(list(nx.common_neighbors(G_train, u, v))))
        return np.array(scores, dtype=float)

    elif method == "AA":
        scores = []
        for u, v in pairs:
            score = sum(
                1.0 / np.log(G_train.degree(w))
                for w in nx.common_neighbors(G_train, u, v)
                if G_train.degree(w) > 1
            )
            scores.append(score)
        return np.array(scores, dtype=float)

    elif method == "PA":
        scores = []
        for u, v in pairs:
            scores.append(G_train.degree(u) * G_train.degree(v))
        return np.array(scores, dtype=float)

    elif method == "Katz":
        S = _katz_scores(G_train, beta=0.01, max_power=3)
        scores = []
        for u, v in pairs:
            scores.append(S[node_idx[u], node_idx[v]])
        return np.array(scores, dtype=float)

    else:
        raise ValueError(f"Unknown method: {method}")


def _run_experiment(G, test_fraction, methods):
    """Run link prediction experiment and return AUC per method."""
    G_train, test_edges = _split_edges(G, test_fraction)
    neg_edges = _negative_samples(G_train, len(test_edges))

    pairs = test_edges + neg_edges
    labels = np.array([1] * len(test_edges) + [0] * len(neg_edges))

    results = {}
    for method in methods:
        scores = _compute_scores(G_train, pairs, method)
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_val = auc(fpr, tpr)
        results[method] = {"fpr": fpr, "tpr": tpr, "auc": auc_val}
    return results


# ── Data ────────────────────────────────────────────────────────────────────
G = nx.karate_club_graph()
methods = ["CN", "AA", "PA", "Katz"]
method_colors = {
    "CN": COLORS["blue"],
    "AA": COLORS["red"],
    "PA": COLORS["green"],
    "Katz": COLORS["orange"],
}

# ── Figure layout ───────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.6.3　链接预测案例结果", fontsize=22, fontweight="bold", y=0.98)

# ── Panel (a): ROC 曲线 ────────────────────────────────────────────────────
results_a = _run_experiment(G, test_fraction=0.2, methods=methods)

for method in methods:
    r = results_a[method]
    ax_a.plot(r["fpr"], r["tpr"], color=method_colors[method], linewidth=2.5,
              label=f'{method} (AUC = {r["auc"]:.3f})')

ax_a.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=1.5,
          label="随机基线")

ax_a.set_xlabel("假阳性率 (FPR)", fontsize=16)
ax_a.set_ylabel("真阳性率 (TPR)", fontsize=16)
ax_a.set_title("(a) ROC 曲线", fontsize=17, fontweight="bold")
ax_a.tick_params(axis="both", labelsize=14)
ax_a.legend(fontsize=14, loc="lower right")
ax_a.set_xlim(-0.02, 1.02)
ax_a.set_ylim(-0.02, 1.05)

# ── Panel (b): 准确率 vs 训练比例 ──────────────────────────────────────────
train_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
auc_cn = []
auc_aa = []

for ratio in train_ratios:
    np.random.seed(42)
    test_frac = 1.0 - ratio
    res = _run_experiment(G, test_fraction=test_frac, methods=["CN", "AA"])
    auc_cn.append(res["CN"]["auc"])
    auc_aa.append(res["AA"]["auc"])

ax_b.plot(train_ratios, auc_cn, color=COLORS["blue"], marker="o", markersize=8,
          linewidth=2.5, label="CN (Common Neighbors)")
ax_b.plot(train_ratios, auc_aa, color=COLORS["red"], marker="s", markersize=8,
          linewidth=2.5, label="AA (Adamic-Adar)")

ax_b.set_xlabel("训练比例", fontsize=16)
ax_b.set_ylabel("AUC", fontsize=16)
ax_b.set_title("(b) 准确率 vs 训练比例", fontsize=17, fontweight="bold")
ax_b.tick_params(axis="both", labelsize=14)
ax_b.legend(fontsize=14, loc="lower right")
ax_b.set_xlim(0.45, 0.95)
ax_b.set_ylim(0, 1.05)

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ────────────────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig10_6_03_case_result")
