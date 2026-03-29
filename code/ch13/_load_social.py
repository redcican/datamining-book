"""
Shared data loader for Zachary Karate Club dataset (§13.4).

Uses networkx built-in dataset — no download needed.
"""
import networkx as nx
import numpy as np


def load_graph():
    """
    Load Zachary Karate Club and return (G, ground_truth, pos).

    G: networkx Graph (34 nodes, 78 edges)
    ground_truth: dict {node: 0 or 1} — Mr. Hi=0, Officer=1
    pos: dict {node: (x, y)} — fixed spring layout for consistent plotting
    """
    G = nx.karate_club_graph()

    # Ground truth: 'Mr. Hi' → 0, 'Officer' → 1
    ground_truth = {}
    for node in G.nodes():
        club = G.nodes[node]["club"]
        ground_truth[node] = 0 if club == "Mr. Hi" else 1

    # Fixed layout for all figures
    pos = nx.spring_layout(G, seed=42, k=0.3)

    return G, ground_truth, pos


def get_community_labels(communities, n_nodes):
    """
    Convert a set of frozensets (community partition) to a label array.

    Returns: np.ndarray of shape (n_nodes,) with community index per node.
    """
    labels = np.zeros(n_nodes, dtype=int)
    for idx, comm in enumerate(sorted(communities, key=min)):
        for node in comm:
            labels[node] = idx
    return labels
