"""
图 5.3.2　四种链接准则在三类数据集上的聚类对比
数据集：各向同性高斯簇、月牙形、带噪声桥接的高斯簇
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from scipy.cluster.hierarchy import linkage, fcluster
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()

np.random.seed(42)

# ── 1. 生成三类数据集 ────────────────────────────────────
# 数据集 1：各向同性高斯簇
X1, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)
# 数据集 2：月牙形
X2, _ = make_moons(n_samples=200, noise=0.06, random_state=42)
# 数据集 3：带噪声桥接的高斯簇
X3a, _ = make_blobs(n_samples=80, centers=[[-3, 0]], cluster_std=0.6, random_state=42)
X3b, _ = make_blobs(n_samples=80, centers=[[3, 0]], cluster_std=0.6, random_state=42)
# 添加桥接噪声
bridge = np.column_stack([np.linspace(-1.5, 1.5, 8),
                          np.random.normal(0, 0.15, 8)])
X3 = np.vstack([X3a, X3b, bridge])

datasets = [
    (X1, 3, '各向同性高斯簇'),
    (X2, 2, '月牙形数据'),
    (X3, 2, '带噪声桥接'),
]
methods = ['single', 'complete', 'average', 'ward']
method_names = ['单链接', '全链接', '均链接', 'Ward']

# ── 2. 绘制 3×4 网格 ─────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for row, (X, n_clusters, title) in enumerate(datasets):
    for col, (method, mname) in enumerate(zip(methods, method_names)):
        ax = axes[row, col]
        Z = linkage(X, method=method)
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        for k in range(1, n_clusters + 1):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c=PALETTE[k - 1],
                       s=20, alpha=0.7)
        if row == 0:
            ax.set_title(mname, fontsize=13)
        if col == 0:
            ax.set_ylabel(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('四种链接准则在不同数据分布上的聚类结果', fontsize=15, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_3_02_linkage_comparison')
