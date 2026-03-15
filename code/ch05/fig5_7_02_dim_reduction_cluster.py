"""
图 5.7.2　PCA 与 UMAP 降维后聚类效果对比（Digits 数据集）
左：PCA(2) + K-means    中：UMAP(2) 真实标签着色    右：PCA(50) + K-means
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 加载数据 ──────────────────────────────────────────
digits = load_digits()
X, y_true = digits.data, digits.target
X_scaled = StandardScaler().fit_transform(X)
n_clusters = 10

# ── 2. 降维与聚类 ────────────────────────────────────────
# PCA(2)
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)
labels_pca2 = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X_pca2)
ari_pca2 = adjusted_rand_score(y_true, labels_pca2)

# PCA(50) + K-means (visualize with PCA(2) projection)
pca50 = PCA(n_components=50, random_state=42)
X_pca50 = pca50.fit_transform(X_scaled)
labels_pca50 = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X_pca50)
ari_pca50 = adjusted_rand_score(y_true, labels_pca50)

# ── 3. UMAP (use PCA(2) as fallback if umap not installed) ──
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    X_umap = reducer.fit_transform(X_scaled)
    has_umap = True
except ImportError:
    # Fallback: use t-SNE from sklearn
    from sklearn.manifold import TSNE
    X_umap = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)
    has_umap = False

# ── 4. 绘图 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
cmap = plt.cm.tab10

# 左：PCA(2) + K-means
ax = axes[0]
for k in range(n_clusters):
    mask = labels_pca2 == k
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=[PALETTE[k % len(PALETTE)]],
               s=10, alpha=0.6)
ax.set_title(f'PCA(2) + K-means\nARI = {ari_pca2:.3f}', fontsize=12)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# 中：UMAP/t-SNE 真实标签
ax = axes[1]
method_name = 'UMAP' if has_umap else 't-SNE'
for digit in range(10):
    mask = y_true == digit
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[PALETTE[digit % len(PALETTE)]],
               s=10, alpha=0.6, label=str(digit))
ax.set_title(f'{method_name}(2) 可视化（真实标签）\n仅用于可视化，不用于聚类', fontsize=12)
ax.set_xlabel(f'{method_name}1')
ax.set_ylabel(f'{method_name}2')
ax.legend(fontsize=7, ncol=5, loc='upper right', markerscale=2)

# 右：PCA(50) + K-means (用 PCA(2) 投影可视化)
ax = axes[2]
for k in range(n_clusters):
    mask = labels_pca50 == k
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=[PALETTE[k % len(PALETTE)]],
               s=10, alpha=0.6)
ax.set_title(f'PCA(50) + K-means\nARI = {ari_pca50:.3f}', fontsize=12)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.suptitle('降维+聚类策略对比（Digits 数据集，64维→降维→K-means）', fontsize=14, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_7_02_dim_reduction_cluster')
