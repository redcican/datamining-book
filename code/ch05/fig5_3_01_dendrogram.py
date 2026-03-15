"""
图 5.3.1　凝聚式层次聚类的树状图（Ward 链接）
左：完整树状图 + 截断线；右：对应散点图
数据集：make_blobs(n=30, centers=3)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

# ── 1. 生成数据 ──────────────────────────────────────────
np.random.seed(42)
X, y_true = make_blobs(n_samples=30, centers=3, cluster_std=1.2, random_state=42)

# ── 2. Ward 链接层次聚类 ─────────────────────────────────
Z = linkage(X, method='ward')
# 确定截断高度：取倒数第3次合并距离的中点
cut_height = (Z[-3, 2] + Z[-2, 2]) / 2
labels = fcluster(Z, t=3, criterion='maxclust')

# ── 3. 绘图 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 左：树状图
color_threshold = Z[-3, 2]
dn = dendrogram(Z, ax=axes[0], leaf_rotation=90, leaf_font_size=9,
                color_threshold=color_threshold,
                above_threshold_color=COLORS['gray'])
axes[0].axhline(y=cut_height, color=COLORS['red'], linestyle='--',
                linewidth=1.5, label=f'截断高度 h={cut_height:.1f}')
axes[0].set_title('Ward 链接树状图')
axes[0].set_xlabel('样本编号')
axes[0].set_ylabel('合并距离')
axes[0].legend(loc='upper right')
# 右：散点图
cluster_colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
for k in range(3):
    mask = labels == k + 1
    axes[1].scatter(X[mask, 0], X[mask, 1], c=cluster_colors[k],
                    label=f'簇 {k+1}', s=60, alpha=0.8, edgecolors='white',
                    linewidths=0.5)
axes[1].set_title('层次聚类结果（K=3）')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].legend()
plt.tight_layout()
save_fig(fig, __file__, 'fig5_3_01_dendrogram')
