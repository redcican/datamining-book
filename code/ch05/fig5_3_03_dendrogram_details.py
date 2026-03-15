"""
图 5.3.3　四种链接准则的树状图对比（同一数据集）
数据集：make_blobs(n=50, centers=3)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

np.random.seed(42)

# ── 1. 生成数据 ──────────────────────────────────────────
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=42)

# ── 2. 四种链接准则 ──────────────────────────────────────
methods = ['ward', 'complete', 'average', 'single']
method_names = ['Ward', '全链接', '均链接', '单链接']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (method, mname) in enumerate(zip(methods, method_names)):
    Z = linkage(X, method=method)
    ax = axes[idx]
    # 截断高度设为从3簇到2簇的合并距离
    color_thresh = Z[-3, 2]
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_rotation=90, leaf_font_size=7,
               color_threshold=color_thresh,
               above_threshold_color=COLORS['gray'])
    ax.set_title(f'{mname}链接', fontsize=13)
    ax.set_xlabel('样本（合并后）')
    ax.set_ylabel('合并距离')
    # 标注截断线
    ax.axhline(y=color_thresh, color=COLORS['red'], linestyle='--',
               linewidth=1, alpha=0.6)

plt.suptitle('四种链接准则的树状图对比', fontsize=15, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_3_03_dendrogram_details')
