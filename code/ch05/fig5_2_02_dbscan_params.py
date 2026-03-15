"""
fig5_2_02_dbscan_params.py
DBSCAN 参数影响对比：3×3 网格 (ε=0.2/0.5/1.0) × (MinPts=3/5/10)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from shared.plot_config import apply_style, save_fig, PALETTE

apply_style()
rng = np.random.default_rng(42)

# 月牙形 + 少量噪声
X, _ = make_moons(n_samples=200, noise=0.08, random_state=42)
X += rng.normal(0, 0.02, X.shape)

EPS_LIST = [0.15, 0.3, 0.6]
MINPTS_LIST = [3, 5, 10]

fig, axes = plt.subplots(3, 3, figsize=(11, 10))

for i, minpts in enumerate(MINPTS_LIST):
    for j, eps in enumerate(EPS_LIST):
        ax = axes[i][j]
        db = DBSCAN(eps=eps, min_samples=minpts).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        unique = sorted(set(labels))
        cmap = PALETTE[:max(n_clusters, 1)]

        for k in unique:
            mask = labels == k
            if k == -1:
                ax.scatter(X[mask, 0], X[mask, 1],
                           c='#999999', s=12, marker='x',
                           linewidths=0.7, zorder=3, label='噪声')
            else:
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=cmap[k % len(cmap)], s=14,
                           alpha=0.8, linewidths=0, zorder=2)

        ax.set_xticks([]); ax.set_yticks([])
        title = f'ε={eps}, MinPts={minpts}\n{n_clusters} 簇, {n_noise} 噪声点'
        ax.set_title(title, fontsize=8.5)

        # 边框颜色区分效果好坏
        if n_clusters == 2 and n_noise < 20:
            for spine in ax.spines.values():
                spine.set_edgecolor('#16a34a')
                spine.set_linewidth(2)
        elif n_clusters == 0 or n_clusters > 5:
            for spine in ax.spines.values():
                spine.set_edgecolor('#dc2626')
                spine.set_linewidth(2)

# 行列标签
for i, minpts in enumerate(MINPTS_LIST):
    axes[i][0].set_ylabel(f'MinPts = {minpts}', fontsize=9, rotation=90,
                          labelpad=5)
for j, eps in enumerate(EPS_LIST):
    axes[0][j].set_title(f'ε = {eps}', fontsize=9)

fig.suptitle('DBSCAN 参数 (ε, MinPts) 影响对比（月牙形数据）\n'
             '绿框=正确识别2簇  红框=参数失效', fontsize=10, y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, __file__, 'fig5_2_02_dbscan_params')
