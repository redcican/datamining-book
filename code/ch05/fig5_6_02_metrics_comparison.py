"""
图 5.6.2　多种内部指标在不同 K 值下的变化曲线
三列：轮廓系数（越大越好）、DBI（越小越好）、CH 指数（越大越好）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成数据（真实 K=3）────────────────────────────────
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42)

# ── 2. 扫描 K ───────────────────────────────────────────
K_range = range(2, 9)
sils, dbis, chs = [], [], []
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    labels = km.labels_
    sils.append(silhouette_score(X, labels))
    dbis.append(davies_bouldin_score(X, labels))
    chs.append(calinski_harabasz_score(X, labels))

best_k_sil = list(K_range)[np.argmax(sils)]
best_k_dbi = list(K_range)[np.argmin(dbis)]
best_k_ch = list(K_range)[np.argmax(chs)]

# ── 3. 绘图 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = [
    (sils, '轮廓系数（越大越好）', best_k_sil, True),
    (dbis, 'Davies-Bouldin 指数（越小越好）', best_k_dbi, False),
    (chs, 'Calinski-Harabasz 指数（越大越好）', best_k_ch, True),
]

for ax, (vals, title, best_k, higher_better) in zip(axes, metrics):
    ax.plot(list(K_range), vals, 'o-', color=COLORS['blue'], linewidth=2,
            markersize=8)
    ax.axvline(x=best_k, color=COLORS['red'], linestyle='--', linewidth=2,
               alpha=0.7, label=f'最优 K={best_k}')
    ax.plot(best_k, vals[best_k - 2], 'o', color=COLORS['red'], markersize=12,
            zorder=10)
    ax.set_xlabel('簇数 K')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(list(K_range))

plt.suptitle('多种内部评估指标选择最优 K（真实 K=3）', fontsize=15, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_6_02_metrics_comparison')
