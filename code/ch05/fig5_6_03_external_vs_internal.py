"""
图 5.6.3　外部指标与内部指标在不同数据分布上的表现对比
上排：三种数据集的 K-means 和 DBSCAN 聚类结果
下排：对应的 ARI、NMI、轮廓系数柱状图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成三种数据集 ─────────────────────────────────────
# 球形簇
X1, y1 = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=42)
# 月牙形
X2, y2 = make_moons(n_samples=400, noise=0.06, random_state=42)
# 各向异性
X3_raw, y3 = make_blobs(n_samples=400, centers=3, cluster_std=0.5, random_state=42)
X3 = np.dot(X3_raw, [[0.6, -0.6], [-0.4, 0.8]])

datasets = [
    ('球形簇', X1, y1, 3, 0.8, 5),
    ('月牙形', X2, y2, 2, 0.15, 5),
    ('各向异性', X3, y3, 3, 0.6, 5),
]

# ── 2. 绘图 ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                         gridspec_kw={'height_ratios': [1.2, 1]})

for col, (name, X, y_true, n_c, eps, min_s) in enumerate(datasets):
    X_s = StandardScaler().fit_transform(X)
    # K-means
    km_labels = KMeans(n_clusters=n_c, n_init=10, random_state=42).fit_predict(X_s)
    # DBSCAN
    db_labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X_s)

    # 上排：散点图（显示 DBSCAN 结果）
    ax = axes[0, col]
    # K-means 用左半，DBSCAN 用右半 -> 用 DBSCAN 结果展示
    n_cl = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    for k in range(n_cl):
        mask = db_labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=PALETTE[k % len(PALETTE)],
                   s=15, alpha=0.7)
    noise = db_labels == -1
    if noise.any():
        ax.scatter(X[noise, 0], X[noise, 1], c=COLORS['gray'], s=8,
                   marker='x', alpha=0.4)
    ax.set_title(f'{name}（DBSCAN 结果）', fontsize=12)
    if col == 0:
        ax.set_ylabel('$x_2$')

    # 下排：指标柱状图
    ax = axes[1, col]
    metrics_km = {}
    metrics_db = {}

    metrics_km['ARI'] = adjusted_rand_score(y_true, km_labels)
    metrics_km['NMI'] = normalized_mutual_info_score(y_true, km_labels)
    metrics_km['轮廓'] = silhouette_score(X_s, km_labels)

    # DBSCAN 排除噪声
    mask_db = db_labels != -1
    if mask_db.sum() > 1 and len(set(db_labels[mask_db])) > 1:
        metrics_db['ARI'] = adjusted_rand_score(y_true[mask_db], db_labels[mask_db])
        metrics_db['NMI'] = normalized_mutual_info_score(y_true[mask_db], db_labels[mask_db])
        metrics_db['轮廓'] = silhouette_score(X_s[mask_db], db_labels[mask_db])
    else:
        metrics_db['ARI'] = 0
        metrics_db['NMI'] = 0
        metrics_db['轮廓'] = 0

    x_pos = np.arange(3)
    width = 0.35
    vals_km = [metrics_km['ARI'], metrics_km['NMI'], metrics_km['轮廓']]
    vals_db = [metrics_db['ARI'], metrics_db['NMI'], metrics_db['轮廓']]
    ax.bar(x_pos - width/2, vals_km, width, label='K-means', color=COLORS['blue'],
           alpha=0.8)
    ax.bar(x_pos + width/2, vals_db, width, label='DBSCAN', color=COLORS['orange'],
           alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['ARI', 'NMI', '轮廓系数'])
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0, color='black', linewidth=0.5)
    if col == 0:
        ax.set_ylabel('指标值')
        ax.legend(fontsize=9)
    ax.set_title(f'{name}', fontsize=11)

plt.suptitle('外部指标（ARI/NMI）与内部指标（轮廓系数）在不同数据分布上的表现', fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_6_03_external_vs_internal')
