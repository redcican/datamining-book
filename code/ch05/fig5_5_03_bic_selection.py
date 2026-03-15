"""
图 5.5.3　BIC/AIC 选择最优成分数 K
左：BIC 和 AIC 曲线
右：K=2 / K=3 / K=5 的 GMM 拟合结果对比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成数据（3 个真实簇）─────────────────────────────
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=[1.0, 0.8, 1.2],
                       random_state=42)

# ── 2. 计算 BIC/AIC ─────────────────────────────────────
K_range = range(1, 9)
bics, aics = [], []
for k in K_range:
    gmm = GaussianMixture(n_components=k, n_init=10, random_state=42)
    gmm.fit(X)
    bics.append(gmm.bic(X))
    aics.append(gmm.aic(X))

best_k_bic = list(K_range)[np.argmin(bics)]
best_k_aic = list(K_range)[np.argmin(aics)]

# ── 3. 绘图 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                         gridspec_kw={'width_ratios': [1.2, 1, 1, 1]})

# 左：BIC/AIC 曲线
ax = axes[0]
ax.plot(list(K_range), bics, 'o-', color=COLORS['blue'], linewidth=2,
        markersize=6, label='BIC')
ax.plot(list(K_range), aics, 's--', color=COLORS['orange'], linewidth=2,
        markersize=6, label='AIC')
ax.axvline(x=best_k_bic, color=COLORS['red'], linestyle=':', alpha=0.7,
           label=f'BIC 最优 K={best_k_bic}')
ax.set_xlabel('成分数 K')
ax.set_ylabel('信息准则')
ax.set_title('BIC 与 AIC 模型选择')
ax.legend(fontsize=10)

# 右三：K=2, 3, 5 的 GMM 拟合
k_values = [2, 3, 5]
k_labels = ['K=2（欠拟合）', 'K=3（最优）', 'K=5（过拟合）']
colors_list = PALETTE[:8]

for col, (k_val, k_label) in enumerate(zip(k_values, k_labels)):
    ax = axes[col + 1]
    gmm = GaussianMixture(n_components=k_val, n_init=10, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    # 散点图
    for k in range(k_val):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_list[k % len(colors_list)],
                   s=15, alpha=0.6)
    # 椭圆
    for k in range(k_val):
        mean = gmm.means_[k]
        cov = gmm.covariances_[k]
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for n_std in [1, 2]:
            ell = Ellipse(xy=mean, width=2*n_std*np.sqrt(vals[1]),
                          height=2*n_std*np.sqrt(vals[0]), angle=angle,
                          edgecolor=colors_list[k % len(colors_list)],
                          facecolor='none',
                          linewidth=1.5 if n_std == 1 else 0.8,
                          linestyle='-' if n_std == 1 else '--')
            ax.add_patch(ell)
    bic_val = gmm.bic(X)
    ax.set_title(f'{k_label}\nBIC={bic_val:.0f}', fontsize=12)
    ax.set_xlabel('$x_1$')
    if col == 0:
        ax.set_ylabel('$x_2$')

plt.tight_layout()
save_fig(fig, __file__, 'fig5_5_03_bic_selection')
