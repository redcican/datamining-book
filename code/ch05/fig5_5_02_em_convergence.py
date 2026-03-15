"""
图 5.5.2　EM 算法的迭代收敛过程（GMM，K=3）
2×2 网格：迭代 1/3/5/20 时的等高线 + 对数似然收敛曲线
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成数据 ──────────────────────────────────────────
X1 = np.random.multivariate_normal([-2, -1], [[0.8, 0.3], [0.3, 0.5]], 120)
X2 = np.random.multivariate_normal([3, 2], [[1.2, -0.4], [-0.4, 0.8]], 100)
X3 = np.random.multivariate_normal([0, 4], [[0.6, 0.1], [0.1, 1.0]], 80)
X = np.vstack([X1, X2, X3])

# ── 2. 手动迭代 EM，记录各步状态 ─────────────────────────
iters_to_show = [1, 3, 5, 20]
states = {}
log_likelihoods = []

for max_iter in range(1, 25):
    gmm = GaussianMixture(n_components=3, max_iter=max_iter, n_init=1,
                          init_params='random', random_state=0,
                          warm_start=False)
    gmm.fit(X)
    ll = gmm.score(X) * len(X)
    log_likelihoods.append(ll)
    if max_iter in iters_to_show:
        states[max_iter] = {
            'means': gmm.means_.copy(),
            'covs': gmm.covariances_.copy(),
            'labels': gmm.predict(X),
            'proba': gmm.predict_proba(X),
        }

# ── 3. 绘图 ──────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
colors_list = [PALETTE[0], PALETTE[1], PALETTE[2]]

for idx, (pos, it) in enumerate(zip(plot_positions, iters_to_show)):
    ax = fig.add_subplot(gs[pos[0], pos[1]])
    state = states[it]
    labels = state['labels']
    # 散点图
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_list[k], s=12, alpha=0.6)
    # 椭圆
    for k in range(3):
        mean = state['means'][k]
        cov = state['covs'][k]
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for n_std in [1, 2]:
            ell = Ellipse(xy=mean, width=2*n_std*np.sqrt(max(vals[1], 0.01)),
                          height=2*n_std*np.sqrt(max(vals[0], 0.01)), angle=angle,
                          edgecolor=colors_list[k], facecolor='none',
                          linewidth=1.5 if n_std == 1 else 0.8,
                          linestyle='-' if n_std == 1 else '--')
            ax.add_patch(ell)
        ax.plot(mean[0], mean[1], '+', color=colors_list[k], markersize=15,
                markeredgewidth=2)
    ax.set_title(f'迭代 {it}', fontsize=13)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-5, 6)
    ax.set_ylim(-4, 7)

# 右侧：对数似然收敛曲线
ax_ll = fig.add_subplot(gs[:, 2])
ax_ll.plot(range(1, len(log_likelihoods)+1), log_likelihoods,
           'o-', color=COLORS['blue'], linewidth=2, markersize=4)
for it in iters_to_show:
    ax_ll.axvline(x=it, color=COLORS['gray'], linestyle=':', alpha=0.5)
    ax_ll.plot(it, log_likelihoods[it-1], 'o', color=COLORS['red'],
               markersize=8, zorder=10)
ax_ll.set_xlabel('迭代次数')
ax_ll.set_ylabel('对数似然 $\\ell(\\Theta)$')
ax_ll.set_title('收敛曲线（单调不减）', fontsize=13)

plt.suptitle('EM 算法迭代收敛过程', fontsize=15, y=1.01)
save_fig(fig, __file__, 'fig5_5_02_em_convergence')
