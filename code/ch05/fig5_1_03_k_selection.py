"""
fig5_1_03_k_selection.py
K 值选择：肘部法则（SSE）+ 轮廓系数 + Gap Statistic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()
rng = np.random.default_rng(42)

# ── 数据：4 簇 ───────────────────────────────────────────────
centers_true = np.array([[1,1],[5,1],[1,5],[5,5]], dtype=float)
X = np.vstack([
    rng.multivariate_normal(c, [[0.5,0],[0,0.5]], 50)
    for c in centers_true
])

K_range = range(2, 10)
sse_list, sil_list = [], []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                random_state=42, max_iter=300)
    labels = km.fit_predict(X)
    sse_list.append(km.inertia_)
    sil_list.append(silhouette_score(X, labels))

K_vals = list(K_range)

# ── Gap Statistic（简化版） ───────────────────────────────────
def compute_gap(X, K_range, B=10, seed=0):
    rng2 = np.random.default_rng(seed)
    x_min, x_max = X.min(0), X.max(0)
    gaps, sks = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        km.fit(X)
        Wk = np.log(km.inertia_)
        Wkbs = []
        for _ in range(B):
            Xb = rng2.uniform(x_min, x_max, size=X.shape)
            km2 = KMeans(n_clusters=k, n_init=3, random_state=0)
            km2.fit(Xb)
            Wkbs.append(np.log(km2.inertia_))
        gap = np.mean(Wkbs) - Wk
        sk = np.std(Wkbs) * np.sqrt(1 + 1/B)
        gaps.append(gap)
        sks.append(sk)
    return np.array(gaps), np.array(sks)

gaps, sks = compute_gap(X, K_range, B=20)

# ── 绘图 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

# Panel 1: Elbow（SSE）
ax = axes[0]
ax.plot(K_vals, sse_list, 'o-', color=PALETTE[0], lw=2, ms=6)
ax.axvline(4, color='crimson', linestyle='--', lw=1.5, label='最优 $K=4$')
# 标注肘部
ax.annotate('肘部\n($K=4$)', xy=(4, sse_list[2]),
            xytext=(5.5, sse_list[2] + (sse_list[0]-sse_list[-1])*0.1),
            arrowprops=dict(arrowstyle='->', color='crimson'),
            fontsize=8.5, color='crimson')
ax.set_xlabel('簇数 $K$', fontsize=10)
ax.set_ylabel('簇内平方和 SSE', fontsize=10)
ax.set_title('（a）肘部法则：SSE 随 $K$ 的变化', fontsize=9)
ax.legend(fontsize=9)
ax.set_xticks(K_vals)

# Panel 2: 轮廓系数
ax = axes[1]
ax.plot(K_vals, sil_list, 's-', color=PALETTE[1], lw=2, ms=6)
best_k_sil = K_vals[int(np.argmax(sil_list))]
ax.axvline(best_k_sil, color='crimson', linestyle='--', lw=1.5,
           label=f'最大值 $K={best_k_sil}$')
ax.set_xlabel('簇数 $K$', fontsize=10)
ax.set_ylabel('平均轮廓系数 $\\bar{s}$', fontsize=10)
ax.set_title('（b）轮廓系数：越大越好', fontsize=9)
ax.legend(fontsize=9)
ax.set_xticks(K_vals)
ax.set_ylim(0, 1)

# Panel 3: Gap Statistic
ax = axes[2]
ax.plot(K_vals, gaps, '^-', color=PALETTE[2], lw=2, ms=6, label='Gap$(k)$')
ax.fill_between(K_vals,
                gaps - sks, gaps + sks,
                alpha=0.25, color=PALETTE[2], label='±$s_k$')
# 找最优K：最小的k使gap(k) >= gap(k+1) - s(k+1)
best_gap_k = None
for i in range(len(K_vals) - 1):
    if gaps[i] >= gaps[i+1] - sks[i+1]:
        best_gap_k = K_vals[i]
        break
if best_gap_k:
    ax.axvline(best_gap_k, color='crimson', linestyle='--', lw=1.5,
               label=f'最优 $K={best_gap_k}$')
ax.set_xlabel('簇数 $K$', fontsize=10)
ax.set_ylabel('Gap$(k)$', fontsize=10)
ax.set_title('（c）Gap Statistic', fontsize=9)
ax.legend(fontsize=9)
ax.set_xticks(K_vals)

fig.suptitle('K 值选择的三种方法（真实簇数 $K^*=4$）', fontsize=11, y=1.01)
fig.tight_layout()
save_fig(fig, __file__, 'fig5_1_03_k_selection')
