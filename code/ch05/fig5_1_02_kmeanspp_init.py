"""
fig5_1_02_kmeanspp_init.py
K-means++ 概率初始化 3 步示意图 vs 随机初始化对比
左: 随机初始化导致次优（100次结果分布）
中: K-means++ 概率选点步骤示意（D²加权）
右: SSE分布对比（随机初始化 vs K-means++）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()
rng = np.random.default_rng(0)

# ── 数据 ─────────────────────────────────────────────────────
centers_true = np.array([[1.0, 1.0], [5.5, 1.0], [3.2, 5.0]])
X = np.vstack([
    rng.multivariate_normal(c, [[0.35, 0], [0, 0.35]], 40)
    for c in centers_true
])
K = 3

def assign(X, c):
    return np.argmin(np.linalg.norm(X[:, None] - c[None], axis=2), axis=1)

def sse(X, c):
    labels = assign(X, c)
    return sum(np.sum((X[labels == k] - c[k])**2) for k in range(K))

def kmeans(X, init, n_iter=50):
    c = init.copy()
    for _ in range(n_iter):
        l = assign(X, c)
        c_new = np.array([X[l == k].mean(0) if (l == k).any() else c[k] for k in range(K)])
        if np.allclose(c, c_new): break
        c = c_new
    return c, sse(X, c)

def random_init(X, K, seed):
    rng2 = np.random.default_rng(seed)
    idx = rng2.choice(len(X), K, replace=False)
    return X[idx]

def kmeanspp_init(X, K, seed):
    rng2 = np.random.default_rng(seed)
    idx = [rng2.integers(len(X))]
    for _ in range(K - 1):
        D2 = np.array([min(np.sum((x - X[i])**2) for i in idx) for x in X])
        probs = D2 / D2.sum()
        idx.append(rng2.choice(len(X), p=probs))
    return X[idx]

# ── 运行100次对比 ─────────────────────────────────────────────
N_RUNS = 100
sse_rand = [kmeans(X, random_init(X, K, s))[1] for s in range(N_RUNS)]
sse_pp   = [kmeans(X, kmeanspp_init(X, K, s))[1] for s in range(N_RUNS)]

# ── 中间面板：K-means++ 选点过程 ─────────────────────────────
# 展示3步选质心的过程
c0 = X[15:16]  # 第1个质心
D2_step1 = np.array([np.sum((x - c0[0])**2) for x in X])
probs1 = D2_step1 / D2_step1.sum()
c1_idx = np.argmax(probs1)
c1 = X[[c1_idx]]
D2_step2 = np.minimum(D2_step1, np.array([np.sum((x - c1[0])**2) for x in X]))
probs2 = D2_step2 / D2_step2.sum()
c2_idx = np.argmax(probs2)
c2 = X[[c2_idx]]

# ── 绘图 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

# Panel 1: 随机初始化问题示例（取最差的一次）
ax = axes[0]
worst_seed = int(np.argmax(sse_rand))
c_worst, _ = kmeans(X, random_init(X, K, worst_seed))
l_worst = assign(X, c_worst)
for k in range(K):
    ax.scatter(X[l_worst==k,0], X[l_worst==k,1],
               c=PALETTE[k], s=20, alpha=0.6, linewidths=0)
ax.scatter(c_worst[:,0], c_worst[:,1], c=PALETTE[:K], s=200,
           marker='*', edgecolors='k', linewidths=0.8, zorder=5)
ax.set_title("（a）随机初始化的次优结果示例", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])

# Panel 2: K-means++ 概率示意
ax = axes[1]
norm = Normalize(vmin=0, vmax=probs2.max())
sc = ax.scatter(X[:,0], X[:,1], c=probs2, cmap='YlOrRd', s=35,
                norm=norm, linewidths=0, zorder=2)
ax.scatter(c0[:,0], c0[:,1], c='royalblue', s=250, marker='*',
           edgecolors='k', linewidths=0.9, zorder=6, label='质心 1')
ax.scatter(c1[:,0], c1[:,1], c='green', s=250, marker='*',
           edgecolors='k', linewidths=0.9, zorder=6, label='质心 2')
ax.scatter(c2[:,0], c2[:,1], c='crimson', s=250, marker='*',
           edgecolors='k', linewidths=0.9, zorder=6, label='质心 3')
plt.colorbar(sc, ax=ax, label='$D^2$ 选择概率', shrink=0.8)
ax.set_title("（b）K-means++ 按 $D^2$ 加权概率选第 3 个质心", fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.set_xticks([]); ax.set_yticks([])

# Panel 3: SSE分布箱线图对比
ax = axes[2]
bp = ax.boxplot([sse_rand, sse_pp], labels=['随机初始化', 'K-means++'],
                patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor(PALETTE[3] + '99')
bp['boxes'][1].set_facecolor(PALETTE[0] + '99')
ax.set_ylabel('最终 SSE', fontsize=9)
ax.set_title(f"（c）SSE 分布对比（{N_RUNS} 次重复）", fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

# 添加均值点
ax.scatter([1, 2], [np.mean(sse_rand), np.mean(sse_pp)],
           c='red', s=60, zorder=5, marker='D', label='均值')
ax.legend(fontsize=8)

fig.suptitle('K-means 初始化策略：随机初始化 vs K-means++', fontsize=11, y=1.01)
fig.tight_layout()
save_fig(fig, __file__, 'fig5_1_02_kmeanspp_init')
