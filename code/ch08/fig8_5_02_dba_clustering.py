"""
fig8_5_02_dba_clustering.py
DBA 质心 vs 逐点平均 + k-means 聚类结果
左：质心对比  右：聚类结果（3 簇 + 质心）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 简化 DTW + 路径回溯 ──────────────────────────────────────────
def dtw_path(x, y):
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = (x[i-1] - y[j-1])**2 + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            idx = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]])
            if idx == 0: i, j = i-1, j-1
            elif idx == 1: i -= 1
            else: j -= 1
    path.reverse()
    return path
# ── DBA 实现 ──────────────────────────────────────────────────────
def dba(series_list, n_iter=10):
    centroid = series_list[0].copy()
    T = len(centroid)
    for _ in range(n_iter):
        assoc = [[] for _ in range(T)]
        for s in series_list:
            path = dtw_path(centroid, s)
            for ci, si in path:
                assoc[ci].append(s[si])
        for j in range(T):
            if assoc[j]:
                centroid[j] = np.mean(assoc[j])
    return centroid
# ── 生成三组序列（不同形状 + 随机相位偏移）─────────────────────
T = 80
t = np.linspace(0, 2 * np.pi, T)
n_per = 8
# 簇 1：正弦
cluster1 = []
for _ in range(n_per):
    phase = np.random.uniform(-0.5, 0.5)
    cluster1.append(np.sin(t + phase) + np.random.normal(0, 0.08, T))
# 簇 2：三角波
cluster2 = []
for _ in range(n_per):
    phase = np.random.uniform(-0.5, 0.5)
    from scipy.signal import sawtooth
    cluster2.append(sawtooth(t + phase, width=0.5) + np.random.normal(0, 0.08, T))
# 簇 3：方波近似
cluster3 = []
for _ in range(n_per):
    phase = np.random.uniform(-0.5, 0.5)
    sq = np.sign(np.sin(t + phase)) * 0.8
    cluster3.append(sq + np.random.normal(0, 0.1, T))
all_series = cluster1 + cluster2 + cluster3
true_labels = [0]*n_per + [1]*n_per + [2]*n_per
# ── 计算质心 ──────────────────────────────────────────────────────
# 逐点平均
pointwise_avg = np.mean(cluster1, axis=0)
# DBA 质心
dba_centroid = dba(cluster1, n_iter=5)
# 各簇 DBA 质心
centroid1 = dba(cluster1, n_iter=5)
centroid2 = dba(cluster2, n_iter=5)
centroid3 = dba(cluster3, n_iter=5)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 8.5.2　DBA 质心与 k-means 聚类",
             fontsize=20, fontweight="bold", y=1.02)
# ── 左面板：逐点平均 vs DBA ───────────────────────────────────────
ax = axes[0]
for s in cluster1:
    ax.plot(np.arange(T), s, color=COLORS["blue"], lw=0.6, alpha=0.3)
ax.plot(np.arange(T), pointwise_avg, color=COLORS["gray"], lw=2.5,
        ls="--", label="逐点平均", zorder=5)
ax.plot(np.arange(T), dba_centroid, color=COLORS["red"], lw=2.5,
        label="DBA 质心", zorder=5)
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("值", fontsize=14)
ax.set_title("(a) 质心对比（簇 1: 正弦波）", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)
# ── 右面板：聚类结果 ──────────────────────────────────────────────
ax = axes[1]
cluster_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
cluster_data = [cluster1, cluster2, cluster3]
centroids = [centroid1, centroid2, centroid3]
cluster_names = ["簇1: 正弦波", "簇2: 三角波", "簇3: 方波"]
for ci, (data, cent, color, name) in enumerate(zip(cluster_data, centroids,
                                                    cluster_colors, cluster_names)):
    for s in data:
        ax.plot(np.arange(T), s + ci * 3.5, color=color, lw=0.5, alpha=0.3)
    ax.plot(np.arange(T), cent + ci * 3.5, color=color, lw=3.0,
            label=name, zorder=5)
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("值（按簇偏移）", fontsize=14)
ax.set_title("(b) DBA k-means 聚类（K=3）", fontsize=15)
ax.legend(fontsize=12, loc="upper right")
ax.tick_params(labelsize=13)
ax.set_yticks([])
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_5_02_dba_clustering")
