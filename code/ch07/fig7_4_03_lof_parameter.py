"""
fig7_4_03_lof_parameter.py
k 值对 LOF 的影响
左：不同 k 值下 LOF 评分分布（箱线图）
右：AUC 随 k 变化曲线 + KNN 基线对比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import roc_auc_score
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成异质密度数据（与 fig7_4_01 相同） ────────────────────────
dense_cluster = np.random.multivariate_normal(
    [2, 2], [[0.09, 0], [0, 0.09]], 200)
sparse_cluster = np.random.multivariate_normal(
    [8, 8], [[1.44, 0], [0, 1.44]], 80)
anomalies = np.array([
    [12, 2], [0, 11], [13, 13], [-2, 8], [6, -1],
])

data = np.vstack([dense_cluster, sparse_cluster, anomalies])
n = len(data)

# 真实标签：0=正常，1=异常
labels = np.zeros(n, dtype=int)
labels[-len(anomalies):] = 1

# ── 不同 k 值的 LOF 评分 ─────────────────────────────────────────
k_values_box = [5, 10, 20, 50]
scores_normal = {}
scores_anomaly = {}

for k in k_values_box:
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
    lof.fit(data)
    lof_scores = -lof.negative_outlier_factor_
    scores_normal[k] = lof_scores[labels == 0]
    scores_anomaly[k] = lof_scores[labels == 1]

# ── AUC 随 k 变化 ────────────────────────────────────────────────
k_range = np.arange(3, 81)
auc_lof = []
auc_knn = []

for k in k_range:
    # LOF AUC
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
    lof.fit(data)
    lof_scores = -lof.negative_outlier_factor_
    auc_lof.append(roc_auc_score(labels, lof_scores))

    # KNN 距离 AUC
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(data)
    knn_dists, _ = nn.kneighbors(data)
    knn_score = knn_dists[:, -1]  # 第 k 近邻距离
    auc_knn.append(roc_auc_score(labels, knn_score))

auc_lof = np.array(auc_lof)
auc_knn = np.array(auc_knn)

# ── 绘图 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.4.3　k 值对 LOF 检测效果的影响",
             fontsize=20, fontweight="bold", y=1.02)

# ── 左：箱线图 ───────────────────────────────────────────────────
ax = axes[0]

positions_normal = np.arange(len(k_values_box)) * 2.5
positions_anomaly = positions_normal + 0.8

bp_normal = ax.boxplot(
    [scores_normal[k] for k in k_values_box],
    positions=positions_normal, widths=0.6,
    patch_artist=True, showfliers=False,
    boxprops=dict(facecolor=COLORS["blue"], alpha=0.5),
    medianprops=dict(color="k", lw=2),
    whiskerprops=dict(color=COLORS["blue"], lw=1.5),
    capprops=dict(color=COLORS["blue"], lw=1.5),
)

bp_anomaly = ax.boxplot(
    [scores_anomaly[k] for k in k_values_box],
    positions=positions_anomaly, widths=0.6,
    patch_artist=True, showfliers=False,
    boxprops=dict(facecolor=COLORS["red"], alpha=0.5),
    medianprops=dict(color="k", lw=2),
    whiskerprops=dict(color=COLORS["red"], lw=1.5),
    capprops=dict(color=COLORS["red"], lw=1.5),
)

# 参考线 LOF = 1
ax.axhline(1.0, color=COLORS["gray"], ls=":", lw=1.5, alpha=0.7)
ax.text(positions_normal[-1] + 1.5, 1.05, "LOF = 1",
        fontsize=13, color=COLORS["gray"], va="bottom")

ax.set_xticks((positions_normal + positions_anomaly) / 2)
ax.set_xticklabels([f"k = {k}" for k in k_values_box], fontsize=14)
ax.set_title("(a) 不同 k 值下 LOF 评分分布", fontsize=17)
ax.set_ylabel("LOF 评分", fontsize=15)
ax.tick_params(labelsize=13)

# 图例
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc=COLORS["blue"], alpha=0.5,
                   edgecolor="k", label="正常点"),
    plt.Rectangle((0, 0), 1, 1, fc=COLORS["red"], alpha=0.5,
                   edgecolor="k", label="异常点"),
]
ax.legend(handles=legend_handles, fontsize=13, loc="upper left")

# ── 右：AUC 随 k 变化 ───────────────────────────────────────────
ax = axes[1]

ax.plot(k_range, auc_lof, color=COLORS["blue"], lw=2.5,
        label="LOF", zorder=4)
ax.plot(k_range, auc_knn, color=COLORS["orange"], lw=2.5,
        ls="--", label="KNN 距离", zorder=3)

# 标注 LOF 平均 AUC
mean_lof = auc_lof.mean()
ax.axhline(mean_lof, color=COLORS["blue"], ls=":", lw=1.2, alpha=0.5)

ax.set_title("(b) AUC 随 k 变化", fontsize=17)
ax.set_xlabel("近邻数 k", fontsize=15)
ax.set_ylabel("ROC-AUC", fontsize=15)
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=14, loc="lower right")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_4_03_lof_parameter")
