"""
fig10_1_03_graph_properties.py
图数据的统计特征
(a) 度分布 (BA 模型)  (b) 聚集系数 vs 度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
# ── 生成 BA 无标度网络 ────────────────────────────────────────────
G = nx.barabasi_albert_graph(5000, 3, seed=42)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 10.1.3　图数据的统计特征",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 度分布 — log-log 幂律分布
# ══════════════════════════════════════════════════════════════════
ax = axes[0]

degrees = [d for _, d in G.degree()]
degree_count = Counter(degrees)
k_vals = np.array(sorted(degree_count.keys()))
counts = np.array([degree_count[k] for k in k_vals])
pk = counts / counts.sum()  # 归一化为概率

# 只拟合有有效数据的点（排除 P(k)=0 的情况已保证）
log_k = np.log10(k_vals)
log_pk = np.log10(pk)

# 线性拟合 log P(k) = -gamma * log k + b
coeffs = np.polyfit(log_k, log_pk, 1)
gamma = -coeffs[0]
intercept = coeffs[1]

# 拟合线
k_fit = np.linspace(log_k.min(), log_k.max(), 200)
pk_fit = coeffs[0] * k_fit + intercept

ax.scatter(k_vals, pk, s=40, c=COLORS["blue"], alpha=0.7,
           edgecolors="k", linewidths=0.3, zorder=3, label="观测值")
ax.plot(10**k_fit, 10**pk_fit, color=COLORS["red"], lw=2.5, zorder=4,
        label=f"拟合: $\\gamma \\approx {gamma:.2f}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("(a) 度分布 (BA 模型, n=5000)", fontsize=17)
ax.set_xlabel("度 $k$", fontsize=16)
ax.set_ylabel("$P(k)$", fontsize=16)
ax.legend(fontsize=14, loc="upper right")
ax.tick_params(labelsize=14)

# ══════════════════════════════════════════════════════════════════
# (b) 聚集系数 vs 度
# ══════════════════════════════════════════════════════════════════
ax = axes[1]

clustering = nx.clustering(G)
node_degrees = np.array([G.degree(v) for v in G.nodes()])
node_clustering = np.array([clustering[v] for v in G.nodes()])

ax.scatter(node_degrees, node_clustering, s=12, c=COLORS["blue"],
           alpha=0.3, edgecolors="none", zorder=2, label="节点")

# 按度分箱，计算每个箱的平均聚集系数
unique_degrees = np.sort(np.unique(node_degrees))
# 使用对数分箱以适应幂律度分布
log_min = np.log10(unique_degrees.min())
log_max = np.log10(unique_degrees.max())
bin_edges = np.logspace(log_min, log_max, 20)

bin_centers = []
bin_means = []
for i in range(len(bin_edges) - 1):
    mask = (node_degrees >= bin_edges[i]) & (node_degrees < bin_edges[i + 1])
    if mask.sum() > 0:
        bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
        bin_means.append(node_clustering[mask].mean())

bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)

ax.plot(bin_centers, bin_means, color=COLORS["red"], lw=2.5,
        marker="o", markersize=7, zorder=4, label="分箱平均")

ax.set_xscale("log")
ax.set_title("(b) 聚集系数 vs 度", fontsize=17)
ax.set_xlabel("度 $d(v)$", fontsize=16)
ax.set_ylabel("聚集系数 $C(v)$", fontsize=16)
ax.legend(fontsize=14, loc="upper right")
ax.tick_params(labelsize=14)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig10_1_03_graph_properties")
