"""
图 2.7.2  LOF 局部离群因子可视化：内部点（LOF≈1）/ 边界点 / 孤立异常点（LOF>>1）
对应节次：2.7 异常值检测与处理
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch02/fig2_7_02_lof_visualization.py
输出路径：public/figures/ch02/fig2_7_02_lof_visualization.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

apply_style()

rng = np.random.default_rng(2024)

# --- 构造双密度簇 + 孤立异常点 ---
# 密集簇：N((0,0), 0.5^2)，80 个点
Xd = rng.normal([0.0, 0.0], 0.5, (80, 2))
# 稀疏簇：N((6,0), 1.3^2)，40 个点
Xs = rng.normal([6.0, 0.0], 1.3, (40, 2))
# 孤立异常点（3 个）
X_out = np.array([[-3.5, 0.0], [3.0, 4.5], [9.5, -3.2]])
X_all = np.vstack([Xd, Xs, X_out])
n_norm = 120

# --- 计算 LOF 分数 ---
clf = LocalOutlierFactor(n_neighbors=15, contamination="auto")
clf.fit_predict(X_all)
lof_scores = -clf.negative_outlier_factor_   # 越大越异常

# --- 绘图 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.subplots_adjust(wspace=0.38)

# ── Panel (a): 散点图，颜色编码 LOF 分数 ─────────────────────────────────────
ax = axes[0]
vmax = min(lof_scores.max(), 8.0)
sc = ax.scatter(X_all[:, 0], X_all[:, 1],
                c=lof_scores, cmap="RdYlBu_r",
                vmin=1.0, vmax=vmax,
                s=55, edgecolors="white", linewidths=0.6, zorder=3)
cb = fig.colorbar(sc, ax=ax, shrink=0.80, pad=0.02)
cb.set_label("LOF 分数（越高越异常）", fontsize=13)
cb.ax.tick_params(labelsize=12)

# 标注孤立异常点的 LOF 值
# (dx, dy) offsets: k=0 左侧点右上, k=1 顶部点下移, k=2 右下点左上移
annot_offsets = [(0.4, 0.7), (0.4, -1.0), (-2.2, 1.0)]
for k in range(3):
    idx = n_norm + k
    dx, dy = annot_offsets[k]
    ax.annotate(f"LOF={lof_scores[idx]:.1f}",
                xy=(X_all[idx, 0], X_all[idx, 1]),
                xytext=(X_all[idx, 0] + dx, X_all[idx, 1] + dy),
                fontsize=12, color="#dc2626", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=1.2))

ax.set_xlabel("特征 $x_1$", fontsize=13)
ax.set_ylabel("特征 $x_2$", fontsize=13)
ax.set_title("(a) LOF 分数空间分布（k=15）\n颜色越深 → 局部密度越低 → 越可能是异常",
             fontsize=13, pad=6)

# 参考文字
ax.text(0.02, 0.98,
        "密集簇内部：LOF≈1\n稀疏簇内部：LOF≈1\n两簇之间/外部：LOF>>1",
        transform=ax.transAxes, ha="left", va="top", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cbd5e1", alpha=0.92))

# ── Panel (b): LOF 分数分布直方图 ─────────────────────────────────────────────
ax = axes[1]
lof_dense  = lof_scores[:80]
lof_sparse = lof_scores[80:120]
lof_outlier = lof_scores[120:]

bins = np.linspace(0.85, min(lof_scores.max() + 0.5, 9.0), 40)
ax.hist(lof_dense,  bins=bins, density=True, alpha=0.65,
        color="#2563eb", label=f"密集簇（n=80），均值={lof_dense.mean():.2f}")
ax.hist(lof_sparse, bins=bins, density=True, alpha=0.65,
        color="#16a34a", label=f"稀疏簇（n=40），均值={lof_sparse.mean():.2f}")

# 异常点标注为竖线
for i, s in enumerate(lof_outlier):
    ax.axvline(s, color="#dc2626", lw=2.5, alpha=0.95,
               label=f"孤立异常点（LOF={s:.1f}）" if i == 0 else None)

ax.axvline(1.0, color="#94a3b8", lw=1.8, ls="--", label="LOF=1（均匀密度理论值）")

ax.set_xlabel("LOF 分数", fontsize=13)
ax.set_ylabel("概率密度", fontsize=13)
ax.set_title("(b) 各类点 LOF 分数分布\n密集/稀疏簇内点集中在 1 附近，孤立点远端",
             fontsize=13, pad=6)
ax.legend(fontsize=12, loc="upper right", labelspacing=0.3)
ax.set_xlim(0.85, min(lof_scores.max() + 0.8, 9.5))

fig.suptitle(
    "LOF 局部离群因子（Breunig et al., 2000）：基于局部密度比的异常检测\n"
    "LOF(p) = 平均邻居密度 / p 自身密度；内部点 LOF≈1，异常点 LOF>>1",
    fontsize=13, y=1.02)

save_fig(fig, __file__, "fig2_7_02_lof_visualization")
