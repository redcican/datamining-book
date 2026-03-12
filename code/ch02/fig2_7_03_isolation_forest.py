"""
图 2.7.3  孤立森林：异常点路径更短（易被孤立）vs 正常点路径更长（需更多分割）
对应节次：2.7 异常值检测与处理
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch02/fig2_7_03_isolation_forest.py
输出路径：public/figures/ch02/fig2_7_03_isolation_forest.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

apply_style()

rng = np.random.default_rng(2024)

# --- 构造数据集 ---
n_norm = 200
# 正常点：椭圆分布（长轴倾斜）
cov = [[3.0, 1.5], [1.5, 1.5]]
X_norm = rng.multivariate_normal([0, 0], cov, n_norm)
# 注入异常点
X_out = np.array([[-5.0, -4.0], [5.5, 3.5], [-4.0, 4.5], [5.0, -4.0]])
X_all = np.vstack([X_norm, X_out])

# --- 孤立森林 ---
iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=2024)
iso.fit(X_all)
scores = iso.score_samples(X_all)        # 越负越异常
anom_score = -scores                     # 正向：越大越异常

# --- 绘图 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.subplots_adjust(wspace=0.40)

# ── Panel (a): 散点图，颜色编码异常得分 ─────────────────────────────────────
ax = axes[0]
vmin_n = anom_score[:n_norm].min()
vmax_n = anom_score[:n_norm].max()
# 正常点
sc = ax.scatter(X_norm[:, 0], X_norm[:, 1],
                c=anom_score[:n_norm], cmap="YlOrRd",
                vmin=vmin_n, vmax=anom_score.max() * 1.02,
                s=40, edgecolors="white", linewidths=0.4, zorder=2, label="正常数据点")
# 异常点（紫色边框标注）
ax.scatter(X_out[:, 0], X_out[:, 1],
           c=anom_score[n_norm:], cmap="YlOrRd",
           vmin=vmin_n, vmax=anom_score.max() * 1.02,
           s=160, edgecolors="#7c3aed", linewidths=2.5, zorder=5, label="注入异常点")
cb = fig.colorbar(sc, ax=ax, shrink=0.80, pad=0.02)
cb.set_label("孤立森林异常得分", fontsize=13)
cb.ax.tick_params(labelsize=12)

# 标注异常点得分
annot_offsets = [(0.4, 0.5), (-1.8, 0.5), (0.4, 0.5), (-1.8, 0.5)]
for k in range(4):
    idx = n_norm + k
    dx, dy = annot_offsets[k]
    ax.annotate(f"{anom_score[idx]:.2f}",
                xy=(X_out[k, 0], X_out[k, 1]),
                xytext=(X_out[k, 0] + dx, X_out[k, 1] + dy),
                fontsize=12, color="#7c3aed", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.2))

ax.set_xlabel("特征 $x_1$", fontsize=13)
ax.set_ylabel("特征 $x_2$", fontsize=13)
ax.set_title("(a) 孤立森林异常得分热图\n颜色越深 → 路径越短 → 越容易被孤立 → 越异常",
             fontsize=13, pad=6)
ax.legend(fontsize=12, loc="upper right")

# ── Panel (b): 正常点 vs 异常点路径长度（决策函数）分布 ──────────────────────
ax = axes[1]
score_norm = anom_score[:n_norm]
score_out  = anom_score[n_norm:]

bins = np.linspace(score_norm.min() - 0.02, anom_score.max() + 0.02, 35)
ax.hist(score_norm, bins=bins, density=True, alpha=0.65,
        color="#2563eb",
        label=f"正常数据点（n={n_norm}）\n均值={score_norm.mean():.3f}")

# 异常点画竖线
for k, s in enumerate(score_out):
    ax.axvline(s, color="#dc2626", lw=2.5, alpha=0.90,
               label=f"注入异常点（均值={score_out.mean():.3f}）" if k == 0 else None)

# 决策阈值
thresh = -iso.offset_
ax.axvline(thresh, color="#94a3b8", lw=2.0, ls="--",
           label=f"决策阈值 ≈ {thresh:.3f}")
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5],
                 thresh, anom_score.max() + 0.05,
                 alpha=0.10, color="#dc2626")

ax.set_xlabel("异常得分（路径长度代理，越大越短）", fontsize=13)
ax.set_ylabel("概率密度", fontsize=13)
ax.set_title("(b) 正常点 vs 异常点得分分布\n阈值右侧（红色区域）被判定为异常",
             fontsize=13, pad=6)
ax.legend(fontsize=12, loc="upper left", labelspacing=0.3)
ax.set_xlim(score_norm.min() - 0.02, anom_score.max() + 0.05)

fig.suptitle(
    "孤立森林（Isolation Forest，Liu et al., 2008）：随机分割孤立异常的原理\n"
    "异常点稀疏/极端 → 更少次随机切割即可孤立 → 路径更短 → 得分更高",
    fontsize=13, y=1.02)

save_fig(fig, __file__, "fig2_7_03_isolation_forest")
