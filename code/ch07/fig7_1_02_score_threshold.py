"""
fig7_1_02_score_threshold.py
异常评分与阈值决策：分布直方图 + 阈值敏感性分析
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.1.2　异常评分与阈值决策", fontsize=20, fontweight="bold", y=1.02)

# ── 生成合成评分 ──────────────────────────────────────────────
n_normal = 1000
n_anomaly = 80
scores_normal = np.random.beta(2, 8, n_normal)       # 偏左
scores_anomaly = np.random.beta(5, 2, n_anomaly)      # 偏右
threshold = 0.45

# ── 左：评分分布 ──────────────────────────────────────────────
ax = axes[0]
bins = np.linspace(0, 1, 40)

ax.hist(scores_normal, bins=bins, alpha=0.55, color=COLORS["blue"],
        label=f"正常 (n={n_normal})", density=True, edgecolor="white", linewidth=0.5)
ax.hist(scores_anomaly, bins=bins, alpha=0.55, color=COLORS["red"],
        label=f"异常 (n={n_anomaly})", density=True, edgecolor="white", linewidth=0.5)

# KDE 曲线
x_kde = np.linspace(0, 1, 300)
kde_n = stats.gaussian_kde(scores_normal)(x_kde)
kde_a = stats.gaussian_kde(scores_anomaly)(x_kde)
ax.plot(x_kde, kde_n, color=COLORS["blue"], lw=2)
ax.plot(x_kde, kde_a, color=COLORS["red"], lw=2)

# 阈值线
ax.axvline(threshold, color=COLORS["gray"], ls="--", lw=2, label=f"阈值 τ={threshold}")

# 标注 TP/FP/FN/TN 区域
ymax = ax.get_ylim()[1]
ax.text(threshold + 0.12, ymax * 0.85, "TP", fontsize=16, color=COLORS["red"],
        fontweight="bold", ha="center")
ax.text(threshold + 0.12, ymax * 0.70, "(真正例)", fontsize=12, color=COLORS["red"],
        ha="center")
ax.text(threshold + 0.12, ymax * 0.55, "FP", fontsize=16, color=COLORS["blue"],
        fontweight="bold", ha="center")
ax.text(threshold + 0.12, ymax * 0.40, "(假正例)", fontsize=12, color=COLORS["blue"],
        ha="center")
ax.text(threshold - 0.15, ymax * 0.85, "FN", fontsize=16, color=COLORS["red"],
        fontweight="bold", ha="center")
ax.text(threshold - 0.15, ymax * 0.70, "(假反例)", fontsize=12, color=COLORS["red"],
        ha="center")
ax.text(threshold - 0.15, ymax * 0.55, "TN", fontsize=16, color=COLORS["blue"],
        fontweight="bold", ha="center")
ax.text(threshold - 0.15, ymax * 0.40, "(真反例)", fontsize=12, color=COLORS["blue"],
        ha="center")

ax.set_title("(a) 两种分布", fontsize=17)
ax.set_xlabel("异常评分", fontsize=15)
ax.set_ylabel("概率密度", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)

# ── 右：阈值敏感性分析 ───────────────────────────────────────
ax = axes[1]
taus = np.linspace(0.05, 0.95, 200)
precisions = []
recalls = []
f1s = []

for tau in taus:
    tp = np.sum(scores_anomaly >= tau)
    fp = np.sum(scores_normal >= tau)
    fn = np.sum(scores_anomaly < tau)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

precisions = np.array(precisions)
recalls = np.array(recalls)
f1s = np.array(f1s)

ax.plot(taus, precisions, color=COLORS["blue"], lw=2.5, label="精确率 (Precision)")
ax.plot(taus, recalls, color=COLORS["red"], lw=2.5, label="召回率 (Recall)")
ax.plot(taus, f1s, color=COLORS["green"], lw=2.5, ls="--", label="F1 分数")

# 最优 F1 点
best_idx = np.argmax(f1s)
best_tau = taus[best_idx]
best_f1 = f1s[best_idx]
ax.scatter([best_tau], [best_f1], c=COLORS["green"], s=120, zorder=5, edgecolors="k", linewidths=1)
ax.annotate(f"最优 F1={best_f1:.2f}\nτ*={best_tau:.2f}",
            xy=(best_tau, best_f1),
            xytext=(best_tau + 0.15, best_f1 - 0.15),
            fontsize=14, fontweight="bold", color=COLORS["green"],
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))

ax.set_title("(b) 阈值敏感性分析", fontsize=17)
ax.set_xlabel("阈值 τ", fontsize=15)
ax.set_ylabel("指标值", fontsize=15)
ax.legend(fontsize=13, loc="lower center")
ax.set_ylim(-0.05, 1.05)
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_1_02_score_threshold")
