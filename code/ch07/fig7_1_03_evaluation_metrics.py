"""
fig7_1_03_evaluation_metrics.py
类不平衡下的评估指标：ROC 曲线 vs PR 曲线
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 合成不平衡数据 (1% 异常率) ────────────────────────────────
n_total = 10000
anomaly_rate = 0.01
n_anomaly = int(n_total * anomaly_rate)
n_normal = n_total - n_anomaly

y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

# 三个检测器的评分：好 / 中等 / 随机
# 好检测器：正常分布偏低，异常分布偏高，区分度大
scores_good_n = np.random.beta(2, 10, n_normal)
scores_good_a = np.random.beta(8, 2, n_anomaly)
scores_good = np.concatenate([scores_good_n, scores_good_a])

# 中等检测器：区分度中等
scores_med_n = np.random.beta(2, 5, n_normal)
scores_med_a = np.random.beta(4, 2, n_anomaly)
scores_med = np.concatenate([scores_med_n, scores_med_a])

# 随机检测器
scores_rand = np.random.uniform(0, 1, n_total)

detectors = [
    ("优秀检测器", scores_good, COLORS["blue"]),
    ("中等检测器", scores_med, COLORS["orange"]),
    ("随机检测器", scores_rand, COLORS["gray"]),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.1.3　类不平衡下的评估指标", fontsize=20, fontweight="bold", y=1.02)

# ── 左：ROC 曲线 ─────────────────────────────────────────────
ax = axes[0]
for name, scores, color in detectors:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], color=COLORS["light"], ls="--", lw=1.5)
ax.set_title("(a) ROC 曲线", fontsize=17)
ax.set_xlabel("假正例率 (FPR)", fontsize=15)
ax.set_ylabel("真正例率 (TPR)", fontsize=15)
ax.legend(fontsize=13, loc="lower right")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.tick_params(labelsize=13)
ax.text(0.55, 0.35, "异常率=1%\nROC 差异不显著",
        fontsize=13, color=COLORS["gray"], transform=ax.transAxes,
        ha="center", style="italic")

# ── 右：PR 曲线 ──────────────────────────────────────────────
ax = axes[1]
for name, scores, color in detectors:
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    ax.plot(rec, prec, color=color, lw=2.5, label=f"{name} (AP={ap:.3f})")

# 基线：随机猜测的精确率 = anomaly_rate
ax.axhline(anomaly_rate, color=COLORS["light"], ls="--", lw=1.5, label=f"随机基线 ({anomaly_rate})")

ax.set_title("(b) PR 曲线", fontsize=17)
ax.set_xlabel("召回率 (Recall)", fontsize=15)
ax.set_ylabel("精确率 (Precision)", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.02, 1.05)
ax.tick_params(labelsize=13)
ax.text(0.55, 0.55, "PR 曲线更能\n区分检测器优劣",
        fontsize=14, color=COLORS["red"], transform=ax.transAxes,
        ha="center", fontweight="bold")

fig.tight_layout()
save_fig(fig, __file__, "fig7_1_03_evaluation_metrics")
