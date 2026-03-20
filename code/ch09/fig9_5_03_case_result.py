"""
fig9_5_03_case_result.py
产品评论情感分析案例结果：分类指标对比 + seaborn KDE 情感得分分布
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.5.3　产品评论情感分析案例结果",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：三种方法分类指标对比 ──────────────────────────────────
ax = axes[0]
methods = ["词典方法", "朴素贝叶斯", "线性 SVM"]
metrics = ["Accuracy", "Precision", "Recall", "F1"]
metric_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
values = {
    "词典方法":  [0.76, 0.78, 0.74, 0.76],
    "朴素贝叶斯": [0.87, 0.88, 0.86, 0.87],
    "线性 SVM":  [0.91, 0.92, 0.90, 0.91],
}
n_methods = len(methods)
bar_width = 0.18
x_base = np.arange(n_methods)
for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    offset = (i - 1.5) * bar_width
    heights = [values[m][i] for m in methods]
    bars = ax.bar(x_base + offset, heights, bar_width * 0.9,
                  color=color, alpha=0.8, label=metric,
                  edgecolor="white", linewidth=0.5)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=color)
ax.set_xticks(x_base)
ax.set_xticklabels(methods, fontsize=14)
ax.set_ylabel("分数", fontsize=16)
ax.set_ylim(0.60, 1.02)
ax.legend(fontsize=14, loc="upper left", ncol=2)
ax.tick_params(labelsize=14)
ax.set_title("(a) 三种方法分类指标对比", fontsize=17)
# ── 右面板：seaborn KDE + 直方图 ──────────────────────────────────
ax = axes[1]
n_samples = 500
pos_scores = np.clip(np.random.normal(0.6, 0.25, n_samples), -1.2, 1.2)
neg_scores = np.clip(np.random.normal(-0.4, 0.30, n_samples), -1.2, 1.2)
# seaborn histplot with KDE
sns.histplot(pos_scores, bins=35, kde=True, color=COLORS["green"],
             alpha=0.4, label="正面评论", ax=ax, stat="count",
             edgecolor="white", linewidth=0.5,
             line_kws={"lw": 2.5})
sns.histplot(neg_scores, bins=35, kde=True, color=COLORS["red"],
             alpha=0.4, label="负面评论", ax=ax, stat="count",
             edgecolor="white", linewidth=0.5,
             line_kws={"lw": 2.5})
# 决策边界
ax.axvline(x=0, color=COLORS["gray"], lw=2.5, ls="--", alpha=0.8,
           label="决策边界 (score=0)")
# 混淆区域
y_max = ax.get_ylim()[1]
ax.fill_betweenx([0, y_max], -0.3, 0.3, alpha=0.08, color=COLORS["orange"])
ax.annotate("易混淆区域",
            xy=(0, y_max * 0.75),
            xytext=(0.65, y_max * 0.88),
            fontsize=13, fontweight="bold", color=COLORS["orange"],
            arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["orange"], alpha=0.9))
ax.set_xlabel("情感得分", fontsize=16)
ax.set_ylabel("评论数量", fontsize=16)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=14)
ax.set_title("(b) 情感得分分布", fontsize=17)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_5_03_case_result")
