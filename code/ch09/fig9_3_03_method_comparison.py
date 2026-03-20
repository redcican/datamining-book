"""
fig9_3_03_method_comparison.py
文本分类方法对比：准确率/F1 柱状图 + 混淆矩阵
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
# ── 模拟分类结果数据 ─────────────────────────────────────────────
methods = ["朴素贝叶斯", "线性 SVM", "TextCNN", "BERT 微调"]
accuracy = [0.867, 0.923, 0.941, 0.968]
macro_f1 = [0.854, 0.918, 0.937, 0.965]
train_time = [0.02, 0.8, 45, 320]  # 秒
# 朴素贝叶斯的混淆矩阵（5 类）
categories = ["体育", "科技", "财经", "娱乐", "教育"]
conf_matrix = np.array([
    [18,  0,  1,  1,  0],
    [ 0, 17,  2,  0,  1],
    [ 1,  2, 16,  0,  1],
    [ 1,  0,  0, 18,  1],
    [ 0,  1,  1,  1, 17],
])
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("图 9.3.3　文本分类方法对比",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：准确率和 Macro-F1 柱状图 ─────────────────────────────
ax = axes[0]
x = np.arange(len(methods))
width = 0.3
bars1 = ax.bar(x - width / 2, accuracy, width, color=COLORS["blue"],
               alpha=0.8, label="准确率", edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x + width / 2, macro_f1, width, color=COLORS["green"],
               alpha=0.8, label="Macro-F1", edgecolor="white", linewidth=1.5)
# 标注数值
for bar, val in zip(bars1, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=COLORS["blue"])
for bar, val in zip(bars2, macro_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=COLORS["green"])
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=13)
ax.set_ylabel("分数", fontsize=16)
ax.set_title("(a) 分类性能对比", fontsize=17)
ax.set_ylim(0.75, 1.05)
ax.legend(fontsize=13, loc="lower right")
ax.tick_params(labelsize=13)
# 训练时间标注（次坐标轴）
ax2 = ax.twinx()
ax2.plot(x, train_time, 'o--', color=COLORS["orange"], lw=2, markersize=8,
         label="训练时间", alpha=0.8)
for xi, t in zip(x, train_time):
    unit = "s" if t < 60 else "min"
    val = t if t < 60 else t / 60
    ax2.text(xi + 0.1, t * 1.1, f"{val:.1f}{unit}", fontsize=10,
             color=COLORS["orange"], fontweight="bold")
ax2.set_ylabel("训练时间 (秒, log)", fontsize=14, color=COLORS["orange"])
ax2.set_yscale("log")
ax2.tick_params(labelsize=12, colors=COLORS["orange"])
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(COLORS["orange"])
ax2.legend(fontsize=12, loc="upper left")
# ── 右面板：混淆矩阵 ─────────────────────────────────────────────
ax = axes[1]
im = ax.imshow(conf_matrix, cmap="Blues", aspect="equal")
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=13)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories, fontsize=13)
ax.set_xlabel("预测类别", fontsize=15)
ax.set_ylabel("真实类别", fontsize=15)
ax.set_title("(b) 朴素贝叶斯混淆矩阵", fontsize=17)
# 标注数值
for i in range(len(categories)):
    for j in range(len(categories)):
        val = conf_matrix[i, j]
        color = "white" if val > 10 else "black"
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=15, fontweight="bold", color=color)
plt.colorbar(im, ax=ax, shrink=0.8)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_3_03_method_comparison")
