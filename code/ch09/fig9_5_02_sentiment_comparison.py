"""
fig9_5_02_sentiment_comparison.py
情感分析方法对比：准确率随数据量变化 + 正面/负面 Top-10 特征词
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
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.5.2　情感分析方法对比",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 方法准确率随数据量变化
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
data_sizes = [100, 200, 500, 1000, 2000, 5000]
x_pos = np.arange(len(data_sizes))

# 词典方法: 水平线, 不依赖训练数据
dict_acc = [0.72] * len(data_sizes)

# 朴素贝叶斯: 从低到高
nb_acc = [0.68, 0.72, 0.78, 0.82, 0.84, 0.86]

# 线性 SVM: 从略高起步, 上升更高
svm_acc = [0.70, 0.74, 0.80, 0.85, 0.87, 0.89]

# BiLSTM+Att: 起步低但上限高
lstm_acc = [0.55, 0.60, 0.72, 0.82, 0.88, 0.93]

ax.plot(x_pos, dict_acc, marker="s", ls="--", lw=2.5, ms=8,
        color=COLORS["gray"], label="词典方法", zorder=3)
ax.plot(x_pos, nb_acc, marker="o", lw=2.5, ms=8,
        color=COLORS["blue"], label="朴素贝叶斯", zorder=3)
ax.plot(x_pos, svm_acc, marker="^", lw=2.5, ms=8,
        color=COLORS["green"], label="线性 SVM", zorder=3)
ax.plot(x_pos, lstm_acc, marker="D", lw=2.5, ms=8,
        color=COLORS["red"], label="BiLSTM+Att", zorder=3)

# 标注交叉点: BiLSTM 超过 SVM (在 ~2000 附近)
cross_x = 4  # index for 2000
ax.annotate("深度模型超越 SVM",
            xy=(cross_x, lstm_acc[cross_x]),
            xytext=(cross_x - 1.5, lstm_acc[cross_x] + 0.07),
            fontsize=12, fontweight="bold", color=COLORS["red"],
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["red"], alpha=0.9))

ax.set_xticks(x_pos)
ax.set_xticklabels([str(s) for s in data_sizes], fontsize=14)
ax.set_xlabel("训练数据量（条）", fontsize=16)
ax.set_ylabel("准确率 (Accuracy)", fontsize=16)
ax.set_ylim(0.50, 1.0)
ax.legend(fontsize=14, loc="lower right")
ax.tick_params(labelsize=14)
ax.set_title("(a) 方法准确率随数据量变化", fontsize=17)

# ══════════════════════════════════════════════════════════════════
# (b) 正面/负面 Top-10 特征词
# ══════════════════════════════════════════════════════════════════
ax = axes[1]

pos_words = ["优秀", "出色", "满意", "推荐", "精美",
             "好用", "流畅", "清晰", "舒适", "方便"]
neg_words = ["差劲", "失望", "退货", "难用", "卡顿",
             "噪音", "发热", "漏水", "粗糙", "虚假"]

# Simulated TF-IDF weights (sorted descending)
np.random.seed(42)
pos_weights = np.sort(np.random.uniform(0.35, 0.80, 10))[::-1]
neg_weights = np.sort(np.random.uniform(0.30, 0.75, 10))[::-1]

# Combine: positive on right, negative on left
all_words = list(reversed(neg_words)) + list(reversed(pos_words))
all_weights = list(-np.sort(neg_weights)) + list(np.sort(pos_weights))
all_colors = [COLORS["red"]] * 10 + [COLORS["green"]] * 10

y_positions = np.arange(len(all_words))

bars = ax.barh(y_positions, all_weights, color=all_colors, alpha=0.75,
               edgecolor="white", linewidth=0.5, height=0.7)

ax.set_yticks(y_positions)
ax.set_yticklabels(all_words, fontsize=12)
ax.set_xlabel("TF-IDF 权重", fontsize=16)
ax.axvline(x=0, color=COLORS["gray"], lw=1.5, ls="-", alpha=0.5)
ax.tick_params(labelsize=14)

# Add labels for the two groups
ax.text(0.35, 14.5, "正面特征词", fontsize=14, fontweight="bold",
        color=COLORS["green"], ha="center")
ax.text(-0.35, 4.5, "负面特征词", fontsize=14, fontweight="bold",
        color=COLORS["red"], ha="center")

# Add a horizontal separator
ax.axhline(y=9.5, color=COLORS["gray"], lw=1, ls="--", alpha=0.4)

ax.set_title("(b) 正面/负面 Top-10 特征词", fontsize=17)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_5_02_sentiment_comparison")
