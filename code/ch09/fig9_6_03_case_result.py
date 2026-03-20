"""
fig9_6_03_case_result.py
文本摘要与实体抽取案例结果：TextRank 句子得分 + 实体识别结果统计
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
fig.suptitle("图 9.6.3　文本摘要与实体抽取案例结果",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) TextRank 句子得分
# ══════════════════════════════════════════════════════════════════
ax = axes[0]

sentence_labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
tr_scores = [0.22, 0.15, 0.25, 0.12, 0.18, 0.08]
n_sentences = len(sentence_labels)
x_pos = np.arange(n_sentences)

# Top-2 句子: S3 (index 2) 和 S1 (index 0)
top2_idx = {0, 2}
bar_colors = [COLORS["red"] if i in top2_idx else COLORS["blue"]
              for i in range(n_sentences)]

bars = ax.bar(x_pos, tr_scores, width=0.6, color=bar_colors, alpha=0.85,
              edgecolor="white", linewidth=1.5)

# 柱顶标注数值
for bar, score in zip(bars, tr_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{score:.2f}", ha="center", va="bottom",
            fontsize=13, fontweight="bold")

# 选取阈值虚线（取 Top-2 的最小值 0.22 作为阈值）
threshold = 0.20
ax.axhline(y=threshold, color=COLORS["orange"], lw=2, ls="--", alpha=0.8,
           label=f"选取阈值 ({threshold:.2f})")

ax.set_xticks(x_pos)
ax.set_xticklabels(sentence_labels, fontsize=14)
ax.set_xlabel("句子", fontsize=16)
ax.set_ylabel("TextRank 得分", fontsize=16)
ax.set_ylim(0, 0.32)
ax.legend(fontsize=14, loc="upper right")
ax.tick_params(labelsize=14)
ax.set_title("(a) TextRank 句子得分", fontsize=17)

# ══════════════════════════════════════════════════════════════════
# (b) 实体识别结果统计
# ══════════════════════════════════════════════════════════════════
ax = axes[1]

entity_types = ["人名", "地名", "机构名"]
precision = [0.88, 0.92, 0.85]
recall = [0.82, 0.89, 0.80]
f1 = [0.85, 0.90, 0.82]

n_types = len(entity_types)
n_metrics = 3
bar_width = 0.22
x_base = np.arange(n_types)

metric_names = ["Precision", "Recall", "F1"]
metric_values = [precision, recall, f1]
metric_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]

for i, (name, vals, color) in enumerate(zip(metric_names, metric_values,
                                             metric_colors)):
    offset = (i - 1) * bar_width
    bars = ax.bar(x_base + offset, vals, bar_width * 0.9,
                  color=color, alpha=0.85, label=name,
                  edgecolor="white", linewidth=0.8)
    # 柱顶标注数值
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)

ax.set_xticks(x_base)
ax.set_xticklabels(entity_types, fontsize=14)
ax.set_xlabel("实体类型", fontsize=16)
ax.set_ylabel("分数", fontsize=16)
ax.set_ylim(0.65, 1.02)
ax.legend(fontsize=14, loc="upper right", ncol=3)
ax.tick_params(labelsize=14)
ax.set_title("(b) 实体识别结果统计", fontsize=17)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_6_03_case_result")
