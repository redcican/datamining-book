"""
fig9_5_01_sentiment_architecture.py
情感分析方法架构：词典流程 vs BiLSTM+Attention 深度模型
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 辅助函数 ──────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, color, fontsize=12, tc="white",
             ec=None, lw=1.5, alpha=1.0):
    if ec is None:
        ec = "white"
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=tc)

def draw_arrow(ax, x1, y1, x2, y2, color=None):
    if color is None:
        color = COLORS["gray"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=15))
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.5.1　情感分析方法架构",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 词典情感分析流程
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 11)
ax.axis("off")
ax.set_aspect("equal")

# 1. 输入评论文本
ax.text(5, 10.3, "这款手机屏幕清晰，但电池不耐用",
        ha="center", fontsize=13, fontweight="bold", color=COLORS["gray"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                  edgecolor=COLORS["gray"], linewidth=1.2))
draw_arrow(ax, 5, 9.6, 5, 9.0)

# 2. 情感词典匹配
draw_box(ax, 1.5, 8.0, 7, 0.8, "情感词典匹配", COLORS["blue"], fontsize=14)
draw_arrow(ax, 5, 7.8, 5, 7.2)

# 3. 匹配结果展示
y_match = 6.0
# "清晰" - 正面词
draw_box(ax, 0.5, y_match, 2.0, 0.8, "清晰 (+1)", COLORS["green"],
         fontsize=13, ec=COLORS["green"], lw=2)
ax.text(1.5, y_match + 1.0, "正面词", ha="center", fontsize=12,
        color=COLORS["green"], fontweight="bold")

# "但" - 转折词
draw_box(ax, 3.5, y_match, 1.5, 0.8, "但", COLORS["orange"],
         fontsize=13, ec=COLORS["orange"], lw=2)
ax.text(4.25, y_match + 1.0, "转折词", ha="center", fontsize=12,
        color=COLORS["orange"], fontweight="bold")

# "不耐用" - 否定+正面=负面
draw_box(ax, 6.0, y_match, 3.5, 0.8, "不 + 耐用 → 负面",
         COLORS["red"], fontsize=13, ec=COLORS["red"], lw=2)
ax.text(7.75, y_match + 1.0, "否定 + 正面词", ha="center", fontsize=12,
        color=COLORS["red"], fontweight="bold")

draw_arrow(ax, 5, 5.8, 5, 5.2)

# 4. 规则处理
draw_box(ax, 1.5, 4.2, 7, 0.8, "规则处理（否定词、程度副词、转折）",
         COLORS["purple"], fontsize=12)
draw_arrow(ax, 5, 4.0, 5, 3.4)

# 5. 得分汇总
ax.text(5, 2.8, "正面: +1.0 × 0.8 = +0.8", ha="center", fontsize=12,
        color=COLORS["green"], fontweight="bold")
ax.text(5, 2.1, "负面: -1.0 × 1.3 = -1.3", ha="center", fontsize=12,
        color=COLORS["red"], fontweight="bold")
ax.text(5, 1.4, "(转折词加权 ×1.3)", ha="center", fontsize=11,
        color=COLORS["orange"])
draw_arrow(ax, 5, 1.0, 5, 0.5)

# 6. 最终结果
ax.text(5, 0.0, "情感得分: -0.5 → 负面", ha="center", fontsize=14,
        fontweight="bold", color=COLORS["red"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["red"], linewidth=2))

ax.set_title("(a) 词典情感分析流程", fontsize=17, pad=5)

# ══════════════════════════════════════════════════════════════════
# (b) BiLSTM + Attention 架构
# ══════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 11)
ax.axis("off")
ax.set_aspect("equal")

# 1. 底层: Word Embedding
words = ["这", "部", "电影", "非常", "精彩"]
n_words = len(words)
x_start = 0.5
word_w = 1.6
word_gap = 0.2
y_embed = 0.0

for i, w in enumerate(words):
    xi = x_start + i * (word_w + word_gap)
    draw_box(ax, xi, y_embed, word_w, 0.6, w, COLORS["blue"],
             fontsize=12, alpha=0.8)
ax.text(5, -0.5, "词嵌入层 (Word Embedding)", ha="center",
        fontsize=13, color=COLORS["blue"], fontweight="bold")

# 2. BiLSTM 层
y_lstm = 2.0
lstm_h = 1.2
draw_box(ax, 0.3, y_lstm, 9.4, lstm_h, "", COLORS["teal"],
         fontsize=12, alpha=0.3, ec=COLORS["teal"], lw=2)

# Forward arrow
ax.annotate("", xy=(8.5, y_lstm + 0.8), xytext=(1.0, y_lstm + 0.8),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["green"],
                            lw=2.5, mutation_scale=18))
ax.text(4.8, y_lstm + 0.85, "Forward →", ha="center", fontsize=12,
        fontweight="bold", color=COLORS["green"])

# Backward arrow
ax.annotate("", xy=(1.0, y_lstm + 0.3), xytext=(8.5, y_lstm + 0.3),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["red"],
                            lw=2.5, mutation_scale=18))
ax.text(4.8, y_lstm + 0.35, "← Backward", ha="center", fontsize=12,
        fontweight="bold", color=COLORS["red"])

ax.text(5, y_lstm - 0.3, "BiLSTM 层", ha="center", fontsize=13,
        color=COLORS["teal"], fontweight="bold")

# Arrows from embedding to BiLSTM
for i in range(n_words):
    xi = x_start + i * (word_w + word_gap) + word_w / 2
    draw_arrow(ax, xi, y_embed + 0.7, xi, y_lstm)

# 3. Attention 层
y_att = 4.5
att_h = 1.8
draw_box(ax, 0.3, y_att, 9.4, att_h, "", COLORS["light"],
         fontsize=12, alpha=0.5, ec=COLORS["purple"], lw=2)
ax.text(5, y_att + att_h + 0.2, "Attention 层", ha="center",
        fontsize=13, color=COLORS["purple"], fontweight="bold")

# Attention weights (bars)
att_weights = [0.05, 0.05, 0.15, 0.30, 0.45]
max_bar_h = 1.4
for i, (w, aw) in enumerate(zip(words, att_weights)):
    xi = x_start + i * (word_w + word_gap)
    bar_h = aw * max_bar_h / 0.45
    alpha_val = 0.3 + 0.7 * (aw / 0.45)
    draw_box(ax, xi + 0.1, y_att + 0.2, word_w - 0.2, bar_h,
             "", COLORS["purple"], alpha=alpha_val * 0.6,
             ec=COLORS["purple"], lw=1)
    ax.text(xi + word_w / 2, y_att + 0.2 + bar_h + 0.1,
            f"α={aw:.2f}", ha="center", fontsize=10,
            color=COLORS["purple"], fontweight="bold")
    ax.text(xi + word_w / 2, y_att + 0.05, w, ha="center",
            fontsize=10, color=COLORS["gray"])

# Arrows from BiLSTM to Attention
draw_arrow(ax, 5, y_lstm + lstm_h + 0.1, 5, y_att)

# 4. Softmax 输出
y_out = 7.8
draw_arrow(ax, 5, y_att + att_h + 0.1, 5, y_out)

# 输出框
draw_box(ax, 1.0, y_out, 3.5, 0.8, "正面: 0.92",
         COLORS["green"], fontsize=14, ec=COLORS["green"], lw=2.5)
draw_box(ax, 5.5, y_out, 3.5, 0.8, "负面: 0.08",
         COLORS["gray"], fontsize=14, alpha=0.5, ec=COLORS["gray"], lw=1.5)
ax.text(5, y_out + 1.1, "Softmax 输出", ha="center", fontsize=13,
        color=COLORS["gray"], fontweight="bold")

ax.set_title("(b) BiLSTM + Attention 架构", fontsize=17, pad=5)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_5_01_sentiment_architecture")
