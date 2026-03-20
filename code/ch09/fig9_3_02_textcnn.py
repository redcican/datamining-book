"""
fig9_3_02_textcnn.py
TextCNN 架构示意图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from shared.plot_config import apply_style, save_fig, COLORS
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
# ── 绘图 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle("图 9.3.2　TextCNN 架构示意",
             fontsize=22, fontweight="bold", y=0.98)
ax.set_xlim(0, 16)
ax.set_ylim(-1, 11)
ax.axis("off")
ax.set_aspect("equal")
# ── 辅助函数 ──────────────────────────────────────────────────────
def draw_rect(ax, x, y, w, h, color, alpha=0.7, ec="white", lw=1.5):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor=ec, linewidth=lw,
                          alpha=alpha)
    ax.add_patch(rect)
    return rect
def draw_arrow(ax, x1, y1, x2, y2, color=None):
    if color is None:
        color = COLORS["gray"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=15))
# ── 1. 输入文本（词序列）────────────────────────────────────────
words = ["这", "部", "电影", "非常", "精彩", "值得", "推荐"]
x_start = 0.3
y_words = 9.5
for i, w in enumerate(words):
    draw_rect(ax, x_start + i * 1.2, y_words, 1.0, 0.6, COLORS["blue"], alpha=0.8)
    ax.text(x_start + i * 1.2 + 0.5, y_words + 0.3, w, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
ax.text(x_start + 3.5 * 1.2, y_words + 1.0, "输入: 词序列 (L 个词)",
        ha="center", fontsize=13, color=COLORS["gray"], fontweight="bold")
# ── 2. 词嵌入矩阵 ────────────────────────────────────────────────
y_embed = 7.5
embed_w = 8.7
embed_h = 1.5
draw_rect(ax, 0.3, y_embed, embed_w, embed_h, COLORS["teal"], alpha=0.3,
          ec=COLORS["teal"], lw=2)
# 画网格线模拟矩阵
for i in range(8):
    ax.plot([0.3 + i * 1.2, 0.3 + i * 1.2], [y_embed, y_embed + embed_h],
            color=COLORS["teal"], alpha=0.3, lw=0.8)
for j in range(4):
    y_line = y_embed + j * embed_h / 3
    ax.plot([0.3, 0.3 + embed_w], [y_line, y_line],
            color=COLORS["teal"], alpha=0.3, lw=0.8)
ax.text(0.3 + embed_w / 2, y_embed + embed_h / 2,
        r"嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{L \times d}$",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=COLORS["teal"])
draw_arrow(ax, 4.5, y_words, 4.5, y_embed + embed_h + 0.1)
# ── 3. 卷积层（三种宽度）────────────────────────────────────────
conv_colors = [COLORS["red"], COLORS["green"], COLORS["orange"]]
conv_widths = [3, 4, 5]
conv_labels = ["h=3", "h=4", "h=5"]
y_conv = 4.8
conv_x_positions = [0.5, 3.5, 6.5]
for i, (cw, cl, cc, cx) in enumerate(zip(conv_widths, conv_labels,
                                          conv_colors, conv_x_positions)):
    # 卷积核
    draw_rect(ax, cx, y_conv, 2.2, 1.8, cc, alpha=0.6, ec=cc, lw=2)
    ax.text(cx + 1.1, y_conv + 1.15, f"卷积核 {cl}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(cx + 1.1, y_conv + 0.55, f"{cw}×d 滤波器\n×n 个核",
            ha="center", va="center", fontsize=9.5, color="white")
    # 箭头从嵌入到卷积
    draw_arrow(ax, cx + 1.1, y_embed, cx + 1.1, y_conv + 1.9, cc)
# ── 4. 最大池化 ───────────────────────────────────────────────────
y_pool = 2.8
for i, (cc, cx) in enumerate(zip(conv_colors, conv_x_positions)):
    draw_rect(ax, cx + 0.3, y_pool, 1.6, 1.0, cc, alpha=0.4, ec=cc, lw=2)
    ax.text(cx + 1.1, y_pool + 0.5, "Max\nPooling",
            ha="center", va="center", fontsize=11, fontweight="bold", color=cc)
    draw_arrow(ax, cx + 1.1, y_conv, cx + 1.1, y_pool + 1.1, cc)
# ── 5. 拼接 ──────────────────────────────────────────────────────
y_concat = 1.5
concat_x = 10.5
draw_rect(ax, concat_x, y_concat, 2.0, 2.3, COLORS["purple"], alpha=0.6,
          ec=COLORS["purple"], lw=2)
ax.text(concat_x + 1.0, y_concat + 1.15, "拼接\nConcat",
        ha="center", va="center", fontsize=12, fontweight="bold", color="white")
# 从三个池化连线到拼接
for i, (cc, cx) in enumerate(zip(conv_colors, conv_x_positions)):
    draw_arrow(ax, cx + 1.9, y_pool + 0.5, concat_x, y_concat + 1.15, cc)
# ── 6. 全连接 + Softmax ──────────────────────────────────────────
y_fc = 1.5
fc_x = 13.0
draw_rect(ax, fc_x, y_fc, 2.2, 2.3, COLORS["blue"], alpha=0.7,
          ec=COLORS["blue"], lw=2)
ax.text(fc_x + 1.1, y_fc + 1.4, "全连接\n+\nSoftmax",
        ha="center", va="center", fontsize=12, fontweight="bold", color="white")
draw_arrow(ax, concat_x + 2.1, y_concat + 1.15, fc_x, y_fc + 1.15,
           COLORS["purple"])
# ── 7. 输出类别 ──────────────────────────────────────────────────
y_out = -0.2
outputs = ["正面", "负面", "中性"]
out_probs = ["0.78", "0.15", "0.07"]
for i, (o, p) in enumerate(zip(outputs, out_probs)):
    ox = fc_x - 0.5 + i * 1.2
    draw_rect(ax, ox, y_out, 1.0, 0.55, COLORS["teal"] if i == 0 else COLORS["gray"],
              alpha=0.7 if i == 0 else 0.3)
    ax.text(ox + 0.5, y_out + 0.28, f"{o}\n{p}", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
draw_arrow(ax, fc_x + 1.1, y_fc, fc_x + 1.1, y_out + 0.65, COLORS["blue"])
# ── 保存 ──────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig9_3_02_textcnn")
