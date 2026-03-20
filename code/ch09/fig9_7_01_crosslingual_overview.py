"""
fig9_7_01_crosslingual_overview.py
跨语言文本挖掘方法概览：三种迁移策略 + 词嵌入对齐可视化
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
fig.suptitle("图 9.7.1　跨语言文本挖掘方法概览",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 三种跨语言迁移策略
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-1.0, 13.0)
ax.set_ylim(-0.5, 10.5)
ax.axis("off")
ax.set_aspect("equal")

# Box dimensions
bw, bh = 2.0, 0.9
arrow_gap = 0.3

# ── Strategy 1 (top): 翻译后测试 ──
y1 = 8.0
ax.text(-0.5, y1 + bh / 2, "策略1", ha="center", va="center",
        fontsize=13, fontweight="bold", color=COLORS["gray"])

boxes_s1 = [
    (1.0, y1, "中文文本", COLORS["red"]),
    (4.0, y1, "机器翻译", COLORS["gray"]),
    (7.0, y1, "英文模型", COLORS["blue"]),
    (10.0, y1, "结果", COLORS["green"]),
]
for (bx, by, label, color) in boxes_s1:
    draw_box(ax, bx, by, bw, bh, label, color, fontsize=12)
for i in range(len(boxes_s1) - 1):
    x_from = boxes_s1[i][0] + bw
    x_to = boxes_s1[i + 1][0]
    cy = y1 + bh / 2
    draw_arrow(ax, x_from + 0.05, cy, x_to - 0.05, cy, color="#333333")

ax.text(6.0, y1 + bh + 0.4, "翻译后测试", ha="center", va="center",
        fontsize=14, fontweight="bold", color=COLORS["gray"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9",
                  edgecolor=COLORS["gray"], linewidth=1.0))

# ── Strategy 2 (middle): 嵌入对齐 ──
y2 = 4.5
ax.text(-0.5, y2 + bh / 2, "策略2", ha="center", va="center",
        fontsize=13, fontweight="bold", color=COLORS["gray"])

boxes_s2 = [
    (1.0, y2, "中文嵌入", COLORS["red"]),
    (4.0, y2, "W* 对齐", COLORS["purple"]),
    (7.0, y2, "英文空间\n分类", COLORS["blue"]),
    (10.0, y2, "结果", COLORS["green"]),
]
for (bx, by, label, color) in boxes_s2:
    draw_box(ax, bx, by, bw, bh, label, color, fontsize=12)
for i in range(len(boxes_s2) - 1):
    x_from = boxes_s2[i][0] + bw
    x_to = boxes_s2[i + 1][0]
    cy = y2 + bh / 2
    draw_arrow(ax, x_from + 0.05, cy, x_to - 0.05, cy, color="#333333")

ax.text(6.0, y2 + bh + 0.4, "嵌入对齐", ha="center", va="center",
        fontsize=14, fontweight="bold", color=COLORS["purple"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f3ff",
                  edgecolor=COLORS["purple"], linewidth=1.0))

# ── Strategy 3 (bottom): 多语言模型 ──
y3 = 1.0
ax.text(-0.5, y3 + bh / 2, "策略3", ha="center", va="center",
        fontsize=13, fontweight="bold", color=COLORS["gray"])

boxes_s3 = [
    (1.0, y3, "中文文本", COLORS["red"]),
    (5.0, y3, "mBERT", COLORS["orange"]),
    (9.0, y3, "结果", COLORS["green"]),
]
for (bx, by, label, color) in boxes_s3:
    w = bw if label != "mBERT" else 2.5
    draw_box(ax, bx, by, w, bh, label, color, fontsize=12)
# Arrows for strategy 3 (variable widths)
draw_arrow(ax, 1.0 + bw + 0.05, y3 + bh / 2,
           5.0 - 0.05, y3 + bh / 2, color="#333333")
draw_arrow(ax, 5.0 + 2.5 + 0.05, y3 + bh / 2,
           9.0 - 0.05, y3 + bh / 2, color="#333333")

ax.text(6.0, y3 + bh + 0.4, "多语言模型", ha="center", va="center",
        fontsize=14, fontweight="bold", color=COLORS["orange"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ed",
                  edgecolor=COLORS["orange"], linewidth=1.0))

ax.set_title("(a) 三种跨语言迁移策略", fontsize=17, pad=5)

# ══════════════════════════════════════════════════════════════════
# (b) 跨语言词嵌入对齐效果
# ══════════════════════════════════════════════════════════════════
ax = axes[1]

# Word pairs: (Chinese, English, ch_x, ch_y, en_x, en_y)
word_pairs = [
    ("猫", "cat",   1.5, 4.2, 4.5, 4.0),
    ("狗", "dog",   1.8, 3.2, 4.8, 3.0),
    ("鸟", "bird",  1.2, 2.5, 5.0, 2.2),
    ("鱼", "fish",  2.0, 1.8, 4.2, 1.5),
    ("书", "book",  1.0, 3.8, 5.2, 3.6),
]

# Aligned Chinese positions (moved closer to English)
aligned_ch = [
    (3.5, 4.1),
    (3.8, 3.1),
    (3.6, 2.3),
    (3.5, 1.6),
    (3.8, 3.7),
]

# Plot original Chinese words
ch_x = [p[2] for p in word_pairs]
ch_y = [p[3] for p in word_pairs]
ax.scatter(ch_x, ch_y, c=COLORS["red"], s=100, zorder=5, alpha=0.6,
           edgecolors="white", linewidth=1.0)

# Plot English words
en_x = [p[4] for p in word_pairs]
en_y = [p[5] for p in word_pairs]
ax.scatter(en_x, en_y, c=COLORS["blue"], s=100, zorder=5,
           edgecolors="white", linewidth=1.0)

# Dashed lines connecting original translation pairs
for p in word_pairs:
    ax.plot([p[2], p[4]], [p[3], p[5]], ls="--", lw=1.0,
            color=COLORS["gray"], alpha=0.4, zorder=2)

# Label original Chinese words
for p in word_pairs:
    ax.annotate(p[0], (p[2], p[3]), textcoords="offset points",
                xytext=(-16, 6), fontsize=12, color=COLORS["red"],
                fontweight="bold")

# Label English words
for p in word_pairs:
    ax.annotate(p[1], (p[4], p[5]), textcoords="offset points",
                xytext=(8, 6), fontsize=12, color=COLORS["blue"],
                fontweight="bold")

# Draw arrows from original Chinese to aligned positions
for i, (p, ap) in enumerate(zip(word_pairs, aligned_ch)):
    ax.annotate("", xy=ap, xytext=(p[2], p[3]),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["orange"],
                                lw=1.5, mutation_scale=12, alpha=0.8))

# Plot aligned Chinese words (smaller, semi-transparent)
al_x = [a[0] for a in aligned_ch]
al_y = [a[1] for a in aligned_ch]
ax.scatter(al_x, al_y, c=COLORS["red"], s=80, zorder=5, alpha=0.9,
           edgecolors=COLORS["orange"], linewidth=1.5, marker="D")

# Dashed lines connecting aligned pairs (shorter distance)
for i, (p, ap) in enumerate(zip(word_pairs, aligned_ch)):
    ax.plot([ap[0], p[4]], [ap[1], p[5]], ls="-", lw=1.2,
            color=COLORS["green"], alpha=0.5, zorder=2)

# Region labels
ax.text(1.4, 4.8, "对齐前", ha="center", fontsize=13,
        fontweight="bold", color=COLORS["red"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2",
                  edgecolor=COLORS["red"], linewidth=1.0, alpha=0.8))
ax.text(5.0, 4.8, "英文词", ha="center", fontsize=13,
        fontweight="bold", color=COLORS["blue"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe",
                  edgecolor=COLORS["blue"], linewidth=1.0, alpha=0.8))

# Annotation
ax.text(3.2, 0.8, "语义相近的词对靠近", ha="center", fontsize=13,
        fontstyle="italic", color=COLORS["green"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7",
                  edgecolor=COLORS["green"], linewidth=1.0, alpha=0.8))

# Legend
legend_handles = [
    mpatches.Patch(color=COLORS["red"], alpha=0.6, label="中文词 (原始)"),
    mpatches.Patch(color=COLORS["blue"], label="英文词"),
    mpatches.Patch(color=COLORS["orange"], label="对齐方向"),
]
ax.legend(handles=legend_handles, fontsize=14, loc="upper left",
          framealpha=0.9)

ax.set_xlabel("嵌入维度 1", fontsize=16)
ax.set_ylabel("嵌入维度 2", fontsize=16)
ax.set_xlim(0.3, 6.0)
ax.set_ylim(0.5, 5.3)
ax.tick_params(labelsize=14)
ax.set_title("(b) 跨语言词嵌入对齐效果", fontsize=17, pad=5)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_7_01_crosslingual_overview")
