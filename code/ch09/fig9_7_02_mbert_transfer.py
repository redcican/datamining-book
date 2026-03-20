"""
fig9_7_02_mbert_transfer.py
多语言模型跨语言迁移：mBERT 架构示意 + 零样本分类性能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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
fig.suptitle("图 9.7.2　多语言模型跨语言迁移",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) mBERT 多语言架构
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.axis("off")
ax.set_aspect("equal")

# Bottom layer: 3 input boxes
input_texts = ["Hello world", "你好世界", "Bonjour monde"]
input_colors = [COLORS["blue"], COLORS["red"], COLORS["green"]]
input_w, input_h = 2.8, 0.9
input_gap = 0.3
total_input_w = 3 * input_w + 2 * input_gap
x_start = (10.5 - total_input_w) / 2

for i, (text, color) in enumerate(zip(input_texts, input_colors)):
    xi = x_start + i * (input_w + input_gap)
    draw_box(ax, xi, 0.5, input_w, input_h, text, color,
             fontsize=11, ec=color, lw=2.0)

ax.text(5.0, -0.2, "多语言输入文本", ha="center", fontsize=13,
        fontweight="bold", color=COLORS["gray"])

# Arrows from inputs to transformer
mid_transformer_y = 3.0
for i in range(3):
    xi = x_start + i * (input_w + input_gap) + input_w / 2
    draw_arrow(ax, xi, 0.5 + input_h + 0.05, xi, mid_transformer_y - 0.05,
               color=input_colors[i])

# Middle: Shared Transformer block
trans_x, trans_y = 0.8, mid_transformer_y
trans_w, trans_h = 8.4, 2.8
draw_box(ax, trans_x, trans_y, trans_w, trans_h,
         "共享 Transformer 层\n(12 层)", COLORS["purple"],
         fontsize=15, ec=COLORS["purple"], lw=2.5, alpha=0.85)

# Small internal layer indicators
layer_y_start = trans_y + 0.3
layer_h = 0.15
for j in range(6):
    ly = layer_y_start + j * 0.35
    rect = FancyBboxPatch(
        (trans_x + 0.3, ly), trans_w - 0.6, layer_h,
        boxstyle="round,pad=0.02",
        facecolor="white", edgecolor="white", alpha=0.15, linewidth=0.5)
    ax.add_patch(rect)

# Arrows from transformer to output
output_y = 7.5
for i in range(3):
    xi = x_start + i * (input_w + input_gap) + input_w / 2
    draw_arrow(ax, xi, trans_y + trans_h + 0.05, xi, output_y - 0.05,
               color=COLORS["orange"])

# Top: Language-agnostic representation
draw_box(ax, 1.2, output_y, 7.6, 1.0,
         "语言无关的语义表示", COLORS["orange"],
         fontsize=14, ec=COLORS["orange"], lw=2.5)

# Annotation
ax.text(5.0, 9.2, "共享参数 → 隐式跨语言对齐", ha="center",
        fontsize=13, fontstyle="italic", color=COLORS["purple"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f3ff",
                  edgecolor=COLORS["purple"], linewidth=1.0, alpha=0.8))

ax.set_title("(a) mBERT 多语言架构", fontsize=17, pad=5)

# ══════════════════════════════════════════════════════════════════
# (b) 零样本跨语言分类性能
# ══════════════════════════════════════════════════════════════════
ax = axes[1]

languages = ["中文", "日文", "德文", "法文", "西班牙文"]
accuracies = [0.82, 0.78, 0.87, 0.89, 0.88]
baseline = 0.92
n_langs = len(languages)
x_pos = np.arange(n_langs)

bars = ax.bar(x_pos, accuracies, width=0.6, color=COLORS["teal"],
              alpha=0.85, edgecolor="white", linewidth=1.5,
              label="零样本迁移")

# Value labels on bars
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=COLORS["teal"])

# Baseline dashed line
ax.axhline(y=baseline, color=COLORS["red"], lw=2.0, ls="--",
           alpha=0.8, label=f"源语言基线 ({baseline:.2f})")

ax.set_xticks(x_pos)
ax.set_xticklabels(languages, fontsize=14)
ax.set_xlabel("目标语言", fontsize=16)
ax.set_ylabel("准确率 (Accuracy)", fontsize=16)
ax.set_ylim(0.5, 1.0)
ax.legend(fontsize=14, loc="lower right", framealpha=0.9)
ax.tick_params(labelsize=14)
ax.set_title("(b) 零样本跨语言分类性能", fontsize=17)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_7_02_mbert_transfer")
