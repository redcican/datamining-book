"""fig11_7_01_transfer_pipeline.py
(a) 迁移学习流程  (b) 三种微调策略对比"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5),
                                gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("图 11.7.1　迁移学习框架与微调策略",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════
LW = 2.0

def draw_box(ax, cx, cy, w, h, text, color, fontsize=15,
             text_color="white", alpha=0.9):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black", linewidth=1.8, alpha=alpha)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)

def arr(ax, x1, y1, x2, y2, color=COLORS["gray"], lw=LW, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.6, label, fontsize=13, fontweight="bold",
                color=color, ha="center", va="bottom")

# ══════════════════════════════════════════════════════════════════
# (a) 迁移学习流程
# ══════════════════════════════════════════════════════════════════
ax1.set_axis_off()
ax1.set_xlim(-1, 25)
ax1.set_ylim(-2, 14)
ax1.set_aspect("equal")

c_blue = COLORS["blue"]
c_teal = COLORS["teal"]
c_orange = COLORS["orange"]
c_red = COLORS["red"]
c_green = COLORS["green"]
c_gray = COLORS["gray"]

# --- Source domain data ---
draw_box(ax1, 2.0, 9.0, 3.5, 2.2, "源域数据\n(ImageNet)", c_blue, fontsize=14)
ax1.text(2.0, 7.2, "大规模标注", fontsize=13, fontweight="bold",
         color=c_blue, ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_blue, alpha=0.8))

# Arrow: source data → pretrained model
arr(ax1, 3.75, 9.0, 6.2, 9.0, color=c_blue, lw=2.5)

# --- Pretrained model (with layer stack) ---
draw_box(ax1, 8.5, 9.0, 4.0, 2.5, "", c_teal, fontsize=14, alpha=0.15)
ax1.text(8.5, 10.8, "预训练模型", fontsize=15, fontweight="bold",
         color=c_teal, ha="center")

# Layer stack inside pretrained model
layer_names_pre = ["Conv1", "Conv2", "Conv3", "Conv4", "FC"]
layer_colors_pre = [c_teal, c_teal, c_teal, c_teal, c_teal]
stack_x, stack_w = 8.5, 3.2
stack_bottom = 7.9
layer_h = 0.35
gap = 0.08
for i, (name, col) in enumerate(zip(layer_names_pre, layer_colors_pre)):
    y = stack_bottom + i * (layer_h + gap)
    rect = FancyBboxPatch(
        (stack_x - stack_w/2, y), stack_w, layer_h,
        boxstyle="round,pad=0.05",
        facecolor=col, edgecolor="white", linewidth=1.0, alpha=0.85)
    ax1.add_patch(rect)
    ax1.text(stack_x, y + layer_h/2, name, fontsize=10, fontweight="bold",
             color="white", ha="center", va="center")

# Arrow: pretrained model → target model (transfer weights)
arr(ax1, 10.5, 9.0, 13.3, 9.0, color=c_gray, lw=2.5, label="迁移权重")

# --- Target domain model ---
draw_box(ax1, 15.8, 9.0, 4.0, 2.5, "", c_orange, fontsize=14, alpha=0.15)
ax1.text(15.8, 10.8, "目标域模型", fontsize=15, fontweight="bold",
         color=c_orange, ha="center")

# Layer stack inside target model
layer_colors_tgt = [c_teal, c_teal, c_teal, c_orange, c_red]
for i, (name, col) in enumerate(zip(layer_names_pre, layer_colors_tgt)):
    y = stack_bottom + i * (layer_h + gap)
    rect = FancyBboxPatch(
        (15.8 - stack_w/2, y), stack_w, layer_h,
        boxstyle="round,pad=0.05",
        facecolor=col, edgecolor="white", linewidth=1.0, alpha=0.85)
    ax1.add_patch(rect)
    ax1.text(15.8, y + layer_h/2, name, fontsize=10, fontweight="bold",
             color="white", ha="center", va="center")

# --- Target domain data ---
draw_box(ax1, 15.8, 4.5, 3.5, 2.0, "目标域数据\n(少量标注)", c_red, fontsize=14)
ax1.text(15.8, 3.0, "少量标注", fontsize=13, fontweight="bold",
         color=c_red, ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_red, alpha=0.8))

# Arrow: target data → target model (fine-tune)
arr(ax1, 15.8, 5.5, 15.8, 7.5, color=c_red, lw=2.5, label="微调")

# Arrow: target model → output
arr(ax1, 17.8, 9.0, 20.5, 9.0, color=c_green, lw=2.5)

# --- Output prediction ---
draw_box(ax1, 22.5, 9.0, 3.5, 2.0, "目标任务\n预测", c_green, fontsize=14)

ax1.set_title("(a) 迁移学习流程", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 三种微调策略对比
# ══════════════════════════════════════════════════════════════════
ax2.set_axis_off()
ax2.set_xlim(-1, 17)
ax2.set_ylim(-2.5, 13)

# Colors for strategies
c_frozen = COLORS["blue"]
c_train = COLORS["red"]
c_frozen_light = "#93c5fd"     # light blue (frozen)
c_train_light = "#fca5a5"     # light red (small lr)
c_mid_lr = "#fdba74"          # light orange (medium lr)

layer_names = ["FC", "Conv4", "Conv3", "Conv2", "Conv1"]
n_layers = len(layer_names)

col_x = [2.5, 7.5, 12.5]
col_titles = ["特征提取", "全量微调", "渐进式解冻"]

rect_w = 3.0
rect_h = 1.2
y_bottom = 2.0
y_gap = 0.25

# Strategy color maps (from bottom=Conv1 to top=FC)
# Column order: Conv1, Conv2, Conv3, Conv4, FC
strategies = [
    # Feature extraction: freeze all conv, train FC only
    [c_frozen, c_frozen, c_frozen, c_frozen, c_train],
    # Full fine-tuning: all trainable (small lr)
    [c_train_light, c_train_light, c_train_light, c_train_light, c_train],
    # Progressive unfreezing: bottom frozen, middle small lr, top large lr
    [c_frozen, c_frozen, c_mid_lr, c_train_light, c_train],
]

# Draw the 3 columns
for col_idx, (cx, title, colors) in enumerate(
        zip(col_x, col_titles, strategies)):
    # Column title below
    ax2.text(cx, y_bottom - 1.5, title, fontsize=15, fontweight="bold",
             color=COLORS["gray"], ha="center", va="center")

    for i, (name, color) in enumerate(zip(layer_names[::-1], colors)):
        y = y_bottom + i * (rect_h + y_gap)
        rect = FancyBboxPatch(
            (cx - rect_w/2, y), rect_w, rect_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.9)
        ax2.add_patch(rect)

        # Determine text color based on background brightness
        if color in [c_frozen, c_train]:
            tc = "white"
        else:
            tc = "#333333"

        ax2.text(cx, y + rect_h/2, name, fontsize=13, fontweight="bold",
                 color=tc, ha="center", va="center")

        # Layer labels on the leftmost column only
        if col_idx == 0:
            ax2.text(cx - rect_w/2 - 0.4, y + rect_h/2, name,
                     fontsize=13, fontweight="bold", color=COLORS["gray"],
                     ha="right", va="center")

    # Add annotation arrows/labels for specific strategies
    if col_idx == 0:
        # Bracket for frozen layers
        y_top_frozen = y_bottom + 3 * (rect_h + y_gap) + rect_h
        ax2.annotate("冻结", xy=(cx + rect_w/2 + 0.3, y_bottom + 2 * (rect_h + y_gap)),
                     fontsize=12, fontweight="bold", color=c_frozen,
                     ha="left", va="center")
        ax2.annotate("训练", xy=(cx + rect_w/2 + 0.3,
                                 y_bottom + 4 * (rect_h + y_gap) + rect_h/2),
                     fontsize=12, fontweight="bold", color=c_train,
                     ha="left", va="center")
    elif col_idx == 1:
        ax2.annotate("小学习率", xy=(cx + rect_w/2 + 0.2,
                                     y_bottom + 2 * (rect_h + y_gap)),
                     fontsize=11, fontweight="bold", color=c_train_light,
                     ha="left", va="center")
    elif col_idx == 2:
        # Annotations for progressive unfreezing
        ax2.annotate("冻结",
                     xy=(cx + rect_w/2 + 0.2, y_bottom + 0.5 * (rect_h + y_gap)),
                     fontsize=11, fontweight="bold", color=c_frozen,
                     ha="left", va="center")
        ax2.annotate("小 lr",
                     xy=(cx + rect_w/2 + 0.2, y_bottom + 2.5 * (rect_h + y_gap)),
                     fontsize=11, fontweight="bold", color="#b45309",
                     ha="left", va="center")
        ax2.annotate("大 lr",
                     xy=(cx + rect_w/2 + 0.2, y_bottom + 4 * (rect_h + y_gap) + rect_h/2),
                     fontsize=11, fontweight="bold", color=c_train,
                     ha="left", va="center")

# Legend
legend_items = [
    mpatches.Patch(facecolor=c_frozen, edgecolor="black", label="冻结 (不更新)"),
    mpatches.Patch(facecolor=c_train, edgecolor="black", label="训练 (大学习率)"),
    mpatches.Patch(facecolor=c_train_light, edgecolor="black", label="训练 (小学习率)"),
    mpatches.Patch(facecolor=c_mid_lr, edgecolor="black", label="渐进解冻 (中学习率)"),
]
ax2.legend(handles=legend_items, fontsize=12, loc="upper center",
           ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))

ax2.set_title("(b) 微调策略对比", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_7_01_transfer_pipeline")
