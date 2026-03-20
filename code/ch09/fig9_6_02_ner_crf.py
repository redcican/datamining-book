"""
fig9_6_02_ner_crf.py
命名实体识别与 CRF：BIO 标注示意 + Viterbi 解码网格
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
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.6.2　命名实体识别与 CRF",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) BIO 标注示意
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-1.0, 3.5)
ax.axis("off")
ax.grid(False)

chars = list("张三在北京大学工作")
tags = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O"]

# 颜色映射
tag_colors = {
    "B-PER": COLORS["red"],
    "I-PER": "#fca5a5",      # 浅红
    "B-ORG": COLORS["green"],
    "I-ORG": "#86efac",      # 浅绿
    "O":     COLORS["light"],
}
tag_text_colors = {
    "B-PER": "white",
    "I-PER": COLORS["red"],
    "B-ORG": "white",
    "I-ORG": COLORS["green"],
    "O":     COLORS["gray"],
}

box_w = 0.85
box_h = 0.7

for i, (ch, tag) in enumerate(zip(chars, tags)):
    x = i + 0.1

    # 字符框（上方）
    char_box = FancyBboxPatch((x, 2.0), box_w, box_h,
                               boxstyle="round,pad=0.05",
                               facecolor="white", edgecolor=COLORS["gray"],
                               linewidth=1.5, zorder=3)
    ax.add_patch(char_box)
    ax.text(x + box_w / 2, 2.0 + box_h / 2, ch,
            fontsize=16, fontweight="bold", ha="center", va="center", zorder=4)

    # 标签框（下方）
    fc = tag_colors[tag]
    tc = tag_text_colors[tag]
    tag_box = FancyBboxPatch((x, 0.8), box_w, box_h,
                              boxstyle="round,pad=0.05",
                              facecolor=fc, edgecolor="white",
                              linewidth=1.5, zorder=3, alpha=0.85)
    ax.add_patch(tag_box)
    ax.text(x + box_w / 2, 0.8 + box_h / 2, tag,
            fontsize=11, fontweight="bold", ha="center", va="center",
            color=tc, zorder=4)

    # 连接箭头
    ax.annotate("", xy=(x + box_w / 2, 0.8 + box_h),
                xytext=(x + box_w / 2, 2.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"],
                                lw=1.2, alpha=0.5))

# 实体类型下划线标注
# PER: 张三 (index 0-1)
ax.annotate("", xy=(0.1, -0.15), xytext=(1.95, -0.15),
            arrowprops=dict(arrowstyle="-", color=COLORS["red"], lw=3))
ax.text(1.025, -0.55, "人名 (PER)", fontsize=12, fontweight="bold",
        ha="center", va="center", color=COLORS["red"])

# ORG: 北京大学 (index 3-6)
ax.annotate("", xy=(3.1, -0.15), xytext=(6.95, -0.15),
            arrowprops=dict(arrowstyle="-", color=COLORS["green"], lw=3))
ax.text(5.025, -0.55, "机构 (ORG)", fontsize=12, fontweight="bold",
        ha="center", va="center", color=COLORS["green"])

# 图例
legend_patches = [
    mpatches.Patch(facecolor=COLORS["red"], label="B-PER / I-PER"),
    mpatches.Patch(facecolor=COLORS["green"], label="B-ORG / I-ORG"),
    mpatches.Patch(facecolor=COLORS["light"], edgecolor=COLORS["gray"],
                   label="O (非实体)"),
]
ax.legend(handles=legend_patches, fontsize=12, loc="upper right",
          framealpha=0.9)
ax.set_title("(a) BIO 标注示意", fontsize=17)

# ══════════════════════════════════════════════════════════════════
# (b) Viterbi 解码网格
# ══════════════════════════════════════════════════════════════════
ax = axes[1]
ax.grid(False)

time_labels = list("张三在北京")
state_labels = ["B-PER", "I-PER", "B-ORG", "O"]
n_time = len(time_labels)
n_states = len(state_labels)

# 最优路径: B-PER(0) → I-PER(1) → O(3) → B-ORG(2) → I-ORG → use I-PER(1) slot for I-ORG display
# 重新映射 state 索引:  B-PER=0, I-PER=1, B-ORG=2, O=3
optimal_path = [0, 1, 3, 2, 1]  # y indices for optimal path

# 坐标设置
x_positions = np.arange(n_time)
y_positions = np.arange(n_states)

# 绘制所有可能的转移边（灰色细线）
for t in range(n_time - 1):
    for s1 in range(n_states):
        for s2 in range(n_states):
            ax.annotate("",
                        xy=(x_positions[t + 1] - 0.12, y_positions[s2]),
                        xytext=(x_positions[t] + 0.12, y_positions[s1]),
                        arrowprops=dict(arrowstyle="->", color=COLORS["light"],
                                        lw=0.6, alpha=0.5))

# 绘制最优路径（红色粗线）
for t in range(n_time - 1):
    s1 = optimal_path[t]
    s2 = optimal_path[t + 1]
    ax.annotate("",
                xy=(x_positions[t + 1] - 0.12, y_positions[s2]),
                xytext=(x_positions[t] + 0.12, y_positions[s1]),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["red"],
                                lw=2.5, mutation_scale=15))

# 绘制节点圆圈
for t in range(n_time):
    for s in range(n_states):
        is_optimal = (optimal_path[t] == s)
        color = COLORS["red"] if is_optimal else COLORS["blue"]
        size = 200 if is_optimal else 100
        edge_lw = 2.0 if is_optimal else 1.0
        ax.scatter(x_positions[t], y_positions[s], s=size, c=color,
                   edgecolors="white", linewidths=edge_lw, zorder=5, alpha=0.9)

# 修改 Y 轴标签：在最优路径的第5个时间步处 state_labels[1] 改为 I-ORG 显示
display_state_labels = ["B-PER", "I-PER\n/ I-ORG", "B-ORG", "O"]

ax.set_xticks(x_positions)
ax.set_xticklabels(time_labels, fontsize=14)
ax.set_yticks(y_positions)
ax.set_yticklabels(display_state_labels, fontsize=13)
ax.set_xlabel("输入字符", fontsize=16)
ax.set_ylabel("状态", fontsize=16)
ax.tick_params(labelsize=14)

# 标注最优路径
ax.text(2.0, 3.7, "最优路径（Viterbi）", fontsize=13, fontweight="bold",
        color=COLORS["red"], ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLORS["red"], alpha=0.9))

ax.set_xlim(-0.5, n_time - 0.5)
ax.set_ylim(-0.6, 4.2)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_title("(b) Viterbi 解码网格", fontsize=17)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_6_02_ner_crf")
