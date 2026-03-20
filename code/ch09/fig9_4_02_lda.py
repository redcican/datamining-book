"""
fig9_4_02_lda.py
LDA 主题模型：生成过程示意 + 主题-词分布柱状图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                         gridspec_kw={"width_ratios": [1, 1.2]})
fig.suptitle("图 9.4.2　LDA 主题模型",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：LDA 生成过程 ──────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.axis("off")
ax.set_aspect("equal")

def draw_node(ax, x, y, r, text, color, filled=True, fontsize=12):
    circle = Circle((x, y), r, facecolor=color if filled else "white",
                    edgecolor=color, linewidth=2, alpha=0.7 if filled else 1)
    ax.add_patch(circle)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white" if filled else color)

def draw_box_label(ax, x, y, text, color, fontsize=11):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, linewidth=1.5))

# Dirichlet priors
draw_box_label(ax, 2, 10, r"$\alpha$", COLORS["red"], fontsize=14)
draw_box_label(ax, 8, 10, r"$\beta$", COLORS["green"], fontsize=14)

# θ_d (document-topic)
draw_node(ax, 2, 7.8, 0.6, r"$\theta_d$", COLORS["red"], filled=False)
ax.annotate("", xy=(2, 8.4), xytext=(2, 9.4),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=2))
ax.text(0.2, 7.8, "文档-主题\n分布", ha="center", fontsize=10,
        color=COLORS["red"])

# φ_k (topic-word)
draw_node(ax, 8, 7.8, 0.6, r"$\phi_k$", COLORS["green"], filled=False)
ax.annotate("", xy=(8, 8.4), xytext=(8, 9.4),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["green"], lw=2))
ax.text(9.8, 7.8, "主题-词\n分布", ha="center", fontsize=10,
        color=COLORS["green"])

# z_dn (topic assignment)
draw_node(ax, 5, 5.5, 0.6, r"$z_{dn}$", COLORS["orange"], filled=False)
ax.annotate("", xy=(2.6, 7.3), xytext=(4.4, 6.0),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=2))
ax.text(3.0, 5.5, "选主题", ha="center", fontsize=10,
        color=COLORS["orange"])

# w_dn (observed word)
draw_node(ax, 5, 3.0, 0.6, r"$w_{dn}$", COLORS["blue"], filled=True)
ax.annotate("", xy=(5, 4.8), xytext=(5, 3.7),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["orange"], lw=2))
ax.annotate("", xy=(7.4, 7.3), xytext=(5.6, 3.5),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["green"], lw=2))
ax.text(6.8, 3.0, "选词", ha="center", fontsize=10, color=COLORS["blue"])

# Plate notation boxes
# Inner plate (words in document)
rect_inner = mpatches.FancyBboxPatch(
    (3.5, 2.0), 3.5, 5.0, boxstyle="round,pad=0.2",
    facecolor="none", edgecolor=COLORS["gray"], linewidth=1.5, linestyle="--")
ax.add_patch(rect_inner)
ax.text(6.7, 2.3, r"$L_d$", fontsize=13, color=COLORS["gray"],
        fontweight="bold")

# Outer plate (documents)
rect_outer = mpatches.FancyBboxPatch(
    (0.5, 1.0), 7.0, 8.0, boxstyle="round,pad=0.2",
    facecolor="none", edgecolor=COLORS["blue"], linewidth=1.5, linestyle="--")
ax.add_patch(rect_outer)
ax.text(7.0, 1.3, r"$N$", fontsize=13, color=COLORS["blue"],
        fontweight="bold")

# Topic plate
rect_topic = mpatches.FancyBboxPatch(
    (7.0, 6.5), 2.5, 3.0, boxstyle="round,pad=0.2",
    facecolor="none", edgecolor=COLORS["green"], linewidth=1.5, linestyle="--")
ax.add_patch(rect_topic)
ax.text(9.2, 6.8, r"$K$", fontsize=13, color=COLORS["green"],
        fontweight="bold")

# Legend
ax.text(5, 0.3, "阴影节点 = 观测变量  空心节点 = 潜在变量",
        ha="center", fontsize=10, color=COLORS["gray"])

ax.set_title("(a) LDA 图模型 (plate notation)", fontsize=17, pad=5)

# ── 右面板：主题-词分布柱状图 ─────────────────────────────────────
ax = axes[1]
topic_names = ["主题1: 体育", "主题2: 科技", "主题3: 财经"]
topic_words = {
    "主题1: 体育": ["比赛", "冠军", "进球", "球员", "联赛", "教练", "赛季", "胜利"],
    "主题2: 科技": ["技术", "发布", "芯片", "智能", "数据", "算法", "模型", "研发"],
    "主题3: 财经": ["市场", "投资", "增长", "经济", "基金", "股市", "利率", "银行"],
}
topic_probs = {
    "主题1: 体育": [0.18, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07],
    "主题2: 科技": [0.16, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07],
    "主题3: 财经": [0.17, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07],
}
topic_colors = [COLORS["blue"], COLORS["red"], COLORS["green"]]
n_topics = len(topic_names)
n_words = 8
bar_height = 0.22
y_positions = np.arange(n_words)

for t_idx, (t_name, t_color) in enumerate(zip(topic_names, topic_colors)):
    words = topic_words[t_name]
    probs = topic_probs[t_name]
    y_offset = t_idx * (n_words + 2.5)
    bars = ax.barh(y_positions + y_offset, probs, height=bar_height * 2.5,
                   color=t_color, alpha=0.7, edgecolor="white", linewidth=1)
    # 词标签
    for j, (w, p) in enumerate(zip(words, probs)):
        ax.text(-0.005, j + y_offset, w, ha="right", va="center",
                fontsize=11, fontweight="bold", color=t_color)
        ax.text(p + 0.003, j + y_offset, f"{p:.0%}", ha="left", va="center",
                fontsize=10, color=t_color)
    # 主题名
    ax.text(0.10, y_offset + n_words - 0.2, t_name,
            fontsize=13, fontweight="bold", color=t_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=t_color, linewidth=1.5))

ax.set_xlim(-0.02, 0.22)
ax.set_xlabel(r"$P(w | \mathrm{topic})$", fontsize=15)
ax.set_yticks([])
ax.set_title("(b) 各主题的 Top-8 词项概率", fontsize=17)
ax.tick_params(labelsize=13)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_4_02_lda")
