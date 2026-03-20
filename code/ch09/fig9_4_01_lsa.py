"""
fig9_4_01_lsa.py
LSA 降维可视化：SVD 分解示意 + 文档在语义空间中的散点图
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
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.4.1　潜在语义分析（LSA）降维与聚类",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：SVD 分解示意 ──────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-0.5, 12)
ax.set_ylim(-1, 7)
ax.axis("off")
ax.set_aspect("equal")

def draw_matrix(ax, x, y, w, h, label, color, sub_label=""):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, alpha=0.6, edgecolor=color, lw=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    if sub_label:
        ax.text(x + w / 2, y - 0.35, sub_label, ha="center", fontsize=10,
                color=color, fontweight="bold")

# X = U Σ V^T
draw_matrix(ax, 0, 2.5, 2.5, 3.5, r"$\mathbf{X}$" + "\nTF-IDF\nN×|V|",
            COLORS["blue"], "文档-词项矩阵")
ax.text(3.0, 4.25, "=", fontsize=22, fontweight="bold", color=COLORS["gray"],
        ha="center", va="center")
draw_matrix(ax, 3.5, 3.0, 1.2, 2.5, r"$\mathbf{U}_k$" + "\nN×k",
            COLORS["red"], "文档-语义")
ax.text(5.0, 4.25, "·", fontsize=26, fontweight="bold", color=COLORS["gray"],
        ha="center", va="center")
draw_matrix(ax, 5.3, 3.5, 1.2, 1.5, r"$\mathbf{\Sigma}_k$" + "\nk×k",
            COLORS["green"], "奇异值")
ax.text(6.8, 4.25, "·", fontsize=26, fontweight="bold", color=COLORS["gray"],
        ha="center", va="center")
draw_matrix(ax, 7.1, 3.0, 2.5, 2.5, r"$\mathbf{V}_k^\top$" + "\nk×|V|",
            COLORS["purple"], "词项-语义")

# 截断说明
ax.annotate("截断: 保留前 k 个\n最大奇异值",
            xy=(5.9, 3.3), xytext=(5.9, 1.5),
            fontsize=11, ha="center", color=COLORS["orange"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.5))

# 降维效果
ax.text(1.25, 1.0, f"|V| = 50,000 维", ha="center", fontsize=11,
        color=COLORS["blue"], fontweight="bold")
ax.text(4.1, 1.0, f"k = 100 维", ha="center", fontsize=11,
        color=COLORS["red"], fontweight="bold")
ax.annotate("", xy=(3.2, 1.0), xytext=(2.3, 1.0),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["orange"], lw=2))
ax.text(2.75, 0.5, "降维\n500×", ha="center", fontsize=11,
        color=COLORS["orange"], fontweight="bold")

ax.set_title("(a) SVD 矩阵分解", fontsize=17, pad=5)

# ── 右面板：LSA 降维后的文档散点图 ────────────────────────────────
ax = axes[1]
# 模拟 5 个主题的文档
n_per_topic = 30
topic_names = ["体育", "科技", "财经", "娱乐", "教育"]
topic_colors = [COLORS["blue"], COLORS["red"], COLORS["green"],
                COLORS["orange"], COLORS["purple"]]
centers = [(-3, 2), (3, 3), (3, -2), (-2, -3), (0, 0)]
spreads = [0.7, 0.8, 0.6, 0.7, 0.9]

for i, (name, color, center, spread) in enumerate(
        zip(topic_names, topic_colors, centers, spreads)):
    x = np.random.normal(center[0], spread, n_per_topic)
    y = np.random.normal(center[1], spread, n_per_topic)
    ax.scatter(x, y, c=color, s=40, alpha=0.6, label=name, edgecolors="white",
               linewidths=0.5)
    # 标注中心
    ax.scatter(*center, c=color, s=150, marker="*", edgecolors="white",
               linewidths=1, zorder=5)

ax.set_xlabel("LSA 维度 1", fontsize=15)
ax.set_ylabel("LSA 维度 2", fontsize=15)
ax.set_title("(b) 文档在语义空间中的分布", fontsize=17)
ax.legend(fontsize=12, loc="upper left", ncol=2)
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_4_01_lsa")
