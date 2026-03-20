"""
fig8_6_02_deep_learning.py
深度学习预测架构示意：LSTM 门控机制 + Transformer 注意力机制
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
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("图 8.6.2　深度学习预测架构",
             fontsize=20, fontweight="bold", y=1.02)
# ── 左面板：LSTM 门控机制 ─────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(a) LSTM 单元结构", fontsize=15)
# 主体框
main_box = FancyBboxPatch((1.5, 2), 7, 6, boxstyle="round,pad=0.3",
                           facecolor="#f0f4ff", edgecolor=COLORS["blue"],
                           lw=2)
ax.add_patch(main_box)
# 三个门
gate_specs = [
    ("遗忘门\n$\\mathbf{f}_t$", 2.5, 6.5, COLORS["red"]),
    ("输入门\n$\\mathbf{i}_t$", 5.0, 6.5, COLORS["green"]),
    ("输出门\n$\\mathbf{o}_t$", 7.5, 6.5, COLORS["orange"]),
]
for name, x, y, color in gate_specs:
    gate = FancyBboxPatch((x - 0.7, y - 0.5), 1.4, 1.0,
                           boxstyle="round,pad=0.15",
                           facecolor=color, alpha=0.3,
                           edgecolor=color, lw=2)
    ax.add_patch(gate)
    ax.text(x, y, name, ha="center", va="center", fontsize=10,
            fontweight="bold")
# 细胞状态
ax.text(5.0, 4.8, "细胞状态 $\\mathbf{c}_t$", ha="center", va="center",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round", fc="#e8f5e9", ec=COLORS["green"],
                  lw=1.5))
# 隐藏状态
ax.text(5.0, 3.2, "隐藏状态 $\\mathbf{h}_t$", ha="center", va="center",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round", fc="#fff3e0", ec=COLORS["orange"],
                  lw=1.5))
# 输入输出标注
ax.annotate("$x_t$", xy=(1.5, 5.0), fontsize=14, fontweight="bold",
            ha="center")
ax.annotate("$\\hat{x}_{t+1}$", xy=(8.8, 3.2), fontsize=14,
            fontweight="bold", ha="center")
ax.annotate("$\\mathbf{h}_{t-1}$", xy=(1.5, 3.2), fontsize=12,
            ha="center", color="gray")
ax.annotate("$\\mathbf{c}_{t-1}$", xy=(1.5, 4.8), fontsize=12,
            ha="center", color="gray")
# 箭头
ax.annotate("", xy=(2.5, 5.0), xytext=(1.8, 5.0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.annotate("", xy=(8.5, 3.2), xytext=(7.8, 3.2),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
# 信息流标注
ax.text(5.0, 1.5, "门控机制：选择性记忆与遗忘",
        ha="center", fontsize=11, style="italic", color="gray")
# ── 右面板：Transformer 注意力 ────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(b) Transformer 自注意力", fontsize=15)
# 输入序列
n_tokens = 6
token_y = 2.0
for i in range(n_tokens):
    x = 1.5 + i * 1.2
    rect = FancyBboxPatch((x - 0.4, token_y - 0.4), 0.8, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=COLORS["blue"], alpha=0.3,
                           edgecolor=COLORS["blue"], lw=1.5)
    ax.add_patch(rect)
    ax.text(x, token_y, f"$x_{i+1}$", ha="center", va="center",
            fontsize=11, fontweight="bold")
# 注意力矩阵
attn_y = 5.5
attn_x = 5.0
np.random.seed(42)
attn_weights = np.random.dirichlet(np.ones(n_tokens), size=n_tokens)
# 画注意力连线（从某个 token 到所有 token）
focus_token = 4  # 关注第 5 个 token
for j in range(n_tokens):
    x_from = 1.5 + focus_token * 1.2
    x_to = 1.5 + j * 1.2
    weight = attn_weights[focus_token, j]
    ax.annotate("", xy=(x_to, token_y + 0.5),
                xytext=(x_from, attn_y - 0.5),
                arrowprops=dict(arrowstyle="->",
                               color=COLORS["red"],
                               alpha=max(0.1, weight * 2),
                               lw=max(0.5, weight * 5)))
# 输出
for i in range(n_tokens):
    x = 1.5 + i * 1.2
    rect = FancyBboxPatch((x - 0.4, attn_y - 0.4), 0.8, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=COLORS["orange"], alpha=0.3,
                           edgecolor=COLORS["orange"], lw=1.5)
    ax.add_patch(rect)
    ax.text(x, attn_y, f"$h_{i+1}$", ha="center", va="center",
            fontsize=11, fontweight="bold")
# Q K V 标注
ax.text(5.0, 4.0, "Attention($\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$)"
        " = softmax($\\frac{\\mathbf{QK}^T}{\\sqrt{d_k}}$)$\\mathbf{V}$",
        ha="center", fontsize=11,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
# 标注
ax.text(1.0, token_y, "输入", ha="center", fontsize=11, color="gray")
ax.text(1.0, attn_y, "输出", ha="center", fontsize=11, color="gray")
ax.text(5.0, 8.5, "每个位置可关注任意历史位置",
        ha="center", fontsize=11, style="italic", color="gray")
ax.text(5.0, 7.8, "全局感受野 · 并行计算",
        ha="center", fontsize=11, style="italic", color="gray")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_6_02_deep_learning")
