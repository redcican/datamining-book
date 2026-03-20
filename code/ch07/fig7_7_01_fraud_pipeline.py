"""
fig7_7_01_fraud_pipeline.py
多层欺诈检测系统架构流程图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
apply_style()
np.random.seed(42)
# ── 辅助函数 ──────────────────────────────────────────────────────
def add_box(ax, xy, w, h, text, fc, ec="#333333", fontsize=14,
            fontweight="bold", text_color="black", lw=1.5, zorder=3):
    """添加圆角矩形并居中文字"""
    box = FancyBboxPatch(xy, w, h,
                         boxstyle="round,pad=0.15", fc=fc, ec=ec,
                         lw=lw, zorder=zorder)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, zorder=zorder + 1)
    return box
def draw_arrow(ax, xy_start, xy_end, color="#333333", lw=2.0,
               style="-|>", mutation_scale=18):
    arrow = FancyArrowPatch(
        xy_start, xy_end,
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=mutation_scale, zorder=2,
        connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    return arrow
# ── 画布 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
fig.suptitle("图 7.7.1　多层欺诈检测系统架构",
             fontsize=20, fontweight="bold", y=1.02)
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-1.0, 7.5)
ax.set_aspect("equal")
ax.axis("off")
# ── 交易输入 ──────────────────────────────────────────────────────
add_box(ax, (0, 2.5), 1.8, 1.5, "交易输入", fc="#e0e7ff", ec=COLORS["blue"],
        fontsize=15)
draw_arrow(ax, (1.8, 3.25), (2.8, 3.25), color=COLORS["blue"])
# ── Layer 1: 规则引擎 ────────────────────────────────────────────
l1_x, l1_w, l1_h = 2.8, 2.2, 2.5
add_box(ax, (l1_x, 2.0), l1_w, l1_h, "", fc="#fee2e2", ec=COLORS["red"])
ax.text(l1_x + l1_w / 2, 2.0 + l1_h - 0.4, "规则引擎",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COLORS["red"], zorder=4)
ax.text(l1_x + l1_w / 2, 2.0 + l1_h / 2 - 0.2, "已知模式过滤",
        ha="center", va="center", fontsize=13, color="#555555", zorder=4)
draw_arrow(ax, (l1_x + l1_w, 3.25), (5.8, 3.25), color=COLORS["red"])
# ── Layer 2: 异常评分模型 ────────────────────────────────────────
l2_x, l2_w, l2_h = 5.8, 3.6, 5.5
add_box(ax, (l2_x, 0.5), l2_w, l2_h, "", fc="#dbeafe", ec=COLORS["blue"])
ax.text(l2_x + l2_w / 2, 0.5 + l2_h - 0.35, "异常评分模型",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COLORS["blue"], zorder=4)
# 子模型盒子
sub_models = ["Z-score", "KNN", "LOF", "K-means", "iForest"]
sub_y_start = 1.0
sub_gap = 0.8
sub_w, sub_h = 1.4, 0.6
for i, name in enumerate(sub_models):
    sy = sub_y_start + i * sub_gap
    add_box(ax, (l2_x + 0.3, sy), sub_w, sub_h, name,
            fc="white", ec=COLORS["blue"], fontsize=12,
            fontweight="normal", lw=1.0)
    # 子模型 → 排名融合 箭头
    draw_arrow(ax, (l2_x + 0.3 + sub_w, sy + sub_h / 2),
               (l2_x + 0.3 + sub_w + 0.6, 3.25),
               color=COLORS["blue"], lw=1.2, mutation_scale=14)
# 排名融合盒子
add_box(ax, (l2_x + 0.3 + sub_w + 0.6, 2.65), 1.2, 1.2, "排名\n融合",
        fc="#bfdbfe", ec=COLORS["blue"], fontsize=13)
draw_arrow(ax, (l2_x + l2_w, 3.25), (10.2, 3.25), color=COLORS["blue"])
# ── Layer 3: 人工审核 ────────────────────────────────────────────
l3_x, l3_w, l3_h = 10.2, 2.2, 2.5
add_box(ax, (l3_x, 2.0), l3_w, l3_h, "", fc="#ffedd5", ec=COLORS["orange"])
ax.text(l3_x + l3_w / 2, 2.0 + l3_h - 0.4, "人工审核",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COLORS["orange"], zorder=4)
ax.text(l3_x + l3_w / 2, 2.0 + l3_h / 2 - 0.2, "专家判断",
        ha="center", va="center", fontsize=13, color="#555555", zorder=4)
# ── 输出箭头 + 结果盒子 ──────────────────────────────────────────
out_x = l3_x + l3_w
results = [
    ("放行", COLORS["green"], "#dcfce7", 5.2),
    ("审核", COLORS["orange"], "#ffedd5", 3.25),
    ("拦截", COLORS["red"], "#fee2e2", 1.3),
]
for label, ec, fc, ry in results:
    draw_arrow(ax, (out_x, 3.25), (out_x + 1.2, ry), color=ec, lw=2.0)
    add_box(ax, (out_x + 1.2, ry - 0.4), 1.6, 0.8, label,
            fc=fc, ec=ec, fontsize=14, fontweight="bold", text_color=ec)
# ── 层级标签 ──────────────────────────────────────────────────────
for label, cx, cy in [("Layer 1", l1_x + l1_w / 2, 1.3),
                       ("Layer 2", l2_x + l2_w / 2, -0.2),
                       ("Layer 3", l3_x + l3_w / 2, 1.3)]:
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=12, fontstyle="italic", color="#888888")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig7_7_01_fraud_pipeline")
