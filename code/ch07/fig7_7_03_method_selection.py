"""
fig7_7_03_method_selection.py
异常检测方法选择决策树
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
def add_node(ax, cx, cy, w, h, text, fc, ec, fontsize=13,
             fontweight="bold", text_color="black"):
    """在 (cx, cy) 为中心绘制圆角矩形节点"""
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.2", fc=fc, ec=ec,
                         lw=2, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, zorder=4,
            linespacing=1.4)
    return (cx, cy, w, h)
def draw_edge(ax, parent, child, label="", label_side="center",
              color="#555555"):
    """从父节点底部到子节点顶部画箭头，附带条件标签"""
    px, py, pw, ph = parent
    cx, cy, cw, ch = child
    start = (px, py - ph / 2)
    end = (cx, cy + ch / 2)
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", color=color, lw=2.0,
        mutation_scale=16, zorder=2,
        connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        offset_x = 0
        if label_side == "left":
            offset_x = -0.35
        elif label_side == "right":
            offset_x = 0.35
        ax.text(mid_x + offset_x, mid_y + 0.15, label,
                ha="center", va="center", fontsize=12,
                color=COLORS["red"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.85),
                zorder=5)
# ── 颜色定义 ──────────────────────────────────────────────────────
FC_DECISION = "#dbeafe"   # 决策节点（浅蓝）
EC_DECISION = COLORS["blue"]
FC_LEAF = "#dcfce7"       # 叶节点（浅绿）
EC_LEAF = COLORS["green"]
# ── 画布 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 10))
fig.suptitle("图 7.7.3　异常检测方法选择决策树",
             fontsize=20, fontweight="bold", y=1.02)
ax.set_xlim(-1, 17)
ax.set_ylim(-1, 11)
ax.set_aspect("equal")
ax.axis("off")
# ── 节点坐标 (cx, cy) ────────────────────────────────────────────
nw, nh = 2.4, 1.0          # 标准节点尺寸
lw, lh = 2.6, 1.0          # 叶节点尺寸（稍宽）
# Level 0 — 根
root = add_node(ax, 8, 10, nw, nh, "数据规模?",
                FC_DECISION, EC_DECISION, fontsize=15)
# Level 1
n_big = add_node(ax, 4, 7.5, nw, nh, "维度?",
                 FC_DECISION, EC_DECISION, fontsize=14)
n_small = add_node(ax, 12, 7.5, nw, nh, "有标签?",
                   FC_DECISION, EC_DECISION, fontsize=14)
draw_edge(ax, root, n_big, ">10万", "left")
draw_edge(ax, root, n_small, "<10万", "right")
# Level 2 — 左子树
leaf_iforest1 = add_node(ax, 2, 5, lw, lh, "隔离森林",
                         FC_LEAF, EC_LEAF, fontsize=14)
n_density = add_node(ax, 6, 5, nw, nh, "密度均匀?",
                     FC_DECISION, EC_DECISION, fontsize=14)
draw_edge(ax, n_big, leaf_iforest1, ">50", "left")
draw_edge(ax, n_big, n_density, "<50", "right")
# Level 2 — 右子树
leaf_ocsvm = add_node(ax, 10.5, 5, lw, lh, "One-Class\nSVM",
                      FC_LEAF, EC_LEAF, fontsize=14)
n_cluster_q = add_node(ax, 14, 5, nw + 0.4, nh, "数据有\n簇结构?",
                       FC_DECISION, EC_DECISION, fontsize=13)
draw_edge(ax, n_small, leaf_ocsvm, "是", "left")
draw_edge(ax, n_small, n_cluster_q, "否", "right")
# Level 3 — 左子树叶
leaf_knn = add_node(ax, 4.5, 2.5, lw, lh, "KNN",
                    FC_LEAF, EC_LEAF, fontsize=14)
leaf_lof = add_node(ax, 7.5, 2.5, lw, lh, "LOF",
                    FC_LEAF, EC_LEAF, fontsize=14)
draw_edge(ax, n_density, leaf_knn, "是", "left")
draw_edge(ax, n_density, leaf_lof, "否", "right")
# Level 3 — 右子树叶
leaf_cluster = add_node(ax, 12.5, 2.5, lw, lh, "聚类方法",
                        FC_LEAF, EC_LEAF, fontsize=14)
leaf_fusion = add_node(ax, 15.5, 2.5, lw + 0.6, lh, "隔离森林\n+LOF融合",
                       FC_LEAF, EC_LEAF, fontsize=13)
draw_edge(ax, n_cluster_q, leaf_cluster, "是", "left")
draw_edge(ax, n_cluster_q, leaf_fusion, "否", "right")
# ── 图例说明 ──────────────────────────────────────────────────────
legend_x, legend_y = 0.2, 0.5
add_node(ax, legend_x, legend_y, 1.6, 0.7, "决策节点",
         FC_DECISION, EC_DECISION, fontsize=11, fontweight="normal")
add_node(ax, legend_x + 2.4, legend_y, 1.6, 0.7, "推荐方法",
         FC_LEAF, EC_LEAF, fontsize=11, fontweight="normal")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig7_7_03_method_selection")
