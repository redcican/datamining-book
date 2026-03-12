"""
图 1.1.1  数据挖掘与相关学科的关系（Venn 风格）
对应节次：1.1 数据挖掘的定义与发展历程
运行方式：python code/ch01/fig1_1_02_fields.py
输出路径：public/figures/ch01/fig1_1_02_fields.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt

apply_style()

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.0)
ax.set_aspect("equal")
ax.axis("off")

# ── 外层虚线圆：领域知识 ──────────────────────────────────────────────
# center (5.0, 4.8), radius 4.7
ax.add_patch(plt.Circle((5.0, 4.8), 4.7, facecolor="none",
                         edgecolor="#64748b", linewidth=2.0,
                         linestyle="--", zorder=0))
# "领域知识" placed inside the outer circle at the top, above the inner circles
# Inner circles reach up to ~5.8+3.1=8.9; outer top is 4.8+4.7=9.5
# → place at y=9.1, inside the outer boundary
ax.text(5.0, 9.15, "领域知识（Domain Knowledge）",
        ha="center", va="center",
        fontsize=12, color="#475569", style="italic", fontweight="bold")

# ── 三主圆 ────────────────────────────────────────────────────────────
# Statistics: center (3.0, 5.5), r=3.1 → top-left region of stats circle
# Non-overlapping area is upper-left; center of that region ≈ (1.6, 7.4)
ax.add_patch(plt.Circle((3.0, 5.5), 3.1, facecolor="#bfdbfe", edgecolor="#2563eb",
                          linewidth=2.2, alpha=0.52, zorder=1))
ax.text(1.5, 7.5, "统计学\nStatistics",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1d4ed8",
        multialignment="center", zorder=2)

# ML: center (7.0, 5.5), r=3.1 → upper-right non-overlapping area ≈ (8.5, 7.4)
ax.add_patch(plt.Circle((7.0, 5.5), 3.1, facecolor="#bbf7d0", edgecolor="#16a34a",
                          linewidth=2.2, alpha=0.52, zorder=1))
ax.text(8.5, 7.5, "机器学习\nMachine Learning",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#15803d",
        multialignment="center", zorder=2)

# Databases: center (5.0, 2.7), r=3.1 → lower non-overlapping area ≈ (5.0, 0.7)
ax.add_patch(plt.Circle((5.0, 2.7), 3.1, facecolor="#fde68a", edgecolor="#d97706",
                          linewidth=2.2, alpha=0.52, zorder=1))
ax.text(5.0, 0.65, "数据库技术\nDatabases",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#b45309",
        multialignment="center", zorder=2)

# ── 两两交叉区标注 ────────────────────────────────────────────────────
ax.text(3.8, 6.1, "统计学习\nStatistical Learning",
        ha="center", va="center", fontsize=12, color="#1d4ed8",
        style="italic", multialignment="center", zorder=4)
ax.text(6.2, 6.1, "ML 理论\nML Theory",
        ha="center", va="center", fontsize=12, color="#065f46",
        style="italic", multialignment="center", zorder=4)
ax.text(5.0, 3.3, "数据仓库 / OLAP",
        ha="center", va="center", fontsize=12, color="#92400e",
        style="italic", zorder=4)

# ── 数据挖掘核心（三圆交汇处） ───────────────────────────────────────
ax.add_patch(plt.Circle((5.0, 4.65), 1.42, facecolor="#f0abfc",
                          edgecolor="#9333ea", linewidth=2.8, alpha=0.88, zorder=3))
ax.text(5.0, 4.65, "数据挖掘\nData Mining",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="#581c87",
        zorder=4, multialignment="center")

save_fig(fig, __file__, "fig1_1_02_fields")
