"""
fig7_6_01_classification_concept.py
两种基于分类的异常检测策略
(a) 隔离策略  (b) 包围策略
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成数据 ──────────────────────────────────────────────────────
n_normal = 120
normal = np.random.multivariate_normal([0, 0], [[0.6, 0.2], [0.2, 0.5]], n_normal)
outliers = np.array([[4.5, 3.5], [-4, 3], [4, -3.5], [-3.5, -4]])
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.6.1　两种基于分类的异常检测策略",
             fontsize=20, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 隔离策略 — 随机分割快速隔离异常点
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.scatter(normal[:, 0], normal[:, 1], c=COLORS["blue"], s=30,
           alpha=0.6, edgecolors="k", linewidths=0.3, zorder=3, label="正常点")
ax.scatter(outliers[:, 0], outliers[:, 1], marker="*", s=300,
           c=COLORS["red"], edgecolors="k", linewidths=0.8, zorder=5,
           label="异常点")
# 隔离异常点 outliers[0] = (4.5, 3.5) 的两次分割线
ax.axvline(x=2.5, color=COLORS["orange"], ls="--", lw=2, alpha=0.8)
ax.axhline(y=2.0, xmin=0.72, color=COLORS["orange"], ls="--", lw=2, alpha=0.8)
# 标注分割次数
ax.annotate("2次分割即隔离",
            xy=(4.5, 3.5), xytext=(1.5, 4.5),
            fontsize=13, fontweight="bold", color=COLORS["red"],
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["red"], alpha=0.9))
# 隔离正常点区域 — 示意需要多次分割
ax.plot([0.5, 0.5], [-1.8, 1.5], color=COLORS["gray"], ls=":", lw=1.5, alpha=0.6)
ax.plot([-1.0, 0.5], [0.3, 0.3], color=COLORS["gray"], ls=":", lw=1.5, alpha=0.6)
ax.plot([-0.3, -0.3], [-1.8, 0.3], color=COLORS["gray"], ls=":", lw=1.5, alpha=0.6)
ax.annotate("正常点需要\n多次分割",
            xy=(0.0, -0.5), xytext=(-4.5, -2.5),
            fontsize=12, color=COLORS["gray"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["gray"], alpha=0.9))
ax.set_title("(a) 隔离策略", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="lower left")
ax.tick_params(labelsize=13)
ax.set_xlim(-5.5, 6)
ax.set_ylim(-5.5, 6)
# ══════════════════════════════════════════════════════════════════
# (b) 包围策略 — One-Class SVM 决策边界
# ══════════════════════════════════════════════════════════════════
ax = axes[1]
ax.scatter(normal[:, 0], normal[:, 1], c=COLORS["blue"], s=30,
           alpha=0.6, edgecolors="k", linewidths=0.3, zorder=3, label="正常点")
ax.scatter(outliers[:, 0], outliers[:, 1], marker="*", s=300,
           c=COLORS["red"], edgecolors="k", linewidths=0.8, zorder=5,
           label="异常点")
# 椭圆决策边界
ellipse = Ellipse(xy=(0, 0), width=4.5, height=3.8, angle=20,
                  fill=False, edgecolor=COLORS["purple"], ls="--", lw=2.5,
                  zorder=4, label="决策边界")
ax.add_patch(ellipse)
# 填充椭圆内部为浅色
ellipse_fill = Ellipse(xy=(0, 0), width=4.5, height=3.8, angle=20,
                        fill=True, facecolor=COLORS["blue"], alpha=0.08,
                        edgecolor="none", zorder=1)
ax.add_patch(ellipse_fill)
# 标注决策边界
ax.annotate("决策边界",
            xy=(1.8, 1.5), xytext=(3.5, 4.5),
            fontsize=14, fontweight="bold", color=COLORS["purple"],
            arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["purple"], alpha=0.9))
# 标注正常区域和异常区域
ax.text(0, -0.3, "正常区域", fontsize=13, ha="center", color=COLORS["blue"],
        fontweight="bold", alpha=0.7)
ax.set_title("(b) 包围策略", fontsize=17)
ax.set_xlabel("$x_1$", fontsize=15)
ax.set_ylabel("$x_2$", fontsize=15)
ax.legend(fontsize=13, loc="lower left")
ax.tick_params(labelsize=13)
ax.set_xlim(-5.5, 6)
ax.set_ylim(-5.5, 6)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig7_6_01_classification_concept")
