"""
图 3.6.1  集成学习框架示意图：Bagging（并行）与 Boosting（顺序）对比
对应节次：3.6 集成学习算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_6_01_ensemble_framework.py
输出路径：public/figures/ch03/fig3_6_01_ensemble_framework.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply_style()

C_DATA   = COLORS["blue"]
C_BOOT   = COLORS["teal"]
C_LEARN  = COLORS["purple"]
C_OUT    = COLORS["red"]
C_ORANGE = COLORS["orange"]
C_GRAY   = COLORS["gray"]
C_CONN   = "#94a3b8"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
for ax in (ax1, ax2):
    ax.set_axis_off()

# ── 公共绘图工具 ──────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, fc, ec="white", lw=1.5, zorder=3, alpha=0.92, radius=0.04):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(p)

def txt(ax, x, y, s, fs=12, color="white", bold=False, zorder=5, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            color=color, fontweight="bold" if bold else "normal", zorder=zorder)

def arr(ax, x1, y1, x2, y2, color=C_CONN, lw=1.5, zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"), zorder=zorder)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): Bagging 并行框架
# ═══════════════════════════════════════════════════════════════════════════════
ax1.set_xlim(0, 10)
ax1.set_ylim(-0.5, 9)

# --- 原始训练集 ---
rbox(ax1, 5, 8.2, 4.2, 0.8, fc=C_DATA)
txt(ax1, 5, 8.2, "原始训练集  $\\mathcal{D}$（$n$ 个样本）", fs=13, bold=True)

# --- Bootstrap 采样 → 子集 ---
boot_xs = [1.5, 3.5, 5.5, 7.5, 9.0]
boot_labels = ["$\\mathcal{D}_1^*$", "$\\mathcal{D}_2^*$", "$\\mathcal{D}_3^*$",
               "$\\mathcal{D}_T^*$", ""]
# 只画4个（最后留省略号）
for i, (bx, bl) in enumerate(zip(boot_xs[:4], boot_labels[:4])):
    arr(ax1, 5, 7.8, bx, 6.8, color=C_BOOT, lw=1.2)
    rbox(ax1, bx, 6.4, 1.4, 0.7, fc=C_BOOT)
    txt(ax1, bx, 6.4, f"Bootstrap\n{bl}", fs=10)

# 省略号
ax1.text(8.25, 6.4, "···", ha="center", va="center", fontsize=20, color=C_CONN)

# Bootstrap 标注
ax1.text(5, 7.28, "Bootstrap 有放回采样（每次抽 $n$ 个样本）",
         ha="center", fontsize=11, color=C_BOOT, style="italic")

# --- 基学习器 ---
learner_xs = [1.5, 3.5, 5.5, 7.5]
for i, lx in enumerate(learner_xs):
    arr(ax1, lx, 6.05, lx, 5.05, color=C_LEARN, lw=1.3)
    rbox(ax1, lx, 4.65, 1.4, 0.7, fc=C_LEARN)
    txt(ax1, lx, 4.65, f"基学习器\n$h_{i+1}(\\mathbf{{x}})$", fs=10)

ax1.text(8.25, 4.65, "···", ha="center", va="center", fontsize=20, color=C_CONN)

# "并行训练" 标注
rbox(ax1, 5, 5.65, 5.5, 0.35, fc="#f0fdf4", ec=C_LEARN, lw=1, alpha=0.9, radius=0.02)
ax1.text(5, 5.65, "并行独立训练（各基学习器互不依赖）",
         ha="center", va="center", fontsize=11, color=C_LEARN)

# --- 聚合 ---
for lx in learner_xs:
    arr(ax1, lx, 4.28, 5, 3.25, color=C_OUT, lw=1.2)

rbox(ax1, 5, 2.95, 4.2, 0.55, fc=C_ORANGE, ec="white")
ax1.text(5, 2.95,
         "聚合：多数投票（分类）/ 均值平均（回归）",
         ha="center", va="center", fontsize=12, color="white", fontweight="bold")

# 公式
ax1.text(5, 2.35,
         "$H(\\mathbf{x}) = \\mathrm{argmax}_k \\sum_{t=1}^T \\mathbf{1}[h_t(\\mathbf{x})=k]$",
         ha="center", va="center", fontsize=12, color=C_ORANGE)

# --- 最终输出 ---
arr(ax1, 5, 2.15, 5, 1.35, color=C_OUT, lw=1.5)
rbox(ax1, 5, 1.05, 2.4, 0.55, fc=C_OUT)
txt(ax1, 5, 1.05, "集成预测 $H(\\mathbf{x})$", fs=13, bold=True)

ax1.set_title("(a) Bagging（Bootstrap Aggregating）：并行集成\n"
              "各基学习器独立训练，降低方差，不降低偏差",
              fontsize=13, pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): Boosting 顺序框架
# ═══════════════════════════════════════════════════════════════════════════════
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.5, 9)

# --- 原始训练集（均匀权重） ---
rbox(ax2, 5, 8.2, 4.6, 0.8, fc=C_DATA)
txt(ax2, 5, 8.2, "训练集 $\\mathcal{D}$（初始等权重 $w_i=1/n$）", fs=12, bold=True)

# --- 轮次 ---
round_ys    = [6.8, 5.0, 3.2]
round_cols  = [C_LEARN, COLORS["teal"], C_ORANGE]
round_alphas = [0.95, 0.92, 0.88]

for r, (ry, rc) in enumerate(zip(round_ys, round_cols)):
    # 基学习器框
    rbox(ax2, 2.5, ry, 2.4, 0.72, fc=rc, alpha=0.93)
    txt(ax2, 2.5, ry + 0.05, f"基学习器 $h_{r+1}$", fs=12, bold=True)
    txt(ax2, 2.5, ry - 0.22, f"权重 $\\alpha_{r+1}$", fs=11)

    # 错误标注框（宽 3.4，高 0.90，中心右移至 7.2）
    BOX_CX = 7.2
    BOX_W  = 3.4
    BOX_H  = 0.90
    rbox(ax2, BOX_CX, ry, BOX_W, BOX_H, fc="#fef3c7", ec=C_ORANGE, lw=1.2, alpha=0.9)
    ax2.text(BOX_CX, ry + 0.18, "更新样本权重",
             ha="center", va="center", fontsize=12, color=C_ORANGE, fontweight="bold")
    ax2.text(BOX_CX, ry - 0.18, "错误样本权重↑  正确样本权重↓",
             ha="center", va="center", fontsize=12, color="#92400e")

    # 基学习器 → 权重更新（箭头终点 = 框左边缘）
    arr(ax2, 3.7, ry, BOX_CX - BOX_W / 2, ry, color=C_ORANGE, lw=1.3)

    if r < 2:
        # 权重更新框底部 → 下一轮框顶部（垂直），再左转 → 基学习器（水平）
        next_top = round_ys[r + 1] + BOX_H / 2
        arr(ax2, BOX_CX, ry - BOX_H / 2, BOX_CX, next_top, color=C_CONN, lw=1.3)
        ax2.annotate("", xy=(2.5, next_top),
                     xytext=(BOX_CX, next_top),
                     arrowprops=dict(arrowstyle="-|>", color=C_CONN, lw=1.3), zorder=2)

# 原始数据 → 第1轮
arr(ax2, 5, 7.8, 5, 7.16, color=C_DATA, lw=1.3)
ax2.annotate("", xy=(3.7, round_ys[0]),
             xytext=(5, round_ys[0]),
             arrowprops=dict(arrowstyle="-|>", color=C_DATA, lw=1.3), zorder=2)

# 省略号（第3轮后还有轮次）
ax2.text(5, 2.35, "··· (共 $T$ 轮) ···", ha="center", va="center",
         fontsize=14, color=C_CONN)

# --- 加权求和 ---
arr(ax2, 5, 2.15, 5, 1.65, color=C_OUT, lw=1.5)
rbox(ax2, 5, 1.3, 5.5, 0.6, fc=C_OUT, ec="white")
ax2.text(5, 1.42, "加权组合：$H(\\mathbf{x}) = \\mathrm{sign}\\!\\left(\\sum_{t=1}^T \\alpha_t h_t(\\mathbf{x})\\right)$",
         ha="center", va="center", fontsize=12, color="white", fontweight="bold")
ax2.text(5, 1.12, "权重 $\\alpha_t$ 越大 → 该基学习器贡献越高",
         ha="center", va="center", fontsize=11, color="white")

# "顺序训练" 标注
rbox(ax2, 5, 4.1, 5.5, 0.35, fc="#fff7ed", ec=C_ORANGE, lw=1, alpha=0.9, radius=0.02)
ax2.text(5, 4.1, "顺序依赖训练（每轮聚焦前轮错误样本）",
         ha="center", va="center", fontsize=11, color=C_ORANGE)

ax2.set_title("(b) Boosting（AdaBoost 示意）：顺序集成\n"
              "每轮聚焦前轮错误，加权投票，降低偏差，有过拟合风险",
              fontsize=13, pad=8)

fig.suptitle("集成学习框架：Bagging（并行降方差）vs Boosting（顺序降偏差）",
             fontsize=14, y=1.02, fontweight="bold")

save_fig(fig, __file__, "fig3_6_01_ensemble_framework")
