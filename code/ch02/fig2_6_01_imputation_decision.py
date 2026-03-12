"""
图 2.6.1  缺失值处理策略选择决策流程
对应节次：2.6 缺失值处理策略
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch02/fig2_6_01_imputation_decision.py
输出路径：public/figures/ch02/fig2_6_01_imputation_decision.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")

# --- 1. 颜色定义 ---
C_START = "#0f172a"
C_DEC   = "#1d4ed8"
C_MCAR  = "#16a34a"
C_MAR   = "#d97706"
C_MNAR  = "#dc2626"
C_DROP  = "#be185d"
C_GRAY  = "#94a3b8"


def rbox(ax, cx, cy, w, h, fc, ec, text, fs=12.5, tc="white", bold=True, a=1.0):
    p = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.20",
        facecolor=fc, edgecolor=ec, linewidth=1.8, alpha=a, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center")


def diam(ax, cx, cy, dx, dy, color, text, fs=12.5):
    xs = [cx, cx + dx, cx, cx - dx, cx]
    ys = [cy + dy, cy, cy - dy, cy, cy + dy]
    ax.fill(xs, ys, color=color, alpha=0.12, zorder=2)
    ax.plot(xs, ys, color=color, lw=2.0, zorder=3)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold", zorder=4, multialignment="center")


def arv(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                lw=1.8, mutation_scale=13), zorder=2)


# --- 2. 节点绘制 ---
# 起始框
rbox(ax, 7.5, 8.50, 4.6, 0.72, C_START, C_START, "缺失值处理策略选择", fs=14.5)

# 决策菱形：缺失率
diam(ax, 7.5, 7.28, 2.15, 0.64, C_DEC, "缺失率 > 40%?", fs=13)

# 是分支：删除/补采
rbox(ax, 2.0, 7.28, 3.2, 0.76, C_DROP, C_DROP,
     "⚠  考虑删除该特征\n或补充数据采集", fs=12)

# 机制诊断框
rbox(ax, 7.5, 6.08, 4.4, 0.76, "#dbeafe", C_DEC,
     "诊断缺失机制（Rubin, 1976）", fs=13, tc=C_DEC)

# 列头：MCAR / MAR / MNAR
for x, txt, col in [(2.5, "MCAR", C_MCAR),
                    (7.5, "MAR",  C_MAR),
                    (12.5, "MNAR", C_MNAR)]:
    ax.text(x, 5.12, txt, ha="center", va="center", fontsize=14.5,
            color=col, fontweight="bold", zorder=4)

# 方法框
rbox(ax, 2.5, 3.82, 4.1, 1.38, "#f0fdf4", C_MCAR,
     "✓ 完整案例分析（CCA）\n✓ 均值 / 中位数填补\n✓ 众数填补（类别变量）",
     fs=12, tc="#14532d", a=0.92)
rbox(ax, 7.5, 3.82, 4.1, 1.38, "#fffbeb", C_MAR,
     "✓ KNN 填补\n✓ 多元回归填补\n✓ 多重填补（MICE）",
     fs=12, tc="#78350f", a=0.92)
rbox(ax, 12.5, 3.82, 4.1, 1.38, "#fef2f2", C_MNAR,
     "！MICE + 敏感性分析\n！选择偏差建模\n！领域知识辅助",
     fs=12, tc="#7f1d1d", a=0.92)

# 偏差风险条
for x, col, txt in [(2.5,  C_MCAR, "偏差风险：低"),
                    (7.5,  C_MAR,  "偏差风险：中等"),
                    (12.5, C_MNAR, "偏差风险：高（不可消除）")]:
    rbox(ax, x, 2.46, 3.82, 0.56, col, col, txt, fs=12)

# 底部注释
ax.text(7.5, 1.60,
        "机制诊断方法：构建缺失指示变量 $r_j\\in\\{0,1\\}$，以其余已观测特征做逻辑斯蒂回归。\n"
        "各系数均不显著 → MCAR；部分系数显著 → MAR；"
        "业务逻辑暗示缺失依赖缺失值本身 → MNAR。",
        ha="center", fontsize=12, color="#64748b", style="italic")

# --- 3. 箭头绘制 ---
arv(ax, 7.5, 8.14, 7.5, 7.92)           # 起始 → 菱形
# 菱形 → 删除（是，向左）
ax.annotate("", xy=(3.6, 7.28), xytext=(5.35, 7.28),
            arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.8, mutation_scale=13))
ax.text(4.48, 7.57, "是", ha="center", fontsize=13, color="#374151", fontweight="bold")
# 菱形 → 机制诊断（否，向下）
arv(ax, 7.5, 6.64, 7.5, 6.46)
ax.text(7.92, 6.54, "否", ha="left", fontsize=13, color="#374151", fontweight="bold")

# 机制诊断 → 水平分流线（y=5.46）
Y_SPLIT = 5.46
ax.plot([7.5, 7.5], [5.70, Y_SPLIT], color=C_GRAY, lw=1.5, zorder=1)
ax.plot([2.5, 12.5], [Y_SPLIT, Y_SPLIT], color=C_GRAY, lw=1.5, zorder=1)

# 水平线 → 列头
for x in [2.5, 7.5, 12.5]:
    arv(ax, x, Y_SPLIT, x, 5.32)
# 列头 → 方法框
for x in [2.5, 7.5, 12.5]:
    arv(ax, x, 4.93, x, 4.51)
# 方法框 → 偏差条
for x in [2.5, 7.5, 12.5]:
    arv(ax, x, 3.13, x, 2.74)

save_fig(fig, __file__, "fig2_6_01_imputation_decision")
