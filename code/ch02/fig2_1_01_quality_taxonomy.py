"""
图 2.1.1  数据质量问题分类体系（不完整 / 噪声 / 不一致 / 重复）
对应节次：2.1 数据清洗技术
运行方式：python code/ch02/fig2_1_01_quality_taxonomy.py
输出路径：public/figures/ch02/fig2_1_01_quality_taxonomy.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

# ── Color palette ──────────────────────────────────────────────────────────
C_ROOT  = "#0f172a"
C_INC   = "#2563eb"   # Incompleteness — blue
C_NOI   = "#ea580c"   # Noise          — orange
C_INC2  = "#16a34a"   # Inconsistency  — green (reuse C_INC2 = green)
C_DUP   = "#7c3aed"   # Duplication    — purple
ARROW_C = "#94a3b8"


def rounded_box(ax, cx, cy, w, h, facecolor, text,
                fontsize=13, text_color="white", bold=True):
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.25",
        facecolor=facecolor, edgecolor=facecolor,
        linewidth=1.6, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center")


def detail_box(ax, cx, cy, w, h, color, text, fontsize=12):
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.2",
        facecolor=color + "22", edgecolor=color,
        linewidth=1.4, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=color,
            fontweight="bold", zorder=4, multialignment="center")


def strategy_box(ax, cx, cy, w, h, color, title, body, fontsize_t=10, fontsize_b=8.5):
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor=color + "12", edgecolor=color + "66",
        linewidth=1.0, zorder=2)
    ax.add_patch(patch)
    ax.text(cx, cy + 0.22, title, ha="center", va="center",
            fontsize=fontsize_t, color=color, fontweight="bold", zorder=3)
    ax.text(cx, cy - 0.20, body, ha="center", va="center",
            fontsize=fontsize_b, color="#475569", zorder=3,
            multialignment="center")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                lw=1.6, mutation_scale=13),
                zorder=1)


# ── Root ───────────────────────────────────────────────────────────────────
rounded_box(ax, 9.0, 9.3, 4.8, 0.95, C_ROOT,
            "数据质量问题  Data Quality Problems", fontsize=15)

# ── Level 1: four main branches ────────────────────────────────────────────
B_Y = 7.35
B_CX = [2.4, 6.8, 11.2, 15.6]
B_COLORS = [C_INC, C_NOI, C_INC2, C_DUP]
B_TITLES = [
    "不完整性\nIncompleteness",
    "噪声\nNoise",
    "不一致性\nInconsistency",
    "重复性\nDuplication",
]
B_SUBTITLES = [
    "约 40% 的预处理时间",
    "影响模型收敛与精度",
    "跨源数据常见问题",
    "浪费计算与存储资源",
]

for cx, color, title, sub in zip(B_CX, B_COLORS, B_TITLES, B_SUBTITLES):
    rounded_box(ax, cx, B_Y, 3.8, 1.1, color, title, fontsize=13)
    ax.text(cx, B_Y - 0.7, sub, ha="center", va="center",
            fontsize=12.5, color=color, style="italic", zorder=4)
    arrow(ax, 9.0, 8.82, cx, B_Y + 0.55)

# ── Level 2: subtypes (2 per branch) ──────────────────────────────────────
S_Y = 5.05
# Incompleteness subtypes
detail_box(ax, 1.35, S_Y, 1.8, 0.85, C_INC, "字段缺失\nField Missing", 10)
detail_box(ax, 3.45, S_Y, 1.8, 0.85, C_INC, "记录缺失\nRecord Missing", 10)
arrow(ax, 2.4, B_Y - 0.55, 1.35, S_Y + 0.425)
arrow(ax, 2.4, B_Y - 0.55, 3.45, S_Y + 0.425)

# Noise subtypes
detail_box(ax, 5.75, S_Y, 1.8, 0.85, C_NOI, "输入错误\nInput Error", 10)
detail_box(ax, 7.85, S_Y, 1.8, 0.85, C_NOI, "传感器噪声\nSensor Noise", 10)
arrow(ax, 6.8, B_Y - 0.55, 5.75, S_Y + 0.425)
arrow(ax, 6.8, B_Y - 0.55, 7.85, S_Y + 0.425)

# Inconsistency subtypes
detail_box(ax, 10.15, S_Y, 1.8, 0.85, C_INC2, "格式冲突\nFormat Clash", 10)
detail_box(ax, 12.25, S_Y, 1.8, 0.85, C_INC2, "语义冲突\nSemantic Clash", 10)
arrow(ax, 11.2, B_Y - 0.55, 10.15, S_Y + 0.425)
arrow(ax, 11.2, B_Y - 0.55, 12.25, S_Y + 0.425)

# Duplication subtypes
detail_box(ax, 14.55, S_Y, 1.8, 0.85, C_DUP, "精确重复\nExact Dup.", 10)
detail_box(ax, 16.65, S_Y, 1.8, 0.85, C_DUP, "近似重复\nNear Dup.", 10)
arrow(ax, 15.6, B_Y - 0.55, 14.55, S_Y + 0.425)
arrow(ax, 15.6, B_Y - 0.55, 16.65, S_Y + 0.425)

# ── Level 3: strategy/example annotation boxes ────────────────────────────
T_Y = 2.6
strategy_data = [
    (2.4,  C_INC,  "检测方法", "missingno 热图\nMCAR / MAR / MNAR 诊断"),
    (6.8,  C_NOI,  "检测方法", "Z-score / IQR\n箱线图 / Grubbs 检验"),
    (11.2, C_INC2, "检测方法", "跨表规则校验\n业务约束检查"),
    (15.6, C_DUP,  "检测方法", "哈希精确匹配\n编辑距离近似匹配"),
]

for cx, color, title, body in strategy_data:
    strategy_box(ax, cx, T_Y, 4.0, 1.0, color, title, body)
    arrow(ax, cx, S_Y - 0.425, cx, T_Y + 0.5)

# ── Example data snippets ─────────────────────────────────────────────────
eg_y = 1.0
ax.text(9.0, eg_y,
        "示例：年龄字段为空（不完整）｜  收入 = -500 元（噪声）｜  "
        "生日格式 '1990/01' vs '01-1990'（不一致）｜  同一客户出现两次（重复）",
        ha="center", va="center", fontsize=12.5, color="#475569",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc",
                  edgecolor="#cbd5e1", linewidth=1.0))

ax.set_title("数据质量问题分类体系", fontsize=17, pad=10)

save_fig(fig, __file__, "fig2_1_01_quality_taxonomy")
