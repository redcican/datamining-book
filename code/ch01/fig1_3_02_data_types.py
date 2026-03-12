"""
图 1.3.2  数据类型层次图（结构化 / 半结构化 / 非结构化）
对应节次：1.3 数据挖掘的流程与方法论
运行方式：python code/ch01/fig1_3_02_data_types.py
输出路径：public/figures/ch01/fig1_3_02_data_types.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")

# ── Color palette ─────────────────────────────────────────────────────────
C_STRUCT = "#1d4ed8"
C_SEMI   = "#0891b2"
C_UNSTRU = "#9333ea"
C_ROOT   = "#0f172a"
ARROW_C  = "#94a3b8"


def node_box(ax, cx, cy, w, h, color, title, subtitle="", body_lines=None,
             fontsize_title=15, fontsize_sub=11, fontsize_body=10.5,
             show_divider=True):
    """Draw a rounded box with title, subtitle, and body lines."""
    bg = color + "18"
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.22",
        facecolor=bg, edgecolor=color, linewidth=2.0, zorder=2)
    ax.add_patch(box)
    title_y = cy + h/2 - 0.35
    ax.text(cx, title_y, title,
            ha="center", va="center",
            fontsize=fontsize_title, fontweight="bold", color=color, zorder=3,
            multialignment="center")
    if subtitle:
        ax.text(cx, title_y - 0.38, subtitle,
                ha="center", va="center",
                fontsize=fontsize_sub, color=color, style="italic", zorder=3)
    if show_divider:
        ax.plot([cx - w/2 + 0.15, cx + w/2 - 0.15],
                [title_y - 0.65, title_y - 0.65],
                color=color + "55", lw=1.0, zorder=3)
    if body_lines:
        start_y = title_y - 0.90 if show_divider else title_y - 0.68
        for line in body_lines:
            ax.text(cx, start_y, line,
                    ha="center", va="center",
                    fontsize=fontsize_body, color="#334155", zorder=3,
                    multialignment="center")
            start_y -= 0.38


def arrow_diag(ax, x0, y0, x1, y1):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.6,
                                mutation_scale=13,
                                connectionstyle="arc3,rad=0"),
                zorder=1)


# ── Root node ─────────────────────────────────────────────────────────────
node_box(ax, 9.0, 10.2, 5.0, 1.1, C_ROOT,
         "数 据（Data）", "——挖掘的起点",
         fontsize_title=18, fontsize_sub=12, fontsize_body=11,
         show_divider=False)

# ── Level 1: three main branches ─────────────────────────────────────────
BRANCH_Y = 8.0
branch_cx = [3.0, 9.0, 15.0]
branch_colors = [C_STRUCT, C_SEMI, C_UNSTRU]
branch_titles = ["结构化数据", "半结构化数据", "非结构化数据"]
branch_en = ["Structured", "Semi-structured", "Unstructured"]
branch_pct = ["约 20% 企业数据", "约 15% 企业数据", "约 65% 企业数据"]

for cx, color, title, en, pct in zip(branch_cx, branch_colors,
                                      branch_titles, branch_en, branch_pct):
    node_box(ax, cx, BRANCH_Y, 5.0, 1.25, color,
             title, en, [pct],
             fontsize_title=15, fontsize_sub=11, fontsize_body=11,
             show_divider=False)
    arrow_diag(ax, 9.0, 9.64, cx, BRANCH_Y + 0.625)

# ── Level 2: subtypes ────────────────────────────────────────────────────
SUB_Y = 5.3
SUB_W = 2.8
SUB_H = 2.5

structured_subs = [
    (1.5, "关系型数据库\nRelational DB",
     "行-列表格形式\n行=样本，列=特征\n$\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$",
     C_STRUCT),
    (4.5, "时间序列\nTime Series",
     "按时间戳索引的序列\n$\\{x_t\\}_{t=1}^T$\n如股价、传感器读数",
     C_STRUCT),
]

semi_subs = [
    (7.5, "JSON / XML\nLog Files",
     "有嵌套结构的键值对\n无固定 Schema\n如 API 响应数据",
     C_SEMI),
    (10.5, "图 / 网络数据\nGraph Data",
     "节点集 $V$，边集 $E$\n如社交网络、知识图谱\n$G = (V, E)$",
     C_SEMI),
]

unstru_subs = [
    (13.5, "文本\nText",
     "自然语言序列\n需分词、向量化\n如新闻、评论、报告",
     C_UNSTRU),
    (16.5, "图像 / 音频\nImage / Audio",
     "像素矩阵或波形序列\n需 CNN/RNN 特征提取\n如医学影像、语音识别",
     C_UNSTRU),
]

all_subs = structured_subs + semi_subs + unstru_subs

for cx, title, body_str, color in all_subs:
    body_lines = body_str.split("\n")
    node_box(ax, cx, SUB_Y, SUB_W, SUB_H, color,
             title, body_lines=body_lines,
             fontsize_title=13, fontsize_body=10.5)
    if color == C_STRUCT:
        parent_cx = 3.0
    elif color == C_SEMI:
        parent_cx = 9.0
    else:
        parent_cx = 15.0
    arrow_diag(ax, parent_cx, BRANCH_Y - 0.625, cx, SUB_Y + SUB_H / 2)

# ── Level 3: mining difficulty annotation ─────────────────────────────────
DIFF_Y = 1.3
diff_labels = [
    (3.0,  C_STRUCT,  "易处理\n直接建模", "★★☆☆☆"),
    (9.0,  C_SEMI,    "中等难度\n需解析", "★★★☆☆"),
    (15.0, C_UNSTRU,  "较难处理\n需表示学习", "★★★★☆"),
]
for cx, color, label, stars in diff_labels:
    diff_box = mpatches.FancyBboxPatch(
        (cx - 2.3, DIFF_Y - 0.50), 4.6, 1.0,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=color + "15", edgecolor=color + "88",
        linewidth=1.0, zorder=2)
    ax.add_patch(diff_box)
    ax.text(cx - 0.5, DIFF_Y, label,
            ha="center", va="center",
            fontsize=12.5, color=color, fontweight="bold", zorder=3,
            multialignment="center")
    ax.text(cx + 1.4, DIFF_Y, stars,
            ha="center", va="center",
            fontsize=14, color=color, zorder=3)
    arrow_diag(ax, cx, SUB_Y - SUB_H / 2, cx, DIFF_Y + 0.50)

# ── Volume annotation ──────────────────────────────────────────────────────
ax.text(9.0, 0.35,
        "★ 挖掘难度（相对）；非结构化数据占企业总数据量约 65%，是当前数据挖掘研究的前沿挑战",
        ha="center", va="center", fontsize=12.5, color="#64748b", style="italic")

ax.set_title("数据类型层次图：从结构化到非结构化", fontsize=18, pad=12)

save_fig(fig, __file__, "fig1_3_02_data_types")
