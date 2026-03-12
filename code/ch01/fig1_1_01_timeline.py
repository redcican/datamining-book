"""
图 1.1.3  数据挖掘发展时间线（1960–2025）
对应节次：1.1 数据挖掘的定义与发展历程
运行方式：python code/ch01/fig1_1_01_timeline.py
输出路径：public/figures/ch01/fig1_1_01_timeline.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt

apply_style()

# level 1 = standard height (±2.45), level 2 = outer height (±3.6)
# level 2 used when same-side neighbors are < 5 years apart
MILESTONES = [
    (1970, "关系型数据库\nCodd, 1970",         +1, 1, "#60a5fa"),
    (1984, "决策树 ID3\nQuinlan, 1984",        -1, 1, "#4ade80"),
    (1986, "反向传播\nRumelhart, 1986",         +1, 1, "#4ade80"),
    (1991, "首届 KDD 研讨会\nAAAI, 1991",      -1, 1, "#fb923c"),
    (1993, "Apriori 算法\nAgrawal, 1993",       +1, 1, "#fb923c"),
    (1996, "KDD 框架论文\nFayyad, 1996",        -1, 2, "#fb923c"),  # 5yr from 1991 (同侧)
    (1998, "PageRank\nBrin & Page, 1998",       +1, 2, "#f472b6"),  # 5yr from 1993 (同侧)
    (2003, "MapReduce\nGoogle, 2003",           -1, 1, "#f472b6"),
    (2006, "Hadoop 开源\nASF, 2006",            +1, 1, "#f472b6"),
    (2012, "AlexNet\nKrizhevsky, 2012",         -1, 1, "#a78bfa"),
    (2013, "Word2Vec\nMikolov, 2013",           +1, 2, "#a78bfa"),  # 1yr from 2012 (相邻)
    (2016, "AlphaGo\nDeepMind, 2016",           -1, 2, "#a78bfa"),  # 4yr from 2012 (同侧)
    (2020, "GPT-3\nOpenAI, 2020",               +1, 1, "#fbbf24"),
    (2022, "ChatGPT\nOpenAI, 2022",             -1, 1, "#fbbf24"),
    (2024, "多模态 LLM\n2024–",                 +1, 2, "#fbbf24"),  # 4yr from 2020 (同侧)
]

Y_LEVELS = {1: 2.45, 2: 3.6}

PHASES = [
    (1960, 1983, "#dbeafe", "统计与\n数据库基础"),
    (1983, 1991, "#dcfce7", "机器学习\n崛起"),
    (1991, 2000, "#ffedd5", "KDD\n形成"),
    (2000, 2010, "#fce7f3", "互联网\n挖掘"),
    (2010, 2020, "#ede9fe", "深度学习\n大数据"),
    (2020, 2026, "#fef9c3", "大语言\n模型"),
]

fig, ax = plt.subplots(figsize=(20, 9))
ax.set_xlim(1958, 2027)
ax.set_ylim(-5.2, 5.0)
ax.axis("off")

# 阶段背景
for x0, x1, color, label in PHASES:
    ax.axvspan(x0, x1, alpha=0.28, color=color, zorder=0)
    # phase labels at y=4.15, clearly below the top boundary
    ax.text((x0 + x1) / 2, 4.55, label,
            ha="center", va="top", fontsize=12, color="#374151",
            fontweight="bold", multialignment="center")

# 时间轴
ax.axhline(0, color="#374151", linewidth=2.5, zorder=1)
for yr in range(1960, 2027, 5):
    ax.plot([yr, yr], [-0.2, 0.2], color="#374151", lw=1.3, zorder=2)
    ax.text(yr, -0.48, str(yr), ha="center", va="top", fontsize=12, color="#6b7280")

# 里程碑
for year, label, side, level, color in MILESTONES:
    y_tip = 0.25 * side
    y_box = Y_LEVELS[level] * side
    ax.annotate("", xy=(year, y_tip), xytext=(year, y_box - 0.52 * side),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.6), zorder=3)
    ax.plot(year, y_tip, "o", color=color, markersize=9, zorder=4,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.text(year, y_box, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.38", facecolor=color,
                      alpha=0.88, edgecolor=color, linewidth=1.2),
            multialignment="center", zorder=5)

save_fig(fig, __file__, "fig1_1_01_timeline")
