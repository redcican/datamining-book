"""
图 1.2.2  数据挖掘典型应用领域全景（3×3 宫格）
对应节次：1.2 数据挖掘的基本任务与应用领域
运行方式：python code/ch01/fig1_2_02_app_domains.py
输出路径：public/figures/ch01/fig1_2_02_app_domains.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

DOMAINS = [
    {
        "name": "金融科技",
        "en":   "FinTech",
        "abbr": "¥ $",
        "tasks": "欺诈检测（异常）\n信用评分（分类）\n量化交易（回归）",
        "color": "#1d4ed8",
        "bg":    "#eff6ff",
    },
    {
        "name": "医疗健康",
        "en":   "Healthcare",
        "abbr": "医",
        "tasks": "疾病预测（分类）\n医学影像（分类）\n药物发现（回归）",
        "color": "#15803d",
        "bg":    "#f0fdf4",
    },
    {
        "name": "零售电商",
        "en":   "E-Commerce",
        "abbr": "零售",
        "tasks": "推荐系统（协同过滤）\n购物篮分析（关联）\n需求预测（回归）",
        "color": "#ea580c",
        "bg":    "#fff7ed",
    },
    {
        "name": "智能制造",
        "en":   "Manufacturing",
        "abbr": "制造",
        "tasks": "预测性维护（分类）\n质量控制（异常）\n产能优化（回归）",
        "color": "#0891b2",
        "bg":    "#ecfeff",
    },
    {
        "name": "交通物流",
        "en":   "Transportation",
        "abbr": "交通",
        "tasks": "路线优化（聚类）\n需求预测（时序）\n事故预警（异常）",
        "color": "#9333ea",
        "bg":    "#faf5ff",
    },
    {
        "name": "教育科技",
        "en":   "EdTech",
        "abbr": "教育",
        "tasks": "学习路径（聚类）\n辍学预警（分类）\n成绩预测（回归）",
        "color": "#0d9488",
        "bg":    "#f0fdfa",
    },
    {
        "name": "社交媒体",
        "en":   "Social Media",
        "abbr": "社交",
        "tasks": "用户画像（聚类）\n舆情分析（分类）\n内容推荐（关联）",
        "color": "#db2777",
        "bg":    "#fdf2f8",
    },
    {
        "name": "政务大数据",
        "en":   "Gov. Data",
        "abbr": "政务",
        "tasks": "税务风险（异常）\n城市规划（聚类）\n政策评估（回归）",
        "color": "#92400e",
        "bg":    "#fefce8",
    },
    {
        "name": "科学研究",
        "en":   "Science",
        "abbr": "科研",
        "tasks": "基因组分析（聚类）\n蛋白质折叠（分类）\n粒子物理（异常）",
        "color": "#4f46e5",
        "bg":    "#eef2ff",
    },
]

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
plt.subplots_adjust(hspace=0.28, wspace=0.22)

for idx, (ax, domain) in enumerate(zip(axes.flatten(), DOMAINS)):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background
    bg = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=2.0,
        edgecolor=domain["color"],
        facecolor=domain["bg"],
        transform=ax.transAxes,
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(bg)

    # Abbr circle + Name
    circ = mpatches.Circle((0.5, 0.87), 0.12,
                            facecolor=domain["color"],
                            transform=ax.transAxes,
                            zorder=2, clip_on=False)
    ax.add_patch(circ)
    ax.text(0.5, 0.87, domain["abbr"],
            ha="center", va="center",
            fontsize=12, color="white", fontweight="bold",
            transform=ax.transAxes, zorder=3)

    ax.text(0.5, 0.65,
            f"{domain['name']}",
            ha="center", va="center",
            fontsize=15, fontweight="bold",
            color=domain["color"],
            transform=ax.transAxes, zorder=2)

    ax.text(0.5, 0.55,
            f"{domain['en']}",
            ha="center", va="center",
            fontsize=12.5, color=domain["color"] + "bb",
            style="italic",
            transform=ax.transAxes, zorder=2)

    # Divider
    ax.plot([0.15, 0.85], [0.47, 0.47],
            transform=ax.transAxes,
            color=domain["color"] + "44", lw=1.2, zorder=2)

    # Tasks
    ax.text(0.5, 0.24,
            domain["tasks"],
            ha="center", va="center",
            fontsize=12.5, color="#334155",
            multialignment="center",
            transform=ax.transAxes, zorder=2,
            linespacing=1.6)

save_fig(fig, __file__, "fig1_2_02_app_domains")
