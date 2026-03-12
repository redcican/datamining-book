"""
图 1.3.1  KDD 流程与 CRISP-DM 流程的双列对比图
对应节次：1.3 数据挖掘的流程与方法论
运行方式：python code/ch01/fig1_3_01_kdd_crisp.py
输出路径：public/figures/ch01/fig1_3_01_kdd_crisp.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

apply_style()

fig, ax = plt.subplots(figsize=(17, 13))
ax.set_xlim(0, 17)
ax.set_ylim(0, 13)
ax.axis("off")

# ── Color scheme ──────────────────────────────────────────────────────────
KDD_C  = "#1d4ed8"
CRIS_C = "#15803d"
SEP_C  = "#94a3b8"

# ── Background panels ──────────────────────────────────────────────────────
kdd_bg = mpatches.FancyBboxPatch(
    (0.3, 0.4), 6.4, 12.2,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor="#eff6ff", edgecolor=KDD_C, linewidth=1.5, alpha=0.4, zorder=0)
ax.add_patch(kdd_bg)

cris_bg = mpatches.FancyBboxPatch(
    (10.3, 0.4), 6.4, 12.2,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor="#f0fdf4", edgecolor=CRIS_C, linewidth=1.5, alpha=0.4, zorder=0)
ax.add_patch(cris_bg)

# ── Column headers ─────────────────────────────────────────────────────────
ax.text(3.5, 12.35, "KDD 知识发现流程", ha="center", va="center",
        fontsize=19, fontweight="bold", color=KDD_C)
ax.text(3.5, 11.95, "Fayyad et al. (1996)", ha="center", va="center",
        fontsize=13, color=KDD_C, style="italic")

ax.text(13.5, 12.35, "CRISP-DM 流程", ha="center", va="center",
        fontsize=19, fontweight="bold", color=CRIS_C)
ax.text(13.5, 11.95, "Chapman et al. (2000)", ha="center", va="center",
        fontsize=13, color=CRIS_C, style="italic")

# Vertical separator
ax.plot([8.5, 8.5], [0.6, 12.0], color=SEP_C, lw=1.5, ls="--", zorder=1)
ax.text(8.5, 6.5, "对\n应\n关\n系", ha="center", va="center",
        fontsize=12, color=SEP_C, style="italic")

# ── KDD steps ────────────────────────────────────────────────────────────
KDD_STEPS = [
    ("①  数据选择",      "Selection",
     "从原始数据库中选取与\n挖掘目标相关的目标数据集\n$\\mathcal{D}' \\subseteq \\mathcal{D}$",
     "#bfdbfe", KDD_C),
    ("②  数据预处理",    "Preprocessing",
     "去除噪声、处理缺失值、\n修正不一致，提升数据质量\n$\\mathcal{D}'' = P(\\mathcal{D}')$",
     "#bfdbfe", KDD_C),
    ("③  数据变换",      "Transformation",
     "特征选择、降维、归一化，\n将数据转化为适合挖掘的表示\n$\\mathcal{D}''' = T(\\mathcal{D}'')$",
     "#bfdbfe", KDD_C),
    ("④  数据挖掘",      "Data Mining",
     "应用学习算法，从变换后的\n数据中提取候选模式\n$f = M(\\mathcal{D}''')$",
     "#fef3c7", "#92400e"),
    ("⑤  解释与评估",    "Interpretation / Evaluation",
     "对发现的模式进行评估，\n与领域专家交互确认，\n筛选出有效的知识 $\\mathcal{P}$",
     "#d1fae5", "#065f46"),
]

KDD_CX = 3.5
KDD_BOX_X = 0.7
KDD_BOX_W = 5.6
KDD_BOX_H = 1.55
KDD_HALF_H = KDD_BOX_H / 2

kdd_y_centers = [10.8, 9.0, 7.2, 5.4, 3.4]

for i, (title, en, desc, bg, fc) in enumerate(KDD_STEPS):
    cy = kdd_y_centers[i]

    box = mpatches.FancyBboxPatch(
        (KDD_BOX_X, cy - KDD_HALF_H), KDD_BOX_W, KDD_BOX_H,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=bg, edgecolor=fc, linewidth=1.8, zorder=2)
    ax.add_patch(box)

    ax.text(KDD_CX, cy + 0.35, title,
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=fc, zorder=3)
    ax.text(KDD_CX, cy + 0.05, en,
            ha="center", va="center", fontsize=12, color=fc,
            style="italic", zorder=3)
    ax.text(KDD_CX, cy - 0.40, desc,
            ha="center", va="center", fontsize=12.5, color="#334155",
            multialignment="center", zorder=3)

    if i < len(KDD_STEPS) - 1:
        ax.annotate("",
                    xy=(KDD_CX, kdd_y_centers[i+1] + KDD_HALF_H),
                    xytext=(KDD_CX, cy - KDD_HALF_H),
                    arrowprops=dict(arrowstyle="-|>", color=KDD_C, lw=1.5,
                                   mutation_scale=14),
                    zorder=4)

# KDD backward loop arrow
ax.annotate("",
            xy=(0.5, kdd_y_centers[0]),
            xytext=(0.5, kdd_y_centers[-1]),
            arrowprops=dict(
                arrowstyle="-|>", color=KDD_C + "88", lw=1.3,
                connectionstyle="arc3,rad=-0.35", mutation_scale=12),
            zorder=1)
ax.text(0.15, 7.0, "迭代\n反馈", ha="center", va="center",
        fontsize=12, color=KDD_C, style="italic")

# ── CRISP-DM phases ───────────────────────────────────────────────────────
CRISP_PHASES = [
    ("① 业务理解",       "Business Understanding",
     "明确业务目标与成功标准，\n将业务问题转化为数据挖掘问题",
     "#bbf7d0", CRIS_C),
    ("② 数据理解",       "Data Understanding",
     "收集初始数据，探索数据特性、\n质量问题，发现有趣的子集",
     "#bbf7d0", CRIS_C),
    ("③ 数据准备",       "Data Preparation",
     "选择、清洗、构造、整合、\n格式化最终用于建模的数据集",
     "#bbf7d0", CRIS_C),
    ("④ 建模",          "Modeling",
     "选择建模技术、生成测试设计、\n构建并评估模型参数",
     "#fef9c3", "#92400e"),
    ("⑤ 评估",          "Evaluation",
     "对照业务目标充分评估模型，\n审查流程，确定下一步行动",
     "#fde68a", "#92400e"),
    ("⑥ 部署",          "Deployment",
     "将模型集成到生产系统，\n制定维护计划，编写最终报告",
     "#d1fae5", "#065f46"),
]

CRIS_CX = 13.5
CRIS_BOX_X = 10.7
CRIS_BOX_W = 5.6
CRIS_BOX_H = 1.38
CRIS_HALF_H = CRIS_BOX_H / 2

crisp_y_centers = [10.8, 9.2, 7.6, 6.0, 4.4, 2.7]

for i, (title, en, desc, bg, fc) in enumerate(CRISP_PHASES):
    cy = crisp_y_centers[i]

    box = mpatches.FancyBboxPatch(
        (CRIS_BOX_X, cy - CRIS_HALF_H), CRIS_BOX_W, CRIS_BOX_H,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=bg, edgecolor=fc, linewidth=1.8, zorder=2)
    ax.add_patch(box)

    ax.text(CRIS_CX, cy + 0.28, title,
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=fc, zorder=3)
    ax.text(CRIS_CX, cy + 0.02, en,
            ha="center", va="center", fontsize=12, color=fc,
            style="italic", zorder=3)
    ax.text(CRIS_CX, cy - 0.35, desc,
            ha="center", va="center", fontsize=12.5, color="#334155",
            multialignment="center", zorder=3)

    if i < len(CRISP_PHASES) - 1:
        ax.annotate("",
                    xy=(CRIS_CX, crisp_y_centers[i+1] + CRIS_HALF_H),
                    xytext=(CRIS_CX, cy - CRIS_HALF_H),
                    arrowprops=dict(arrowstyle="-|>", color=CRIS_C, lw=1.5,
                                   mutation_scale=14),
                    zorder=4)

# CRISP-DM outer cycle arrow
ax.annotate("",
            xy=(16.5, crisp_y_centers[0]),
            xytext=(16.5, crisp_y_centers[-1]),
            arrowprops=dict(
                arrowstyle="-|>", color=CRIS_C + "aa", lw=1.5,
                connectionstyle="arc3,rad=0.35", mutation_scale=14),
            zorder=1)
ax.text(16.85, 6.5, "下一\n轮次", ha="center", va="center",
        fontsize=12.5, color=CRIS_C, style="italic")

# ── Correspondence connectors ──────────────────────────────────────────────
correspondences = [
    (kdd_y_centers[0], crisp_y_centers[0]),
    (kdd_y_centers[2], crisp_y_centers[2]),
    (kdd_y_centers[3], crisp_y_centers[3]),
    (kdd_y_centers[4], (crisp_y_centers[4] + crisp_y_centers[5]) / 2),
]

for ky, cy in correspondences:
    ax.annotate("",
                xy=(CRIS_BOX_X, cy),
                xytext=(KDD_BOX_X + KDD_BOX_W, ky),
                arrowprops=dict(
                    arrowstyle="<->", color="#94a3b8", lw=1.2,
                    connectionstyle="arc3,rad=0"),
                zorder=1)

# ── Bottom note ────────────────────────────────────────────────────────────
ax.text(8.5, 0.8,
        "两种框架均强调迭代性：数据挖掘不是线性流程，而是在各阶段间反复循环直至满足质量标准",
        ha="center", va="center", fontsize=12, color="#475569",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc",
                  edgecolor="#cbd5e1", linewidth=1.0))

ax.set_title("KDD 流程 vs. CRISP-DM 流程：阶段对应关系", fontsize=18, pad=12)

save_fig(fig, __file__, "fig1_3_01_kdd_crisp")
