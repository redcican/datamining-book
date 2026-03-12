"""
图 1.2.1  数据挖掘任务分类树（描述性 vs 预测性）
对应节次：1.2 数据挖掘的基本任务与应用领域
运行方式：python code/ch01/fig1_2_01_task_taxonomy.py
输出路径：public/figures/ch01/fig1_2_01_task_taxonomy.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

fig, ax = plt.subplots(figsize=(18, 8.5))
ax.set_xlim(0, 18)
ax.set_ylim(0, 8.5)
ax.axis("off")

# ── Color palette ─────────────────────────────────────────────────────────
C_ROOT   = "#0f172a"
C_PRED   = "#1d4ed8"
C_DESC   = "#15803d"
C_CLS    = "#2563eb"
C_REG    = "#0891b2"
C_CLU    = "#16a34a"
C_ASS    = "#65a30d"
C_ANO    = "#ea580c"
C_LIGHT  = "#f8fafc"
ARROW    = "#94a3b8"


def rounded_box(ax, cx, cy, w, h, color, text, fontsize=15,
                text_color="white", radius=0.3, bold=True):
    fancy = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=1.8,
        edgecolor=color,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(fancy)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal", zorder=4,
            multialignment="center")


def detail_box(ax, cx, cy, w, h, color, text, fontsize=12):
    fancy = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.22",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color + "22",
        zorder=3,
    )
    ax.add_patch(fancy)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=color,
            fontweight="bold", zorder=4,
            multialignment="center")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>",
                                color=ARROW,
                                lw=1.8,
                                mutation_scale=15),
                zorder=2)


# ── Root node ─────────────────────────────────────────────────────────────
rounded_box(ax, 7.8, 7.5, 4.2, 1.05, C_ROOT,
            "数据挖掘任务\nData Mining Tasks", fontsize=16)

# ── Level 1: predictive / descriptive ─────────────────────────────────────
rounded_box(ax, 3.5, 5.8, 3.6, 0.98, C_PRED,
            "预测性任务\nPredictive", fontsize=15)
rounded_box(ax, 12.2, 5.8, 3.6, 0.98, C_DESC,
            "描述性任务\nDescriptive", fontsize=15)

arrow(ax, 7.8, 6.975, 3.5, 6.29)
arrow(ax, 7.8, 6.975, 12.2, 6.29)

# Divider line (visual guide)
ax.plot([7.3, 7.3], [4.0, 6.8], color="#e2e8f0", lw=1.0,
        ls="--", zorder=1)

# ── Level 2: classification & regression ──────────────────────────────────
rounded_box(ax, 1.8, 4.1, 2.8, 0.95, C_CLS,
            "分类\nClassification", fontsize=14)
rounded_box(ax, 5.2, 4.1, 2.8, 0.95, C_REG,
            "回归\nRegression", fontsize=14)

arrow(ax, 3.5, 5.31, 1.8, 4.575)
arrow(ax, 3.5, 5.31, 5.2, 4.575)

# ── Level 2: clustering / association / anomaly ──────────────────────────
rounded_box(ax, 9.0, 4.1, 2.6, 0.95, C_CLU,
            "聚类\nClustering", fontsize=14)
rounded_box(ax, 12.15, 4.1, 2.6, 0.95, C_ASS,
            "关联规则\nAssociation", fontsize=14)
rounded_box(ax, 15.8, 4.1, 2.4, 0.95, C_ANO,
            "异常检测\nAnomaly Det.", fontsize=13.5)

arrow(ax, 12.2, 5.31, 9.0, 4.575)
arrow(ax, 12.2, 5.31, 12.15, 4.575)
arrow(ax, 12.2, 5.31, 15.8, 4.575)

# ── Level 3: sub-types ────────────────────────────────────────────────────
# Classification subtypes
detail_box(ax, 0.9, 2.65, 1.6, 0.82, C_CLS, "二分类\nBinary", 12)
detail_box(ax, 2.7, 2.65, 1.6, 0.82, C_CLS, "多分类\nMulti-class", 12)
arrow(ax, 1.8, 3.625, 0.9, 3.06)
arrow(ax, 1.8, 3.625, 2.7, 3.06)

# Regression subtypes
detail_box(ax, 4.3, 2.65, 1.6, 0.82, C_REG, "线性回归\nLinear", 12)
detail_box(ax, 6.1, 2.65, 1.6, 0.82, C_REG, "非线性\nNon-linear", 12)
arrow(ax, 5.2, 3.625, 4.3, 3.06)
arrow(ax, 5.2, 3.625, 6.1, 3.06)

# Clustering subtypes
detail_box(ax, 8.2, 2.65, 1.35, 0.82, C_CLU, "划分式\nPartitional", 11.5)
detail_box(ax, 9.7, 2.65, 1.35, 0.82, C_CLU, "层次式\nHierarchical", 11.5)
arrow(ax, 9.0, 3.625, 8.2, 3.06)
arrow(ax, 9.0, 3.625, 9.7, 3.06)

# Association subtypes
detail_box(ax, 11.4, 2.65, 1.35, 0.82, C_ASS, "频繁项集\nFreq. Items", 11.5)
detail_box(ax, 12.9, 2.65, 1.35, 0.82, C_ASS, "关联规则\nRules", 11.5)
arrow(ax, 12.15, 3.625, 11.4, 3.06)
arrow(ax, 12.15, 3.625, 12.9, 3.06)

# Anomaly subtypes
detail_box(ax, 14.9, 2.65, 1.35, 0.82, C_ANO, "有监督\nSupervised", 11.5)
detail_box(ax, 16.8, 2.65, 1.35, 0.82, C_ANO, "无监督\nUnsup.", 11.5)
arrow(ax, 15.8, 3.625, 14.9, 3.06)
arrow(ax, 15.8, 3.625, 16.8, 3.06)

# ── Level 4: example algorithms (annotation boxes) ───────────────────────
def algo_note(ax, cx, cy, text, color):
    ax.text(cx, cy, text,
            ha="center", va="top",
            fontsize=12, color=color,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color + "88", linewidth=0.8),
            zorder=5)

algo_note(ax, 0.9,  1.85, "决策树\nSVM\n朴素贝叶斯", C_CLS)
algo_note(ax, 2.7,  1.85, "RandomForest\nXGBoost\nKNN", C_CLS)
algo_note(ax, 4.3,  1.85, "线性回归\n岭回归", C_REG)
algo_note(ax, 6.1,  1.85, "SVR\n决策树\n神经网络", C_REG)
algo_note(ax, 8.2,  1.85, "K-means\nGMM", C_CLU)
algo_note(ax, 9.7,  1.85, "Ward\nDBSCAN", C_CLU)
algo_note(ax, 11.4, 1.85, "Apriori\nFP-Growth", C_ASS)
algo_note(ax, 12.9, 1.85, "置信度/支持度\n提升度", C_ASS)
algo_note(ax, 14.9, 1.85, "One-class SVM\nIsolation Forest", C_ANO)
algo_note(ax, 16.8, 1.85, "LOF\nAutoEncoder", C_ANO)

# ── Background label for two branches ────────────────────────────────────
ax.text(3.5, 6.63, "需要标注标签 $y$", ha="center", va="center",
        fontsize=12, color=C_PRED, style="italic")
ax.text(12.2, 6.63, "仅需特征矩阵 $\\mathbf{X}$", ha="center", va="center",
        fontsize=12, color=C_DESC, style="italic")

save_fig(fig, __file__, "fig1_2_01_task_taxonomy")
