"""
fig9_4_03_topic_analysis.py
主题分析结果：seaborn 热力图 + LSA 降维散点图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 模拟 LDA 主题-词分布 ─────────────────────────────────────────
topic_names = ["体育", "科技", "财经", "娱乐", "教育"]
all_words = ["比赛", "冠军", "技术", "发布", "市场", "投资",
             "电影", "明星", "学校", "考试", "进球", "芯片",
             "增长", "票房", "教学", "球员", "数据", "经济",
             "导演", "招生"]
n_topics = 5
n_words = 20
phi = np.random.dirichlet(np.ones(n_words) * 0.5, n_topics)
for k in range(n_topics):
    idx = list(range(k * 4, min((k + 1) * 4, n_words)))
    phi[k, idx] += 0.15
    phi[k] /= phi[k].sum()
# ── 模拟 LSA 文档降维数据 ────────────────────────────────────────
n_per_topic = 40
topic_colors_list = [COLORS["blue"], COLORS["red"], COLORS["green"],
                     COLORS["orange"], COLORS["purple"]]
centers = [(-3, 2), (3, 3.5), (4, -2), (-3, -3), (0, -0.5)]
all_x, all_y, all_labels, all_colors = [], [], [], []
for i, (name, color, center) in enumerate(
        zip(topic_names, topic_colors_list, centers)):
    x = np.random.normal(center[0], 0.8, n_per_topic)
    y = np.random.normal(center[1], 0.8, n_per_topic)
    all_x.extend(x)
    all_y.extend(y)
    all_labels.extend([name] * n_per_topic)
    all_colors.extend([color] * n_per_topic)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                         gridspec_kw={"width_ratios": [1.1, 1]})
fig.suptitle("图 9.4.3　主题分析结果",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：seaborn 热力图 ─────────────────────────────────────
ax = axes[0]
import pandas as pd
phi_df = pd.DataFrame(phi, index=topic_names, columns=all_words)
sns.heatmap(phi_df, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
            annot_kws={"size": 9, "fontweight": "bold"},
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": r"$P(w|\mathrm{topic})$", "shrink": 0.8},
            mask=phi_df < 0.06)
# 重新绘制全部数值（不被 mask 遮挡的已由 annot 处理，被遮挡的显式添加）
for i in range(n_topics):
    for j in range(n_words):
        val = phi[i, j]
        if val < 0.06:
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="#999999")
# 标注特征词区域
for k in range(n_topics):
    rect = plt.Rectangle((k * 4, k), 4, 1,
                          fill=False, edgecolor=topic_colors_list[k],
                          linewidth=2.5, linestyle="--")
    ax.add_patch(rect)
ax.set_title("(a) LDA 主题-词概率分布", fontsize=17)
ax.tick_params(labelsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)
# ── 右面板：LSA 降维散点图 ────────────────────────────────────────
ax = axes[1]
for name, color in zip(topic_names, topic_colors_list):
    idx = [i for i, l in enumerate(all_labels) if l == name]
    ax.scatter([all_x[i] for i in idx], [all_y[i] for i in idx],
               c=color, s=35, alpha=0.6, label=name,
               edgecolors="white", linewidths=0.5)
ax.set_xlabel("LSA 维度 1", fontsize=15)
ax.set_ylabel("LSA 维度 2", fontsize=15)
ax.set_title("(b) 文档语义空间分布 (LSA 降维)", fontsize=17)
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_4_03_topic_analysis")
