"""
fig9_3_01_naive_bayes.py
朴素贝叶斯文本分类示意：分类决策过程 + 类别词项概率分布
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 模拟数据 ──────────────────────────────────────────────────────
categories = ["体育", "科技", "财经", "娱乐", "教育"]
top_words = {
    "体育": ["比赛", "冠军", "进球", "联赛", "教练", "球员", "赛季"],
    "科技": ["技术", "发布", "芯片", "智能", "研发", "数据", "算法"],
    "财经": ["市场", "增长", "投资", "经济", "股市", "基金", "利率"],
    "娱乐": ["演员", "电影", "票房", "综艺", "导演", "明星", "娱乐"],
    "教育": ["学校", "考试", "教学", "课程", "招生", "高考", "学生"],
}
# 各类别下词项的概率 (模拟)
word_probs = {}
for cat, words in top_words.items():
    probs = np.random.dirichlet(np.ones(7) * 3)
    probs = np.sort(probs)[::-1]
    word_probs[cat] = dict(zip(words, probs))
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.3.1　朴素贝叶斯文本分类",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：分类决策过程 ──────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 10)
ax.axis("off")
ax.set_aspect("equal")
import matplotlib.patches as mpatches
def draw_box(ax, x, y, w, h, text, color, fontsize=12, tc="white"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="white", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=tc)
# 新文档
ax.text(5, 9.5, '新文档: "芯片 技术 发布 市场"', ha="center", fontsize=13,
        fontweight="bold", color=COLORS["gray"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                  edgecolor=COLORS["gray"], linewidth=1.2))
# 箭头
ax.annotate("", xy=(5, 7.9), xytext=(5, 8.8),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=2))
draw_box(ax, 2, 7.2, 6, 0.7,
         "计算各类别后验: log P(c) + Σ xⱼ log P(wⱼ|c)",
         COLORS["blue"], fontsize=11)
# 各类别得分
scores = {
    "体育": -12.4,
    "科技": -6.8,
    "财经": -9.2,
    "娱乐": -14.1,
    "教育": -13.7,
}
cat_colors = [COLORS["blue"], COLORS["red"], COLORS["green"],
              COLORS["orange"], COLORS["purple"]]
y_start = 5.8
for i, (cat, score) in enumerate(scores.items()):
    y_pos = y_start - i * 1.0
    color = cat_colors[i]
    is_best = (cat == "科技")
    lw = 3 if is_best else 1.5
    ec = COLORS["red"] if is_best else "white"
    rect = mpatches.FancyBboxPatch(
        (1.5, y_pos), 3.0, 0.6, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=ec, linewidth=lw, alpha=0.7 if not is_best else 1.0)
    ax.add_patch(rect)
    ax.text(3.0, y_pos + 0.3, cat, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    # 分数条
    bar_w = abs(score) / 15 * 4
    rect2 = mpatches.FancyBboxPatch(
        (5.5, y_pos + 0.08), bar_w, 0.44, boxstyle="round,pad=0.05",
        facecolor=color, alpha=0.4)
    ax.add_patch(rect2)
    ax.text(5.5 + bar_w + 0.2, y_pos + 0.3, f"{score:.1f}",
            ha="left", va="center", fontsize=12, fontweight="bold",
            color=color)
    if is_best:
        ax.text(9.5, y_pos + 0.3, "← 最大", ha="center", va="center",
                fontsize=12, fontweight="bold", color=COLORS["red"])
# 结果
ax.text(5, 0.3, '预测类别: 科技', ha="center", fontsize=15,
        fontweight="bold", color=COLORS["red"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["red"], linewidth=2))
ax.set_title("(a) 贝叶斯分类决策", fontsize=17, pad=5)
# ── 右面板：类别词项概率分布 ──────────────────────────────────────
ax = axes[1]
n_cats = len(categories)
n_words = 7
bar_width = 0.14
x_base = np.arange(n_cats)
for i in range(n_words):
    heights = []
    for cat in categories:
        words = list(word_probs[cat].keys())
        probs = list(word_probs[cat].values())
        heights.append(probs[i])
    offset = (i - 3) * bar_width
    bars = ax.bar(x_base + offset, heights, bar_width * 0.9,
                  alpha=0.75, color=PALETTE[i % len(PALETTE)])
    # 标注词名（仅对第一个类别的前3个词）
    if i < 3:
        for j, cat in enumerate(categories):
            w = list(word_probs[cat].keys())[i]
            ax.text(j + offset, heights[j] + 0.005, w,
                    ha="center", va="bottom", fontsize=8.5, rotation=60)
ax.set_xticks(x_base)
ax.set_xticklabels(categories, fontsize=14)
ax.set_ylabel("P(w|c)", fontsize=16)
ax.set_title("(b) 各类别 Top 词项概率", fontsize=17)
ax.tick_params(labelsize=13)
ax.set_ylim(0, 0.3)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_3_01_naive_bayes")
