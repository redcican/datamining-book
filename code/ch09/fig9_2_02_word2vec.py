"""
fig9_2_02_word2vec.py
Word2Vec 词嵌入可视化：Skip-gram 训练示意 + 词向量 PCA 降维
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("图 9.2.2　Word2Vec 词嵌入可视化",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：Skip-gram 训练示意 ────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_aspect("equal")
def draw_box(ax, x, y, w, h, text, color, fontsize=12, text_color="white"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="white", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)
# 输入句子
sentence_words = ["数据", "挖掘", "是", "一门", "重要", "的", "技术"]
y_sent = 9.0
for i, w in enumerate(sentence_words):
    x_pos = 0.5 + i * 1.3
    bg = COLORS["red"] if i == 2 else ("#e2e8f0" if abs(i - 2) > 2 else COLORS["blue"])
    tc = "white" if i == 2 or abs(i - 2) <= 2 else COLORS["gray"]
    draw_box(ax, x_pos, y_sent, 1.0, 0.6, w, bg, fontsize=11, text_color=tc)
# 标注
ax.text(3.15, 8.3, "中心词", ha="center", fontsize=11,
        color=COLORS["red"], fontweight="bold")
ax.text(1.8, 8.3, "上下文", ha="center", fontsize=11,
        color=COLORS["blue"], fontweight="bold")
ax.text(4.5, 8.3, "上下文", ha="center", fontsize=11,
        color=COLORS["blue"], fontweight="bold")
# 中心词向量
ax.annotate("", xy=(5, 7.3), xytext=(3.15, 8.5),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=2))
draw_box(ax, 3.5, 6.0, 3.0, 1.2, "嵌入层\n" + r"$\mathbf{v}_{w_c}$" + "\n(d 维向量)",
         COLORS["purple"], fontsize=12)
# 预测上下文
ax.annotate("", xy=(5, 4.5), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=2))
draw_box(ax, 3.0, 3.0, 4.0, 1.4,
         "Softmax / 负采样\n$P(w_o | w_c)$",
         COLORS["orange"], fontsize=12)
# 输出概率
pred_words = ["挖掘", "一门", "数据", "重要"]
pred_probs = ["0.35", "0.28", "0.20", "0.12"]
y_pred = 1.5
for i, (w, p) in enumerate(zip(pred_words, pred_probs)):
    x_pos = 1.5 + i * 1.8
    draw_box(ax, x_pos, y_pred, 1.4, 0.55, f"{w}: {p}",
             COLORS["teal"], fontsize=10.5)
ax.annotate("", xy=(5, 2.3), xytext=(5, 2.8),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=2))
ax.text(5, 0.8, "最大化正确上下文的概率", ha="center", fontsize=12,
        color=COLORS["gray"], fontweight="bold")
ax.set_title("(a) Skip-gram 训练过程", fontsize=17, pad=10)
# ── 右面板：词向量 PCA 降维可视化 ─────────────────────────────────
ax = axes[1]
# 模拟词向量聚类效果
word_groups = {
    "动物": {"猫": (1.2, 2.5), "狗": (1.5, 2.0), "鸟": (0.8, 2.8),
             "鱼": (1.8, 3.0)},
    "食物": {"米饭": (4.0, 1.0), "面条": (4.5, 1.5), "面包": (3.5, 0.5),
             "水果": (4.2, 0.2)},
    "国家": {"中国": (-2.0, 4.5), "日本": (-1.5, 4.0), "美国": (-2.5, 3.8),
             "法国": (-1.8, 5.0)},
    "首都": {"北京": (-0.5, 4.8), "东京": (0.0, 4.3), "华盛顿": (-1.0, 4.1),
             "巴黎": (-0.3, 5.3)},
}
group_colors = {
    "动物": COLORS["blue"], "食物": COLORS["green"],
    "国家": COLORS["orange"], "首都": COLORS["red"],
}
for group, words_dict in word_groups.items():
    xs = [v[0] for v in words_dict.values()]
    ys = [v[1] for v in words_dict.values()]
    color = group_colors[group]
    ax.scatter(xs, ys, c=color, s=80, zorder=5, alpha=0.8)
    for w, (x, y) in words_dict.items():
        ax.annotate(w, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=12, color=color, fontweight="bold")
# 画类比关系箭头：中国→北京 ≈ 日本→东京
ax.annotate("", xy=(-0.5, 4.8), xytext=(-2.0, 4.5),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["purple"],
                            lw=2, ls="--"))
ax.annotate("", xy=(0.0, 4.3), xytext=(-1.5, 4.0),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["purple"],
                            lw=2, ls="--"))
ax.text(-0.8, 3.3, "国家→首都\n(平行向量)", ha="center", fontsize=11,
        color=COLORS["purple"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["purple"], alpha=0.8))
# 画聚类椭圆（近似）
from matplotlib.patches import Ellipse
for group, words_dict in word_groups.items():
    xs = [v[0] for v in words_dict.values()]
    ys = [v[1] for v in words_dict.values()]
    cx, cy = np.mean(xs), np.mean(ys)
    w = (max(xs) - min(xs)) + 1.5
    h = (max(ys) - min(ys)) + 1.2
    ell = Ellipse((cx, cy), w, h, alpha=0.1, color=group_colors[group])
    ax.add_patch(ell)
# 图例
legend_patches = [mpatches.Patch(color=c, label=g, alpha=0.7)
                  for g, c in group_colors.items()]
ax.legend(handles=legend_patches, fontsize=12, loc="lower left")
ax.set_xlabel("PCA 维度 1", fontsize=14)
ax.set_ylabel("PCA 维度 2", fontsize=14)
ax.set_title("(b) 词向量空间 (PCA 降维)", fontsize=17)
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_2_02_word2vec")
