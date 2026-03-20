"""
fig9_1_01_pipeline.py
文本预处理管线示意：从原始文本到标准化词项序列的完整流程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
# ── 配置 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle("图 9.1.1　文本预处理管线",
             fontsize=22, fontweight="bold", y=0.98)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_aspect("equal")
# ── 辅助函数 ──────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, color, fontsize=13, text_color="white"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"],
                                lw=2.5, mutation_scale=18))

def draw_text_box(ax, x, y, lines, fontsize=12):
    text = "\n".join(lines)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                      edgecolor=COLORS["gray"], linewidth=1.2))
# ── 原始文本（顶部）────────────────────────────────────────────────
raw_text = [
    '"The 3 Data Mining',
    'researchers can\'t STOP',
    'analyzing data!!"'
]
draw_text_box(ax, 7, 9.0, raw_text, fontsize=13)
ax.text(7, 9.8, "原始文本", ha="center", va="center",
        fontsize=15, fontweight="bold", color=COLORS["gray"])
# ── 步骤 1：分词 ──────────────────────────────────────────────────
draw_arrow(ax, 7, 8.35, 7, 7.55)
draw_box(ax, 5.0, 7.0, 4.0, 0.55, "步骤 1：分词 (Tokenization)", COLORS["blue"])
step1_result = [
    '[The, 3, Data, Mining, researchers,',
    " can't, STOP, analyzing, data, !, !]"
]
draw_text_box(ax, 7, 6.2, step1_result, fontsize=11)
# ── 步骤 2：规范化 ────────────────────────────────────────────────
draw_arrow(ax, 7, 5.65, 7, 4.95)
draw_box(ax, 4.5, 4.4, 5.0, 0.55, "步骤 2：规范化 (小写化 + 去标点)", COLORS["green"])
step2_result = [
    "[the, 3, data, mining, researchers,",
    " can't, stop, analyzing, data]"
]
draw_text_box(ax, 7, 3.6, step2_result, fontsize=11)
# ── 步骤 3：停用词过滤 ────────────────────────────────────────────
draw_arrow(ax, 7, 3.05, 7, 2.35)
draw_box(ax, 4.5, 1.8, 5.0, 0.55, "步骤 3：去停用词 (the, can't, 3)", COLORS["orange"])
step3_result = [
    "[data, mining, researchers,",
    " stop, analyzing, data]"
]
draw_text_box(ax, 7, 1.0, step3_result, fontsize=11)
# ── 步骤 4：词形还原 ──────────────────────────────────────────────
draw_arrow(ax, 7, 0.45, 7, -0.25)
draw_box(ax, 4.5, -0.8, 5.0, 0.55, "步骤 4：词形还原 (Lemmatization)", COLORS["purple"])
final_result = ["[datum, mine, researcher, stop, analyze, datum]"]
draw_text_box(ax, 7, -1.55, final_result, fontsize=12)
ax.text(7, -2.15, "标准化词项序列", ha="center", va="center",
        fontsize=15, fontweight="bold", color=COLORS["purple"])
# ── 右侧标注：词汇表变化 ──────────────────────────────────────────
annotations = [
    (12.5, 6.2, "|V| = 10", COLORS["blue"]),
    (12.5, 3.6, "|V| = 8", COLORS["green"]),
    (12.5, 1.0, "|V| = 5", COLORS["orange"]),
    (12.5, -1.55, "|V| = 5", COLORS["purple"]),
]
for x, y, txt, c in annotations:
    ax.text(x, y, txt, ha="center", va="center", fontsize=15,
            fontweight="bold", color=c,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=c, linewidth=1.5))
ax.text(12.5, 7.5, "词汇表大小", ha="center", va="center",
        fontsize=14, fontweight="bold", color=COLORS["gray"])
# ── 调整和保存 ────────────────────────────────────────────────────
ax.set_ylim(-2.8, 10.3)
save_fig(fig, __file__, "fig9_1_01_pipeline")
