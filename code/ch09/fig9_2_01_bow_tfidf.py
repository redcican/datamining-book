"""
fig9_2_01_bow_tfidf.py
词袋模型与 TF-IDF 对比：词频热力图 vs TF-IDF 热力图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
# ── 模拟文档-词项数据 ────────────────────────────────────────────
doc_labels = [
    "体育-1", "体育-2", "体育-3",
    "科技-1", "科技-2", "科技-3",
    "财经-1", "财经-2",
]
words = ["的", "是", "在", "比赛", "冠军", "进球",
         "技术", "芯片", "发布", "市场", "经济", "增长"]
# 词频矩阵（模拟）
tf_matrix = np.array([
    [5, 3, 2, 4, 3, 2, 0, 0, 0, 0, 0, 0],  # 体育-1
    [4, 2, 3, 3, 2, 3, 0, 0, 0, 0, 0, 0],  # 体育-2
    [6, 4, 2, 5, 4, 1, 0, 0, 0, 0, 0, 0],  # 体育-3
    [5, 3, 3, 0, 0, 0, 4, 3, 2, 0, 0, 0],  # 科技-1
    [4, 2, 2, 0, 0, 0, 3, 4, 3, 0, 0, 0],  # 科技-2
    [3, 3, 1, 0, 0, 0, 5, 2, 4, 0, 0, 0],  # 科技-3
    [5, 4, 3, 0, 0, 0, 0, 0, 0, 4, 3, 2],  # 财经-1
    [4, 3, 2, 0, 0, 0, 0, 0, 0, 3, 4, 3],  # 财经-2
], dtype=float)
N = len(doc_labels)
# ── 计算 TF-IDF ──────────────────────────────────────────────────
df = np.sum(tf_matrix > 0, axis=0)
idf = np.log(N / (df + 1e-8))
tfidf_matrix = tf_matrix * idf
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.2.1　词袋模型与 TF-IDF 对比",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：词频热力图 ───────────────────────────────────────────
ax = axes[0]
im1 = ax.imshow(tf_matrix, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, fontsize=12, rotation=45, ha="right")
ax.set_yticks(range(len(doc_labels)))
ax.set_yticklabels(doc_labels, fontsize=12)
ax.set_title("(a) 词频矩阵 (BoW)", fontsize=17)
# 标注数值
for i in range(tf_matrix.shape[0]):
    for j in range(tf_matrix.shape[1]):
        val = tf_matrix[i, j]
        if val > 0:
            color = "white" if val > 3 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")
cb1 = plt.colorbar(im1, ax=ax, shrink=0.8)
cb1.set_label("词频", fontsize=13)
# 标注高频词问题
ax.annotate("高频功能词\n(无区分力)", xy=(1, 3.5), xytext=(-2.8, 4.5),
            fontsize=11, color=COLORS["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))
# ── 右面板：TF-IDF 热力图 ────────────────────────────────────────
ax = axes[1]
im2 = ax.imshow(tfidf_matrix, cmap="Oranges", aspect="auto")
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, fontsize=12, rotation=45, ha="right")
ax.set_yticks(range(len(doc_labels)))
ax.set_yticklabels(doc_labels, fontsize=12)
ax.set_title("(b) TF-IDF 加权矩阵", fontsize=17)
# 标注数值
for i in range(tfidf_matrix.shape[0]):
    for j in range(tfidf_matrix.shape[1]):
        val = tfidf_matrix[i, j]
        if val > 0.1:
            color = "white" if val > 4 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
cb2 = plt.colorbar(im2, ax=ax, shrink=0.8)
cb2.set_label("TF-IDF 权重", fontsize=13)
# 标注效果
ax.annotate("功能词权重→0\n(IDF 低)", xy=(1, 3.5), xytext=(-2.8, 4.5),
            fontsize=11, color=COLORS["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_2_01_bow_tfidf")
