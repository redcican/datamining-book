"""
fig9_1_02_vocab_reduction.py
预处理各步骤的词汇压缩效果 + Zipf 分布
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
# ── 模拟词汇压缩数据 ─────────────────────────────────────────────
stages = ["原始词汇", "小写化", "+去停用词", "+词干提取"]
vocab_sizes = [45200, 38400, 37800, 26500]
doc_lengths = [520, 520, 385, 385]  # 平均文档长度（词）
compression = [0, 15.0, 16.4, 41.4]  # 累计压缩比 %
# ── 模拟 Zipf 分布 ───────────────────────────────────────────────
n_words = 45200
ranks = np.arange(1, n_words + 1)
# Zipf: f(r) ∝ 1/r^α, α ≈ 1.07
alpha = 1.07
freqs = (ranks ** (-alpha))
freqs = freqs / freqs.sum() * 500000  # 总词频 50 万
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.1.2　预处理的词汇压缩效果",
             fontsize=22, fontweight="bold", y=1.02)
# ── 左面板：词汇表大小（柱状图）──────────────────────────────────
ax = axes[0]
colors_bar = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]]
bars = ax.bar(stages, vocab_sizes, color=colors_bar, alpha=0.8, width=0.55,
              edgecolor="white", linewidth=1.5)
# 标注数值
for bar, v, c in zip(bars, vocab_sizes, compression):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{v:,}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    if c > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"-{c:.0f}%", ha="center", va="center", fontsize=14,
                fontweight="bold", color="white")
ax.set_ylabel("词汇表大小 |V|", fontsize=16)
ax.set_title("(a) 逐步预处理的词汇表变化", fontsize=17)
ax.tick_params(labelsize=14)
ax.set_ylim(0, 52000)
# 添加箭头连接
for i in range(len(stages) - 1):
    ax.annotate("", xy=(i + 1, vocab_sizes[i + 1] + 1500),
                xytext=(i, vocab_sizes[i] + 1500),
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"],
                                lw=1.5, connectionstyle="arc3,rad=-0.2"))
# ── 右面板：Zipf log-log 分布 ────────────────────────────────────
ax = axes[1]
# 只画部分点以保持清晰
sample_idx = np.unique(np.logspace(0, np.log10(n_words - 1), 2000).astype(int))
ax.scatter(ranks[sample_idx], freqs[sample_idx], s=3, alpha=0.4,
           color=COLORS["blue"], label="词频数据")
# 拟合线
fit_ranks = np.array([1, n_words])
fit_freqs = freqs[0] * (fit_ranks ** (-alpha)) / (1 ** (-alpha))
ax.plot(fit_ranks, fit_freqs, "--", color=COLORS["red"], lw=2,
        label=f"Zipf 拟合 (α={alpha})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("词频排名 r (log)", fontsize=16)
ax.set_ylabel("词频 f(r) (log)", fontsize=16)
ax.set_title("(b) 词频的 Zipf 分布", fontsize=17)
ax.legend(fontsize=14, loc="upper right")
ax.tick_params(labelsize=14)
# 标注高频区和低频区
ax.axvspan(1, 100, alpha=0.08, color=COLORS["green"])
ax.axvspan(10000, n_words, alpha=0.08, color=COLORS["orange"])
ax.text(15, freqs[0] * 0.1, "高频词\n(~100 词)", ha="center", fontsize=12,
        color=COLORS["green"], fontweight="bold")
ax.text(25000, freqs[25000] * 5, "低频长尾\n(占词汇 >75%)", ha="center",
        fontsize=12, color=COLORS["orange"], fontweight="bold")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_1_02_vocab_reduction")
