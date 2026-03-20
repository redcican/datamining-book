"""
fig9_7_03_case_result.py
跨语言情感分类案例结果：三种方法准确率对比 + 错误类型分析
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 9.7.3　跨语言情感分类案例结果",
             fontsize=22, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 三种方法准确率对比
# ══════════════════════════════════════════════════════════════════
ax = axes[0]

methods = ["翻译后测试", "嵌入对齐", "mBERT零样本"]
accuracies = [0.78, 0.82, 0.88]
errors = [0.03, 0.04, 0.02]
bar_colors = [COLORS["blue"], COLORS["purple"], COLORS["red"]]
baseline = 0.92

n_methods = len(methods)
x_pos = np.arange(n_methods)

bars = ax.bar(x_pos, accuracies, width=0.55, color=bar_colors,
              alpha=0.85, edgecolor="white", linewidth=1.5,
              yerr=errors, capsize=5,
              error_kw=dict(lw=1.5, capthick=1.5, color="#333333"))

# Value labels on bars
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold")

# Baseline dashed line
ax.axhline(y=baseline, color=COLORS["orange"], lw=2.0, ls="--",
           alpha=0.8, label=f"源语言性能 ({baseline:.2f})")

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=14)
ax.set_xlabel("迁移方法", fontsize=16)
ax.set_ylabel("准确率 (Accuracy)", fontsize=16)
ax.set_ylim(0.6, 1.0)
ax.legend(fontsize=14, loc="upper left", framealpha=0.9)
ax.tick_params(labelsize=14)
ax.set_title("(a) 三种方法准确率对比", fontsize=17)

# ══════════════════════════════════════════════════════════════════
# (b) 跨语言错误类型分析
# ══════════════════════════════════════════════════════════════════
ax = axes[1]

error_types = ["翻译歧义", "文化特异表达", "否定结构差异", "领域术语", "其他"]
error_counts = [32, 25, 18, 15, 10]
total = sum(error_counts)
error_pcts = [c / total * 100 for c in error_counts]

# Color gradient from red (most frequent) to blue (least)
gradient_colors = [COLORS["red"], COLORS["orange"], COLORS["purple"],
                   COLORS["teal"], COLORS["blue"]]

n_errors = len(error_types)
y_pos = np.arange(n_errors)

bars = ax.barh(y_pos, error_counts, height=0.6, color=gradient_colors,
               alpha=0.85, edgecolor="white", linewidth=1.5)

# Percentage labels on bars
for bar, count, pct in zip(bars, error_counts, error_pcts):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct:.0f}%)", va="center", ha="left",
            fontsize=13, fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(error_types, fontsize=14)
ax.set_xlabel("错误数量", fontsize=16)
ax.set_xlim(0, max(error_counts) * 1.35)
ax.invert_yaxis()
ax.tick_params(labelsize=14)
ax.set_title("(b) 跨语言错误类型分析", fontsize=17)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig9_7_03_case_result")
