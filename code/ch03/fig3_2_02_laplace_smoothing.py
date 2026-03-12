"""
图 3.2.2  拉普拉斯平滑：零频率问题与平滑后的词概率估计
对应节次：3.2 贝叶斯分类方法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_2_02_laplace_smoothing.py
输出路径：public/figures/ch03/fig3_2_02_laplace_smoothing.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

# ── 模拟语料 ──────────────────────────────────────────────────────────────────
vocab = ["免费", "中奖", "优惠", "会议", "项目", "报告", "折扣", "安排"]
V = len(vocab)

# 垃圾邮件词频（词表8个词）
spam_counts = np.array([4, 1, 3, 0, 0, 0, 2, 0])   # total = 10
ham_counts  = np.array([0, 0, 0, 3, 3, 3, 0, 3])   # total = 12

spam_total = spam_counts.sum()
ham_total  = ham_counts.sum()

# 未平滑 MLE
p_spam_raw = spam_counts / spam_total
p_ham_raw  = ham_counts  / ham_total

# 拉普拉斯平滑 (α=1)
alpha = 1
p_spam_smooth = (spam_counts + alpha) / (spam_total + alpha * V)
p_ham_smooth  = (ham_counts  + alpha) / (ham_total  + alpha * V)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.40)

x = np.arange(V)
w = 0.36

# ── Panel (a): 未平滑（零频率问题）────────────────────────────────────────────
ax = axes[0]
bars_spam = ax.bar(x - w/2, p_spam_raw, width=w, color=COLORS["red"],
                   alpha=0.82, label="垃圾邮件（spam）", zorder=3)
bars_ham  = ax.bar(x + w/2, p_ham_raw,  width=w, color=COLORS["blue"],
                   alpha=0.82, label="正常邮件（ham）",  zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(vocab, fontsize=12)
ax.set_xlabel("词汇表", fontsize=13)
ax.set_ylabel("$\\hat{P}(w_j \\mid c_k)$（未平滑 MLE）", fontsize=12)
ax.set_title("(a) 无平滑 MLE：零频率词概率为 0",
             fontsize=13, pad=6)
ax.legend(fontsize=12, loc="upper right")
ax.set_ylim(0, 0.55)
ax.axhline(0, color="#94a3b8", lw=0.8)

# 仅在零概率柱上标 ★
for i, (ps, ph) in enumerate(zip(p_spam_raw, p_ham_raw)):
    if ps == 0:
        ax.text(i - w/2, 0.008, "★", ha="center", va="bottom",
                fontsize=13, color=COLORS["red"])
    if ph == 0:
        ax.text(i + w/2, 0.008, "★", ha="center", va="bottom",
                fontsize=13, color=COLORS["blue"])

# ── Panel (b): 拉普拉斯平滑（α=1）────────────────────────────────────────────
ax = axes[1]
ax.bar(x - w/2, p_spam_smooth, width=w, color=COLORS["red"],
       alpha=0.82, label="垃圾邮件（spam）", zorder=3)
ax.bar(x + w/2, p_ham_smooth,  width=w, color=COLORS["blue"],
       alpha=0.82, label="正常邮件（ham）",  zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(vocab, fontsize=12)
ax.set_xlabel("词汇表", fontsize=13)
ax.set_ylabel(r"$\hat{P}_\alpha(w_j \mid c_k)$（$\alpha=1$ 平滑后）", fontsize=12)
ax.set_title(f"(b) 拉普拉斯平滑（$\\alpha=1$）：所有词概率严格大于 0\n"
             f"$\\hat{{P}}_\\alpha(w_j \\mid c_k) = (N_{{jk}}+\\alpha)/(N_k+\\alpha|V|)$",
             fontsize=13, pad=6)
ax.legend(fontsize=12, loc="upper right")
ax.set_ylim(0, 0.55)
ax.axhline(0, color="#94a3b8", lw=0.8)

fig.suptitle(
    "多项式朴素贝叶斯：零频率问题与拉普拉斯平滑\n"
    "训练语料（|V|=8）：spam 10 词次，ham 12 词次；平滑后所有条件概率严格大于零",
    fontsize=13, y=1.02)

save_fig(fig, __file__, "fig3_2_02_laplace_smoothing")
