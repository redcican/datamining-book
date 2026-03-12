"""
图 3.1.1  三种节点不纯度度量的比较：信息熵 / 基尼不纯度 / 错分率
对应节次：3.1 决策树算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_1_01_impurity_measures.py
输出路径：public/figures/ch03/fig3_1_01_impurity_measures.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt

apply_style()

p = np.linspace(1e-9, 1 - 1e-9, 1000)

# --- 三种不纯度度量（二分类，p = P(正类)）---
entropy  = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))   # 最大值 1.0
gini     = 2 * p * (1 - p)                                  # 最大值 0.5
mce      = 1 - np.maximum(p, 1 - p)                         # 最大值 0.5

# 归一化到 [0, 1]：各除以各自最大值
entropy_n = entropy / 1.0
gini_n    = gini    / 0.5
mce_n     = mce     / 0.5

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.38)

# ── Panel (a): 原始尺度 ───────────────────────────────────────────────────────
ax = axes[0]
lw = 2.4
ax.plot(p, entropy, color=COLORS["blue"],   lw=lw, label="信息熵 $H(p)$（最大值 = 1.0）")
ax.plot(p, gini,    color=COLORS["red"],    lw=lw, label="基尼不纯度 $G(p) = 2p(1-p)$（最大值 = 0.5）")
ax.plot(p, mce,     color=COLORS["teal"],   lw=lw, ls="--", label="错分率 $M(p) = 1 - \\max(p,1-p)$（最大值 = 0.5）")
# 标注峰值
ax.axvline(0.5, color="#94a3b8", lw=1.4, ls=":", alpha=0.8)
ax.annotate("峰值 $p=0.5$", xy=(0.5, 0.02), ha="center", fontsize=12, color="#64748b")
ax.scatter([0.5], [1.0], s=60, color=COLORS["blue"],  zorder=5)
ax.scatter([0.5], [0.5], s=60, color=COLORS["red"],   zorder=5)
ax.scatter([0.5], [0.5], s=60, color=COLORS["teal"],  zorder=5, marker="D")
ax.set_xlabel("正类概率 $p$", fontsize=13)
ax.set_ylabel("不纯度值", fontsize=13)
ax.set_title("(a) 原始尺度：三种不纯度度量\n（熵量程为 [0, 1]，基尼与错分率量程为 [0, 0.5]）",
             fontsize=13, pad=6)
ax.legend(fontsize=12, loc="upper center", labelspacing=0.35)
ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.08)

# ── Panel (b): 归一化到 [0, 1] ────────────────────────────────────────────────
ax = axes[1]
ax.plot(p, entropy_n, color=COLORS["blue"],  lw=lw, label="归一化信息熵 $H(p)$")
ax.plot(p, gini_n,    color=COLORS["red"],   lw=lw, label="归一化基尼系数 $G(p)/0.5$")
ax.plot(p, mce_n,     color=COLORS["teal"],  lw=lw, ls="--", label="归一化错分率 $M(p)/0.5$")
ax.axvline(0.5, color="#94a3b8", lw=1.4, ls=":", alpha=0.8)
# 标注三曲线的差异区域
ax.fill_between(p, gini_n, entropy_n, where=(entropy_n > gini_n),
                alpha=0.12, color=COLORS["blue"], label="熵与基尼之差（差异带）")
ax.set_xlabel("正类概率 $p$", fontsize=13)
ax.set_ylabel("归一化不纯度", fontsize=13)
ax.set_title("(b) 归一化后的对比：三者均峰值于 $p=0.5$\n（说明在大多数情况下特征排序一致）",
             fontsize=13, pad=6)
ax.legend(fontsize=12, loc="upper center", labelspacing=0.35)
ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.08)

fig.suptitle(
    "决策树节点不纯度度量：信息熵 / 基尼不纯度 / 错分率（二分类情形）\n"
    "三者在 $p=0$ 和 $p=1$（纯节点）时均为 0，在 $p=0.5$（最大不确定性）时达到峰值",
    fontsize=13, y=1.02)

save_fig(fig, __file__, "fig3_1_01_impurity_measures")
