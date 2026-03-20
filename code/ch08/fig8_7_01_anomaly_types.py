"""
fig8_7_01_anomaly_types.py
时间序列的三种异常类型：点异常、上下文异常、集体异常
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
# ── 生成基础信号 ──────────────────────────────────────────────────
T = 200
t = np.arange(T)
base = 50 + 15 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 2, T)
# ── 三种异常 ──────────────────────────────────────────────────────
# (a) 点异常
signal_a = base.copy()
spike_locs = [45, 120, 170]
for loc in spike_locs:
    signal_a[loc] = base[loc] + 35
# (b) 上下文异常
signal_b = base.copy()
# 夜间（低值区间 100-130）出现异常高值
ctx_start, ctx_end = 100, 115
signal_b[ctx_start:ctx_end] += 20  # 在应该低的地方偏高
# (c) 集体异常
signal_c = base.copy()
coll_start, coll_end = 80, 120
# 正常的正弦模式被替换为异常的锯齿模式
signal_c[coll_start:coll_end] = 50 + np.linspace(0, 20, coll_end - coll_start) + \
    5 * np.mod(np.arange(coll_end - coll_start), 8)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("图 8.7.1　时间序列的三种异常类型",
             fontsize=20, fontweight="bold", y=1.02)
# ── (a) 点异常 ────────────────────────────────────────────────────
ax = axes[0]
ax.plot(t, signal_a, color=COLORS["blue"], lw=1.0, alpha=0.8)
for loc in spike_locs:
    ax.plot(loc, signal_a[loc], 'o', color=COLORS["red"], markersize=10,
            zorder=5)
# 3-sigma 线
mu, sigma = base.mean(), base.std()
ax.axhline(mu + 3 * sigma, color=COLORS["red"], ls="--", lw=1.2,
           alpha=0.6, label="3σ 上限")
ax.set_xlabel("时间步", fontsize=13)
ax.set_ylabel("值", fontsize=13)
ax.set_title("(a) 点异常", fontsize=14)
ax.legend(fontsize=11, loc="lower right")
ax.tick_params(labelsize=12)
# ── (b) 上下文异常 ────────────────────────────────────────────────
ax = axes[1]
ax.plot(t, signal_b, color=COLORS["blue"], lw=1.0, alpha=0.8)
ax.axvspan(ctx_start, ctx_end, alpha=0.2, color=COLORS["orange"],
           label="上下文异常区间")
# 标注：全局不极端但在上下文中异常
ax.axhline(mu + 3 * sigma, color=COLORS["red"], ls="--", lw=1.0,
           alpha=0.4)
ax.annotate("全局范围内正常\n但在此上下文中异常",
            xy=((ctx_start + ctx_end) / 2, signal_b[ctx_start:ctx_end].mean()),
            xytext=(140, 80),
            fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", color=COLORS["orange"]),
            color=COLORS["orange"])
ax.set_xlabel("时间步", fontsize=13)
ax.set_title("(b) 上下文异常", fontsize=14)
ax.legend(fontsize=11, loc="lower right")
ax.tick_params(labelsize=12)
# ── (c) 集体异常 ──────────────────────────────────────────────────
ax = axes[2]
ax.plot(t, signal_c, color=COLORS["blue"], lw=1.0, alpha=0.8)
ax.axvspan(coll_start, coll_end, alpha=0.2, color=COLORS["purple"],
           label="集体异常区间")
ax.annotate("整段模式异常\n单个点可能不极端",
            xy=((coll_start + coll_end) / 2, signal_c[coll_start:coll_end].mean()),
            xytext=(150, 80),
            fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", color=COLORS["purple"]),
            color=COLORS["purple"])
ax.set_xlabel("时间步", fontsize=13)
ax.set_title("(c) 集体异常", fontsize=14)
ax.legend(fontsize=11, loc="lower right")
ax.tick_params(labelsize=12)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_7_01_anomaly_types")
