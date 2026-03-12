"""
图 2.7.1  四种异常值类型示意图（全局异常 / 局部异常 / 上下文异常 / 集体异常）
对应节次：2.7 异常值检测与处理
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch02/fig2_7_01_outlier_types.py
输出路径：public/figures/ch02/fig2_7_01_outlier_types.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

apply_style()

rng = np.random.default_rng(42)

C_NORM  = COLORS["blue"]
C_OUT   = COLORS["red"]
C_TEAL  = COLORS["teal"]
C_COLL  = COLORS["orange"]
C_GRAY  = COLORS["gray"]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.subplots_adjust(hspace=0.40, wspace=0.35, top=0.92, bottom=0.08)

# ── Panel (a): 全局异常 ─────────────────────────────────────────────────────
ax = axes[0, 0]
n = 140
X_norm = rng.normal(0, 1, (n, 2))
X_out  = np.array([[4.5, 4.0], [-4.8, 2.5], [3.6, -4.2]])
ax.scatter(X_norm[:, 0], X_norm[:, 1], s=28, alpha=0.55, color=C_NORM, zorder=2, label="正常数据点")
ax.scatter(X_out[:, 0], X_out[:, 1], s=110, color=C_OUT, zorder=4,
           edgecolors="white", linewidths=1.4, label="全局异常点", marker="*")
# Ellipse for normal region
ell = Ellipse((0, 0), width=5.2, height=5.2, fill=False,
              edgecolor=C_NORM, linewidth=1.5, linestyle="--", alpha=0.6, zorder=3)
ax.add_patch(ell)
ax.set_title("(a) 全局异常\n（Global Outlier）", fontsize=13, pad=6)
ax.set_xlabel("特征 $x_1$", fontsize=12)
ax.set_ylabel("特征 $x_2$", fontsize=12)
ax.legend(fontsize=12, loc="lower right", labelspacing=0.3)
ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
ax.set_aspect("equal")

# ── Panel (b): 局部异常 ─────────────────────────────────────────────────────
ax = axes[0, 1]
# 密集簇
Xd = rng.normal([0.0, 0.0], 0.5, (70, 2))
# 稀疏簇
Xs = rng.normal([5.5, 5.5], 1.6, (45, 2))
# 局部异常点：夹在两簇之间，既不属于密集簇也不属于稀疏簇的局部空洞中
X_lo = np.array([[2.3, 2.0]])
ax.scatter(Xd[:, 0], Xd[:, 1], s=28, alpha=0.60, color=C_NORM, zorder=2, label="密集簇")
ax.scatter(Xs[:, 0], Xs[:, 1], s=28, alpha=0.60, color=C_TEAL, zorder=2, label="稀疏簇")
ax.scatter(X_lo[:, 0], X_lo[:, 1], s=130, color=C_OUT, zorder=5,
           edgecolors="white", linewidths=1.4, label="局部异常点", marker="*")
ax.annotate("局部\n异常", xy=(2.3, 2.0), xytext=(3.4, 1.2),
            fontsize=12, color=C_OUT, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_OUT, lw=1.3))
ax.set_title("(b) 局部异常\n（Local Outlier）", fontsize=13, pad=6)
ax.set_xlabel("特征 $x_1$", fontsize=12)
ax.set_ylabel("特征 $x_2$", fontsize=12)
ax.legend(fontsize=12, loc="upper left", labelspacing=0.3)
ax.set_xlim(-2.2, 9.5); ax.set_ylim(-2.2, 9.5)
ax.set_aspect("equal")

# ── Panel (c): 上下文异常 ───────────────────────────────────────────────────
ax = axes[1, 0]
months = np.arange(1, 25)
# 上海月均气温近似季节曲线：1 月约 5°C，7 月约 28°C
T_seasonal = 16.5 + 13.0 * np.sin(2 * np.pi * (months - 1) / 12 - np.pi / 2)
noise = rng.normal(0, 1.2, len(months))
T_obs = T_seasonal + noise
# 注入上下文异常：1 月（冬季）突然出现 28°C
outlier_idx = [0, 12]
T_obs[outlier_idx[0]] = 27.5
T_obs[outlier_idx[1]] = 26.8
outlier_mask = np.zeros(len(months), dtype=bool)
for idx in outlier_idx:
    outlier_mask[idx] = True
ax.plot(months, T_seasonal, color=C_GRAY, lw=1.8, ls="--", alpha=0.70, label="季节性基线", zorder=1)
ax.plot(months, T_obs, color=C_NORM, lw=1.8, zorder=2)
ax.scatter(months[~outlier_mask], T_obs[~outlier_mask], s=32, color=C_NORM, zorder=3)
ax.scatter(months[outlier_mask], T_obs[outlier_mask], s=120, color=C_OUT, zorder=5,
           edgecolors="white", linewidths=1.4, label="上下文异常（1 月高温）", marker="*")
# Shade winter months
for yr in [0, 1]:
    ax.axvspan(12 * yr + 1, 12 * yr + 3, alpha=0.08, color=C_TEAL, label=("冬季区间" if yr == 0 else None))
ax.set_title("(c) 上下文异常\n（Contextual Outlier）", fontsize=13, pad=6)
ax.set_xlabel("月份（第 1–24 月）", fontsize=12)
ax.set_ylabel("气温（℃）", fontsize=12)
ax.legend(fontsize=12, loc="lower right", labelspacing=0.3)

# ── Panel (d): 集体异常 ─────────────────────────────────────────────────────
ax = axes[1, 1]
X_bg = rng.uniform(0, 10, (160, 2))
# 集体异常：12 个点集体构成窄带线性聚集（单独每个点不特殊，整体为异常模式）
t = np.linspace(0, 1, 14)
X_coll = np.column_stack([6.0 + 2.2 * t + rng.normal(0, 0.07, 14),
                           1.2 + 0.6 * t + rng.normal(0, 0.07, 14)])
ax.scatter(X_bg[:, 0], X_bg[:, 1], s=22, alpha=0.40, color=C_NORM, zorder=2, label="正常数据点")
ax.scatter(X_coll[:, 0], X_coll[:, 1], s=72, color=C_COLL, zorder=4,
           edgecolors="white", linewidths=1.2, label="集体异常点群", marker="D")
# 椭圆框出集体异常区域
ell2 = Ellipse((7.1, 1.55), width=2.8, height=0.95, angle=15,
               fill=False, edgecolor=C_COLL, linewidth=2.2, linestyle="--", zorder=5)
ax.add_patch(ell2)
ax.annotate("集体异常\n（各点单独\n并不极端）", xy=(7.5, 1.2), xytext=(3.5, 0.3),
            fontsize=12, color=C_COLL, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_COLL, lw=1.3))
ax.set_title("(d) 集体异常\n（Collective Outlier）", fontsize=13, pad=6)
ax.set_xlabel("特征 $x_1$", fontsize=12)
ax.set_ylabel("特征 $x_2$", fontsize=12)
ax.legend(fontsize=12, loc="upper right", labelspacing=0.3)
ax.set_xlim(-0.5, 11.0); ax.set_ylim(-0.5, 11.0)
ax.set_aspect("equal")

fig.suptitle(
    "四种异常值类型：全局/局部/上下文/集体异常的几何直觉\n"
    "★ 标注为异常点；(a) 全局：远离总体分布；(b) 局部：相对于局部邻域密度偏低；"
    "(c) 上下文：在特定时间/空间上下文下偏离；(d) 集体：个体不异常但群体模式异常",
    fontsize=12, y=0.98)

save_fig(fig, __file__, "fig2_7_01_outlier_types")
