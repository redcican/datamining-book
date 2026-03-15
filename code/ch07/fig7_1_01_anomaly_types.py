"""
fig7_1_01_anomaly_types.py
三种异常类型示意图：点异常、上下文异常、集体异常
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("图 7.1.1　三种异常类型", fontsize=20, fontweight="bold", y=1.02)

# ── 左：点异常 ────────────────────────────────────────────────
ax = axes[0]
# 主簇
n_main = 120
cx, cy = 5.0, 5.0
X_main = np.column_stack([
    np.random.normal(cx, 0.8, n_main),
    np.random.normal(cy, 0.8, n_main),
])
# 离群点
outliers = np.array([
    [1.0, 1.5],
    [9.0, 9.0],
    [1.5, 8.5],
    [8.5, 1.0],
])
ax.scatter(X_main[:, 0], X_main[:, 1], c=COLORS["blue"], s=30, alpha=0.6, label="正常点")
ax.scatter(outliers[:, 0], outliers[:, 1], c=COLORS["red"], s=100, marker="*",
           edgecolors="k", linewidths=0.5, zorder=5, label="点异常")
for ox, oy in outliers:
    ax.annotate("", xy=(ox, oy), xytext=(ox + 0.3, oy + 0.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))
ax.set_title("(a) 点异常", fontsize=17)
ax.set_xlabel("特征 $x_1$", fontsize=15)
ax.set_ylabel("特征 $x_2$", fontsize=15)
ax.legend(fontsize=14, loc="lower right")
ax.tick_params(labelsize=13)

# ── 中：上下文异常 ────────────────────────────────────────────
ax = axes[1]
t = np.arange(0, 365)
# 正弦季节模式（温度类）
temp = 15 + 12 * np.sin(2 * np.pi * (t - 80) / 365) + np.random.normal(0, 1.2, len(t))
ax.plot(t, temp, color=COLORS["blue"], lw=1.5, label="正常温度")

# 上下文异常：冬天出现夏天温度
anomaly_idx = 15  # 1月中旬
anomaly_val = 28.0  # 异常高温
ax.scatter([t[anomaly_idx]], [anomaly_val], c=COLORS["red"], s=150, marker="*",
           edgecolors="k", linewidths=0.5, zorder=5, label="上下文异常")
ax.annotate("冬季高温\n(值正常,时间异常)",
            xy=(t[anomaly_idx], anomaly_val),
            xytext=(120, 20),
            fontsize=13, color=COLORS["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))

# 标注季节
for label, x_pos in [("冬", 30), ("春", 120), ("夏", 210), ("秋", 300)]:
    ax.text(x_pos, -2, label, fontsize=14, ha="center", color=COLORS["gray"],
            fontweight="bold")

ax.set_title("(b) 上下文异常", fontsize=17)
ax.set_xlabel("天（一年）", fontsize=15)
ax.set_ylabel("温度 (°C)", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)

# ── 右：集体异常 ────────────────────────────────────────────
ax = axes[2]
t2 = np.arange(0, 200)
signal = np.sin(2 * np.pi * t2 / 40) + np.random.normal(0, 0.15, len(t2))

# 集体异常区间
anom_start, anom_end = 110, 140
signal[anom_start:anom_end] = 0.5 * np.sin(2 * np.pi * np.arange(anom_end - anom_start) / 8) + 1.5

ax.plot(t2[:anom_start], signal[:anom_start], color=COLORS["blue"], lw=1.5)
ax.plot(t2[anom_start:anom_end], signal[anom_start:anom_end], color=COLORS["red"], lw=2.5,
        label="集体异常子序列")
ax.plot(t2[anom_end:], signal[anom_end:], color=COLORS["blue"], lw=1.5)

# 高亮区域
ax.axvspan(anom_start, anom_end, alpha=0.15, color=COLORS["red"])
ax.annotate("异常子序列",
            xy=((anom_start + anom_end) / 2, 1.8),
            xytext=(30, 2.2),
            fontsize=14, color=COLORS["red"], fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))

# 添加正常标签
ax.plot([], [], color=COLORS["blue"], lw=1.5, label="正常信号")

ax.set_title("(c) 集体异常", fontsize=17)
ax.set_xlabel("时间步", fontsize=15)
ax.set_ylabel("信号值", fontsize=15)
ax.legend(fontsize=13, loc="lower left")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_1_01_anomaly_types")
