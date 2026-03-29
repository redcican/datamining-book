"""
图 13.7.1　北京 PM2.5 数据探索
(a) PM2.5 月度均值时间序列  (b) 24 小时日内变化模式
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_pm25 import load_pm25

df = load_pm25()

# ── 基本统计 ──────────────────────────────────────────────
print("=== 北京 PM2.5 数据集 ===")
print(f"  时间范围: {df.index.min()} — {df.index.max()}")
print(f"  记录数: {len(df):,} (小时)")
print(f"  PM2.5 统计:")
print(f"    均值: {df['pm2.5'].mean():.1f} μg/m³")
print(f"    中位数: {df['pm2.5'].median():.1f} μg/m³")
print(f"    标准差: {df['pm2.5'].std():.1f} μg/m³")
print(f"    范围: [{df['pm2.5'].min():.0f}, {df['pm2.5'].max():.0f}]")

# 月度统计
monthly = df.resample("ME")["pm2.5"].mean()
print(f"\n  月度均值范围: [{monthly.min():.0f}, {monthly.max():.0f}]")

# 季节性
hourly_avg = df.groupby(df.index.hour)["pm2.5"].mean()
print(f"\n  24 小时模式:")
print(f"    最低: {hourly_avg.idxmin()}时 ({hourly_avg.min():.1f})")
print(f"    最高: {hourly_avg.idxmax()}时 ({hourly_avg.max():.1f})")

# 各年均值
for year in range(2010, 2015):
    yearly = df[df.index.year == year]["pm2.5"].mean()
    print(f"    {year}年: {yearly:.1f} μg/m³")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# (a) 月度均值时间序列
ax1.plot(monthly.index, monthly.values, "-", color=COLORS["blue"],
         linewidth=1.5, alpha=0.9)
ax1.fill_between(monthly.index, monthly.values, alpha=0.15,
                 color=COLORS["blue"])

# 标注 WHO 标准
ax1.axhline(75, color=COLORS["orange"], linestyle="--", linewidth=1,
            alpha=0.7, label="中国 II 级标准 (75 μg/m³)")
ax1.axhline(35, color=COLORS["red"], linestyle="--", linewidth=1,
            alpha=0.7, label="中国 I 级标准 (35 μg/m³)")

ax1.set_ylabel("PM2.5 月均值 (μg/m³)")
ax1.set_title("(a) PM2.5 月均值时间序列 (2010–2014)", fontweight="bold")
ax1.legend(fontsize=10, loc="upper right")

# (b) 24 小时日内模式
ax2.bar(hourly_avg.index, hourly_avg.values, color=COLORS["orange"],
        edgecolor="white", width=0.7, alpha=0.85)
ax2.axhline(hourly_avg.mean(), color=COLORS["red"], linestyle="--",
            linewidth=1.5,
            label=f"全天均值 = {hourly_avg.mean():.1f}")
ax2.set_xlabel("小时 (0-23)")
ax2.set_ylabel("PM2.5 均值 (μg/m³)")
ax2.set_title("(b) PM2.5 日内变化模式", fontweight="bold")
ax2.set_xticks(range(0, 24))
ax2.legend(fontsize=10)

plt.tight_layout(h_pad=3)
save_fig(fig, __file__, "fig13_7_01_eda")
