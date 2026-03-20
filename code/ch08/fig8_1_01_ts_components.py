"""
fig8_1_01_ts_components.py
时间序列的三成分分解（趋势 + 季节性 + 残差）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成三年合成日温度数据 ────────────────────────────────────────
n_days = 1095
dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
t = np.arange(n_days)
# 三成分
trend = 0.1 * t / n_days                          # 线性趋势 0 → 0.1
seasonal = 15 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)  # 年周期
residual = np.random.normal(0, 3, n_days)          # 高斯噪声 σ=3
observed = 12 + trend + seasonal + residual        # 观测 = 基线 + 各成分
# 真实趋势成分（教学用途：直接展示已知成分）
trend_show = 12 + trend
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("图 8.1.1　时间序列的三成分分解",
             fontsize=20, fontweight="bold", y=1.02)
# ── (1) 原始序列 ─────────────────────────────────────────────────
ax = axes[0]
ax.plot(dates, observed, color=COLORS["blue"], lw=1.0, alpha=0.85)
ax.set_ylabel("温度 (°C)", fontsize=14)
ax.set_title("原始序列", fontsize=15)
ax.tick_params(labelsize=13)
# ── (2) 趋势成分 ─────────────────────────────────────────────────
ax = axes[1]
ax.plot(dates, trend_show, color=COLORS["red"], lw=2.0)
ax.set_ylabel("温度 (°C)", fontsize=14)
ax.set_title("趋势成分", fontsize=15)
ax.tick_params(labelsize=13)
# ── (3) 季节性成分 ───────────────────────────────────────────────
ax = axes[2]
ax.plot(dates, seasonal, color=COLORS["green"], lw=1.5)
ax.set_ylabel("温度 (°C)", fontsize=14)
ax.set_title("季节性成分", fontsize=15)
ax.tick_params(labelsize=13)
# ── (4) 残差成分 ─────────────────────────────────────────────────
ax = axes[3]
ax.plot(dates, residual, color=COLORS["gray"], lw=0.8, alpha=0.8)
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_ylabel("温度 (°C)", fontsize=14)
ax.set_title("残差成分", fontsize=15)
ax.set_xlabel("日期", fontsize=14)
ax.tick_params(labelsize=13)
# ── 日期格式 ─────────────────────────────────────────────────────
for a in axes:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.2)
save_fig(fig, __file__, "fig8_1_01_ts_components")
