"""
fig8_2_03_pipeline.py
预处理管线各步骤效果（原始 → 异常修正+填充 → 平滑 → 差分）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成工业传感器数据 ────────────────────────────────────────────
n = 1440
t = np.arange(n)
true_signal = 80 + 0.003 * t + 5 * np.sin(2 * np.pi * t / 1440)
noise = np.random.normal(0, 0.8, n)
raw = true_signal + noise
# ── 注入随机缺失 5% ──────────────────────────────────────────────
missing_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
raw[missing_idx] = np.nan
# ── 注入两段连续缺失（各 30 点）──────────────────────────────────
block_gaps = [(350, 380), (900, 930)]
for s, e in block_gaps:
    raw[s:e] = np.nan
# ── 注入 8 个异常尖峰（±10）─────────────────────────────────────
outlier_idx = np.random.choice(
    np.where(~np.isnan(raw))[0], size=8, replace=False)
raw[outlier_idx] += np.random.choice([-10, 10], size=8)
# ── Step 1: 保留原始数据副本 ──────────────────────────────────────
raw_display = raw.copy()
# ── Step 2: 异常修正（全局中位数 ± 阈值）+ 线性插值填充 ─────────
step2 = raw.copy()
# 先用全局统计量检测明显异常尖峰（偏离 > 6°C）
valid_vals = step2[~np.isnan(step2)]
global_med = np.median(valid_vals)
global_mad = np.median(np.abs(valid_vals - global_med))
threshold = global_med + 6 * max(global_mad, 0.5)
for i in range(n):
    if np.isnan(step2[i]):
        continue
    if np.abs(step2[i] - global_med) > 6 * max(global_mad, 0.5):
        step2[i] = np.nan
# 用 pandas 线性插值填充所有 NaN（对块状缺失更稳定）
import pandas as pd
step2_series = pd.Series(step2).interpolate(method='linear')
step2_filled = step2_series.ffill().bfill().values
# ── Step 3: Savitzky-Golay 平滑 ──────────────────────────────────
step3 = savgol_filter(step2_filled, window_length=31, polyorder=3)
# ── Step 4: 一阶差分 ─────────────────────────────────────────────
step4 = np.diff(step3, prepend=step3[0])
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("图 8.2.3　预处理管线各步骤效果",
             fontsize=20, fontweight="bold", y=1.02)
# ── 面板 1: 原始数据 ─────────────────────────────────────────────
ax = axes[0]
ax.plot(t, raw_display, color=COLORS["gray"], lw=0.8, alpha=0.8,
        label="原始数据")
ax.plot(t, true_signal, color="black", ls="--", lw=1.2, alpha=0.6,
        label="真实信号")
ax.set_ylabel("温度(°C)", fontsize=14)
ax.set_title("原始数据", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)
# ── 面板 2: 异常修正 + 缺失填充 ──────────────────────────────────
ax = axes[1]
ax.plot(t, step2_filled, color=COLORS["blue"], lw=1.2)
ax.plot(t, true_signal, color="black", ls="--", lw=1.0, alpha=0.4)
ax.set_ylabel("温度(°C)", fontsize=14)
ax.set_title("异常修正 + 缺失填充", fontsize=15)
ax.tick_params(labelsize=13)
# ── 面板 3: Savitzky-Golay 平滑 ──────────────────────────────────
ax = axes[2]
ax.plot(t, step3, color=COLORS["green"], lw=1.5)
ax.plot(t, true_signal, color="black", ls="--", lw=1.0, alpha=0.4)
ax.set_ylabel("温度(°C)", fontsize=14)
ax.set_title("Savitzky-Golay 平滑", fontsize=15)
ax.tick_params(labelsize=13)
# ── 面板 4: 一阶差分 ─────────────────────────────────────────────
ax = axes[3]
ax.plot(t, step4, color=COLORS["purple"], lw=1.0)
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_ylabel("Δ温度", fontsize=14)
ax.set_title("一阶差分", fontsize=15)
ax.set_xlabel("时间步(分钟)", fontsize=14)
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.2)
save_fig(fig, __file__, "fig8_2_03_pipeline")
