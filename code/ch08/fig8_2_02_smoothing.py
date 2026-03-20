"""
fig8_2_02_smoothing.py
平滑方法效果对比：SMA / EWMA / Savitzky-Golay
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成信号 ──────────────────────────────────────────────────────
n = 300
t = np.arange(n)
true_signal = 5 * np.sin(2 * np.pi * t / 80) + 0.01 * t
noise = np.random.normal(0, 1.5, n)
noisy = true_signal + noise
# ── 平滑方法 ──────────────────────────────────────────────────────
# 简单移动平均 SMA(w=15)
w_sma = 15
sma = np.convolve(noisy, np.ones(w_sma) / w_sma, mode="same")
# 边界修正：卷积 same 模式在两端使用零填充，重新计算边界值
for i in range(w_sma // 2):
    left = noisy[:i + w_sma // 2 + 1]
    sma[i] = left.mean()
    right = noisy[n - w_sma // 2 + i:]
    sma[n - w_sma // 2 + i] = right.mean()
# 指数加权移动平均 EWMA(α=0.1)
alpha = 0.1
ewma = np.zeros(n)
ewma[0] = noisy[0]
for i in range(1, n):
    ewma[i] = alpha * noisy[i] + (1 - alpha) * ewma[i - 1]
# Savitzky-Golay 滤波 (w=31, p=3)
sg = savgol_filter(noisy, window_length=31, polyorder=3)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("图 8.2.2　平滑方法效果对比",
             fontsize=20, fontweight="bold", y=1.02)
ax.plot(t, noisy, color=COLORS["light"], alpha=0.5, lw=1.0,
        label="含噪信号")
ax.plot(t, true_signal, color="black", ls="--", lw=1.5,
        label="真实信号")
ax.plot(t, sma, color=COLORS["blue"], lw=2.0, label="SMA(w=15)")
ax.plot(t, ewma, color=COLORS["red"], lw=2.0, label="EWMA(α=0.1)")
ax.plot(t, sg, color=COLORS["green"], lw=2.0, label="Savitzky-Golay")
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("信号值", fontsize=14)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_2_02_smoothing")
