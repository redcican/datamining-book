"""
fig8_2_01_interpolation.py
插值方法效果对比：随机缺失 vs 连续缺失
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成真实信号 ──────────────────────────────────────────────────
n = 200
t = np.arange(n)
true_signal = 5 * np.sin(2 * np.pi * t / 60) + 0.02 * t
# ── 辅助：三种插值方法 ───────────────────────────────────────────
def interpolate_methods(t_full, signal_with_nan):
    """返回线性、三次样条、前向填充三种插值结果。"""
    valid = ~np.isnan(signal_with_nan)
    t_valid = t_full[valid]
    y_valid = signal_with_nan[valid]
    # 线性插值
    linear = np.interp(t_full, t_valid, y_valid)
    # 三次样条插值
    cs = CubicSpline(t_valid, y_valid, extrapolate=True)
    cubic = cs(t_full)
    # 前向填充
    ffill = signal_with_nan.copy()
    for i in range(1, len(ffill)):
        if np.isnan(ffill[i]):
            ffill[i] = ffill[i - 1]
    return linear, cubic, ffill
# ── (a) 随机缺失 10% ─────────────────────────────────────────────
random_missing = np.random.choice(n, size=int(n * 0.1), replace=False)
signal_rand = true_signal.copy()
signal_rand[random_missing] = np.nan
lin_r, cub_r, ff_r = interpolate_methods(t, signal_rand)
# ── (b) 连续缺失 30 点 ──────────────────────────────────────────
block_start, block_end = 80, 110
signal_block = true_signal.copy()
signal_block[block_start:block_end] = np.nan
lin_b, cub_b, ff_b = interpolate_methods(t, signal_block)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 8.2.1　插值方法效果对比",
             fontsize=20, fontweight="bold", y=1.02)
# ── 面板 (a) 随机缺失 ────────────────────────────────────────────
ax = axes[0]
ax.plot(t, true_signal, color="gray", ls="--", lw=1.2, alpha=0.6,
        label="真实信号")
obs_mask = ~np.isnan(signal_rand)
ax.scatter(t[obs_mask], signal_rand[obs_mask], s=8, color="black",
           zorder=5, label="观测点")
ax.plot(t, lin_r, color=COLORS["blue"], lw=1.8, label="线性插值")
ax.plot(t, cub_r, color=COLORS["red"], lw=1.8, label="三次样条插值")
ax.step(t, ff_r, where="post", color=COLORS["green"], lw=1.8,
        label="前向填充")
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("信号值", fontsize=14)
ax.set_title("(a) 随机缺失", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)
# ── 面板 (b) 连续缺失 ────────────────────────────────────────────
ax = axes[1]
ax.axvspan(block_start, block_end, alpha=0.15, color="gray",
           label="缺失区间")
ax.plot(t, true_signal, color="gray", ls="--", lw=1.2, alpha=0.6,
        label="真实信号")
obs_mask_b = ~np.isnan(signal_block)
ax.scatter(t[obs_mask_b], signal_block[obs_mask_b], s=8, color="black",
           zorder=5, label="观测点")
ax.plot(t, lin_b, color=COLORS["blue"], lw=1.8, label="线性插值")
ax.plot(t, cub_b, color=COLORS["red"], lw=1.8, label="三次样条插值")
ax.step(t, ff_b, where="post", color=COLORS["green"], lw=1.8,
        label="前向填充")
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("信号值", fontsize=14)
ax.set_title("(b) 连续缺失", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_2_01_interpolation")
