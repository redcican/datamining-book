"""
fig8_7_02_detection_methods.py
三种异常检测方法对比：原始序列 + Shewhart + 预测残差
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
# ── 生成含异常的时间序列 ──────────────────────────────────────────
T = 500
t = np.arange(T)
seasonal = 20 * np.sin(2 * np.pi * t / 50)
noise = np.random.normal(0, 3, T)
signal = 50 + seasonal + noise
# 注入异常
# 点异常
spike_locs = [80, 250, 400]
for loc in spike_locs:
    signal[loc] += 30
# 上下文异常（应该低的地方偏高）
signal[150:165] += 18
# 缓慢漂移
signal[300:350] += np.linspace(0, 15, 50)
# ── Shewhart 控制图 ──────────────────────────────────────────────
mu_base = signal[:50].mean()
sigma_base = signal[:50].std()
ucl = mu_base + 3 * sigma_base
lcl = mu_base - 3 * sigma_base
shewhart_alerts = signal > ucl
# ── 预测残差（滑动窗口均值）────────────────────────────────────────
W = 25
predictions = np.convolve(signal, np.ones(W) / W, mode='same')
residuals = np.abs(signal - predictions)
res_threshold = np.percentile(residuals[:100], 97)
pred_alerts = residuals > res_threshold * 1.5
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("图 8.7.2　异常检测方法对比",
             fontsize=20, fontweight="bold", y=1.02)
# ── 面板 1：原始序列 ──────────────────────────────────────────────
ax = axes[0]
ax.plot(t, signal, color=COLORS["blue"], lw=0.8, alpha=0.8)
# 标注异常区间
for loc in spike_locs:
    ax.plot(loc, signal[loc], 'o', color=COLORS["red"], markersize=6,
            zorder=5)
ax.axvspan(150, 165, alpha=0.15, color=COLORS["orange"])
ax.axvspan(300, 350, alpha=0.15, color=COLORS["purple"])
ax.set_ylabel("值", fontsize=14)
ax.set_title("原始时间序列（含注入异常）", fontsize=15)
ax.tick_params(labelsize=13)
# 图例
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_el = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["red"],
           markersize=8, label="点异常"),
    Patch(facecolor=COLORS["orange"], alpha=0.3, label="上下文异常"),
    Patch(facecolor=COLORS["purple"], alpha=0.3, label="缓慢漂移"),
]
ax.legend(handles=legend_el, fontsize=11, loc="upper right")
# ── 面板 2：Shewhart 控制图 ───────────────────────────────────────
ax = axes[1]
ax.plot(t, signal, color=COLORS["blue"], lw=0.8, alpha=0.6)
ax.axhline(ucl, color=COLORS["red"], ls="--", lw=1.5,
           label=f"UCL = {ucl:.1f}")
ax.axhline(lcl, color=COLORS["red"], ls="--", lw=1.5)
ax.axhline(mu_base, color="gray", ls=":", lw=1.0)
# 标注检测到的点
alert_idx = np.where(shewhart_alerts)[0]
ax.scatter(alert_idx, signal[alert_idx], color=COLORS["red"], s=15,
           zorder=5, label=f"告警 ({len(alert_idx)} 个)")
ax.set_ylabel("值", fontsize=14)
ax.set_title("Shewhart 控制图（3σ 规则）", fontsize=15)
ax.legend(fontsize=11, loc="upper right")
ax.tick_params(labelsize=13)
# ── 面板 3：预测残差 ──────────────────────────────────────────────
ax = axes[2]
ax.plot(t, residuals, color=COLORS["purple"], lw=0.8, alpha=0.8)
ax.axhline(res_threshold * 1.5, color=COLORS["red"], ls="--", lw=1.5,
           label=f"自适应阈值")
ax.fill_between(t, 0, res_threshold * 1.5, alpha=0.05, color=COLORS["green"])
alert_idx_p = np.where(pred_alerts)[0]
ax.scatter(alert_idx_p, residuals[alert_idx_p], color=COLORS["red"], s=10,
           zorder=5, label=f"告警 ({len(alert_idx_p)} 个)")
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("|残差|", fontsize=14)
ax.set_title("基于预测的异常评分", fontsize=15)
ax.legend(fontsize=11, loc="upper right")
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.2)
save_fig(fig, __file__, "fig8_7_02_detection_methods")
