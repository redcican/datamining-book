"""
fig8_1_02_acf_pacf.py
典型时间序列的 ACF 与 PACF（白噪声 / AR(1) / 季节性过程）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成三种典型序列 ─────────────────────────────────────────────
T = 500
nlags = 40
eps = np.random.randn(T)
# (a) 白噪声
wn = np.random.randn(T)
# (b) AR(1), φ=0.8
ar1 = np.zeros(T)
ar1[0] = eps[0]
for i in range(1, T):
    ar1[i] = 0.8 * ar1[i - 1] + eps[i]
# (c) 季节性过程 (period=50)
t_idx = np.arange(T)
seasonal = np.sin(2 * np.pi * t_idx / 50) + 0.5 * np.random.randn(T)
# ── 计算 ACF / PACF ──────────────────────────────────────────────
series_list = [wn, ar1, seasonal]
titles = ["白噪声", "AR(1), $\\varphi$=0.8", "季节性过程"]
acf_vals = [acf(s, nlags=nlags, fft=True) for s in series_list]
pacf_vals = [pacf(s, nlags=nlags, method="ywm") for s in series_list]
conf_band = 1.96 / np.sqrt(T)
# ── 辅助绘图函数 ─────────────────────────────────────────────────
def plot_correlogram(ax, values, nlags, label, color):
    """绘制 ACF / PACF 柱状图。"""
    lags = np.arange(len(values))
    markerline, stemlines, baseline = ax.stem(
        lags, values, linefmt="-", markerfmt="o", basefmt=" ")
    plt.setp(stemlines, color=color, lw=1.5)
    plt.setp(markerline, color=color, markersize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(conf_band, color=COLORS["red"], ls="--", lw=1.2, alpha=0.8)
    ax.axhline(-conf_band, color=COLORS["red"], ls="--", lw=1.2, alpha=0.8)
    ax.fill_between(lags, -conf_band, conf_band,
                     color=COLORS["red"], alpha=0.06)
    ax.set_xlim(-0.5, nlags + 0.5)
    ax.tick_params(labelsize=12)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("图 8.1.2　典型时间序列的 ACF 与 PACF",
             fontsize=20, fontweight="bold", y=1.02)
row_labels = ["ACF", "PACF"]
for col in range(3):
    # ACF（第一行）
    ax_acf = axes[0, col]
    plot_correlogram(ax_acf, acf_vals[col], nlags, row_labels[0], COLORS["blue"])
    ax_acf.set_title(titles[col], fontsize=15)
    if col == 0:
        ax_acf.set_ylabel("ACF", fontsize=14)
    # PACF（第二行）
    ax_pacf = axes[1, col]
    plot_correlogram(ax_pacf, pacf_vals[col], nlags, row_labels[1], COLORS["blue"])
    ax_pacf.set_xlabel("滞后阶数", fontsize=14)
    if col == 0:
        ax_pacf.set_ylabel("PACF", fontsize=14)
# ── 添加 95% 置信带图例（仅一次） ─────────────────────────────────
axes[0, 2].plot([], [], color=COLORS["red"], ls="--", lw=1.2,
                label="95% 置信带")
axes[0, 2].legend(fontsize=12, loc="upper right")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.5, w_pad=1.5)
save_fig(fig, __file__, "fig8_1_02_acf_pacf")
