"""
fig8_1_03_ts_visualization.py
时间序列的多维度可视化（时间图 / 季节子图 / ACF / 滞后散点图）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from statsmodels.tsa.stattools import acf
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成三年合成日温度数据（同 Script 1） ────────────────────────
n_days = 1095
dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
t = np.arange(n_days)
trend = 0.1 * t / n_days
seasonal = 15 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
residual = np.random.normal(0, 3, n_days)
observed = 12 + trend + seasonal + residual
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("图 8.1.3　时间序列的多维度可视化",
             fontsize=20, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 时间图
# ══════════════════════════════════════════════════════════════════
ax = axes[0, 0]
ax.plot(dates, observed, color=COLORS["blue"], lw=0.9, alpha=0.85)
ax.set_xlabel("日期", fontsize=14)
ax.set_ylabel("温度 (°C)", fontsize=14)
ax.set_title("(a) 时间图", fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.tick_params(labelsize=12)
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha("right")
# ══════════════════════════════════════════════════════════════════
# (b) 季节子图：三年叠加在 1–365 天轴上
# ══════════════════════════════════════════════════════════════════
ax = axes[0, 1]
df = pd.DataFrame({"date": dates, "temp": observed})
df["year"] = df["date"].dt.year
df["dayofyear"] = df["date"].dt.dayofyear
year_colors = [COLORS["blue"], COLORS["red"], COLORS["green"]]
for idx, (year, grp) in enumerate(df.groupby("year")):
    ax.plot(grp["dayofyear"].values, grp["temp"].values,
            color=year_colors[idx], lw=1.0, alpha=0.8, label=str(year))
# 月刻度
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = [f"{m}月" for m in range(1, 13)]
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels, fontsize=11)
ax.set_xlabel("月", fontsize=14)
ax.set_ylabel("温度 (°C)", fontsize=14)
ax.set_title("(b) 季节子图", fontsize=15)
ax.legend(fontsize=12, loc="lower right", title="年份", title_fontsize=13)
ax.tick_params(labelsize=12)
# ══════════════════════════════════════════════════════════════════
# (c) ACF 图
# ══════════════════════════════════════════════════════════════════
ax = axes[1, 0]
nlags = 60
acf_values = acf(observed, nlags=nlags, fft=True)
lags = np.arange(nlags + 1)
conf_band = 1.96 / np.sqrt(n_days)
markerline, stemlines, baseline = ax.stem(
    lags, acf_values, linefmt="-", markerfmt="o", basefmt=" ")
plt.setp(stemlines, color=COLORS["blue"], lw=1.5)
plt.setp(markerline, color=COLORS["blue"], markersize=3)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(conf_band, color=COLORS["red"], ls="--", lw=1.2, alpha=0.8,
           label="95% 置信带")
ax.axhline(-conf_band, color=COLORS["red"], ls="--", lw=1.2, alpha=0.8)
ax.fill_between(lags, -conf_band, conf_band, color=COLORS["red"], alpha=0.06)
ax.set_xlim(-0.5, nlags + 0.5)
ax.set_xlabel("滞后阶数", fontsize=14)
ax.set_ylabel("ACF", fontsize=14)
ax.set_title("(c) ACF 图", fontsize=15)
ax.legend(fontsize=12, loc="upper right")
ax.tick_params(labelsize=12)
# ══════════════════════════════════════════════════════════════════
# (d) 滞后散点图 x_t vs x_{t-1}
# ══════════════════════════════════════════════════════════════════
ax = axes[1, 1]
x_lag0 = observed[1:]
x_lag1 = observed[:-1]
ax.scatter(x_lag1, x_lag0, s=8, alpha=0.4, color=COLORS["blue"], edgecolors="none")
# 趋势线
coeffs = np.polyfit(x_lag1, x_lag0, 1)
x_line = np.linspace(x_lag1.min(), x_lag1.max(), 100)
y_line = np.polyval(coeffs, x_line)
ax.plot(x_line, y_line, color=COLORS["red"], lw=2.0, ls="--", label="趋势线")
# 相关系数
r = np.corrcoef(x_lag1, x_lag0)[0, 1]
ax.annotate(f"$r$ = {r:.3f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=15, fontweight="bold", color=COLORS["red"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["red"], alpha=0.9))
ax.set_xlabel("$x_{t-1}$", fontsize=14)
ax.set_ylabel("$x_t$", fontsize=14)
ax.set_title("(d) 滞后散点图", fontsize=15)
ax.legend(fontsize=12, loc="lower right")
ax.tick_params(labelsize=12)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=2.0, w_pad=2.0)
save_fig(fig, __file__, "fig8_1_03_ts_visualization")
