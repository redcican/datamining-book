"""
fig8_6_01_arima.py
ARIMA 建模与预测
上：原始序列 + 拟合 + 预测区间  下：残差 ACF
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
# ── 生成合成电力负荷数据 ──────────────────────────────────────────
T = 300
t = np.arange(T)
trend = 0.05 * t
seasonal = 20 * np.sin(2 * np.pi * t / 24)
noise = np.random.normal(0, 4, T)
load = 200 + trend + seasonal + noise
# ── 简化 ARIMA 拟合（AR(2) 近似）─────────────────────────────────
train_end = 260
train = load[:train_end]
test = load[train_end:]
# 一阶差分
diff = np.diff(train)
# AR(2) 系数估计（Yule-Walker 近似）
from numpy.linalg import solve
gamma = np.array([np.mean(diff[k:] * diff[:len(diff)-k]) for k in range(3)])
R = np.array([[gamma[0], gamma[1]], [gamma[1], gamma[0]]])
r = np.array([gamma[1], gamma[2]])
phi = solve(R, r)
# 拟合值
fitted = np.zeros(len(train))
fitted[:3] = train[:3]
for i in range(3, len(train)):
    fitted[i] = train[i-1] + phi[0] * (train[i-1] - train[i-2]) + \
                phi[1] * (train[i-2] - train[i-3])
# 预测
forecast = np.zeros(len(test))
last_vals = list(train[-3:])
for i in range(len(test)):
    pred = last_vals[-1] + phi[0] * (last_vals[-1] - last_vals[-2]) + \
           phi[1] * (last_vals[-2] - last_vals[-3])
    forecast[i] = pred
    last_vals.append(pred)
# 残差
residuals = train[3:] - fitted[3:]
residual_std = np.std(residuals)
# 预测区间
forecast_upper = forecast + 2 * residual_std
forecast_lower = forecast - 2 * residual_std
# 残差 ACF
n_lags = 20
acf_vals = []
res_centered = residuals - residuals.mean()
gamma0 = np.sum(res_centered ** 2) / len(res_centered)
for k in range(n_lags + 1):
    if k == 0:
        acf_vals.append(1.0)
    else:
        gk = np.sum(res_centered[k:] * res_centered[:len(res_centered)-k]) / len(res_centered)
        acf_vals.append(gk / gamma0)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                         gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("图 8.6.1　ARIMA 建模与预测",
             fontsize=20, fontweight="bold", y=1.02)
# ── 上面板：序列 + 拟合 + 预测 ────────────────────────────────────
ax = axes[0]
ax.plot(t[:train_end], train, color=COLORS["blue"], lw=1.0, alpha=0.8,
        label="训练数据")
ax.plot(t[train_end:], test, color=COLORS["blue"], lw=1.0, ls="--",
        alpha=0.6, label="真实值")
ax.plot(t[3:train_end], fitted[3:], color=COLORS["red"], lw=1.2,
        alpha=0.7, label="模型拟合")
ax.plot(t[train_end:], forecast, color=COLORS["red"], lw=2.0,
        label="预测值")
ax.fill_between(t[train_end:], forecast_lower, forecast_upper,
                color=COLORS["red"], alpha=0.15, label="95% 预测区间")
ax.axvline(train_end, color="gray", ls=":", lw=1.5, alpha=0.7)
ax.text(train_end + 2, load.max() - 5, "预测起点", fontsize=12,
        color="gray")
ax.set_ylabel("负荷 (MW)", fontsize=14)
ax.set_title("原始序列与 ARIMA 预测", fontsize=15)
ax.legend(fontsize=12, loc="upper left", ncol=2)
ax.tick_params(labelsize=13)
# ── 下面板：残差 ACF ──────────────────────────────────────────────
ax = axes[1]
lags = np.arange(n_lags + 1)
ax.bar(lags, acf_vals, color=COLORS["purple"], alpha=0.7, width=0.6)
ci = 1.96 / np.sqrt(len(residuals))
ax.axhline(ci, color=COLORS["red"], ls="--", lw=1.2, alpha=0.7,
           label="95% 置信区间")
ax.axhline(-ci, color=COLORS["red"], ls="--", lw=1.2, alpha=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("滞后阶数", fontsize=14)
ax.set_ylabel("ACF", fontsize=14)
ax.set_title("残差自相关函数", fontsize=15)
ax.legend(fontsize=12, loc="upper right")
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.5)
save_fig(fig, __file__, "fig8_6_01_arima")
