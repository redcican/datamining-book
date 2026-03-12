"""
图 4.1.3  线性回归诊断四联图
对应节次：4.1 线性回归基础（OLS、正规方程、Gauss-Markov 定理）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_1_03_diagnostics.py
输出路径：public/figures/ch04/fig4_1_03_diagnostics.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

apply_style()

C_DATA  = COLORS["blue"]
C_FIT   = COLORS["red"]
C_RESID = COLORS["orange"]
C_INFL  = COLORS["red"]
C_GRAY  = COLORS["gray"]
C_GREEN = COLORS["green"]

# ── 数据：California Housing（取前 500 条简化） ───────────────────────────────
housing = fetch_california_housing()
X_raw = housing.data[:500, :4]   # 取前4个特征
y_raw = housing.target[:500]
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)
X_full = np.column_stack([np.ones(len(X_sc)), X_sc])
beta = np.linalg.lstsq(X_full, y_raw, rcond=None)[0]
y_hat = X_full @ beta
residuals = y_raw - y_hat
n, p = X_full.shape

# 标准化残差
s = np.sqrt(np.sum(residuals**2) / (n - p))
# Hat matrix diagonal (leverage)
XtX_inv = np.linalg.inv(X_full.T @ X_full)
h = np.array([X_full[i] @ XtX_inv @ X_full[i] for i in range(n)])
std_resid = residuals / (s * np.sqrt(1 - h))
# Cook's Distance
cook_d = std_resid**2 * h / (p * (1 - h))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── 面板(a): 残差 vs 拟合值 ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(y_hat, residuals, color=C_DATA, s=18, alpha=0.6, zorder=4)
ax.axhline(0, color=C_FIT, lw=1.8, ls="--")
# LOWESS 趋势线（用多项式近似）
from numpy.polynomial.polynomial import polyfit as npfit
sort_idx = np.argsort(y_hat)
trend_coef = np.polyfit(y_hat, residuals, 3)
trend_y = np.polyval(trend_coef, np.sort(y_hat))
ax.plot(np.sort(y_hat), trend_y, color=C_RESID, lw=2.0, label="趋势（LOWESS 近似）")
ax.set_xlabel("拟合值 $\\hat{y}$", fontsize=13)
ax.set_ylabel("残差 $e = y - \\hat{y}$", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(a) 残差 vs 拟合值\n理想：残差随机散布在 0 线两侧（无模式）",
             fontsize=13, pad=6)

# ── 面板(b): 正态 Q-Q 图 ─────────────────────────────────────────────────────
ax = axes[0, 1]
(osm, osr), (slope, intercept, r) = stats.probplot(std_resid, dist="norm")
ax.scatter(osm, osr, color=C_DATA, s=18, alpha=0.6, zorder=4)
x_qq = np.array([osm.min(), osm.max()])
ax.plot(x_qq, slope * x_qq + intercept, color=C_FIT, lw=2.0, ls="--",
        label=f"理论正态线 ($r={r:.3f}$)")
ax.set_xlabel("理论分位数", fontsize=13)
ax.set_ylabel("标准化残差分位数", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(b) 正态 Q-Q 图\n点越接近对角线，残差越接近正态分布",
             fontsize=13, pad=6)

# ── 面板(c): Scale-Location 图 ───────────────────────────────────────────────
ax = axes[1, 0]
sqrt_abs_resid = np.sqrt(np.abs(std_resid))
ax.scatter(y_hat, sqrt_abs_resid, color=C_DATA, s=18, alpha=0.6, zorder=4)
trend_coef2 = np.polyfit(y_hat, sqrt_abs_resid, 3)
trend_y2 = np.polyval(trend_coef2, np.sort(y_hat))
ax.plot(np.sort(y_hat), trend_y2, color=C_RESID, lw=2.0, label="趋势线")
ax.set_xlabel("拟合值 $\\hat{y}$", fontsize=13)
ax.set_ylabel("$\\sqrt{|e_i^*|}$（标准化残差绝对值）", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(c) Scale-Location 图（等方差检验）\n理想：水平趋势线（同方差性成立）",
             fontsize=13, pad=6)

# ── 面板(d): Cook's Distance ─────────────────────────────────────────────────
ax = axes[1, 1]
idx_arr = np.arange(n)
ax.bar(idx_arr, cook_d, color=C_DATA, alpha=0.6, width=1.0)
thresh = 4 / n
ax.axhline(thresh, color=C_FIT, lw=1.8, ls="--",
           label=f"影响点阈值 $4/n={thresh:.3f}$")
high_infl = idx_arr[cook_d > thresh]
ax.bar(high_infl, cook_d[high_infl], color=C_INFL, alpha=0.9,
       width=1.0, label=f"高影响点（{len(high_infl)} 个）")
ax.set_xlabel("样本编号", fontsize=13)
ax.set_ylabel("Cook's Distance", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(d) Cook's Distance（影响点检测）\n"
             "超过 $4/n$ 阈值的点对回归系数有显著影响",
             fontsize=13, pad=6)

fig.suptitle("线性回归诊断四联图（California Housing 子集，$n=500$）\n"
             "用于检验 OLS 假设：正态性 · 同方差性 · 独立性 · 无强影响点",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, __file__, "fig4_1_03_diagnostics")
