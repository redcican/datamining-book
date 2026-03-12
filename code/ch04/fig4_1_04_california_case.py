"""
图 4.1.4  案例 4.1：California Housing OLS 基线建模分析
对应节次：4.1 线性回归基础（OLS、正规方程、Gauss-Markov 定理）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_1_04_california_case.py
输出路径：public/figures/ch04/fig4_1_04_california_case.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

apply_style()

C_DATA  = COLORS["blue"]
C_FIT   = COLORS["red"]
C_COEF  = COLORS["teal"]
C_GRAY  = COLORS["gray"]
C_PURP  = COLORS["purple"]
C_GREEN = COLORS["green"]

# ── 数据准备 ─────────────────────────────────────────────────────────────────
housing = fetch_california_housing()
X, y = housing.data, housing.target
feat_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
              "Population", "AveOccup", "Latitude", "Longitude"]
feat_names_cn = ["收入中位数", "房龄", "平均房间数", "平均卧室数",
                 "人口数量", "平均入住人数", "纬度", "经度"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

lr = LinearRegression()
lr.fit(X_tr_s, y_tr)
y_pred = lr.predict(X_te_s)
r2   = r2_score(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
residuals = y_te - y_pred

# t 统计量（用于系数显著性）
X_full = np.column_stack([np.ones(len(X_tr_s)), X_tr_s])
beta_full = np.linalg.lstsq(X_full, y_tr, rcond=None)[0]
n_tr, p_full = X_full.shape
mse = np.sum((y_tr - X_full @ beta_full)**2) / (n_tr - p_full)
XtX_inv = np.linalg.inv(X_full.T @ X_full)
se = np.sqrt(mse * np.diag(XtX_inv)[1:])   # 去掉截距
t_stat = lr.coef_ / se
p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_tr - p_full))

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# ── 面板(a): 实际值 vs 预测值 ─────────────────────────────────────────────────
ax = axes[0]
ax.scatter(y_te, y_pred, alpha=0.25, s=12, color=C_DATA, zorder=4)
lim = [y_te.min() - 0.1, y_te.max() + 0.1]
ax.plot(lim, lim, color=C_FIT, lw=2.0, ls="--", label="完美预测线 $y=\\hat{y}$")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("实际房价（×10万美元）", fontsize=13)
ax.set_ylabel("预测房价（×10万美元）", fontsize=13)
ax.legend(fontsize=11)
ax.set_title(f"(a) 实际值 vs 预测值\n$R^2={r2:.3f}$，$\\mathrm{{RMSE}}={rmse:.3f}$（测试集）",
             fontsize=13, pad=8)

# ── 面板(b): 标准化系数与显著性 ──────────────────────────────────────────────
ax = axes[1]
coef = lr.coef_
sorted_idx = np.argsort(coef)
colors_bar = [C_GREEN if c > 0 else C_FIT for c in coef[sorted_idx]]
bars = ax.barh(np.arange(len(coef)), coef[sorted_idx],
               color=colors_bar, alpha=0.82)
ax.set_yticks(np.arange(len(coef)))
ax.set_yticklabels([feat_names_cn[i] for i in sorted_idx], fontsize=12)
ax.axvline(0, color=C_GRAY, lw=1.2, ls="--")
# 标注 p 值
for i, (bar, idx) in enumerate(zip(bars, sorted_idx)):
    pv = p_vals[idx]
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    xpos = bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.06)
    ax.text(xpos, i, sig, va="center", fontsize=11,
            color=C_FIT if pv < 0.05 else C_GRAY)
ax.set_xlabel("标准化回归系数 $\\hat{\\beta}_j$（输入已标准化）", fontsize=13)
ax.set_title("(b) OLS 标准化系数与显著性\n"
             "*** p<0.001  ** p<0.01  * p<0.05  ns=不显著",
             fontsize=13, pad=8)

# ── 面板(c): 残差分布 ─────────────────────────────────────────────────────────
ax = axes[2]
ax.hist(residuals, bins=60, color=C_DATA, alpha=0.75, density=True,
        edgecolor="white", lw=0.4, label="残差频率分布")
x_norm = np.linspace(residuals.min(), residuals.max(), 300)
norm_pdf = stats.norm.pdf(x_norm, loc=residuals.mean(), scale=residuals.std())
ax.plot(x_norm, norm_pdf, color=C_FIT, lw=2.2, label="正态分布参考")
ax.axvline(0, color=C_GRAY, ls="--", lw=1.5)
# 正态性检验
stat_sw, p_sw = stats.shapiro(residuals[:500] if len(residuals)>500 else residuals)
ax.text(0.97, 0.96,
        f"Shapiro-Wilk\n$W={stat_sw:.3f}$，$p={p_sw:.4f}$",
        transform=ax.transAxes, fontsize=11, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRAY, alpha=0.9))
ax.set_xlabel("残差 $e = y - \\hat{y}$", fontsize=13)
ax.set_ylabel("密度", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(c) 残差分布（正态性检验）\n$e \\approx \\mathcal{N}(0,\\sigma^2)$ 是 OLS 推断有效的前提",
             fontsize=13, pad=8)

fig.suptitle("案例 4.1：California Housing OLS 基线建模（$n_{\\mathrm{train}}=16512$，$d=8$）",
             fontsize=14, fontweight="bold", y=1.01)
save_fig(fig, __file__, "fig4_1_04_california_case")
