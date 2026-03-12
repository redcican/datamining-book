"""
图 4.2.2  残差诊断：异方差检验（Breusch-Pagan）与影响点气泡图
对应节次：4.2 统计推断与模型诊断（t/F 检验、残差分析）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_2_02_residual_tests.py
输出路径：public/figures/ch04/fig4_2_02_residual_tests.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
apply_style()
# --- 1. 模拟同方差与异方差数据 ---
rng = np.random.default_rng(0)
x_sim = np.linspace(0.5, 6, 200)
y_homo = 0.5 + 1.2 * x_sim + rng.normal(0, 0.8, 200)
y_hete = 0.5 + 1.2 * x_sim + rng.normal(0, 0.15 * x_sim, 200)
# --- 2. OLS 拟合两组数据，计算残差与拟合值 ---
def ols_fit(x, y):
    X_mat = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    yhat = X_mat @ beta
    resid = y - yhat
    return yhat, resid, beta
yhat_homo, resid_homo, _ = ols_fit(x_sim, y_homo)
yhat_hete, resid_hete, _ = ols_fit(x_sim, y_hete)
# --- 3. BP 检验：e² 对 ŷ 回归，BP = n * R² ~ χ²(1) ---
def bp_test(yhat, resid):
    n = len(resid)
    e2 = resid ** 2
    X_bp = np.column_stack([np.ones(n), yhat])
    beta_bp, _, _, _ = np.linalg.lstsq(X_bp, e2, rcond=None)
    e2_hat = X_bp @ beta_bp
    ss_res = np.sum((e2 - e2_hat) ** 2)
    ss_tot = np.sum((e2 - e2.mean()) ** 2)
    r2_bp = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    bp_stat = n * r2_bp
    p_val = 1 - chi2.cdf(bp_stat, df=1)
    return bp_stat, p_val, beta_bp
bp_homo, p_homo, beta_bp_homo = bp_test(yhat_homo, resid_homo)
bp_hete, p_hete, beta_bp_hete = bp_test(yhat_hete, resid_hete)
# --- 4. California Housing 前 2000 行：杠杆值、标准化残差、Cook's D ---
housing = fetch_california_housing()
X_full_data = housing.data[:2000]
y_full_data = housing.target[:2000]
scaler = StandardScaler()
X_s = scaler.fit_transform(X_full_data)
n_c, p_c = X_s.shape
X_c = np.column_stack([np.ones(n_c), X_s])
beta_c, _, _, _ = np.linalg.lstsq(X_c, y_full_data, rcond=None)
yhat_c = X_c @ beta_c
resid_c = y_full_data - yhat_c
rss_c = np.sum(resid_c ** 2)
s_c = np.sqrt(rss_c / (n_c - p_c - 1))
XtX_inv_c = np.linalg.inv(X_c.T @ X_c)
H_diag = np.einsum("ij,jk,ik->i", X_c, XtX_inv_c, X_c)
std_resid = resid_c / (s_c * np.sqrt(np.clip(1 - H_diag, 1e-10, None)))
cook_d = (resid_c ** 2 * H_diag) / ((p_c + 1) * s_c ** 2 * (1 - H_diag) ** 2)
# --- 5. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
# --- 6. 面板(a)：残差 vs 拟合值 ---
ax = axes[0]
ax.scatter(yhat_homo, resid_homo, color=COLORS["blue"], alpha=0.55, s=22,
           label="同方差", zorder=4)
ax.scatter(yhat_hete, resid_hete, color=COLORS["orange"], alpha=0.55, s=22,
           label="异方差", zorder=4)
sort_idx = np.argsort(yhat_hete)
abs_trend = np.abs(resid_hete[sort_idx])
trend_X = np.column_stack([np.ones_like(yhat_hete[sort_idx]), yhat_hete[sort_idx]])
trend_beta, _, _, _ = np.linalg.lstsq(trend_X, abs_trend, rcond=None)
trend_line = trend_X @ trend_beta
ax.plot(yhat_hete[sort_idx], trend_line, color=COLORS["orange"], lw=2.0,
        ls="--", label="异方差 $|e|$ 趋势线", zorder=5)
ax.axhline(0, color=COLORS["gray"], lw=1.2, ls="--")
ax.set_xlabel("拟合值 $\\hat{y}$", fontsize=13)
ax.set_ylabel("残差 $e$", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(a) 同方差 vs 异方差：残差 vs 拟合值\nGM2 假设要求残差带宽与 $\\hat{y}$ 无关",
            fontsize=13, pad=8)
# --- 7. 面板(b)：BP 检验原理图 ---
ax = axes[1]
e2_homo = resid_homo ** 2
e2_hete = resid_hete ** 2
ax.scatter(yhat_homo, e2_homo, color=COLORS["blue"], alpha=0.5, s=22, label="同方差 $e^2$")
ax.scatter(yhat_hete, e2_hete, color=COLORS["orange"], alpha=0.5, s=22, label="异方差 $e^2$")
xs_h = np.column_stack([np.ones_like(yhat_homo), yhat_homo])
beta_h2, _, _, _ = np.linalg.lstsq(xs_h, e2_homo, rcond=None)
xs_sorted = np.sort(yhat_homo)
ax.plot(xs_sorted, np.column_stack([np.ones_like(xs_sorted), xs_sorted]) @ beta_h2,
        color=COLORS["blue"], lw=2.0, ls="--")
xs_he = np.column_stack([np.ones_like(yhat_hete), yhat_hete])
beta_he2, _, _, _ = np.linalg.lstsq(xs_he, e2_hete, rcond=None)
xs_sorted2 = np.sort(yhat_hete)
ax.plot(xs_sorted2, np.column_stack([np.ones_like(xs_sorted2), xs_sorted2]) @ beta_he2,
        color=COLORS["orange"], lw=2.0, ls="--")
p_homo_str = f"{p_homo:.4f}" if p_homo >= 0.0001 else "<0.0001"
p_hete_str = f"{p_hete:.4f}" if p_hete >= 0.0001 else "<0.0001"
ax.text(0.03, 0.97,
        f"同方差：$BP={bp_homo:.2f}$，$p={p_homo_str}$\n异方差：$BP={bp_hete:.2f}$，$p={p_hete_str}$",
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COLORS["gray"], alpha=0.9))
ax.set_xlabel("拟合值 $\\hat{y}$", fontsize=13)
ax.set_ylabel("$e_i^2$", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(b) Breusch-Pagan 检验原理：$e_i^2$ vs $\\hat{y}$\n趋势显著→ 异方差；$BP=nR^2_{e^2}\\sim\\chi^2(1)$",
            fontsize=13, pad=8)
# --- 8. 面板(c)：杠杆值–标准化残差气泡图 ---
ax = axes[2]
bubble_size = np.clip(cook_d * 8000, 5, 400)
threshold_cook = 4 / n_c
colors_c = np.where(cook_d > threshold_cook, COLORS["red"], COLORS["blue"])
ax.scatter(H_diag, std_resid, s=bubble_size, c=colors_c, alpha=0.45, edgecolors="none")
h_thresh = 2 * (p_c + 1) / n_c
ax.axvline(h_thresh, color=COLORS["orange"], lw=1.8, ls="--",
           label=f"高杠杆阈值 $2(p+1)/n={h_thresh:.3f}$")
ax.axhline(3, color=COLORS["red"], lw=1.4, ls=":", alpha=0.8)
ax.axhline(-3, color=COLORS["red"], lw=1.4, ls=":", alpha=0.8, label="$|e^*|=3$")
ax.axhline(0, color=COLORS["gray"], lw=1.0, ls="--", alpha=0.5)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["blue"], ms=9,
           label=f"普通点（Cook's $D\\leq 4/n$）"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["red"], ms=9,
           label=f"影响点（Cook's $D>4/n$）"),
]
ax.legend(handles=legend_elements + [
    Line2D([0], [0], color=COLORS["orange"], lw=2, ls="--", label=f"高杠杆阈值"),
    Line2D([0], [0], color=COLORS["red"], lw=1.5, ls=":", label="$|e^*|=3$"),
], fontsize=10)
ax.set_xlabel("杠杆值 $h_{ii}$", fontsize=13)
ax.set_ylabel("标准化残差 $e_i^*$", fontsize=13)
ax.set_title("(c) 杠杆值–标准化残差气泡图（气泡大小$\\propto$Cook's $D$）\n高杠杆+大残差→高影响点；仅高残差→普通离群点",
            fontsize=13, pad=8)
# --- 9. 总标题与保存 ---
fig.suptitle("残差形式诊断：异方差 Breusch-Pagan 检验与影响点识别",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, __file__, "fig4_2_02_residual_tests")
