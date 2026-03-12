"""
图 4.2.1  系数 t 检验、整体 F 检验与置信区间 Forest Plot
对应节次：4.2 统计推断与模型诊断（t/F 检验、残差分析）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_2_01_t_f_tests.py
输出路径：public/figures/ch04/fig4_2_01_t_f_tests.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
apply_style()
# --- 1. 数据准备与 OLS 拟合 ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
feat_names_cn = ["收入中位数", "房龄", "平均房间数", "平均卧室数",
                 "人口数量", "平均入住人数", "纬度", "经度"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
n_tr = len(X_tr_s)
p = X_tr_s.shape[1]
X_full = np.column_stack([np.ones(n_tr), X_tr_s])
beta, _, _, _ = np.linalg.lstsq(X_full, y_tr, rcond=None)
y_hat_tr = X_full @ beta
rss = np.sum((y_tr - y_hat_tr) ** 2)
df_res = n_tr - p - 1
mse_val = rss / df_res
XtX_inv = np.linalg.inv(X_full.T @ X_full)
se = np.sqrt(mse_val * np.diag(XtX_inv)[1:])
t_stats = beta[1:] / se
p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_res))
ci_low = beta[1:] - stats.t.ppf(0.975, df=df_res) * se
ci_high = beta[1:] + stats.t.ppf(0.975, df=df_res) * se
ss_tot = np.sum((y_tr - y_tr.mean()) ** 2)
ss_reg = ss_tot - rss
msr = ss_reg / p
mse_f = rss / df_res
f_stat = msr / mse_f
t_crit = stats.t.ppf(0.975, df=df_res)
f_crit = stats.f.ppf(0.95, dfn=p, dfd=df_res)
# --- 2. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
# --- 3. 面板(a)：t 分布与拒绝域 ---
ax = axes[0]
t_medinc = t_stats[0]
x_t = np.linspace(-15, 15, 1000)
y_t = stats.t.pdf(x_t, df=df_res)
ax.plot(x_t, y_t, color=COLORS["blue"], lw=2.2, label=f"$t({df_res})$ 分布")
mask_left = x_t <= -t_crit
mask_right = x_t >= t_crit
ax.fill_between(x_t, y_t, where=mask_left, color=COLORS["red"], alpha=0.35, label=f"拒绝域 $|t|>t_{{0.025}}={t_crit:.2f}$")
ax.fill_between(x_t, y_t, where=mask_right, color=COLORS["red"], alpha=0.35)
ax.annotate(
    f"MedInc $t={t_medinc:.1f}$\n(极大，超出图范围)",
    xy=(14.5, stats.t.pdf(14.5, df=df_res) + 0.01),
    xytext=(8, 0.25),
    arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.8),
    fontsize=11, color=COLORS["orange"],
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["orange"], alpha=0.9)
)
ax.axvline(14.5, color=COLORS["orange"], lw=1.5, ls="--")
ax.set_xlim(-15, 15)
ax.set_xlabel("$t$ 统计量", fontsize=13)
ax.set_ylabel("概率密度", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(a) 单系数 $t$ 检验：$H_0: \\beta_j=0$\n$t_j = \\hat{\\beta}_j/\\mathrm{se}(\\hat{\\beta}_j) \\sim t(n-p-1)$",
            fontsize=13, pad=8)
# --- 4. 面板(b)：F 分布与拒绝域 ---
ax = axes[1]
x_f = np.linspace(0, f_crit * 3, 1000)
y_f = stats.f.pdf(x_f, dfn=p, dfd=df_res)
ax.plot(x_f, y_f, color=COLORS["blue"], lw=2.2, label=f"$F({p},{df_res})$ 分布")
mask_f = x_f >= f_crit
ax.fill_between(x_f, y_f, where=mask_f, color=COLORS["red"], alpha=0.35, label=f"拒绝域 $F>F_{{0.05}}={f_crit:.2f}$")
ax.axvline(f_crit, color=COLORS["red"], lw=1.5, ls="--")
ax.text(0.97, 0.82,
        f"观测 $F={f_stat:.0f}$\n$p\\approx 0$（远超图范围）",
        transform=ax.transAxes, fontsize=12, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COLORS["orange"], alpha=0.92))
ax.set_xlabel("$F$ 统计量", fontsize=13)
ax.set_ylabel("概率密度", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(b) 整体显著性 $F$ 检验：$H_0: \\beta_1=\\cdots=\\beta_p=0$\n$F = \\mathrm{MSR}/\\mathrm{MSE} \\sim F(p, n-p-1)$",
            fontsize=13, pad=8)
# --- 5. 面板(c)：Forest Plot ---
ax = axes[2]
sorted_idx = np.argsort(np.abs(beta[1:]))
y_pos = np.arange(p)
for i, idx in enumerate(sorted_idx):
    color = COLORS["green"] if beta[1:][idx] > 0 else COLORS["red"]
    ax.plot([ci_low[idx], ci_high[idx]], [i, i], color=color, lw=2.5, solid_capstyle="round")
    ax.plot(beta[1:][idx], i, "o", color=color, ms=8, zorder=5)
    pv = p_vals[idx]
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    ci_right = ci_high[idx]
    ax.text(ci_right + 0.01, i, f" {sig}", va="center", fontsize=12,
            color=COLORS["red"] if pv < 0.05 else COLORS["gray"])
ax.axvline(0, color=COLORS["gray"], lw=1.4, ls="--", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_names_cn[i] for i in sorted_idx], fontsize=12)
ax.set_xlabel("回归系数 $\\hat{\\beta}_j$ 及 95% 置信区间", fontsize=13)
ax.set_title("(c) 系数置信区间图（Forest Plot）\nCI 不含 0 → 拒绝 $H_0: \\beta_j=0$；*** $p<0.001$",
            fontsize=13, pad=8)
# --- 6. 总标题与保存 ---
fig.suptitle("$t$ 检验与 $F$ 检验：回归系数的假设检验与置信区间（California Housing）",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, __file__, "fig4_2_01_t_f_tests")
