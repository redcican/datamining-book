"""
图 2.4.2  包裹式与嵌入式特征选择：RFE 学习曲线、Lasso 正则化路径与方法对比
对应节次：2.4 特征选择与特征工程
运行方式：python code/ch02/fig2_4_02_wrapper_embedded.py
输出路径：public/figures/ch02/fig2_4_02_wrapper_embedded.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LassoCV, lasso_path
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

apply_style()

C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_GREEN  = "#16a34a"
C_ORANGE = "#ea580c"
C_GRAY   = "#94a3b8"
C_DARK   = "#1e293b"

rng_seed = 2024

# ── Dataset: 15 features, 5 informative ────────────────────────────────────
n, d, n_info = 300, 15, 5
X_raw, y, coef_true = make_regression(
    n_samples=n, n_features=d, n_informative=n_info,
    noise=25, coef=True, random_state=rng_seed,
)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)
y_sc = (y - y.mean()) / y.std()

feat_labels = [f"x{i+1}" for i in range(d)]
# True informative feature indices (by |true_coef| magnitude)
true_idx = set(np.argsort(np.abs(coef_true))[::-1][:n_info].tolist())

# ── (a) RFECV ──────────────────────────────────────────────────────────────
rfecv = RFECV(
    LinearRegression(), step=1,
    cv=KFold(5, shuffle=True, random_state=rng_seed),
    scoring="r2", min_features_to_select=1,
)
rfecv.fit(X_sc, y_sc)
rfe_support = rfecv.support_
try:
    cv_mean = rfecv.cv_results_["mean_test_score"]
    cv_std  = rfecv.cv_results_["std_test_score"]
except (AttributeError, KeyError):
    cv_mean = rfecv.grid_scores_
    cv_std  = np.zeros_like(cv_mean)
opt_k = rfecv.n_features_

# ── (b) Lasso path ─────────────────────────────────────────────────────────
alphas_path, coefs_path, _ = lasso_path(X_sc, y_sc, n_alphas=80, eps=5e-4)
log_alphas = np.log10(alphas_path)

lcv = LassoCV(cv=5, random_state=rng_seed, max_iter=5000).fit(X_sc, y_sc)
alpha_cv = lcv.alpha_
lasso_support = np.abs(lcv.coef_) > 1e-6

# ── (c) Comparison heatmap ─────────────────────────────────────────────────
mi = mutual_info_regression(X_sc, y_sc, random_state=rng_seed)
mi_support = mi >= np.sort(mi)[::-1][n_info - 1]

gt_support = np.zeros(d, dtype=int)
for i in true_idx:
    gt_support[i] = 1

selections = np.array([
    gt_support,
    mi_support.astype(int),
    rfe_support.astype(int),
    lasso_support.astype(int),
])
method_labels = ["真实信号特征", "互信息过滤", "RFE（包裹式）", "Lasso（嵌入式）"]

# ── Layout 1×3 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.subplots_adjust(wspace=0.40)

# ── Panel (a): RFECV curve ──────────────────────────────────────────────────
ax = axes[0]
ks = np.arange(1, d + 1)
ax.plot(ks, cv_mean, color=C_BLUE, lw=2.5,
        marker="o", markersize=5, markerfacecolor="white", markeredgewidth=2)
ax.fill_between(ks, cv_mean - cv_std, cv_mean + cv_std,
                alpha=0.18, color=C_BLUE)
ax.axvline(opt_k, color=C_RED, lw=1.8, ls="--", alpha=0.85)
ax.text(opt_k + 0.25, cv_mean.min() + 0.02,
        f"最优 K={opt_k}", fontsize=12.5, color=C_RED)

ax.set_xlabel("保留特征数 K", fontsize=12)
ax.set_ylabel("5 折交叉验证 R²", fontsize=12)
ax.set_title("(a) 递归特征消除（RFECV）", fontsize=13, pad=6)
ax.set_xlim(0.5, d + 0.5)
ax.tick_params(labelsize=10)

# ── Panel (b): Lasso path ───────────────────────────────────────────────────
ax = axes[1]
for i in range(d):
    is_sig = i in true_idx
    ax.plot(log_alphas, coefs_path[i],
            color=C_BLUE if is_sig else C_GRAY,
            lw=2.0 if is_sig else 0.9,
            alpha=0.9 if is_sig else 0.40)

log_a_cv = np.log10(alpha_cv)
ax.axvline(log_a_cv, color=C_RED, lw=1.8, ls="--", alpha=0.85)
ax.text(log_a_cv + 0.05, ax.get_ylim()[1] * 0.88,
        "CV 最优\nalpha", fontsize=12.5, color=C_RED, va="top")

p_sig   = mpatches.Patch(color=C_BLUE, label="信号特征（真实非零）")
p_noise = mpatches.Patch(color=C_GRAY, label="噪声特征")
ax.legend(handles=[p_sig, p_noise], fontsize=12)
ax.set_xlabel("log₁₀(α)（正则化强度）", fontsize=12)
ax.set_ylabel("系数值", fontsize=12)
ax.set_title("(b) Lasso 正则化路径", fontsize=13, pad=6)
ax.tick_params(labelsize=10)

# ── Panel (c): Method comparison heatmap ────────────────────────────────────
ax = axes[2]
im = ax.imshow(selections, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
for i in range(4):
    for j in range(d):
        v = selections[i, j]
        symbol = "✓" if v else "—"
        color = "white" if v else C_GRAY
        ax.text(j, i, symbol, ha="center", va="center",
                fontsize=12, color=color, fontweight="bold")

ax.set_xticks(range(d))
ax.set_xticklabels(feat_labels, fontsize=12)
ax.set_yticks(range(4))
ax.set_yticklabels(method_labels, fontsize=12.5)
ax.set_title("(c) 三类方法选择结果对比", fontsize=13, pad=6)
# Separator below ground truth row
ax.axhline(0.5, color=C_RED, lw=1.5, ls="--", alpha=0.5)
ax.tick_params(labelsize=9)

fig.suptitle("包裹式与嵌入式特征选择：RFE 学习曲线与 Lasso 正则化路径", fontsize=14, y=1.02)
fig.text(
    0.5, -0.05,
    "合成回归数据（n=300，15 个特征，5 个真实信号特征）。"
    "(a) RFECV：CV R² 随 K 变化，选 CV 性能饱和时的最小 K。"
    "(b) Lasso 路径：α 增大时系数依次收缩至零，噪声特征（灰）先于信号特征（蓝）归零。"
    "(c) 三类方法与真实特征集的选择结果对比（蓝=选中，灰=排除）。",
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_4_02_wrapper_embedded")
