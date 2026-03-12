"""
图 4.6.3  交叉验证超参数选择：K折验证曲线与稳定性分析
对应节次：4.6 偏差–方差分解与超参数选择
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_6_03_cv_selection.py
输出路径：public/figures/ch04/fig4_6_03_cv_selection.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

apply_style()
np.random.seed(42)

# ── 数据加载（California Housing，子集加速） ──────────────────────────────────
data = fetch_california_housing()
X_full, y_full = data.data, data.target
idx = np.random.choice(len(X_full), 3000, replace=False)
X, y = X_full[idx], y_full[idx]

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ── 1. K=5 折 CV：验证曲线（α 敏感性）────────────────────────────────────────
alphas   = np.logspace(-3, 4, 30)
kf       = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mean  = []
cv_std   = []
for a in alphas:
    scores = -cross_val_score(Ridge(alpha=a), X_s, y,
                               cv=kf, scoring='neg_root_mean_squared_error')
    cv_mean.append(scores.mean())
    cv_std.append(scores.std())
cv_mean = np.array(cv_mean)
cv_std  = np.array(cv_std)

best_alpha_idx = int(np.argmin(cv_mean))
best_alpha = alphas[best_alpha_idx]

# ── 2. K 对 CV 稳定性的影响（K=2,5,10,LOO） ──────────────────────────────────
k_vals  = [2, 5, 10, 20]
k_means = []
k_stds  = []
for k in k_vals:
    kf_k = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = -cross_val_score(Ridge(alpha=best_alpha), X_s, y,
                               cv=kf_k, scoring='neg_root_mean_squared_error')
    k_means.append(scores.mean())
    k_stds.append(scores.std() / np.sqrt(k))   # SE = std / sqrt(K)

# ── 3. RidgeCV 选出的 α vs 手动 CV ──────────────────────────────────────────
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
ridge_cv.fit(X_s, y)
auto_alpha = ridge_cv.alpha_

# ── 作图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
ax_a, ax_b, ax_c = axes

# ── 面板 (a): K=5 折 CV 验证曲线 + 误差带 ────────────────────────────────────
ax_a.semilogx(alphas, cv_mean, 'o-', color=COLORS['blue'], lw=2, ms=5, label='5折 CV RMSE（均值）')
ax_a.fill_between(alphas, cv_mean - cv_std, cv_mean + cv_std,
                   color=COLORS['blue'], alpha=0.15, label='± 1 std')

# 标注最优 α
ax_a.axvline(x=best_alpha, color='green', ls='--', lw=1.8, alpha=0.9)
ax_a.scatter([best_alpha], [cv_mean[best_alpha_idx]], color='green', s=150, zorder=5,
             label=f'最优 $\\alpha^*={best_alpha:.3f}$（手动 CV）')
ax_a.axvline(x=auto_alpha, color='orange', ls=':', lw=1.8, alpha=0.9)
ax_a.scatter([auto_alpha], [cv_mean[np.argmin(np.abs(alphas - auto_alpha))]],
             color='orange', s=150, marker='D', zorder=5, label=f'RidgeCV 选 $\\alpha={auto_alpha:.3f}$')

# 标注 1-SE 规则
se = cv_std[best_alpha_idx]
rmse_1se = cv_mean[best_alpha_idx] + se / np.sqrt(5)
ax_a.axhline(y=rmse_1se, color='red', ls='-.', lw=1, alpha=0.6, label=f'1-SE 规则（取最简单使CV≤{rmse_1se:.4f}的模型）')

ax_a.set_xlabel('正则化参数 $\\alpha$（对数刻度）', fontsize=12)
ax_a.set_ylabel('5折 CV RMSE', fontsize=12)
ax_a.set_title('(a) Ridge 正则化路径的交叉验证曲线\n（California Housing，$n=3000$，5折）',
               fontsize=13, fontweight='bold')
ax_a.legend(fontsize=8.5, loc='upper right')

# ── 面板 (b): 5折 CV 数据集划分示意图（概念图）──────────────────────────────
K = 5
colors_fold = [COLORS['blue'], COLORS['teal'], COLORS['green'], COLORS['orange'], COLORS['red']]
fold_h = 0.5
fold_gap = 0.15

for k in range(K):
    y_pos = k * (fold_h + fold_gap)
    for f in range(K):
        is_val = (f == k)
        fc = colors_fold[k] if is_val else COLORS['light']
        ec = colors_fold[k] if is_val else 'gray'
        lw_rect = 1.5 if is_val else 0.5
        rect = mpatches.FancyBboxPatch(
            (f / K, y_pos), 1 / K - 0.01, fold_h,
            boxstyle="round,pad=0.01",
            facecolor=fc, edgecolor=ec, linewidth=lw_rect
        )
        ax_b.add_patch(rect)
        if is_val:
            ax_b.text(f / K + 0.5 / K, y_pos + fold_h / 2, f'验证\n折 {f+1}',
                      ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        else:
            ax_b.text(f / K + 0.5 / K, y_pos + fold_h / 2, '训练',
                      ha='center', va='center', fontsize=7.5, color=COLORS['gray'])
    ax_b.text(-0.01, y_pos + fold_h / 2, f'实验 {k+1}',
              ha='right', va='center', fontsize=9)

ax_b.set_xlim(-0.1, 1.05)
ax_b.set_ylim(-0.1, K * (fold_h + fold_gap))
ax_b.axis('off')
ax_b.set_title(f'(b) {K} 折交叉验证数据集划分示意图\n每次用 4 折训练、1 折验证，轮换 {K} 次',
               fontsize=13, fontweight='bold')
ax_b.text(0.5, -0.08, '← 完整数据集 →', ha='center', va='top',
          fontsize=10, transform=ax_b.transAxes, style='italic')

# 添加列标注
for f in range(K):
    ax_b.text((f + 0.5) / K, K * (fold_h + fold_gap) + 0.05,
              f'折 {f+1}', ha='center', va='bottom', fontsize=9)

# ── 面板 (c): K 折数量对 CV 误差估计稳定性的影响 ───────────────────────────────
ax_c.errorbar(k_vals, k_means, yerr=k_stds, fmt='o-', color=COLORS['blue'],
              lw=2, ms=8, capsize=5, capthick=1.5, elinewidth=1.5, label='CV RMSE ± SE')

for k, m, s in zip(k_vals, k_means, k_stds):
    ax_c.annotate(f'SE={s:.4f}', xy=(k, m), xytext=(5, 5),
                  textcoords='offset points', fontsize=8.5)

ax_c.axhline(y=np.mean(k_means), color='red', ls='--', lw=1.2, alpha=0.5,
             label=f'均值={np.mean(k_means):.4f}')
ax_c.set_xlabel('$K$（折数）', fontsize=12)
ax_c.set_ylabel('CV RMSE 均值 ± 标准误（SE）', fontsize=12)
ax_c.set_title('(c) 折数 $K$ 对 CV 误差稳定性的影响\n$K$ 越大估计越稳定（SE 越小），但计算代价越高',
               fontsize=13, fontweight='bold')
ax_c.legend(fontsize=10)
ax_c.set_xticks(k_vals)
ax_c.set_xticklabels([f'K={k}' for k in k_vals], fontsize=11)

plt.tight_layout()
save_fig(fig, __file__, 'fig4_6_03_cv_selection')
