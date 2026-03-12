"""
图 2.6.2  四种填补方法对双峰分布的形态保真度对比
对应节次：2.6 缺失值处理策略
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch02/fig2_6_02_imputation_comparison.py
输出路径：public/figures/ch02/fig2_6_02_imputation_comparison.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # 启用实验性 API（当前 sklearn 版本要求）
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

apply_style()

# --- 1. 生成双峰数据与辅助变量 ---
rng = np.random.default_rng(2024)
n = 600
# 双峰分布：N(25, 4²) ∪ N(55, 4²)
x_true = np.concatenate([rng.normal(25, 4, n // 2), rng.normal(55, 4, n // 2)])
rng.shuffle(x_true)
# 相关辅助变量（完整观测），与 x 线性相关
y_aux = 1.8 * x_true + rng.normal(0, 6, n)

# --- 2. 注入 35% MCAR 缺失 ---
miss_mask = rng.random(n) < 0.35
x_miss = x_true.copy().astype(float)
x_miss[miss_mask] = np.nan
X_inc = np.column_stack([x_miss, y_aux])

# --- 3. 三种填补方案 ---
X_mean = SimpleImputer(strategy="mean").fit_transform(X_inc)
X_knn  = KNNImputer(n_neighbors=7, weights="distance").fit_transform(X_inc)
X_iter = IterativeImputer(max_iter=10, random_state=42).fit_transform(X_inc)

# --- 4. 四面板对比图 ---
configs = [
    ("(a) 完整数据（参考）", x_true,       "#2563eb"),
    ("(b) 均值填补",         X_mean[:, 0], "#dc2626"),
    ("(c) KNN 填补",         X_knn[:, 0],  "#16a34a"),
    ("(d) 迭代填补（MICE）", X_iter[:, 0], "#7c3aed"),
]

fig, axes = plt.subplots(1, 4, figsize=(22, 8))
fig.subplots_adjust(wspace=0.30, top=0.86, bottom=0.14)

xs_ref = np.linspace(5, 75, 300)
kde_true = stats.gaussian_kde(x_true, bw_method=0.22)

for ax, (title, data, color) in zip(axes, configs):
    # 直方图
    ax.hist(data, bins=40, density=True, color=color, alpha=0.32,
            edgecolor="white", linewidth=0.4)
    # 当前方法的 KDE 曲线
    kde = stats.gaussian_kde(data, bw_method=0.22)
    ax.plot(xs_ref, kde(xs_ref), color=color, lw=2.4)
    # 参考分布叠加（灰色虚线），仅非参考面板显示
    if color != "#2563eb":
        ax.plot(xs_ref, kde_true(xs_ref), color="#94a3b8",
                lw=1.6, ls="--", alpha=0.75, label="完整数据 KDE")
        ax.legend(fontsize=12, loc="upper center", framealpha=0.85)
    # 真实众数位置参考线
    for mode_val in [25, 55]:
        ax.axvline(mode_val, color="#94a3b8", lw=1.0, ls=":", alpha=0.55)
    ax.set_title(title, fontsize=13, pad=6)
    ax.set_xlabel("特征值", fontsize=12)
    ax.set_ylabel("概率密度", fontsize=12)
    ax.set_xlim(5, 75)

fig.suptitle(
    "缺失值填补对分布形态的影响（n=600，35% MCAR 缺失，双峰分布）\n"
    "灰色虚线为完整数据参考分布；竖线标注真实众数位置（25 和 55）",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig2_6_02_imputation_comparison")
