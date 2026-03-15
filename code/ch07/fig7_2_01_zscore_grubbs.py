"""
fig7_2_01_zscore_grubbs.py
Z-score 与 Grubbs 检验示意图
左：直方图 + 正态拟合 + ±3σ 阈值
右：Grubbs 检验迭代过程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成数据：正态 + 少量离群点 ──────────────────────────────────
normal_data = np.random.normal(50, 8, 200)
outliers = np.array([15, 18, 85, 90, 92])
data = np.concatenate([normal_data, outliers])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.2.1　Z-score 与 Grubbs 检验", fontsize=20, fontweight="bold", y=1.02)

# ── 左：Z-score 检测 ─────────────────────────────────────────────
ax = axes[0]
mu, sigma = np.mean(data), np.std(data)

# 直方图
counts, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.6,
                                 color=COLORS["blue"], edgecolor="white", label="数据分布")

# 正态拟合曲线
x_fit = np.linspace(mu - 4.5 * sigma, mu + 4.5 * sigma, 300)
y_fit = stats.norm.pdf(x_fit, mu, sigma)
ax.plot(x_fit, y_fit, color=COLORS["blue"], lw=2.5, label="正态拟合")

# ±3σ 阈值线
for sign, label in [(1, "+3σ"), (-1, "−3σ")]:
    threshold = mu + sign * 3 * sigma
    ax.axvline(threshold, color=COLORS["red"], ls="--", lw=2, zorder=4)
    ax.text(threshold + sign * 1.5, max(y_fit) * 0.85, label,
            fontsize=14, color=COLORS["red"], fontweight="bold",
            ha="left" if sign > 0 else "right", va="center")

# 标记超出阈值的离群点
z_scores = np.abs((data - mu) / sigma)
outlier_mask = z_scores > 3
inlier_mask = ~outlier_mask

# 在 x 轴上方标记离群点
ax.scatter(data[outlier_mask], np.full(outlier_mask.sum(), -0.003),
           c=COLORS["red"], s=100, marker="^", zorder=5,
           edgecolors="k", linewidths=0.5, label=f"异常点 (|Z|>3, n={outlier_mask.sum()})")

ax.set_title("(a) Z-score 检测", fontsize=17)
ax.set_xlabel("数据值", fontsize=15)
ax.set_ylabel("概率密度", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13)

# ── 右：Grubbs 检验迭代过程 ──────────────────────────────────────
ax = axes[1]

# 对数据排序
sorted_data = np.sort(data.copy())
n = len(sorted_data)

# 计算 Grubbs 统计量
mean_val = np.mean(sorted_data)
std_val = np.std(sorted_data, ddof=1)

# 找最极端点的 G 值
max_dev_idx = np.argmax(np.abs(sorted_data - mean_val))
G = np.abs(sorted_data[max_dev_idx] - mean_val) / std_val

# Grubbs 临界值 (alpha=0.05, two-sided)
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))

# 标记被 Grubbs 检验标记为异常的点
grubbs_scores = np.abs(sorted_data - mean_val) / std_val
grubbs_outlier = grubbs_scores > G_crit
grubbs_inlier = ~grubbs_outlier

indices = np.arange(len(sorted_data))

ax.scatter(indices[grubbs_inlier], sorted_data[grubbs_inlier],
           c=COLORS["blue"], s=20, alpha=0.6, label="正常点")
ax.scatter(indices[grubbs_outlier], sorted_data[grubbs_outlier],
           c=COLORS["red"], s=80, marker="*", edgecolors="k", linewidths=0.5,
           zorder=5, label=f"Grubbs 异常 (n={grubbs_outlier.sum()})")

# 均值线
ax.axhline(mean_val, color=COLORS["gray"], ls="-", lw=1.5, alpha=0.7, label=f"均值 μ={mean_val:.1f}")

# 临界值范围
upper_crit = mean_val + G_crit * std_val
lower_crit = mean_val - G_crit * std_val
ax.axhline(upper_crit, color=COLORS["orange"], ls="--", lw=2, label=f"临界值 (G$_{{crit}}$={G_crit:.2f})")
ax.axhline(lower_crit, color=COLORS["orange"], ls="--", lw=2)

# 标注最极端点的 G 值
extreme_x = indices[max_dev_idx]
extreme_y = sorted_data[max_dev_idx]
ax.annotate(f"G = {G:.2f}",
            xy=(extreme_x, extreme_y),
            xytext=(extreme_x - 40, extreme_y - 8),
            fontsize=14, color=COLORS["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))

ax.set_title("(b) Grubbs 检验", fontsize=17)
ax.set_xlabel("排序索引", fontsize=15)
ax.set_ylabel("数据值", fontsize=15)
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=13)

fig.tight_layout()
save_fig(fig, __file__, "fig7_2_01_zscore_grubbs")
