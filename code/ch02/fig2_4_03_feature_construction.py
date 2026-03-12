"""
图 2.4.3  特征构造：分箱离散化、多项式特征与交互特征
对应节次：2.4 特征选择与特征工程
运行方式：python code/ch02/fig2_4_03_feature_construction.py
输出路径：public/figures/ch02/fig2_4_03_feature_construction.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

apply_style()

C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_GREEN  = "#16a34a"
C_ORANGE = "#ea580c"
C_PURPLE = "#7c3aed"
C_GRAY   = "#94a3b8"
C_DARK   = "#1e293b"

rng = np.random.default_rng(42)

# ── Layout 1×3 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.subplots_adjust(wspace=0.42)

# ── Panel (a): Binning — age vs salary ─────────────────────────────────────
ax = axes[0]
n_pts = 400
age = rng.uniform(22, 62, n_pts)
# Salary peaks in mid-career: concave quadratic
salary = (30 + 1.8 * age - 0.022 * age**2
          + rng.normal(0, 3, n_pts))   # unit: thousand yuan

# Define 5 equal-width bins
bins = np.linspace(22, 62, 6)
bin_labels = ["22-30", "30-38", "38-46", "46-54", "54-62"]
bin_idx = np.digitize(age, bins[1:-1])  # 0..4

# Scatter (raw data, light)
ax.scatter(age, salary, s=10, alpha=0.3, color=C_GRAY, zorder=2)

# Draw bin boundaries
for b in bins[1:-1]:
    ax.axvline(b, color=C_DARK, lw=1.2, ls="--", alpha=0.5, zorder=3)

# Mean salary per bin
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]
colors_bar = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE]
for k in range(5):
    mask = bin_idx == k
    if mask.sum() == 0:
        continue
    mean_sal = salary[mask].mean()
    bx = bins[k]
    ax.bar(bx + bin_width / 2, mean_sal, width=bin_width * 0.7,
           color=colors_bar[k], alpha=0.75, zorder=4,
           bottom=salary.min() - 2)
    ax.text(bx + bin_width / 2, mean_sal + 1.0,
            f"{mean_sal:.0f}k", ha="center", fontsize=12,
            color=colors_bar[k], fontweight="bold")

ax.set_xlabel("年龄（连续）", fontsize=12)
ax.set_ylabel("月薪（千元）", fontsize=12)
ax.set_title("(a) 分箱离散化：连续特征 → 区间均值", fontsize=13, pad=6)
ax.set_xlim(20, 64)
ax.tick_params(labelsize=10)

bin_patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors_bar, bin_labels)]
ax.legend(handles=bin_patches, fontsize=12, title="年龄区间",
          title_fontsize=12.5, loc="lower right", ncol=1)
ax.text(0.03, 0.97,
        "等宽分箱：5 个区间\n竖线为分割边界\n彩色柱为各箱均值",
        transform=ax.transAxes, fontsize=12.5, va="top", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cbd5e1", alpha=0.9))

# ── Panel (b): Polynomial features ─────────────────────────────────────────
ax = axes[1]
n_poly = 80
x_raw = rng.uniform(-3, 3, n_poly)
y_raw = 0.7 * x_raw**2 - 1.2 * x_raw + 1.5 + rng.normal(0, 0.8, n_poly)

ax.scatter(x_raw, y_raw, s=35, color=C_GRAY, alpha=0.7, zorder=3,
           label="观测数据")

x_fit = np.linspace(-3.2, 3.2, 200).reshape(-1, 1)

# Linear fit
lin = LinearRegression().fit(x_raw.reshape(-1, 1), y_raw)
y_lin = lin.predict(x_fit)
ax.plot(x_fit, y_lin, color=C_RED, lw=2.2, ls="--",
        label=f"线性拟合（R²={lin.score(x_raw.reshape(-1,1), y_raw):.2f}）",
        zorder=4)

# Quadratic fit
poly2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly2.fit(x_raw.reshape(-1, 1), y_raw)
y_poly = poly2.predict(x_fit)
r2_poly = poly2.score(x_raw.reshape(-1, 1), y_raw)
ax.plot(x_fit, y_poly, color=C_BLUE, lw=2.5,
        label=f"二次多项式拟合（R²={r2_poly:.2f}）", zorder=5)

ax.set_xlabel("原始特征 x", fontsize=12)
ax.set_ylabel("目标变量 y", fontsize=12)
ax.set_title("(b) 多项式特征：线性 vs 二次拟合", fontsize=13, pad=6)
ax.legend(fontsize=12.5, loc="upper right")
ax.tick_params(labelsize=10)

ax.text(0.03, 0.04,
        "添加 x² 特征后\nR² 大幅提升",
        transform=ax.transAxes, fontsize=12, va="bottom", color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C_BLUE, alpha=0.9))

# ── Panel (c): Interaction features — XOR pattern ──────────────────────────
ax = axes[2]
n_cls = 300
x1 = rng.normal(0, 1, n_cls)
x2 = rng.normal(0, 1, n_cls)
# XOR: class 1 when x1 and x2 have same sign, class 0 otherwise
y_cls = ((x1 > 0) == (x2 > 0)).astype(int)

colors_cls = [C_RED if yi == 0 else C_BLUE for yi in y_cls]
markers_cls = ["o" if yi == 0 else "^" for yi in y_cls]

for xi, yi, c, m in zip(x1, x2, colors_cls, markers_cls):
    ax.scatter(xi, yi, color=c, marker=m, s=28, alpha=0.6, zorder=3)

# Draw quadrant boundaries
ax.axvline(0, color=C_DARK, lw=1.2, ls="-", alpha=0.4, zorder=2)
ax.axhline(0, color=C_DARK, lw=1.2, ls="-", alpha=0.4, zorder=2)

# Shade the "same sign" regions (class 1)
from matplotlib.patches import Rectangle
lim = 3.2
ax.add_patch(Rectangle((0, 0), lim, lim, alpha=0.07, color=C_BLUE, zorder=1))
ax.add_patch(Rectangle((-lim, -lim), lim, lim, alpha=0.07, color=C_BLUE, zorder=1))

ax.text(1.5, 1.5, "类别 1\n(x1>0, x2>0)", ha="center", va="center",
        fontsize=12, color=C_BLUE, fontweight="bold")
ax.text(-1.5, -1.5, "类别 1\n(x1<0, x2<0)", ha="center", va="center",
        fontsize=12, color=C_BLUE, fontweight="bold")
ax.text(1.5, -1.5, "类别 0\n(x1>0, x2<0)", ha="center", va="center",
        fontsize=12, color=C_RED, fontweight="bold")
ax.text(-1.5, 1.5, "类别 0\n(x1<0, x2>0)", ha="center", va="center",
        fontsize=12, color=C_RED, fontweight="bold")

ax.set_xlabel("特征 x1", fontsize=12)
ax.set_ylabel("特征 x2", fontsize=12)
ax.set_title("(c) 交互特征：x1*x2 解决异或分类", fontsize=13, pad=6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.tick_params(labelsize=10)

p0 = mpatches.Patch(color=C_RED, label="类别 0（异号区）")
p1 = mpatches.Patch(color=C_BLUE, label="类别 1（同号区）")
ax.legend(handles=[p0, p1], fontsize=12, loc="upper left")

ax.text(0.97, 0.03,
        "原始特征 x1、x2\n无法线性分类\n交互特征 x1·x2 > 0\n即为类别 1",
        transform=ax.transAxes, fontsize=12.5, ha="right", va="bottom",
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cbd5e1", alpha=0.9))

fig.suptitle("特征构造三种策略：分箱离散化 / 多项式扩展 / 交互特征", fontsize=14, y=1.02)
fig.text(
    0.5, -0.05,
    "(a) 等宽分箱将连续年龄转化为区间特征，柱高为该区间月薪均值，捕捉非线性职业发展规律。"
    "(b) 添加 x² 多项式特征后 R² 显著提升，线性模型即可拟合二次关系。"
    "(c) 异或分类问题：x1、x2 单独无法线性分割，引入交互特征 x1·x2 后问题变为线性可分。",
    ha="center", fontsize=12, color="#64748b", style="italic",
)

save_fig(fig, __file__, "fig2_4_03_feature_construction")
