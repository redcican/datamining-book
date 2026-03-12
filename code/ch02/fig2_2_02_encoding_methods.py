"""
图 2.2.2  类别变量四种编码方法对比（原始 / 序数 / 独热 / 目标编码）
对应节次：2.2 数据集成与数据变换
运行方式：python code/ch02/fig2_2_02_encoding_methods.py
输出路径：public/figures/ch02/fig2_2_02_encoding_methods.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

# ── Dataset: 房间类型 (categorical) + 租金 (target) ───────────────────────
categories  = ["整套公寓", "独立卧室", "合租隔间"]
n_per_cat   = [5, 6, 4]
target_means = [8500, 4200, 1800]   # 月租金（元）

# Generate sample for illustration (15 samples)
rng = np.random.default_rng(42)
raw_cat = []
raw_target = []
for cat, n, mean in zip(categories, n_per_cat, target_means):
    raw_cat.extend([cat] * n)
    raw_target.extend(rng.normal(mean, mean * 0.12, n).astype(int).tolist())

cat_arr = np.array(raw_cat)
y_arr   = np.array(raw_target)

# Ordinal encoding
ordinal_map = {"整套公寓": 2, "独立卧室": 1, "合租隔间": 0}  # ordered by typical price
ord_enc = np.array([ordinal_map[c] for c in raw_cat])

# Target encoding (mean per category)
target_map = {cat: np.mean(y_arr[cat_arr == cat]) for cat in categories}
tgt_enc = np.array([target_map[c] for c in raw_cat])

# ── Layout: 2×2 grid ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.42, wspace=0.35)

C_ORIG = "#0f172a"
C_ORD  = "#2563eb"
C_OH   = ["#2563eb", "#16a34a", "#ea580c"]
C_TGT  = "#7c3aed"
cat_colors = {"整套公寓": "#2563eb", "独立卧室": "#16a34a", "合租隔间": "#ea580c"}

# ── (a) 原始类别变量分布 ──────────────────────────────────────────────────
ax = axes[0, 0]
counts = [n_per_cat[i] for i in range(3)]
bars = ax.bar(categories, counts,
              color=[cat_colors[c] for c in categories], alpha=0.8, width=0.55)
for bar, n in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(n), ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="#1e293b")
ax.set_title("(a) 原始：类别变量", fontsize=12.5, pad=6)
ax.set_ylabel("样本数", fontsize=12)
ax.set_ylim(0, 8)
ax.tick_params(axis="x", labelsize=9.5)

# Annotation box
ax.text(0.97, 0.95,
        "字符串值\n无法直接\n用于数值模型",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12.5, color="#dc2626",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#dc2626", alpha=0.9))

# ── (b) 序数编码 ──────────────────────────────────────────────────────────
ax = axes[0, 1]
ord_labels = [f"{ordinal_map[c]}\n（{c}）" for c in categories]
bars = ax.bar(categories, [ordinal_map[c] for c in categories],
              color=[cat_colors[c] for c in categories], alpha=0.8, width=0.55)
for bar, c in zip(bars, categories):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            str(ordinal_map[c]), ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=cat_colors[c])
ax.set_title("(b) 序数编码：保留有序关系", fontsize=12.5, pad=6)
ax.set_ylabel("编码值", fontsize=12)
ax.set_yticks([0, 1, 2])
ax.set_ylim(0, 2.8)
ax.tick_params(axis="x", labelsize=9.5)

ax.text(0.97, 0.95,
        "适用：有自然\n顺序的类别\n（如评级、教育）",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12.5, color=C_ORD,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=C_ORD, alpha=0.9))

# ── (c) 独热编码 ──────────────────────────────────────────────────────────
ax = axes[1, 0]
n_show = 8
oh_matrix = np.zeros((n_show, 3), dtype=int)
sample_cats = raw_cat[:n_show]
for i, c in enumerate(sample_cats):
    oh_matrix[i, categories.index(c)] = 1

im = ax.imshow(oh_matrix.T, aspect="auto", cmap="Blues",
               vmin=0, vmax=1.5, origin="upper")

ax.set_xticks(range(n_show))
ax.set_xticklabels([f"s{i+1}" for i in range(n_show)], fontsize=12)
ax.set_yticks(range(3))
ax.set_yticklabels(categories, fontsize=12.5)
ax.set_xlabel("样本编号", fontsize=12)
ax.set_title("(c) 独热编码：K-1 列二进制矩阵", fontsize=12.5, pad=6)

for i in range(n_show):
    for j in range(3):
        ax.text(i, j, str(oh_matrix[i, j]),
                ha="center", va="center",
                fontsize=12, color="white" if oh_matrix[i, j] else "#94a3b8",
                fontweight="bold")

ax.text(0.97, 0.03,
        "适用：名义类别\n列数 = K，\n模型用 K-1 列",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=12.5, color="#16a34a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#16a34a", alpha=0.9))

# ── (d) 目标编码 ──────────────────────────────────────────────────────────
ax = axes[1, 1]
means = [target_map[c] for c in categories]
bars = ax.bar(categories, means,
              color=[cat_colors[c] for c in categories], alpha=0.8, width=0.55)
ax.axhline(np.mean(y_arr), color="#64748b", ls="--", lw=1.5)

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{m:.0f}元", ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            color=cat_colors[categories[bars.index(bar)]]
            if False else "#1e293b")

for i, (bar, m) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{m:.0f}元", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=C_TGT)

ax.set_title("(d) 目标编码：类别均值替换", fontsize=12.5, pad=6)
ax.set_ylabel("月均租金（元）", fontsize=12)
ax.set_ylim(0, 10500)
ax.tick_params(axis="x", labelsize=9.5)

ax.text(0.97, 0.95,
        "适用：高基数类别\n⚠ 注意目标泄露\n（需交叉验证）",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12.5, color=C_TGT,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=C_TGT, alpha=0.9))

fig.suptitle("类别变量编码方法对比：原始 / 序数 / 独热 / 目标编码",
             fontsize=15, y=1.01)
fig.text(0.5, -0.02,
         "示例数据：租房平台房间类型（n=15）。序数编码保留价格顺序；"
         "独热编码为名义类别创建二进制列；"
         "目标编码用类别均值替换字符串，但需防范目标信息泄露。",
         ha="center", fontsize=12.5, color="#64748b", style="italic")

save_fig(fig, __file__, "fig2_2_02_encoding_methods")
