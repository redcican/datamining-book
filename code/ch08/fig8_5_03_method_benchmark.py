"""
fig8_5_03_method_benchmark.py
分类与聚类方法性能对比
左：分类准确率柱状图  右：聚类结果可视化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成合成数据（3 类活动信号）────────────────────────────────────
T = 120
t = np.arange(T)
n_per = 15
# 走路：中频中幅正弦
walking = []
for _ in range(n_per):
    freq = 2.0 + np.random.normal(0, 0.15)
    amp = 1.0 + np.random.normal(0, 0.1)
    phase = np.random.uniform(0, 2 * np.pi)
    walking.append(amp * np.sin(2 * np.pi * freq * t / T + phase) +
                   np.random.normal(0, 0.15, T))
# 跑步：高频大幅
running = []
for _ in range(n_per):
    freq = 4.0 + np.random.normal(0, 0.2)
    amp = 2.0 + np.random.normal(0, 0.15)
    phase = np.random.uniform(0, 2 * np.pi)
    running.append(amp * np.sin(2 * np.pi * freq * t / T + phase) +
                   np.random.normal(0, 0.2, T))
# 静坐：低幅噪声
sitting = []
for _ in range(n_per):
    sitting.append(np.random.normal(0, 0.25, T))
# ── 左面板：分类准确率（模拟结果）──────────────────────────────────
methods = ["1-NN+ED", "1-NN+DTW", "Shapelet\nTransform", "BOSS", "FCN"]
accuracies = [0.89, 0.96, 0.94, 0.91, 0.97]
method_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"],
                 COLORS["purple"], COLORS["red"]]
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 8.5.3　分类与聚类方法性能对比",
             fontsize=20, fontweight="bold", y=1.02)
# ── 左面板：分类准确率 ────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(np.arange(len(methods)), accuracies, color=method_colors,
              alpha=0.8, width=0.6, edgecolor="white", linewidth=1.5)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.0%}", ha="center", va="bottom", fontsize=12,
            fontweight="bold")
ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel("分类准确率", fontsize=14)
ax.set_title("(a) 分类方法对比", fontsize=15)
ax.set_ylim(0.8, 1.02)
ax.tick_params(labelsize=12)
ax.axhline(0.96, color="gray", ls=":", lw=1, alpha=0.5)
# ── 右面板：聚类可视化 ────────────────────────────────────────────
ax = axes[1]
cluster_data = [walking, running, sitting]
cluster_colors_r = [COLORS["blue"], COLORS["red"], COLORS["green"]]
cluster_names = ["走路", "跑步", "静坐"]
offsets = [0, 6, 12]
for ci, (data, color, name, offset) in enumerate(zip(cluster_data,
                                                      cluster_colors_r,
                                                      cluster_names, offsets)):
    for s in data:
        ax.plot(t, s + offset, color=color, lw=0.4, alpha=0.3)
    mean_s = np.mean(data, axis=0)
    ax.plot(t, mean_s + offset, color=color, lw=2.5,
            label=f"簇{ci+1}: {name}", zorder=5)
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("加速度（按簇偏移）", fontsize=14)
ax.set_title("(b) k-means 聚类结果（K=3）", fontsize=15)
ax.legend(fontsize=12, loc="upper right")
ax.tick_params(labelsize=12)
ax.set_yticks([])
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_5_03_method_benchmark")
