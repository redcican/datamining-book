"""
fig8_4_02_shapelet.py
Shapelet 提取与分类
左上：两类时间序列；右上：最优 Shapelet 及其匹配位置
下：各序列到 Shapelet 的距离分布
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
# ── 生成两类时间序列 ──────────────────────────────────────────────
T = 100
t = np.linspace(0, 2 * np.pi, T)
n_per_class = 8
# 类 A：在中段有一个尖峰（位置 40-60）
class_a = []
for _ in range(n_per_class):
    base = 0.3 * np.sin(t) + np.random.normal(0, 0.1, T)
    peak = np.zeros(T)
    peak[40:60] = np.sin(np.linspace(0, np.pi, 20)) * 1.5
    class_a.append(base + peak + np.random.normal(0, 0.05, T))
# 类 B：在中段有一个凹陷（位置 40-60）
class_b = []
for _ in range(n_per_class):
    base = 0.3 * np.sin(t) + np.random.normal(0, 0.1, T)
    dip = np.zeros(T)
    dip[40:60] = -np.sin(np.linspace(0, np.pi, 20)) * 1.5
    class_b.append(base + dip + np.random.normal(0, 0.05, T))
# ── 提取最优 Shapelet ────────────────────────────────────────────
all_series = class_a + class_b
labels = [0] * n_per_class + [1] * n_per_class
shapelet_len = 20
# 从判别区域提取 Shapelet（简化：直接从类 A 的第一个样本的 40:60 位置）
shapelet = class_a[0][40:60]
shapelet_norm = (shapelet - shapelet.mean()) / (shapelet.std() + 1e-8)
# 计算所有序列到 Shapelet 的距离
def sdist(s, ts, m):
    s_norm = (s - s.mean()) / (s.std() + 1e-8)
    best_dist = np.inf
    best_pos = 0
    for j in range(len(ts) - m + 1):
        sub = ts[j:j + m]
        sub_norm = (sub - sub.mean()) / (sub.std() + 1e-8)
        d = np.sqrt(np.sum((s_norm - sub_norm) ** 2))
        if d < best_dist:
            best_dist = d
            best_pos = j
    return best_dist, best_pos
distances_a = []
positions_a = []
for ts in class_a:
    d, p = sdist(shapelet, ts, shapelet_len)
    distances_a.append(d)
    positions_a.append(p)
distances_b = []
positions_b = []
for ts in class_b:
    d, p = sdist(shapelet, ts, shapelet_len)
    distances_b.append(d)
    positions_b.append(p)
# ── 绘图 ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.suptitle("图 8.4.2　Shapelet 提取与分类",
             fontsize=20, fontweight="bold", y=1.02)
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
# ── 左上：两类时间序列 ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for ts in class_a:
    ax.plot(np.arange(T), ts, color=COLORS["blue"], lw=0.8, alpha=0.4)
for ts in class_b:
    ax.plot(np.arange(T), ts, color=COLORS["red"], lw=0.8, alpha=0.4)
ax.axvspan(40, 60, alpha=0.15, color=COLORS["green"])
ax.set_xlabel("时间步", fontsize=13)
ax.set_ylabel("值", fontsize=13)
ax.set_title("(a) 两类时间序列", fontsize=14)
ax.tick_params(labelsize=12)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["blue"], alpha=0.4, label="类 A（尖峰）"),
    Patch(facecolor=COLORS["red"], alpha=0.4, label="类 B（凹陷）"),
    Patch(facecolor=COLORS["green"], alpha=0.15, label="判别区域"),
]
ax.legend(handles=legend_elements, fontsize=11, loc="upper right")
# ── 右上：Shapelet 及其匹配 ───────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
# 展示 Shapelet
ax.plot(np.arange(shapelet_len), shapelet, color=COLORS["green"], lw=3.0,
        label="Shapelet", zorder=5)
# 展示类 A 的最佳匹配片段
for i in range(min(3, n_per_class)):
    p = positions_a[i]
    seg = class_a[i][p:p + shapelet_len]
    ax.plot(np.arange(shapelet_len), seg, color=COLORS["blue"],
            lw=1.0, alpha=0.5)
# 展示类 B 的最佳匹配片段
for i in range(min(3, n_per_class)):
    p = positions_b[i]
    seg = class_b[i][p:p + shapelet_len]
    ax.plot(np.arange(shapelet_len), seg, color=COLORS["red"],
            lw=1.0, alpha=0.5)
ax.set_xlabel("时间步", fontsize=13)
ax.set_ylabel("值", fontsize=13)
ax.set_title("(b) Shapelet 与最佳匹配", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.tick_params(labelsize=12)
# ── 下面板：距离分布 ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1, :])
x_a = np.arange(n_per_class)
x_b = np.arange(n_per_class, 2 * n_per_class)
bars_a = ax.bar(x_a, distances_a, color=COLORS["blue"], alpha=0.7,
                label="类 A（尖峰）", width=0.8)
bars_b = ax.bar(x_b, distances_b, color=COLORS["red"], alpha=0.7,
                label="类 B（凹陷）", width=0.8)
# 最优阈值线
threshold = (max(distances_a) + min(distances_b)) / 2
ax.axhline(threshold, color=COLORS["green"], ls="--", lw=2.0,
           label=f"最优阈值 τ = {threshold:.2f}")
ax.set_xlabel("序列编号", fontsize=13)
ax.set_ylabel("到 Shapelet 的距离", fontsize=13)
ax.set_title("(c) 距离分布与分类阈值", fontsize=14)
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(2 * n_per_class))
ax.set_xticklabels([f"A{i+1}" for i in range(n_per_class)] +
                   [f"B{i+1}" for i in range(n_per_class)], fontsize=10)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_4_02_shapelet")
