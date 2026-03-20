"""
fig8_5_01_classification.py
三种时间序列分类范式对比示意图
左：1-NN 距离分类  中：Shapelet Transform  右：深度学习 FCN
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
t = np.linspace(0, 4 * np.pi, T)
n_each = 6
class_a, class_b = [], []
for _ in range(n_each):
    phase = np.random.uniform(-0.3, 0.3)
    class_a.append(np.sin(t + phase) + np.random.normal(0, 0.1, T))
for _ in range(n_each):
    phase = np.random.uniform(-0.3, 0.3)
    class_b.append(np.sin(2 * t + phase) * 0.7 + np.random.normal(0, 0.1, T))
# 新测试样本
test_sample = np.sin(t + 0.1) + np.random.normal(0, 0.08, T)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("图 8.5.1　三种时间序列分类范式",
             fontsize=20, fontweight="bold", y=1.02)
# ── (a) 1-NN 距离分类 ────────────────────────────────────────────
ax = axes[0]
for s in class_a:
    ax.plot(np.arange(T), s, color=COLORS["blue"], lw=0.6, alpha=0.3)
for s in class_b:
    ax.plot(np.arange(T), s, color=COLORS["red"], lw=0.6, alpha=0.3)
ax.plot(np.arange(T), test_sample, color=COLORS["green"], lw=2.5,
        label="测试样本", zorder=5)
# 找最近邻
dists_a = [np.sqrt(np.sum((test_sample - s) ** 2)) for s in class_a]
dists_b = [np.sqrt(np.sum((test_sample - s) ** 2)) for s in class_b]
nn_idx = np.argmin(dists_a)
ax.plot(np.arange(T), class_a[nn_idx], color=COLORS["blue"], lw=2.0,
        ls="--", label="最近邻(类A)", zorder=4)
ax.set_title("(a) 1-NN + DTW", fontsize=14)
ax.set_xlabel("时间步", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.tick_params(labelsize=11)
# ── (b) Shapelet Transform ───────────────────────────────────────
ax = axes[1]
# 模拟 Shapelet 特征空间
np.random.seed(42)
feat_a = np.column_stack([np.random.normal(1, 0.3, n_each),
                          np.random.normal(3, 0.4, n_each)])
feat_b = np.column_stack([np.random.normal(3, 0.3, n_each),
                          np.random.normal(1, 0.4, n_each)])
feat_test = np.array([[1.2, 2.8]])
ax.scatter(feat_a[:, 0], feat_a[:, 1], color=COLORS["blue"], s=80,
           zorder=3, label="类 A")
ax.scatter(feat_b[:, 0], feat_b[:, 1], color=COLORS["red"], s=80,
           zorder=3, label="类 B")
ax.scatter(feat_test[:, 0], feat_test[:, 1], color=COLORS["green"],
           s=150, marker="*", zorder=5, label="测试样本")
# SVM 决策边界
xx = np.linspace(0, 4.5, 100)
ax.plot(xx, -xx + 4.2, color="gray", ls="--", lw=1.5, label="SVM 边界")
ax.set_title("(b) Shapelet Transform + SVM", fontsize=14)
ax.set_xlabel("sdist(S₁, T)", fontsize=12)
ax.set_ylabel("sdist(S₂, T)", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.tick_params(labelsize=11)
ax.set_xlim(-0.2, 4.5)
ax.set_ylim(-0.2, 4.5)
# ── (c) 深度学习 FCN ─────────────────────────────────────────────
ax = axes[2]
# 绘制简化的网络架构示意
layer_x = [0.1, 0.3, 0.5, 0.7, 0.9]
layer_h = [1.0, 0.8, 0.6, 0.4, 0.2]
layer_names = ["输入\n(T×1)", "Conv1D\n(128)", "Conv1D\n(256)",
               "GAP", "Softmax\n(K)"]
layer_colors = [COLORS["blue"], COLORS["green"], COLORS["green"],
                COLORS["orange"], COLORS["red"]]
for i, (x, h, name, c) in enumerate(zip(layer_x, layer_h, layer_names,
                                         layer_colors)):
    rect = plt.Rectangle((x - 0.06, 0.5 - h / 2), 0.12, h,
                          facecolor=c, alpha=0.5, edgecolor=c, lw=2)
    ax.add_patch(rect)
    ax.text(x, 0.5 - h / 2 - 0.08, name, ha="center", va="top",
            fontsize=9, fontweight="bold")
    if i < len(layer_x) - 1:
        ax.annotate("", xy=(layer_x[i + 1] - 0.06, 0.5),
                    xytext=(x + 0.06, 0.5),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1.1)
ax.set_title("(c) FCN 端到端学习", fontsize=14)
ax.axis("off")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_5_01_classification")
