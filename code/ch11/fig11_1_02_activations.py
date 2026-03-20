"""fig11_1_02_activations.py
常用激活函数及其导数"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()

# ── 定义激活函数与导数 ────────────────────────────────────────────
z = np.linspace(-6, 6, 500)

# Sigmoid
sigmoid = 1 / (1 + np.exp(-z))
sigmoid_d = sigmoid * (1 - sigmoid)

# Tanh
tanh = np.tanh(z)
tanh_d = 1 - tanh ** 2

# ReLU
relu = np.maximum(0, z)
relu_d = np.where(z > 0, 1.0, 0.0)

# Leaky ReLU (alpha = 0.01)
leaky_relu = np.where(z > 0, z, 0.01 * z)
leaky_relu_d = np.where(z > 0, 1.0, 0.01)

# ── 颜色映射 ─────────────────────────────────────────────────────
c_sigmoid = COLORS["blue"]       # #2563eb
c_tanh = COLORS["orange"]        # #ea580c
c_relu = COLORS["green"]         # #16a34a
c_leaky = COLORS["purple"]       # #9333ea

LW = 2.5  # 主曲线线宽

# ── 绘图 ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 11.1.2　常用激活函数及其导数",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) 激活函数曲线
# ══════════════════════════════════════════════════════════════════
ax1.plot(z, sigmoid, color=c_sigmoid, lw=LW, label="Sigmoid")
ax1.plot(z, tanh, color=c_tanh, lw=LW, label="Tanh")
ax1.plot(z, relu, color=c_relu, lw=LW, label="ReLU")
ax1.plot(z, leaky_relu, color=c_leaky, lw=LW, label="Leaky ReLU")

# 参考水平线
ax1.axhline(y=0, color=COLORS["gray"], ls="--", lw=1.0, alpha=0.5)
ax1.axhline(y=1, color=COLORS["gray"], ls="--", lw=1.0, alpha=0.5)

ax1.set_title("(a) 激活函数", fontsize=17, fontweight="bold")
ax1.set_xlabel("$z$", fontsize=16)
ax1.set_ylabel("$f(z)$", fontsize=16)
ax1.tick_params(labelsize=14)
ax1.legend(fontsize=14, loc="upper left")
ax1.set_xlim(-6, 6)
ax1.set_ylim(-2, 6)
ax1.grid(alpha=0.3)

# ══════════════════════════════════════════════════════════════════
# (b) 导数曲线
# ══════════════════════════════════════════════════════════════════
ax2.plot(z, sigmoid_d, color=c_sigmoid, lw=LW, label="Sigmoid'")
ax2.plot(z, tanh_d, color=c_tanh, lw=LW, label="Tanh'")
ax2.plot(z, relu_d, color=c_relu, lw=LW, label="ReLU'")
ax2.plot(z, leaky_relu_d, color=c_leaky, lw=LW, label="Leaky ReLU'")

# 参考水平线
ax2.axhline(y=0, color=COLORS["gray"], ls="--", lw=1.0, alpha=0.5)

# 标注 Sigmoid 最大导数 = 0.25
ax2.annotate(
    "最大值 = 0.25",
    xy=(0, 0.25), xytext=(2.5, 0.55),
    fontsize=13, fontweight="bold", color=c_sigmoid,
    arrowprops=dict(arrowstyle="->", color=c_sigmoid, lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white",
              ec=c_sigmoid, alpha=0.9),
)

# 标注 ReLU 导数 z>0 恒为 1
ax2.annotate(
    "恒为 1.0",
    xy=(4, 1.0), xytext=(4.5, 0.6),
    fontsize=13, fontweight="bold", color=c_relu,
    arrowprops=dict(arrowstyle="->", color=c_relu, lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white",
              ec=c_relu, alpha=0.9),
)

ax2.set_title("(b) 导数", fontsize=17, fontweight="bold")
ax2.set_xlabel("$z$", fontsize=16)
ax2.set_ylabel("$f'(z)$", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=14, loc="upper left")
ax2.set_xlim(-6, 6)
ax2.set_ylim(-0.1, 1.2)
ax2.grid(alpha=0.3)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_1_02_activations")
