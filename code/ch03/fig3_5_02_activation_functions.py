"""
图 3.5.2  常用激活函数及其导数对比
对应节次：3.5 神经网络分类方法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_5_02_activation_functions.py
输出路径：public/figures/ch03/fig3_5_02_activation_functions.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt

apply_style()

x = np.linspace(-4, 4, 800)

# ── 激活函数定义 ────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.1):
    return np.where(x > 0, 1.0, alpha)

# ── 颜色 ────────────────────────────────────────────────────────────────────────
C_SIG   = COLORS["blue"]
C_TANH  = COLORS["teal"]
C_RELU  = COLORS["red"]
C_LRELU = COLORS["orange"]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.subplots_adjust(hspace=0.42, wspace=0.38)

funcs = [
    ("Sigmoid $\\sigma(x)$",      sigmoid(x),         sigmoid_deriv(x),    C_SIG,   "$\\sigma(x) = \\frac{1}{1+e^{-x}}$",       "$\\sigma'(x) = \\sigma(x)(1-\\sigma(x))$"),
    ("Tanh $\\tanh(x)$",          np.tanh(x),         tanh_deriv(x),       C_TANH,  "$\\tanh(x) = \\frac{e^x-e^{-x}}{e^x+e^{-x}}$", "$\\tanh'(x)=1-\\tanh^2(x)$"),
    ("ReLU $f(x)$",               relu(x),            relu_deriv(x),       C_RELU,  "$f(x)=\\max(0,x)$",                        "$f'(x)=\\mathbf{1}[x>0]$"),
    ("Leaky ReLU $f(x)$",         leaky_relu(x),      leaky_relu_deriv(x), C_LRELU, "$f(x)=\\max(0.1x,\\, x)$",                 "$f'(x)=1$ if $x>0$ else $0.1$"),
]

for col, (name, fx, dfx, color, formula, dformula) in enumerate(funcs):
    # ── 上行：函数本身 ──────────────────────────────────────────────────────────
    ax = axes[0, col]
    ax.plot(x, fx, color=color, lw=2.5)
    ax.axhline(0, color="#94a3b8", lw=0.8, ls="--")
    ax.axvline(0, color="#94a3b8", lw=0.8, ls="--")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_title(f"{name}\n{formula}", fontsize=12, pad=6)
    # 饱和区标注（sigmoid/tanh）
    if col <= 1:
        ax.fill_between(x, fx, where=(x < -2.5), alpha=0.12, color=color,
                        label="饱和区")
        ax.fill_between(x, fx, where=(x > 2.5), alpha=0.12, color=color)
    # 死神经元标注（ReLU）
    if col == 2:
        ax.fill_betweenx([0, 0], -4, 0, alpha=0.0)
        ax.text(-2, 0.2, "死亡区\n($f=0$)", fontsize=11, color=color, ha="center")

    # ── 下行：导数 ──────────────────────────────────────────────────────────────
    ax = axes[1, col]
    ax.plot(x, dfx, color=color, lw=2.5, ls="--")
    ax.axhline(0, color="#94a3b8", lw=0.8, ls="--")
    ax.axvline(0, color="#94a3b8", lw=0.8, ls="--")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_title(f"导数\n{dformula}", fontsize=12, pad=6)
    # 梯度消失标注
    if col <= 1:
        ax.text(0, dfx.max() * 0.5,
                f"最大梯度={dfx.max():.2f}",
                ha="center", fontsize=11, color=color,
                bbox=dict(fc="white", ec=color, alpha=0.85, pad=2, boxstyle="round"))
    if col == 2:
        ax.text(2, 0.6, "梯度=1\n（不消失）", fontsize=11, color=color, ha="center")

# 行标签
for row, label in enumerate(["激活函数", "导数（梯度）"]):
    axes[row, 0].set_ylabel(label, fontsize=13)

fig.suptitle(
    "常用激活函数对比：函数形状、输出范围与梯度特性\n"
    "Sigmoid/Tanh 有梯度消失风险；ReLU 系列在正半轴保持梯度=1",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_5_02_activation_functions")
