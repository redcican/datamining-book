"""fig11_3_03_gradient_and_prediction.py
(a) RNN vs LSTM 梯度幅度对比  (b) 正弦波时间序列预测"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle("图 11.3.3　RNN 梯度流与序列预测",
             fontsize=22, fontweight="bold", y=0.98)

LW = 2.5

# ══════════════════════════════════════════════════════════════════
# (a) 梯度幅度随时间步衰减
# ══════════════════════════════════════════════════════════════════
steps = np.arange(1, 51)

# Vanilla RNN: gradient ∝ (||W_hh|| * γ)^k, with ||W_hh||*γ ≈ 0.85
rnn_grad = 0.85 ** steps
# Add some noise
np.random.seed(42)
rnn_grad = rnn_grad * (1 + 0.08 * np.random.randn(len(steps)))
rnn_grad = np.clip(rnn_grad, 1e-8, 2.0)

# LSTM: gradient roughly stable due to cell state highway
# f_t ≈ 0.95, so gradient ∝ 0.95^k but much slower decay
lstm_grad = 0.98 ** steps
lstm_grad = lstm_grad * (1 + 0.05 * np.random.randn(len(steps)))
lstm_grad = np.clip(lstm_grad, 0.3, 1.5)
# Normalize to start at 1
lstm_grad = lstm_grad / lstm_grad[0]
rnn_grad = rnn_grad / rnn_grad[0]

ax1.semilogy(steps, rnn_grad, color=COLORS["red"], lw=LW,
             label="基本 RNN", ls="--", alpha=0.9)
ax1.semilogy(steps, lstm_grad, color=COLORS["blue"], lw=LW,
             label="LSTM", alpha=0.9)

# Shade the "gradient vanishing zone"
ax1.axhspan(1e-4, 1e-2, color=COLORS["red"], alpha=0.08)
ax1.text(40, 3e-3, "梯度消失区域", fontsize=12, fontweight="bold",
         color=COLORS["red"], ha="center", va="center", alpha=0.7)

# Annotations
ax1.annotate(
    "约20步后\n梯度接近零",
    xy=(25, rnn_grad[24]),
    xytext=(35, 0.15),
    fontsize=12, fontweight="bold", color=COLORS["red"],
    arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white",
              ec=COLORS["red"], alpha=0.9),
)

ax1.annotate(
    "细胞状态\n保持梯度稳定",
    xy=(40, lstm_grad[39]),
    xytext=(30, 1.5),
    fontsize=12, fontweight="bold", color=COLORS["blue"],
    arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white",
              ec=COLORS["blue"], alpha=0.9),
)

ax1.set_xlabel("回传时间步数", fontsize=16)
ax1.set_ylabel("相对梯度幅度", fontsize=16)
ax1.tick_params(labelsize=14)
ax1.legend(fontsize=14, loc="lower left")
ax1.set_xlim(0, 50)
ax1.set_ylim(5e-5, 5)
ax1.grid(alpha=0.3)
ax1.set_title("(a) 梯度幅度随回传距离的变化", fontsize=17,
              fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 正弦波预测
# ══════════════════════════════════════════════════════════════════
# Generate ground truth sine wave
t_full = np.linspace(0, 8 * np.pi, 500)
y_true = np.sin(t_full)

# Simulated LSTM prediction (slightly noisy but accurate)
np.random.seed(123)
# Training region: t < 6π, Testing region: t >= 6π
split_idx = 375  # approximately 6π

y_pred = y_true.copy()
# Training: exact (not plotted)
# Testing: add small noise to simulate prediction
noise = 0.03 * np.random.randn(500 - split_idx)
y_pred[split_idx:] = y_true[split_idx:] + noise

# Plot
ax2.plot(t_full, y_true, color=COLORS["blue"], lw=LW,
         label="真实值", alpha=0.9)
ax2.plot(t_full[split_idx:], y_pred[split_idx:],
         color=COLORS["red"], lw=LW, ls="--",
         label="LSTM 预测", alpha=0.9)

# Mark train/test split
ax2.axvline(x=t_full[split_idx], color=COLORS["gray"], ls=":",
            lw=1.5, alpha=0.7)
ax2.text(t_full[split_idx] - 0.3, 1.3, "训练集", fontsize=13,
         fontweight="bold", color=COLORS["gray"], ha="right")
ax2.text(t_full[split_idx] + 0.3, 1.3, "测试集", fontsize=13,
         fontweight="bold", color=COLORS["gray"], ha="left")

# Shade training region
ax2.axvspan(t_full[0], t_full[split_idx], color=COLORS["blue"],
            alpha=0.03)
ax2.axvspan(t_full[split_idx], t_full[-1], color=COLORS["red"],
            alpha=0.03)

# MSE annotation
mse = np.mean((y_pred[split_idx:] - y_true[split_idx:]) ** 2)
ax2.text(t_full[-1] - 1.5, -1.4,
         f"测试 MSE = {mse:.4f}",
         fontsize=13, fontweight="bold", color=COLORS["red"],
         ha="right", va="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=COLORS["red"], alpha=0.9))

ax2.set_xlabel("时间 $t$", fontsize=16)
ax2.set_ylabel("$y(t) = \\sin(t)$", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=14, loc="upper right")
ax2.set_xlim(t_full[0], t_full[-1])
ax2.set_ylim(-1.7, 1.7)
ax2.grid(alpha=0.3)
ax2.set_title("(b) LSTM 正弦波预测", fontsize=17,
              fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_3_03_gradient_and_prediction")
