"""fig11_7_03_finetune_results.py
(a) 三种迁移策略验证准确率  (b) 域适应特征可视化"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig = plt.figure(figsize=(16, 7))
fig.suptitle("图 11.7.3　迁移学习效果对比",
             fontsize=22, fontweight="bold", y=1.02)

ax1 = fig.add_axes([0.06, 0.10, 0.42, 0.72])   # (a) accuracy curves
ax2 = fig.add_axes([0.55, 0.10, 0.20, 0.72])    # (b) before adaptation
ax3 = fig.add_axes([0.78, 0.10, 0.20, 0.72])    # (b) after adaptation
fig.text(0.67, 0.92, "(b) 特征可视化",
         fontsize=17, fontweight="bold", ha="center")

# ══════════════════════════════════════════════════════════════════
# (a) 验证准确率曲线
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)
epochs = np.arange(1, 21)

# 从头训练 — starts ~20%, slowly rises to ~55%, with overfitting oscillation
scratch = 0.2 + 0.35 * (1 - np.exp(-epochs / 8))
scratch += np.random.normal(0, 0.03, size=len(epochs))
scratch = np.clip(scratch, 0.15, 0.55)

# 特征提取 — starts ~60%, quickly rises to ~85%, plateaus
feature_ext = 0.6 + 0.25 * (1 - np.exp(-epochs / 2))
feature_ext += np.random.normal(0, 0.015, size=len(epochs))
feature_ext = np.clip(feature_ext, 0.55, 0.88)

# 全量微调 — starts ~50%, rises to ~93%, best performance
finetune = 0.5 + 0.43 * (1 - np.exp(-epochs / 4))
finetune += np.random.normal(0, 0.012, size=len(epochs))
finetune = np.clip(finetune, 0.45, 0.95)

ax1.plot(epochs, scratch, color=COLORS["gray"], lw=2.5,
         marker="s", markersize=5, label="从头训练")
ax1.plot(epochs, feature_ext, color=COLORS["blue"], lw=2.5,
         marker="o", markersize=5, label="特征提取")
ax1.plot(epochs, finetune, color=COLORS["red"], lw=2.5,
         marker="^", markersize=5, label="全量微调")

# 90% baseline
ax1.axhline(y=0.90, color=COLORS["green"], ls="--", lw=2,
            alpha=0.8, label="90% 基准线")

ax1.set_xlabel("训练轮次 (Epoch)", fontsize=16)
ax1.set_ylabel("验证准确率", fontsize=16)
ax1.set_xlim(1, 20)
ax1.set_ylim(0.1, 1.0)
ax1.legend(fontsize=14, loc="lower right")
ax1.tick_params(labelsize=14)
ax1.grid(alpha=0.3)
ax1.set_title("(a) 验证准确率", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) t-SNE 特征可视化 — 域适应前 / 后
# ══════════════════════════════════════════════════════════════════
np.random.seed(123)
n_pts = 40  # points per cluster

# --- Before adaptation: source and target clearly separated ---
# Source domain — 3 clusters (blue)
src_c1 = np.random.randn(n_pts, 2) * 0.5 + np.array([-3, 2])
src_c2 = np.random.randn(n_pts, 2) * 0.5 + np.array([-3, -2])
src_c3 = np.random.randn(n_pts, 2) * 0.5 + np.array([1, 3])
src_before = np.vstack([src_c1, src_c2, src_c3])

# Target domain — 3 clusters (red), shifted away from source
tgt_c1 = np.random.randn(n_pts, 2) * 0.5 + np.array([3, 2])
tgt_c2 = np.random.randn(n_pts, 2) * 0.5 + np.array([3, -2])
tgt_c3 = np.random.randn(n_pts, 2) * 0.5 + np.array([-1, -3])
tgt_before = np.vstack([tgt_c1, tgt_c2, tgt_c3])

ax2.scatter(src_before[:, 0], src_before[:, 1],
            c=COLORS["blue"], marker="o", s=22, alpha=0.7,
            edgecolors="white", linewidths=0.3, label="源域")
ax2.scatter(tgt_before[:, 0], tgt_before[:, 1],
            c=COLORS["red"], marker="^", s=22, alpha=0.7,
            edgecolors="white", linewidths=0.3, label="目标域")

ax2.set_xlabel("$z_1$", fontsize=16)
ax2.set_ylabel("$z_2$", fontsize=16)
ax2.set_title("适应前", fontsize=15, fontweight="bold", pad=25)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=13, loc="upper left", markerscale=1.5)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.grid(alpha=0.3)

# --- After adaptation: domains mixed, classes still separable ---
# 3 mixed clusters at distinct positions
mix_c1_src = np.random.randn(n_pts, 2) * 0.45 + np.array([-2.5, 2.5])
mix_c1_tgt = np.random.randn(n_pts, 2) * 0.45 + np.array([-2.5, 2.5])
mix_c2_src = np.random.randn(n_pts, 2) * 0.45 + np.array([2.5, 2.5])
mix_c2_tgt = np.random.randn(n_pts, 2) * 0.45 + np.array([2.5, 2.5])
mix_c3_src = np.random.randn(n_pts, 2) * 0.45 + np.array([0, -2.5])
mix_c3_tgt = np.random.randn(n_pts, 2) * 0.45 + np.array([0, -2.5])

src_after = np.vstack([mix_c1_src, mix_c2_src, mix_c3_src])
tgt_after = np.vstack([mix_c1_tgt, mix_c2_tgt, mix_c3_tgt])

ax3.scatter(src_after[:, 0], src_after[:, 1],
            c=COLORS["blue"], marker="o", s=22, alpha=0.7,
            edgecolors="white", linewidths=0.3, label="源域")
ax3.scatter(tgt_after[:, 0], tgt_after[:, 1],
            c=COLORS["red"], marker="^", s=22, alpha=0.7,
            edgecolors="white", linewidths=0.3, label="目标域")

ax3.set_xlabel("$z_1$", fontsize=16)
ax3.set_ylabel("$z_2$", fontsize=16)
ax3.set_title("适应后", fontsize=15, fontweight="bold", pad=25)
ax3.tick_params(labelsize=14)
ax3.legend(fontsize=13, loc="upper left", markerscale=1.5)
ax3.set_xlim(-5, 5)
ax3.set_ylim(-5, 5)
ax3.grid(alpha=0.3)

# ── 保存 ──────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig11_7_03_finetune_results")
