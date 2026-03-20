"""fig11_6_03_cartpole_training.py
(a) CartPole 训练奖励曲线  (b) DQN 变体对比"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("图 11.6.3　DQN CartPole 训练结果",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) 训练奖励曲线
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)
episodes = np.arange(1, 301)

# Sigmoid-like growth curve with noise
base = 500.0 / (1.0 + np.exp(-0.05 * (episodes - 80)))
noise = np.random.normal(0, 30, size=len(episodes))
rewards = np.clip(base + noise, 10, 500)

# Moving average (window=20)
window = 20
ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
ma_x = episodes[window - 1:]

# Raw episode rewards
ax1.plot(episodes, rewards, color=COLORS["blue"], alpha=0.3,
         lw=0.8, label="回合奖励")
# Moving average
ax1.plot(ma_x, ma, color=COLORS["red"], lw=2.5,
         label=f"移动平均 (窗口={window})")
# Target line
ax1.axhline(y=475, color=COLORS["green"], ls="--", lw=2,
            alpha=0.8, label="目标: 475")

# Annotation arrow at the "learning" transition point
ax1.annotate("智能体开始收敛",
             xy=(95, 420), xytext=(160, 250),
             fontsize=14, fontweight="bold", color=COLORS["purple"],
             arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=2),
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=COLORS["purple"], alpha=0.9))

ax1.set_xlabel("回合 (Episode)", fontsize=16)
ax1.set_ylabel("回合奖励", fontsize=16)
ax1.legend(fontsize=14, loc="lower right")
ax1.set_xlim(0, 300)
ax1.set_ylim(0, 550)
ax1.tick_params(labelsize=14)
ax1.set_title("(a) 训练奖励曲线", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) DQN 变体对比
# ══════════════════════════════════════════════════════════════════
np.random.seed(99)
eps = np.arange(1, 301)

# DQN — reaches 475 around episode 120
dqn_base = 500.0 / (1.0 + np.exp(-0.05 * (eps - 90)))
dqn_noise = np.random.normal(0, 8, size=len(eps))
dqn_curve = np.clip(dqn_base + dqn_noise, 10, 500)
# Smooth with moving average
dqn_ma = np.convolve(dqn_curve, np.ones(window) / window, mode="valid")

# Double DQN — reaches 475 around episode 100, more stable
ddqn_base = 500.0 / (1.0 + np.exp(-0.06 * (eps - 70)))
ddqn_noise = np.random.normal(0, 6, size=len(eps))
ddqn_curve = np.clip(ddqn_base + ddqn_noise, 10, 500)
ddqn_ma = np.convolve(ddqn_curve, np.ones(window) / window, mode="valid")

# Dueling DQN — reaches 475 around episode 80, fastest
dueling_base = 500.0 / (1.0 + np.exp(-0.07 * (eps - 55)))
dueling_noise = np.random.normal(0, 5, size=len(eps))
dueling_curve = np.clip(dueling_base + dueling_noise, 10, 500)
dueling_ma = np.convolve(dueling_curve, np.ones(window) / window, mode="valid")

ma_eps = eps[window - 1:]

ax2.plot(ma_eps, dqn_ma, color=COLORS["blue"], lw=2.5, label="DQN")
ax2.plot(ma_eps, ddqn_ma, color=COLORS["red"], lw=2.5, label="Double DQN")
ax2.plot(ma_eps, dueling_ma, color=COLORS["green"], lw=2.5, label="Dueling DQN")

# Shaded target region
ax2.axhspan(475, 550, color=COLORS["green"], alpha=0.06, label="目标区间")

ax2.set_xlabel("回合 (Episode)", fontsize=16)
ax2.set_ylabel("回合奖励", fontsize=16)
ax2.legend(fontsize=14, loc="lower right")
ax2.set_xlim(0, 300)
ax2.set_ylim(0, 550)
ax2.tick_params(labelsize=14)
ax2.set_title("(b) DQN 变体对比", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_6_03_cartpole_training")
