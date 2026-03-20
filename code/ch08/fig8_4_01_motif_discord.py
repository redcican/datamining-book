"""
fig8_4_01_motif_discord.py
Matrix Profile 统一 Motif 与 Discord 发现
上：原始时间序列（含重复模式 + 异常片段）
下：Matrix Profile，标注 Motif（谷值）和 Discord（峰值）
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
# ── 生成合成信号 ──────────────────────────────────────────────────
# 基线 + 3 次重复模式 + 1 个异常片段
n = 1000
t = np.arange(n)
base = np.random.normal(0, 0.3, n)
# 定义一个模式（形状）
pattern_len = 50
pt = np.linspace(0, 2 * np.pi, pattern_len)
pattern = np.sin(pt) + 0.5 * np.sin(3 * pt)
# 在 3 个位置插入模式
motif_positions = [100, 400, 700]
for pos in motif_positions:
    base[pos:pos + pattern_len] += pattern * 2
# 在 1 个位置插入异常
discord_pos = 550
discord_pattern = np.ones(pattern_len) * 3 + np.random.normal(0, 0.1, pattern_len)
base[discord_pos:discord_pos + pattern_len] += discord_pattern
signal = base
# ── 计算 Matrix Profile（简化版） ─────────────────────────────────
m = pattern_len
n_subs = n - m + 1
profile = np.full(n_subs, np.inf)
exclusion = m // 4
for i in range(n_subs):
    sub_i = signal[i:i + m]
    mu_i, std_i = sub_i.mean(), sub_i.std() + 1e-8
    sub_i_norm = (sub_i - mu_i) / std_i
    for j in range(n_subs):
        if abs(i - j) < exclusion:
            continue
        sub_j = signal[j:j + m]
        mu_j, std_j = sub_j.mean(), sub_j.std() + 1e-8
        sub_j_norm = (sub_j - mu_j) / std_j
        dist = np.sqrt(np.sum((sub_i_norm - sub_j_norm) ** 2))
        if dist < profile[i]:
            profile[i] = dist
# ── 找 Motif 和 Discord ──────────────────────────────────────────
motif_idx = np.argmin(profile)
discord_idx = np.argmax(profile)
# 找 top-3 motif（排除重叠）
motif_indices = []
profile_copy = profile.copy()
for _ in range(3):
    idx = np.argmin(profile_copy)
    motif_indices.append(idx)
    lo = max(0, idx - exclusion)
    hi = min(n_subs, idx + exclusion)
    profile_copy[lo:hi] = np.inf
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("图 8.4.1　Matrix Profile 统一 Motif 与 Discord 发现",
             fontsize=20, fontweight="bold", y=1.02)
# ── 上面板：原始信号 ──────────────────────────────────────────────
ax = axes[0]
ax.plot(t, signal, color=COLORS["blue"], lw=1.0, alpha=0.8)
# 标注 Motif 位置
for mi in motif_indices:
    ax.axvspan(mi, mi + m, alpha=0.2, color=COLORS["green"])
# 标注 Discord 位置
ax.axvspan(discord_idx, discord_idx + m, alpha=0.3, color=COLORS["red"])
ax.set_ylabel("信号值", fontsize=14)
ax.set_title("原始时间序列", fontsize=15)
ax.tick_params(labelsize=13)
# 添加标注
ax.text(motif_positions[0] + m // 2, signal[motif_positions[0]:motif_positions[0] + m].max() + 0.5,
        "Motif", fontsize=12, ha="center", color=COLORS["green"], fontweight="bold")
ax.text(discord_idx + m // 2, signal[discord_idx:discord_idx + m].max() + 0.5,
        "Discord", fontsize=12, ha="center", color=COLORS["red"], fontweight="bold")
# ── 下面板：Matrix Profile ────────────────────────────────────────
ax = axes[1]
ax.plot(np.arange(n_subs), profile, color=COLORS["purple"], lw=1.0)
# 标注 Motif 位置（谷值）
for mi in motif_indices:
    ax.plot(mi, profile[mi], 'v', color=COLORS["green"], markersize=10,
            zorder=5)
# 标注 Discord 位置（峰值）
ax.plot(discord_idx, profile[discord_idx], '^', color=COLORS["red"],
        markersize=10, zorder=5)
ax.set_ylabel("MP 值", fontsize=14)
ax.set_xlabel("子序列起始位置", fontsize=14)
ax.set_title("Matrix Profile", fontsize=15)
ax.tick_params(labelsize=13)
# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS["green"],
           markersize=10, label='Motif（谷值）'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["red"],
           markersize=10, label='Discord（峰值）'),
]
ax.legend(handles=legend_elements, fontsize=13, loc="upper right")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.2)
save_fig(fig, __file__, "fig8_4_01_motif_discord")
