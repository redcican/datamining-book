"""
fig8_4_03_pattern_comparison.py
模式发现方法的适用场景对比
左：Matrix Profile 在长序列上发现 Motif + Discord
右：Shapelet 在两类短序列上提取判别形状
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
# ── 左面板数据：长序列 + Matrix Profile ───────────────────────────
n = 600
t = np.arange(n)
signal = np.random.normal(0, 0.3, n)
# 插入 3 个 Motif
motif_pattern = np.sin(np.linspace(0, 2 * np.pi, 40)) * 2
motif_locs = [50, 250, 450]
for loc in motif_locs:
    signal[loc:loc + 40] += motif_pattern
# 插入 1 个 Discord
discord_loc = 350
signal[discord_loc:discord_loc + 40] += np.ones(40) * 3
# 简化 Matrix Profile 计算
m = 40
n_subs = n - m + 1
profile = np.full(n_subs, np.inf)
excl = m // 4
for i in range(0, n_subs, 2):  # 步长2加速
    sub_i = signal[i:i + m]
    mu_i, std_i = sub_i.mean(), sub_i.std() + 1e-8
    sub_i_n = (sub_i - mu_i) / std_i
    for j in range(0, n_subs, 2):
        if abs(i - j) < excl:
            continue
        sub_j = signal[j:j + m]
        mu_j, std_j = sub_j.mean(), sub_j.std() + 1e-8
        sub_j_n = (sub_j - mu_j) / std_j
        d = np.sqrt(np.sum((sub_i_n - sub_j_n) ** 2))
        if d < profile[i]:
            profile[i] = d
# 插值填充跳过的位置
for i in range(1, n_subs - 1, 2):
    profile[i] = (profile[i - 1] + profile[i + 1]) / 2
if n_subs % 2 == 0:
    profile[-1] = profile[-2]
# ── 右面板数据：两类短序列 ────────────────────────────────────────
T_short = 60
t_short = np.linspace(0, 2 * np.pi, T_short)
n_each = 5
# 类 A：平滑正弦
class_a_short = []
for _ in range(n_each):
    s = np.sin(t_short) + np.random.normal(0, 0.1, T_short)
    class_a_short.append(s)
# 类 B：正弦 + 中段尖刺
class_b_short = []
for _ in range(n_each):
    s = np.sin(t_short) + np.random.normal(0, 0.1, T_short)
    s[25:35] += np.sin(np.linspace(0, np.pi, 10)) * 1.5
    class_b_short.append(s)
# 最优 Shapelet 区域
shapelet_region = (25, 35)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                         gridspec_kw={"height_ratios": [1.2, 1]})
fig.suptitle("图 8.4.3　模式发现方法的适用场景对比",
             fontsize=20, fontweight="bold", y=1.02)
# ── 左上：长序列原始信号 ──────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t, signal, color=COLORS["blue"], lw=0.8, alpha=0.8)
for loc in motif_locs:
    ax.axvspan(loc, loc + m, alpha=0.15, color=COLORS["green"])
ax.axvspan(discord_loc, discord_loc + m, alpha=0.2, color=COLORS["red"])
ax.set_ylabel("信号值", fontsize=13)
ax.set_title("(a) Matrix Profile：单条长序列", fontsize=14)
ax.tick_params(labelsize=12)
ax.text(motif_locs[0] + m // 2, 3.5, "Motif", fontsize=11,
        ha="center", color=COLORS["green"], fontweight="bold")
ax.text(discord_loc + m // 2, 4.5, "Discord", fontsize=11,
        ha="center", color=COLORS["red"], fontweight="bold")
# ── 左下：Matrix Profile ─────────────────────────────────────────
ax = axes[1, 0]
ax.plot(np.arange(n_subs), profile, color=COLORS["purple"], lw=1.0)
# 标注
motif_mp_idx = np.argmin(profile)
discord_mp_idx = np.argmax(profile)
ax.plot(motif_mp_idx, profile[motif_mp_idx], 'v', color=COLORS["green"],
        markersize=10, zorder=5, label="Motif")
ax.plot(discord_mp_idx, profile[discord_mp_idx], '^', color=COLORS["red"],
        markersize=10, zorder=5, label="Discord")
ax.set_xlabel("位置", fontsize=13)
ax.set_ylabel("MP 值", fontsize=13)
ax.set_title("Matrix Profile", fontsize=14)
ax.legend(fontsize=12, loc="upper right")
ax.tick_params(labelsize=12)
# ── 右上：两类短序列 ──────────────────────────────────────────────
ax = axes[0, 1]
for s in class_a_short:
    ax.plot(np.arange(T_short), s, color=COLORS["blue"], lw=0.8, alpha=0.5)
for s in class_b_short:
    ax.plot(np.arange(T_short), s, color=COLORS["red"], lw=0.8, alpha=0.5)
ax.axvspan(shapelet_region[0], shapelet_region[1], alpha=0.15,
           color=COLORS["green"])
ax.set_ylabel("值", fontsize=13)
ax.set_title("(b) Shapelet：两类短序列", fontsize=14)
ax.tick_params(labelsize=12)
from matplotlib.patches import Patch
legend_el = [
    Patch(facecolor=COLORS["blue"], alpha=0.5, label="类 A"),
    Patch(facecolor=COLORS["red"], alpha=0.5, label="类 B"),
    Patch(facecolor=COLORS["green"], alpha=0.15, label="Shapelet 区域"),
]
ax.legend(handles=legend_el, fontsize=11, loc="upper right")
# ── 右下：Shapelet 判别区域放大 ───────────────────────────────────
ax = axes[1, 1]
# 放大 Shapelet 区域
for s in class_a_short:
    seg = s[shapelet_region[0]:shapelet_region[1]]
    ax.plot(np.arange(len(seg)), seg, color=COLORS["blue"], lw=1.5, alpha=0.4)
for s in class_b_short:
    seg = s[shapelet_region[0]:shapelet_region[1]]
    ax.plot(np.arange(len(seg)), seg, color=COLORS["red"], lw=1.5, alpha=0.4)
# 平均形状
mean_a = np.mean([s[shapelet_region[0]:shapelet_region[1]]
                  for s in class_a_short], axis=0)
mean_b = np.mean([s[shapelet_region[0]:shapelet_region[1]]
                  for s in class_b_short], axis=0)
ax.plot(np.arange(len(mean_a)), mean_a, color=COLORS["blue"], lw=2.5,
        label="类 A 均值")
ax.plot(np.arange(len(mean_b)), mean_b, color=COLORS["red"], lw=2.5,
        label="类 B 均值")
ax.set_xlabel("时间步（局部放大）", fontsize=13)
ax.set_ylabel("值", fontsize=13)
ax.set_title("Shapelet 区域放大", fontsize=14)
ax.legend(fontsize=12, loc="upper right")
ax.tick_params(labelsize=12)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.5)
save_fig(fig, __file__, "fig8_4_03_pattern_comparison")
