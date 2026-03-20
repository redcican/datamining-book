"""
fig8_3_02_paa_sax.py
PAA 降维与 SAX 符号化过程
上：原始序列 + PAA 阶梯线  下：SAX 符号映射
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成原始时间序列 ──────────────────────────────────────────────
T = 128
t = np.arange(T)
raw = np.sin(2 * np.pi * t / 40) + 0.5 * np.sin(2 * np.pi * t / 17)
raw += np.random.normal(0, 0.15, T)
# Z-标准化
raw = (raw - raw.mean()) / raw.std()
# ── PAA 计算 ──────────────────────────────────────────────────────
w = 8  # PAA 段数
seg_len = T // w
paa_vals = np.array([raw[i * seg_len:(i + 1) * seg_len].mean()
                     for i in range(w)])
# PAA 阶梯线（用于绘图）
paa_line = np.repeat(paa_vals, seg_len)
# ── SAX 符号化 ────────────────────────────────────────────────────
alpha = 4  # 字母表大小
# 正态分布等概率分位点
breakpoints = [norm.ppf(i / alpha) for i in range(1, alpha)]
alphabet = ['a', 'b', 'c', 'd']
sax_symbols = []
for v in paa_vals:
    idx = 0
    for bp in breakpoints:
        if v > bp:
            idx += 1
    sax_symbols.append(alphabet[idx])
sax_string = ''.join(sax_symbols)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("图 8.3.2　PAA 降维与 SAX 符号化过程",
             fontsize=20, fontweight="bold", y=1.02)
# ── 上面板：原始序列 + PAA ────────────────────────────────────────
ax = axes[0]
ax.plot(t, raw, color=COLORS["blue"], lw=1.2, alpha=0.7, label="原始序列(Z-标准化)")
ax.step(t, paa_line, where="post", color=COLORS["red"], lw=2.5,
        label=f"PAA 表示 (w={w})")
# 画段分隔虚线
for i in range(1, w):
    ax.axvline(i * seg_len, color="gray", ls=":", lw=0.8, alpha=0.5)
ax.set_ylabel("值", fontsize=14)
ax.set_title("PAA 降维", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)
# ── 下面板：SAX 符号映射 ──────────────────────────────────────────
ax = axes[1]
# 先画 PAA 阶梯
ax.step(t, paa_line, where="post", color=COLORS["red"], lw=2.0, alpha=0.6,
        label="PAA 值")
# 画分位线
bp_colors = [COLORS["gray"]] * len(breakpoints)
for i, bp in enumerate(breakpoints):
    ax.axhline(bp, color="gray", ls="--", lw=1.2, alpha=0.5)
    label_text = f"β{i + 1} = {bp:.2f}"
    ax.text(T + 1, bp, label_text, fontsize=11, va="center", color="gray")
# 给每个区域着色
y_min, y_max = ax.get_ylim()
region_colors = [PALETTE[0], PALETTE[2], PALETTE[3], PALETTE[4]]
region_bounds = [raw.min() - 0.5] + breakpoints + [raw.max() + 0.5]
for i in range(len(region_bounds) - 1):
    ax.axhspan(region_bounds[i], region_bounds[i + 1],
               alpha=0.08, color=region_colors[i])
# 在每个 PAA 段中心标注 SAX 符号
for i in range(w):
    cx = (i + 0.5) * seg_len
    cy = paa_vals[i]
    ax.text(cx, cy + 0.25, sax_symbols[i], fontsize=16,
            fontweight="bold", ha="center", va="bottom",
            color=COLORS["purple"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["purple"],
                      alpha=0.9))
ax.set_xlabel("时间步", fontsize=14)
ax.set_ylabel("值", fontsize=14)
ax.set_title(f"SAX 符号化 → \"{sax_string}\"", fontsize=15)
ax.tick_params(labelsize=13)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.5)
save_fig(fig, __file__, "fig8_3_02_paa_sax")
