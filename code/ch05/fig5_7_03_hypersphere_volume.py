"""
图 5.7.3　单位超球体积随维度的衰减曲线
左：V_d(1) 随维度变化（对数纵轴）
右：V_d(1) / 2^d（超球占超立方体的比例）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()

# ── 1. 计算超球体积 ──────────────────────────────────────
dims = np.arange(1, 101)
volumes = np.array([np.pi**(d/2) / gamma_func(d/2 + 1) for d in dims])
ratios = volumes / (2.0 ** dims)  # V_d / (2^d) = 占超立方体比例

peak_d = dims[np.argmax(volumes)]
peak_v = volumes[np.argmax(volumes)]

# ── 2. 绘图 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 左：体积
ax1.semilogy(dims, volumes, '-', color=COLORS['blue'], linewidth=2.5)
ax1.plot(peak_d, peak_v, 'o', color=COLORS['red'], markersize=10, zorder=10)
ax1.annotate(f'峰值 d={peak_d}\n$V_d$={peak_v:.3f}',
             xy=(peak_d, peak_v), xytext=(peak_d + 15, peak_v * 10),
             arrowprops=dict(arrowstyle='->', color=COLORS['red']),
             fontsize=11, color=COLORS['red'])
ax1.set_xlabel('维度 $d$')
ax1.set_ylabel('单位超球体积 $V_d(1)$')
ax1.set_title('超球体积随维度的衰减（对数纵轴）', fontsize=13)
ax1.set_xlim(0, 100)
ax1.axhline(y=1e-10, color=COLORS['gray'], linestyle=':', alpha=0.5)
ax1.text(80, 2e-10, '$10^{-10}$', fontsize=10, color=COLORS['gray'])

# 右：比例
ax2.semilogy(dims, ratios, '-', color=COLORS['orange'], linewidth=2.5)
ax2.set_xlabel('维度 $d$')
ax2.set_ylabel('$V_d(1) / 2^d$')
ax2.set_title('超球体积占超立方体比例（双指数衰减）', fontsize=13)
ax2.set_xlim(0, 100)

# 标注几个关键值
for d_mark in [2, 10, 20, 50]:
    idx = d_mark - 1
    ax2.plot(d_mark, ratios[idx], 'o', color=COLORS['red'], markersize=6, zorder=10)
    ax2.annotate(f'd={d_mark}: {ratios[idx]:.1e}',
                 xy=(d_mark, ratios[idx]),
                 xytext=(d_mark + 5, ratios[idx] * 5),
                 fontsize=9, color=COLORS['red'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=0.8))

plt.suptitle('维度诅咒：高维超球体积与空间占比', fontsize=14, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_7_03_hypersphere_volume')
