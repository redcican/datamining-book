"""
图 6.4.2　零不变性实验
固定 f11, f10, f01，逐步增大 f00，观察各度量变化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

# ── 固定频次 ──────────────────────────────────────────────────
f11, f10, f01 = 100, 200, 150
f00_values = np.arange(500, 50001, 500)

# ── 计算各度量 ────────────────────────────────────────────────
lifts, cosines, kulcs, all_confs, chi2s = [], [], [], [], []

for f00 in f00_values:
    n = f11 + f10 + f01 + f00
    pA = (f11 + f10) / n
    pB = (f11 + f01) / n
    pAB = f11 / n

    lifts.append(pAB / (pA * pB))
    cosines.append(pAB / np.sqrt(pA * pB))
    kulcs.append(0.5 * (pAB / pA + pAB / pB))
    all_confs.append(pAB / max(pA, pB))

    e11 = (f11+f10) * (f11+f01) / n
    e10 = (f11+f10) * (f10+f00) / n
    e01 = (f01+f00) * (f11+f01) / n
    e00 = (f01+f00) * (f10+f00) / n
    chi2 = sum((f-e)**2/e if e > 0 else 0
               for f, e in [(f11,e11),(f10,e10),(f01,e01),(f00,e00)])
    chi2s.append(chi2)

# ── 绘图 ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左：零不变度量（应为常数）
ax1.plot(f00_values, cosines, '-', color=COLORS['blue'], lw=2.5, label='余弦相似度')
ax1.plot(f00_values, kulcs, '-', color=COLORS['green'], lw=2.5, label='Kulczynski')
ax1.plot(f00_values, all_confs, '-', color=COLORS['orange'], lw=2.5, label='全置信度')
ax1.set_xlabel('$f_{00}$（双缺席事务数）', fontsize=14)
ax1.set_ylabel('度量值', fontsize=14)
ax1.set_title('零不变度量（不受 $f_{00}$ 影响）', fontsize=16)
ax1.legend(fontsize=13)
ax1.set_ylim(0, 0.8)

# 右：非零不变度量
ax2r = ax2.twinx()
l1, = ax2.plot(f00_values, lifts, '-', color=COLORS['red'], lw=2.5, label='提升度')
l2, = ax2r.plot(f00_values, chi2s, '--', color=COLORS['purple'], lw=2.5, label='$\\chi^2$')
ax2.set_xlabel('$f_{00}$（双缺席事务数）', fontsize=14)
ax2.set_ylabel('提升度', fontsize=14, color=COLORS['red'])
ax2r.set_ylabel('$\\chi^2$ 统计量', fontsize=14, color=COLORS['purple'])
ax2.set_title('非零不变度量（随 $f_{00}$ 变化）', fontsize=16)
ax2.legend(handles=[l1, l2], fontsize=13, loc='center right')

# 标注
ax2.annotate('$f_{00}$ 增大\n→ 数据更稀疏\n→ 提升度虚增',
             xy=(30000, lifts[59]), fontsize=13, color=COLORS['red'],
             xytext=(15000, max(lifts)*0.85),
             arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', fc='#fef2f2', ec=COLORS['red']))

plt.suptitle('零不变性实验（定义 6.18, 定理 6.10）：'
             f'$f_{{11}}$={f11}, $f_{{10}}$={f10}, $f_{{01}}$={f01}',
             fontsize=18, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_4_02_null_invariance')
