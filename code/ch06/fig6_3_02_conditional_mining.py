"""
图 6.3.2　FP-Growth 条件模式基与递归挖掘过程
展示从项头表逆序提取条件模式基并构建条件 FP-Tree 的过程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.subplots_adjust(hspace=0.40, wspace=0.20)

steps = [
    {'item': 'D', 'color': COLORS['purple'],
     'cpb': ['(A,B,E):1', '(A,C):1'], 'cfreq': 'A:2',
     'pats': ['{D}:2  {A,D}:2']},
    {'item': 'E', 'color': COLORS['red'],
     'cpb': ['(A,B,C):2', '(A,B):1', '(B,C):1'], 'cfreq': 'A:3, B:4, C:3',
     'pats': ['{E}:4  {B,E}:4  {A,E}:3',
              '{C,E}:3  {A,B,E}:3  {B,C,E}:3',
              '{A,C,E}:2  {A,B,C,E}:2']},
    {'item': 'C', 'color': COLORS['orange'],
     'cpb': ['(A,B):2', '(A):1', '(B):1'], 'cfreq': 'A:3, B:3',
     'pats': ['{C}:4  {A,C}:3  {B,C}:3', '{A,B,C}:2']},
    {'item': 'B', 'color': COLORS['green'],
     'cpb': ['(A):3'], 'cfreq': 'A:3',
     'pats': ['{B}:4  {A,B}:3']},
    {'item': 'A', 'color': COLORS['blue'],
     'cpb': [], 'cfreq': '',
     'pats': ['{A}:4']},
]

FS_TITLE = 18
FS_LABEL = 16
FS_DATA = 16
FS_PAT = 15

for idx, s in enumerate(steps):
    ax = axes[idx // 3][idx % 3]
    c = s['color']
    ax.set_title(f"处理项 {s['item']}", fontsize=FS_TITLE, color=c, fontweight='bold')

    y = 0.93
    ax.text(0.04, y, '条件模式基:', fontsize=FS_LABEL, fontweight='bold',
            color='#64748b', transform=ax.transAxes, va='top')
    y -= 0.11
    if s['cpb']:
        for line in s['cpb']:
            ax.text(0.06, y, line, fontsize=FS_DATA, fontfamily='monospace',
                    color='black', transform=ax.transAxes, va='top')
            y -= 0.10
    else:
        ax.text(0.06, y, '(空)', fontsize=FS_DATA, color='#94a3b8',
                transform=ax.transAxes, va='top')
        y -= 0.10

    y -= 0.02
    ax.text(0.04, y, '条件频繁项:', fontsize=FS_LABEL, fontweight='bold',
            color='#64748b', transform=ax.transAxes, va='top')
    y -= 0.11
    if s['cfreq']:
        ax.text(0.06, y, s['cfreq'], fontsize=FS_DATA, fontfamily='monospace',
                color='black', transform=ax.transAxes, va='top')
    else:
        ax.text(0.06, y, '(无)', fontsize=FS_DATA, color='#94a3b8',
                transform=ax.transAxes, va='top')
    y -= 0.13

    ax.text(0.04, y, '频繁模式:', fontsize=FS_LABEL, fontweight='bold',
            color=c, transform=ax.transAxes, va='top')
    y -= 0.11
    for row in s['pats']:
        ax.text(0.06, y, row, fontsize=FS_PAT, fontweight='bold',
                color=c, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc=c, alpha=0.08,
                          ec=c, lw=1))
        y -= 0.11

    ax.axis('off')

# ── 汇总面板 ──────────────────────────────────────────────────
ax = axes[1][2]
ax.set_title('挖掘汇总', fontsize=FS_TITLE, fontweight='bold', color='#64748b')

data = [('D', 2, 2, COLORS['purple']),
        ('E', 3, 8, COLORS['red']),
        ('C', 3, 4, COLORS['orange']),
        ('B', 1, 2, COLORS['green']),
        ('A', 0, 1, COLORS['blue'])]

for j, h in enumerate(['项', 'CPB', '模式']):
    ax.text(0.15 + j * 0.30, 0.93, h, fontsize=FS_LABEL, fontweight='bold',
            color='#64748b', ha='center', transform=ax.transAxes, va='top')
ax.plot([0.02, 0.95], [0.88, 0.88], '-', color='#cbd5e1', lw=1.5,
        transform=ax.transAxes)

for i, (item, cpb_n, pat_n, col) in enumerate(data):
    y = 0.83 - i * 0.13
    ax.text(0.15, y, item, fontsize=FS_DATA, fontweight='bold', color=col,
            ha='center', transform=ax.transAxes, va='top')
    ax.text(0.45, y, str(cpb_n), fontsize=FS_DATA, color='black',
            ha='center', transform=ax.transAxes, va='top')
    ax.text(0.75, y, str(pat_n), fontsize=FS_DATA, color='black',
            ha='center', transform=ax.transAxes, va='top')

ax.plot([0.02, 0.95], [0.17, 0.17], '-', color='#cbd5e1', lw=1.5,
        transform=ax.transAxes)
ax.text(0.15, 0.11, '合计', fontsize=FS_DATA, fontweight='bold', color='black',
        ha='center', transform=ax.transAxes, va='top')
ax.text(0.75, 0.11, '17', fontsize=20, fontweight='bold', color=COLORS['red'],
        ha='center', transform=ax.transAxes, va='top')

ax.axis('off')

plt.suptitle('FP-Growth 条件模式基递归挖掘过程（定义 6.9–6.11）',
             fontsize=20, y=1.01)
save_fig(fig, __file__, 'fig6_3_02_conditional_mining')
