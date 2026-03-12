"""
图 2.2.1  多源数据集成管道（Schema匹配→实体对齐→数据融合）
对应节次：2.2 数据集成与数据变换
运行方式：python code/ch02/fig2_2_01_integration_pipeline.py
输出路径：public/figures/ch02/fig2_2_01_integration_pipeline.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

apply_style()

# ── Bright color palette ──────────────────────────────────────────────────
C_A      = "#2563eb"
C_A_BG   = "#eff6ff"
C_B      = "#16a34a"
C_B_BG   = "#f0fdf4"
C_PIPE   = "#7c3aed"
C_PIPE_BG = "#f5f3ff"
C_OUT    = "#1e293b"
C_OUT_BG = "#f8fafc"
C_ARROW  = "#64748b"
C_WARN   = "#dc2626"
C_WARN_BG = "#fef2f2"

fig, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")


def fancy_box(ax, x, y, w, h, fc, ec, alpha=1.0, zorder=2, lw=2.0):
    p = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.25",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def arrow_h(ax, x0, y, x1, color=C_ARROW, lw=2.2):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=18), zorder=5)


def arrow_v(ax, x, y0, y1, color=C_ARROW):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                mutation_scale=16), zorder=5)


# ── Source A (blue, bright) ───────────────────────────────────────────────
fancy_box(ax, 0.3, 6.3, 6.0, 5.0, C_A_BG, C_A, zorder=2)
ax.text(3.3, 10.95, "数据源 A（医院 EHR 系统）", ha="center", va="center",
        fontsize=16, fontweight="bold", color=C_A, zorder=4)

schema_a = [
    ("PatientID",  "INT",      "001, 002, 003 …"),
    ("Name",       "VARCHAR",  "张伟, 李明 …"),
    ("DOB",        "VARCHAR",  "1985/03/15, 1990-07-20"),
    ("SystolicBP", "FLOAT",    "130, 125, 118 …"),
    ("DiagCode",   "VARCHAR",  "I10, E11 …"),
]
y_a = 10.35
for fname, ftype, example in schema_a:
    ax.text(0.7, y_a, fname, ha="left", va="center",
            fontsize=13, color="#1e40af", fontweight="bold", zorder=4)
    ax.text(3.3, y_a, ftype, ha="left", va="center",
            fontsize=12, color="#64748b", zorder=4)
    ax.text(6.0, y_a, example, ha="right", va="center",
            fontsize=12, color="#475569", style="italic", zorder=4)
    y_a -= 0.65
ax.plot([0.5, 6.1], [10.05, 10.05], color=C_A + "55", lw=0.8, zorder=3)

# Warning placed below all schema rows (last row DiagCode at y=7.75)
fancy_box(ax, 0.45, 6.45, 5.7, 0.55, C_WARN_BG, C_WARN, alpha=0.9, zorder=3, lw=1.5)
ax.text(6.0, 6.73, "⚠ 日期格式不统一", ha="right", va="center",
        fontsize=12, color=C_WARN, fontweight="bold", zorder=5)

# ── Source B (green, bright) ──────────────────────────────────────────────
fancy_box(ax, 0.3, 0.5, 6.0, 5.2, C_B_BG, C_B, zorder=2)
ax.text(3.3, 5.35, "数据源 B（体检中心系统）", ha="center", va="center",
        fontsize=16, fontweight="bold", color=C_B, zorder=4)

schema_b = [
    ("patient_no", "VARCHAR",  "A001, B003 …"),
    ("full_name",  "VARCHAR",  "张 伟, 王芳 …"),
    ("birthday",   "DATE",     "15-Mar-1985, 1995.09.10"),
    ("sys_press",  "INT",      "130, 110 …"),
    ("disease_id", "VARCHAR",  "ICD: I10 …"),
]
y_b = 4.75
for fname, ftype, example in schema_b:
    ax.text(0.7, y_b, fname, ha="left", va="center",
            fontsize=13, color="#14532d", fontweight="bold", zorder=4)
    ax.text(3.3, y_b, ftype, ha="left", va="center",
            fontsize=12, color="#64748b", zorder=4)
    ax.text(6.0, y_b, example, ha="right", va="center",
            fontsize=12, color="#475569", style="italic", zorder=4)
    y_b -= 0.65
ax.plot([0.5, 6.1], [4.45, 4.45], color=C_B + "55", lw=0.8, zorder=3)

# Warnings placed below all schema rows (last row disease_id at y=2.15)
fancy_box(ax, 0.45, 1.35, 5.7, 0.55, C_WARN_BG, C_WARN, alpha=0.9, zorder=3, lw=1.5)
ax.text(6.0, 1.63, "⚠ 与 A 格式冲突（日期）", ha="right", va="center",
        fontsize=12, color=C_WARN, fontweight="bold", zorder=5)

fancy_box(ax, 0.45, 0.63, 5.7, 0.55, C_WARN_BG, C_WARN, alpha=0.9, zorder=3, lw=1.5)
ax.text(6.0, 0.91, "⚠ 命名冲突（patient_no vs PatientID）", ha="right", va="center",
        fontsize=12, color=C_WARN, fontweight="bold", zorder=5)

# ── Arrows: sources → pipeline ─────────────────────────────────────────────
arrow_h(ax, 6.3, 8.8, 7.3, color=C_A, lw=2.5)
arrow_h(ax, 6.3, 3.0, 7.3, color=C_B, lw=2.5)

# ── Pipeline stages (center column, bright purple) ────────────────────────
stage_data = [
    (7.5, 8.0, "① Schema 匹配",
     "字段名映射：PatientID ↔ patient_no\n"
     "类型对齐：FLOAT ↔ INT\n"
     "语义对齐：DOB ↔ birthday → ISO 8601"),
    (7.5, 5.0, "② 实体对齐",
     "规范化键：姓名标准化 + 出生日期\n"
     "相似度：张伟 ≈ 张 伟（编辑距离=1）\n"
     "匹配：001 ↔ A001（同一患者）"),
    (7.5, 1.7, "③ 数据融合",
     "冲突消解：日期→ 1985-03-15\n"
     "空值策略：B 无诊断码→ NaN\n"
     "字段重命名→统一 Schema"),
]

for bx, by, title, desc in stage_data:
    fancy_box(ax, bx, by - 1.05, 5.2, 2.2, C_PIPE_BG, C_PIPE, zorder=3)
    ax.text(bx + 2.6, by + 0.80, title,
            ha="center", va="center", fontsize=16,
            fontweight="bold", color=C_PIPE, zorder=5)
    ax.plot([bx + 0.15, bx + 5.05], [by + 0.50, by + 0.50],
            color=C_PIPE + "55", lw=0.8, zorder=4)
    for i, line in enumerate(desc.split("\n")):
        ax.text(bx + 2.6, by + 0.18 - i * 0.42, line,
                ha="center", va="center", fontsize=12,
                color="#4c1d95", zorder=5)

arrow_v(ax, 10.1, 6.95, 6.20, color=C_PIPE)
arrow_v(ax, 10.1, 3.95, 2.87, color=C_PIPE)

# ── Arrow: pipeline → output ────────────────────────────────────────────────
arrow_h(ax, 12.7, 5.5, 13.7, color=C_OUT, lw=2.8)

# ── Output (bright) ──────────────────────────────────────────────────────
fancy_box(ax, 13.7, 0.5, 6.0, 10.8, C_OUT_BG, C_OUT, zorder=2)
ax.text(16.7, 10.85, "集成数据集", ha="center", va="center",
        fontsize=17, fontweight="bold", color=C_OUT, zorder=4)
ax.text(16.7, 10.45, "（统一 Schema）", ha="center", va="center",
        fontsize=13, color="#475569", style="italic", zorder=4)
ax.plot([13.9, 19.5], [10.15, 10.15], color="#cbd5e1", lw=1.0, zorder=3)

out_schema = [
    ("unified_id",  "VARCHAR",  "P001, P002, P003…"),
    ("name",        "VARCHAR",  "张伟, 李明, 王芳…"),
    ("birth_date",  "DATE",     "1985-03-15…"),
    ("systolic_bp", "INT",      "130, 125, 110…"),
    ("diag_code",   "VARCHAR",  "I10, E11, NaN…"),
    ("source",      "VARCHAR",  "A, A, B…"),
]
y_o = 9.65
for fname, ftype, example in out_schema:
    ax.text(14.0, y_o, fname, ha="left", va="center",
            fontsize=13, color=C_OUT, fontweight="bold", zorder=4)
    ax.text(16.5, y_o, ftype, ha="left", va="center",
            fontsize=12, color="#64748b", zorder=4)
    ax.text(19.5, y_o, example, ha="right", va="center",
            fontsize=12, color="#475569", style="italic", zorder=4)
    y_o -= 0.65
ax.plot([13.9, 19.5], [9.35, 9.35], color="#cbd5e1", lw=0.8, zorder=3)

# Sample data rows
rows = [
    ("P001", "张伟", "1985-03-15", "130", "I10", "A"),
    ("P002", "李明", "1990-07-20", "NaN", "E11", "A"),
    ("P003", "王芳", "1995-09-10", "110", "NaN", "B"),
]
row_colors = [C_A_BG, "white", C_B_BG]
y_r = 4.8
for row, rc in zip(rows, row_colors):
    fancy_box(ax, 13.85, y_r - 0.28, 5.65, 0.58, rc, "#cbd5e1",
              alpha=0.9, zorder=3, lw=1.2)
    ax.text(16.7, y_r, "  ".join(row), ha="center", va="center",
            fontsize=12, color="#1e293b", zorder=4)
    y_r -= 0.65

ax.text(16.7, 1.8,
        "记录数：3（来源 A: 2, 来源 B: 1）\n"
        "不匹配：1 条（李明，仅存于 A 系统）",
        ha="center", va="center", fontsize=12, color="#475569",
        style="italic", zorder=4)

# ── Global title ──────────────────────────────────────────────────────────
ax.set_title("多源数据集成三阶段管道：Schema 匹配 → 实体对齐 → 数据融合",
             fontsize=20, pad=12)

save_fig(fig, __file__, "fig2_2_01_integration_pipeline")
