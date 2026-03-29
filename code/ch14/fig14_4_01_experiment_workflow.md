# Figure 14.4.1 — Model Development Iterative Workflow

## Prompt

Create a **professional, publication-quality infographic** depicting the five-stage model development workflow, from baseline to ensemble, with a diminishing-returns curve and experiment tracking panel.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Three horizontal bands**:
  - **Top band** (~15% height): Title + Sprint timeline.
  - **Middle band** (~55% height): Five-stage pipeline (left) + AUC performance curve (right).
  - **Bottom band** (~30% height): Experiment tracking table (MLflow-style).

### Middle Band — Five-Stage Pipeline (Left ~55%)

A **horizontal pipeline** with five rounded-rectangle stage cards connected by arrows:

| # | Stage | Color | Label | Typical Experiments | Expected Lift |
|---|-------|-------|-------|---------------------|---------------|
| 1 | 基线建立 | Blue `#2563eb` | MVM（默认参数） | 1–2 | — |
| 2 | 特征迭代 | Green `#16a34a` | 增删特征、变换、交互 | 5–15 | 最大（50–80%） |
| 3 | 模型迭代 | Orange `#ea580c` | LR → RF → GBDT | 3–5 | 中等 |
| 4 | 超参数调优 | Purple `#9333ea` | 网格搜索 / 贝叶斯 | 10–50 | 较小（< 5%） |
| 5 | 集成/融合 | Red `#dc2626` | Stacking / Blending | 2–5 | 最小 |

Each card contains:
- **Stage number** in a circle (bold, white text on colored background).
- **Stage name** (Chinese, 11pt bold).
- **Key activities** (8pt, 2–3 bullet points).
- **Experiment count badge**: e.g., "5–15 次实验" in a small pill badge.

Connecting arrows between stages:
- Solid right-pointing arrows (gradient from left-stage color to right-stage color).
- Small label above each arrow: "最大提升", "中等提升", "较小提升", "最小提升".

### Middle Band — Performance Curve (Right ~45%)

A **line chart** showing AUC on the y-axis (0.70 to 0.95) and cumulative experiment count on the x-axis (0 to ~75):

- **Curve shape**: Steep rise in stages 1–2, moderate rise in stage 3, plateau in stages 4–5.
- **Background shading**: Five vertical bands in stage colors (same as pipeline), with transparency at 10%.
- **Curve line**: Bold dark line (2pt, `#1e293b`).
- **Data points**: Small circles at key experiment milestones.
- **Red dashed horizontal line**: At a target AUC level (e.g., 0.90), labeled "目标指标".
- **Red dashed vertical annotation**: A vertical line at the "diminishing returns point" (around experiment 40), with annotation: "收益递减点：连续 3 次提升 < 0.5%" in red 8pt text.

### Bottom Band — Experiment Tracking Table

A compact **table** (MLflow-inspired) with the following columns:

| Exp ID | 阶段 | 变量 | AUC | Recall@5% | 延迟(ms) | 模型大小 | 状态 |
|--------|------|------|-----|-----------|----------|---------|------|
| exp-001 | 基线 | LR 默认 | 0.78 | 0.52 | 2 | 0.1 MB | ✅ 基线 |
| exp-003 | 特征 | +交互特征 | 0.85 | 0.64 | 3 | 0.1 MB | ✅ 提升 |
| exp-007 | 模型 | GBDT | 0.89 | 0.72 | 15 | 2.3 MB | ✅ 提升 |
| exp-012 | 调参 | lr=0.05 | 0.912 | 0.78 | 15 | 2.3 MB | ⭐ 最终 |

- Rows alternate in light gray/white.
- The "最终" row highlighted with a light yellow background.
- Status column uses colored badges: green for ✅, yellow-gold for ⭐.

### Top Band — Sprint Timeline

A thin **horizontal timeline** showing a 4-week Sprint:
- Week 1: "基线 + 特征" (blue+green segment)
- Week 2: "特征 + 模型" (green+orange segment)
- Week 3: "调参" (purple segment)
- Week 4: "集成 + 报告" (red segment)
- Each segment proportional to suggested time allocation.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Light gray `#f8fafc`.
- **Card backgrounds**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 30 px between elements, ≥ 40 px outer margins.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_4_01_experiment_workflow.png`
- Place in `public/figures/ch14/`
