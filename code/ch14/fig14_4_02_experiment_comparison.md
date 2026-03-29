# Figure 14.4.2 — Experiment Comparison and Model Selection

## Prompt

Create a **professional, publication-quality infographic** depicting the experiment comparison heatmap and model selection decision pipeline. This figure helps data scientists systematically select the best model from multiple candidates.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two rows**:
  - **Top row** (~55% height): Experiment comparison heatmap.
  - **Bottom row** (~45% height): Model selection decision flowchart.

### Top Row — Experiment Comparison Heatmap

A **matrix heatmap** with the following structure:

- **Columns** (x-axis): 12 experiments labeled `exp-001` through `exp-012`, grouped by stage:
  - exp-001–002 (blue header): 基线
  - exp-003–007 (green header): 特征迭代
  - exp-008–010 (orange header): 模型迭代
  - exp-011–012 (purple header): 超参数调优

- **Rows** (y-axis): 4 evaluation metrics:
  - AUC (higher is better)
  - Recall@5% FPR (higher is better)
  - 推理延迟 ms (lower is better)
  - 模型大小 MB (lower is better)

- **Cell coloring**: Green-to-red diverging colormap:
  - For "higher is better" metrics: dark green `#16a34a` = best, red `#dc2626` = worst.
  - For "lower is better" metrics: dark green = lowest, red = highest.
  - Cell text: the actual numeric value in white (on dark cells) or dark gray (on light cells).

- **Highlighted column**: `exp-012` (the selected model) has a **yellow border** (3px, `#eab308`) and a small star icon ⭐ above the column header.

- **Stage group headers**: Colored bar above each group of experiments matching stage color.

### Bottom Row — Model Selection Decision Flowchart

A **horizontal decision flow** with four filtering steps, flowing left to right:

| Step | Shape | Label | Description | Color |
|------|-------|-------|-------------|-------|
| 1 | Rounded rect | 候选模型集 | 12 个实验结果 | Gray `#64748b` |
| 2 | Diamond | 统计显著性 | DeLong/McNemar p < 0.05 | Blue `#2563eb` |
| 3 | Diamond | 约束条件 | 延迟 < SLA, 大小 < 限制 | Orange `#ea580c` |
| 4 | Rounded rect | 北极星排序 | 按主指标排序取 Top-3 | Green `#16a34a` |
| 5 | Rounded rect (bold border) | 业务方评审 | 最终选择 | Purple `#9333ea` |

- Between each step, arrows with annotations:
  - Step 1→2: "12 → 8" (4 eliminated, not significantly better than baseline)
  - Step 2→3: "8 → 5" (3 eliminated, violate constraints)
  - Step 3→4: "5 → 3" (Top-3 by AUC)
  - Step 4→5: "3 → 1" (business review selects final)

- Each diamond has two outgoing arrows: "通过" (green, continues right) and "淘汰" (red, points down to a small "淘汰池" box).

- A funnel visual overlay (optional): gradually narrowing from left (wide) to right (narrow), reinforcing the filtering metaphor.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Light gray `#f8fafc`.
- **Card backgrounds**: Pure white `#ffffff`.
- **Margins**: ≥ 30 px between elements, ≥ 40 px outer margins.
- **Heatmap borders**: Light gray `#e2e8f0` between cells (1px).

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_4_02_experiment_comparison.png`
- Place in `public/figures/ch14/`
