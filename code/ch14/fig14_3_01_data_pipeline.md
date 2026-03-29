# Figure 14.3.1 — Data Acquisition Pipeline for Data Mining Projects

## Prompt

Create a **professional, publication-quality horizontal pipeline infographic** depicting the end-to-end data acquisition workflow for a data mining project, from requirements analysis to quality gate G2.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Main element**: A **horizontal pipeline** flowing left-to-right, with four major stages connected by blue gradient arrows.
- **Bottom layer**: A thin blue pipeline/conveyor-belt visual running the full width, unifying the four stages.
- **Top layer**: Stage cards (detailed content) positioned above the pipeline at each stage.

### Four Pipeline Stages

Each stage is a **rounded-rectangle card** (~320 × 400 px) with a colored top border (4px) and white background:

| # | Stage Name (Chinese) | Color | Key Activities | Deliverable |
|---|---|---|---|---|
| 1 | 需求分析 | Blue `#2563eb` | 从挖掘目标反推; 列数据需求清单; 标注优先级 | 📋 数据需求清单 |
| 2 | 源评估 | Green `#16a34a` | 四维度评估 (质量/可及性/时效/合规); 内部 vs 外部; 合规一票否决 | 📊 数据源评估报告 |
| 3 | 质量验证 | Orange `#ea580c` | 五维度检查; 综合评分 Q; 样本量评估; 标签审计 | 📈 数据质量报告 |
| 4 | 文档化 | Purple `#9333ea` | 数据字典; 质量报告; 风险标注; G2 审查 | 📄 数据字典 + 质量报告 |

### Card Internal Layout

Each card contains:
- **Header**: Stage number (large, bold, colored) + stage name (bold, 12pt).
- **Body**: 3–4 bullet points of key activities (9pt, dark gray).
- **Footer**: Deliverable name in a small badge (colored background, white text, 8pt bold).

### Pipeline Arrows

- Between each stage card, a **gradient arrow** (left-stage-color → right-stage-color, 2pt, with arrowhead) runs along the bottom pipeline.
- Label above each arrow: brief transition description (8pt gray):
  - Stage 1→2: "需要什么？→ 从哪获取？"
  - Stage 2→3: "有数据 → 质量如何？"
  - Stage 3→4: "质量达标 → 正式记录"

### Feedback Paths (Red Dashed Arrows)

- A **red dashed arrow** (`#dc2626`, 1pt) from Stage 3 back to Stage 2, arching above the cards.
  - Label: "质量不达标 → 寻找替代数据源" in 8pt red.
- A smaller **red dashed arrow** from Stage 3 back to Stage 1, arching higher.
  - Label: "数据不可行 → 重新评估目标" in 7pt red.

### Right Endpoint — Quality Gate G2

- At the far right, a **gate icon** (stylized barrier, red `#dc2626` with green checkmark overlay).
- Label: "质量关口 G2" in bold, 11pt.
- Below: "通过 → 进入建模阶段 (§14.4)" in 9pt green `#16a34a`.
- Below that: "未通过 → 返回修正" in 9pt red `#dc2626`.

### Bottom Annotation

- Below the pipeline, a thin horizontal bar with three key statistics:
  - "数据工作 = 项目 60–80% 时间" (blue text)
  - "五维度质量评分 Q ≥ 0.85" (orange text)
  - "合规性: 一票否决" (red text)

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 40 px on all sides.
- **Overall aesthetic**: Clean, structured, process-oriented — like a well-designed DevOps pipeline diagram.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_3_01_data_pipeline.png`
- Place in `public/figures/ch14/`
