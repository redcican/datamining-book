# Figure 14.7.1 — Three-Case Project Retrospective Comparison

## Prompt

Create a **professional, publication-quality infographic** comparing three data mining case studies (fraud detection, predictive maintenance, PM2.5 prediction) across CRISP-DM completion stages and success/failure patterns.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Three sections**:
  - **Left** (~40%): Overlaid radar chart (CRISP-DM stages).
  - **Right** (~60%): Success/failure pattern matrix.
  - **Bottom strip**: Key metrics summary bar.

### Left — CRISP-DM Completion Radar Chart

A **hexagonal radar chart** with six axes representing CRISP-DM stages:

1. ① 业务理解 (Business Understanding)
2. ② 数据理解 (Data Understanding)
3. ③ 数据准备 (Data Preparation)
4. ④ 建模 (Modeling)
5. ⑤ 评估 (Evaluation)
6. ⑥ 部署 (Deployment)

- **Scale**: 1 to 5 (corresponding to ★ ratings), with grid lines at 1, 2, 3, 4, 5.
- **Three overlaid polygons**:

| Case | Color | Line Style | Fill Opacity |
|------|-------|-----------|-------------|
| 欺诈检测 (§13.2) | Blue `#2563eb` | Solid 2pt | 10% |
| 预测性维护 (§13.6) | Orange `#ea580c` | Dashed 2pt | 10% |
| PM2.5 预测 (§13.7) | Green `#16a34a` | Dotted 2pt | 10% |

- **Data points** (from the text):
  - 欺诈检测: [3, 4, 5, 4, 3, 1]
  - 预测性维护: [4, 5, 3, 4, 4, 1]
  - PM2.5 预测: [3, 5, 5, 4, 4, 1]

- **Common weak point**: Stage 6 (部署) is visibly the smallest for all three → annotate with a red callout: "共性短板：部署阶段" with an arrow pointing to the ⑥ axis.

- **Legend**: Three colored lines with case names, positioned below the chart.
- **Title**: "CRISP-DM 六阶段完成度" in bold 11pt.

### Right — Success/Failure Pattern Matrix

A **matrix table** with:

- **Columns** (x-axis): Three cases:
  - 欺诈检测 (blue header)
  - 预测性维护 (orange header)
  - PM2.5 预测 (green header)

- **Rows** (y-axis): Two groups:

**成功模式** (green section header):
| Pattern | 欺诈 | 维护 | PM2.5 |
|---------|------|------|-------|
| S1 基线优先 | ✅ | ✅ | ✅ |
| S2 时间分割 | ✅ | — | ✅ |
| S3 非对称损失 | ✅ | ✅ | — |
| S4 诚实报告 | — | — | ✅ |
| S5 双路径设计 | — | ✅ | — |
| S6 领域特征 | — | ✅ | ✅ |

**失败陷阱** (red section header):
| Pattern | 欺诈 | 维护 | PM2.5 |
|---------|------|------|-------|
| F1 忽略业务约束 | ❌ | — | — |
| F2 静态化时序 | — | ❌ | — |
| F3 忽略部署 | ❌ | ❌ | ❌ |
| F4 缺子群体分析 | ❌ | — | — |
| F5 未定义时域 | — | — | ❌ |

- **Cell icons**:
  - ✅: Green checkmark icon `#16a34a` (pattern successfully applied).
  - ❌: Red cross icon `#dc2626` (trap triggered).
  - —: Gray dash `#94a3b8` (not applicable).

- Cell backgrounds: light green tint for ✅ rows in success section, light red tint for ❌ rows in failure section.

- **Title**: "成功模式与失败陷阱" in bold 11pt.

### Bottom Strip — Key Metrics Summary

A thin **horizontal bar** (full width, ~80px height) with three summary cards:

| Case | Metric | Data Size | Key Challenge |
|------|--------|-----------|---------------|
| 欺诈检测 | AUPRC = 0.86 | 284,807 | 0.172% 极端不平衡 |
| 预测性维护 | Recall = 0.91 | 10,000 | 5 种故障模式 |
| PM2.5 预测 | R² = 0.947 | 43,824 | 强自相关时序 |

- Each card has the case color as left border, case name in bold, and metrics in a compact layout.
- Separated by thin vertical dividers.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Light gray `#f8fafc`.
- **Card backgrounds**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 20 px between sections, ≥ 40 px outer margins.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_7_01_case_comparison.png`
- Place in `public/figures/ch14/`
