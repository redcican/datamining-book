# Figure 14.3.2 — Data Quality Assessment Dashboard

## Prompt

Create a **professional, publication-quality dashboard infographic** depicting a data quality assessment report for a data mining project. The dashboard combines a radar chart, statistical panels, and a quality gate verdict.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two rows**:
  - **Top row** (~55% height): Radar chart (left, ~40%) + Quality score card (right, ~60%).
  - **Bottom row** (~45% height): Three statistical panels side by side.

### Top Left — Five-Dimension Radar Chart

- A **pentagon radar chart** with five axes:
  1. 完整性 (Completeness) — Score: 0.96
  2. 唯一性 (Uniqueness) — Score: 0.99
  3. 一致性 (Consistency) — Score: 0.94
  4. 准确性 (Accuracy) — Score: 0.97
  5. 时效性 (Timeliness) — Score: 0.88

- **Filled polygon**: Blue `#2563eb` at 20% opacity, with solid blue border (2pt).
- **Grid lines**: Five concentric pentagons at 0.2, 0.4, 0.6, 0.8, 1.0, in light gray `#e2e8f0`.
- **Axis labels**: Bilingual — Chinese name (10pt bold) above, score value (9pt, colored) below.
- **Score coloring**: ≥ 0.95 = green `#16a34a`, 0.85–0.95 = blue `#2563eb`, < 0.85 = orange `#ea580c`.
- **Title**: "五维度质量评估" in bold, 12pt, centered above the chart.

### Top Right — Quality Score Card

A **vertical card** with the following elements, stacked top-to-bottom:

1. **综合评分**: Large number "Q = 0.95" in bold green `#16a34a`, 36pt font.
   - Below: A horizontal bar (full width, green fill to 95%, gray remainder).
   - Threshold marker at 0.85 position, labeled "合格线 0.85" in 8pt.

2. **Individual scores table** (compact, 5 rows):
   | 维度 | 得分 | 权重 | 加权 | 状态 |
   |---|---|---|---|---|
   | 完整性 | 0.96 | 0.30 | 0.288 | ✅ |
   | 准确性 | 0.97 | 0.25 | 0.243 | ✅ |
   | 一致性 | 0.94 | 0.20 | 0.188 | ✅ |
   | 时效性 | 0.88 | 0.15 | 0.132 | ⚠️ |
   | 唯一性 | 0.99 | 0.10 | 0.099 | ✅ |

   Table font: 9pt, with row coloring matching the status (green rows for ✅, yellow for ⚠️).

3. **Verdict badge**: "质量关口 G2: ✅ 通过" in a green `#16a34a` rounded badge, centered, bold 11pt white text.

### Bottom Row — Three Statistical Panels

Three equal-width panels, each with a thin colored top border and white background:

#### Panel 1: 特征缺失率排行 (Missing Rate Ranking)

- **Color accent**: Blue `#2563eb` top border.
- A **horizontal bar chart** showing top-8 features sorted by missing rate (descending).
- Features with missing rate > 5%: bars in red `#dc2626`.
- Features with missing rate ≤ 5%: bars in blue `#2563eb`.
- Example data:
  - `wind_direction`: 2.1% (blue)
  - `precipitation`: 4.7% (blue)
  - `pm25_lag24`: 0.1% (blue)
  - (most features have very low missing rates)
- A vertical dashed line at 5% labeled "阈值 5%" in red.
- Title: "特征缺失率" in bold 10pt.

#### Panel 2: 标签分布 (Label Distribution)

- **Color accent**: Orange `#ea580c` top border.
- A **histogram/bar chart** showing the target variable distribution.
- For a classification example: two bars — "正常" (large, blue) and "异常/故障" (small, red).
- Annotations: sample counts and percentages on each bar.
- Label: "不平衡比: 28.5:1" in bold orange text below.
- Title: "标签分布" in bold 10pt.

#### Panel 3: 时间覆盖度 (Temporal Coverage)

- **Color accent**: Green `#16a34a` top border.
- A **calendar heatmap** or **horizontal timeline bar** showing data availability across time.
- Green cells/segments: data available.
- Gray cells/segments: data missing.
- Red outlined cells: data gaps that may affect model training.
- Example: 2010-2014 monthly grid, with a few gaps in early 2010.
- Title: "时间覆盖度" in bold 10pt.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Light gray `#f8fafc` overall background (to make white cards pop).
- **Card backgrounds**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 30 px between cards, ≥ 40 px outer margins.
- **Overall aesthetic**: Clean, dashboard-style — like a Grafana or Metabase monitoring panel, but for data quality.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_3_02_data_quality_report.png`
- Place in `public/figures/ch14/`
