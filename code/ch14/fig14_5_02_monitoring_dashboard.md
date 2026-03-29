# Figure 14.5.2 — Model Monitoring Dashboard

## Prompt

Create a **professional, publication-quality dashboard infographic** depicting a comprehensive model monitoring dashboard with three monitoring layers (data, model, system) and alert timeline.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Three rows**:
  - **Top row** (~25% height): Three overview cards (data health, model performance, system status).
  - **Middle row** (~50% height): Feature PSI heatmap (left) + prediction distribution comparison (right).
  - **Bottom row** (~25% height): Alert timeline.

### Top Row — Three Overview Cards

Three equal-width cards with colored left borders (4px):
fig14_5_01_deployment_patterns
#### Card 1: 数据健康 — Blue `#2563eb`

- **Traffic light**: Large green circle with "正常" label.
- **Key metric**: "PSI 均值 = 0.06" in blue, 14pt.
- **Mini line chart** (sparkline): PSI trend over 30 days, showing a gradual upward trend.
  - Y-axis: 0.00 to 0.30.
  - A dashed red horizontal line at 0.25 (alert threshold).
  - A dashed orange horizontal line at 0.10 (watch threshold).
- **Sub-label**: "5/5 关键特征正常" in green 8pt.

#### Card 2: 模型性能 — Orange `#ea580c`

- **Traffic light**: Yellow circle with "关注" label.
- **Key metric**: "7日 AUC = 0.891 (↓0.02)" in orange, 14pt.
- **Mini line chart**: AUC over 30 days showing slight decline in last week.
  - A dashed red line at target AUC threshold.
- **Sub-label**: "转化率: -3.2% (3日)" in orange 8pt with a small down-arrow icon.

#### Card 3: 系统状态 — Green `#16a34a`

- **Traffic light**: Green circle with "正常" label.
- **Three mini-metrics** in a row:
  - P50: 12ms (green)
  - P95: 45ms (green)
  - P99: 98ms (green, near threshold)
- **Mini bar**: Error rate = 0.3% (green bar, threshold at 1%).
- **Sub-label**: "QPS: 1,250/s | 可用性: 99.97%" in green 8pt.

### Middle Row Left (~55%) — Feature PSI Heatmap

A **matrix heatmap** with:

- **Rows** (y-axis): 8 feature names (e.g., `amount`, `frequency`, `recency`, `age`, `tenure`, `balance`, `channel`, `region`).
- **Columns** (x-axis): 8 weekly time windows (W1 through W8, most recent on right).
- **Cell colors**: Three-level color coding:
  - PSI < 0.10: Green `#16a34a` (low opacity ~30%).
  - 0.10 ≤ PSI < 0.25: Orange `#ea580c` (medium opacity ~50%).
  - PSI ≥ 0.25: Red `#dc2626` (high opacity ~70%).
- **Cell text**: PSI value (2 decimal places) in each cell, white on dark cells, dark on light cells.
- **Highlight**: One or two cells in the most recent column showing orange/red, indicating emerging drift.
- **Title**: "特征 PSI 热力图" in bold 10pt.
- **Legend**: Three colored boxes with labels: "< 0.10 正常", "0.10–0.25 关注", "≥ 0.25 告警".

### Middle Row Right (~45%) — Prediction Distribution Comparison

A **dual histogram** (overlaid):

- **X-axis**: Prediction score (0.0 to 1.0).
- **Y-axis**: Density.
- **Histogram 1**: Training period distribution — filled blue `#2563eb` at 30% opacity, blue outline.
  - Label: "训练期分布" with blue legend marker.
- **Histogram 2**: Current period distribution — filled orange `#ea580c` at 30% opacity, orange outline.
  - Label: "当前分布" with orange legend marker.
- **Overlap region**: Purple-ish where both histograms overlap.
- **Annotation**: "PSI = 0.12" in a callout box, with an arrow pointing to the region of maximum divergence.
- **Title**: "预测分布对比" in bold 10pt.

### Bottom Row — Alert Timeline

A **horizontal timeline** spanning 30 days (most recent on right):

- **Background**: Light gray bar running the full width.
- **Alert markers**: Small icons on the timeline at specific dates:

| Date | Alert Type | Icon | Color | Label |
|------|-----------|------|-------|-------|
| Day 5 | 系统延迟 | ⚡ | Orange | "P99 > 200ms (已解决)" |
| Day 12 | 数据缺失 | 📊 | Blue | "wind_dir 缺失率 8% (已修复)" |
| Day 22 | 模型性能 | 📉 | Orange | "AUC 下降 2% (观察中)" |
| Day 27 | 数据漂移 | 🔄 | Red | "channel PSI=0.28 (处理中)" |

- Each marker has a tooltip-style callout above with the description.
- **Resolved** alerts have a green checkmark overlay.
- **Active** alerts have a pulsing red/orange dot.
- **Title**: "告警时间线（近 30 天）" in bold 10pt.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Light gray `#f8fafc` overall.
- **Card backgrounds**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 20 px between cards, ≥ 40 px outer margins.
- **Overall aesthetic**: Grafana/Datadog-inspired monitoring dashboard, but static and publication-ready.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_5_02_monitoring_dashboard.png`
- Place in `public/figures/ch14/`
