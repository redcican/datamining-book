# Figure 12.1.3 — Data Mining Platform Positioning Bubble Chart

## Prompt

Create a **professional, publication-quality bubble chart** that maps major data mining platforms on a two-dimensional space of ease-of-use versus scalability, with bubble size encoding community size / adoption.

### Layout & Composition

- **Orientation**: Landscape (16:9, ~1600 × 900 px at 150 DPI).
- **Axes**:
  - **X-axis** (horizontal): "Ease of Use" scored 1–10, with tick marks at integers. Label at bottom center.
  - **Y-axis** (vertical): "Scalability (Data Volume)" scored 1–10, with tick marks at integers. Label at left center, rotated 90°.
- **Axis intersection** at (1, 1) in the bottom-left corner. Light gray gridlines at each integer.

### Quadrant System

Divide the chart into **four quadrants** using two dashed gray lines at x = 5.5 and y = 5.5. Label each quadrant in the corner with a subtle, italic, uppercase tag:

| Quadrant | Position | Label |
|----------|----------|-------|
| Bottom-Left | Low ease-of-use, low scalability | "NICHE TOOLS" |
| Top-Left | Low ease-of-use, high scalability | "BIG DATA ENGINES" |
| Bottom-Right | High ease-of-use, low scalability | "ENTRY-LEVEL TOOLS" |
| Top-Right | High ease-of-use, high scalability | "ALL-IN-ONE PLATFORMS" |

Each quadrant label should be rendered in 11 pt, `#94a3b8` gray, italic, positioned 16 px from the corner.

### Platform Bubbles

Each platform is represented as a **filled circle** with:
- **Position**: (ease_of_use, scalability) coordinates.
- **Size**: Proportional to community activity / user base (relative, not absolute).
- **Color**: From the book's `COLORS` palette, assigned by platform category.
- **Label**: Platform name in bold 11 pt, positioned adjacent to the bubble (auto-offset to avoid overlap).

| Platform | Ease of Use | Scalability | Relative Size | Color | Category |
|----------|------------|-------------|---------------|-------|----------|
| WEKA | 8.5 | 2.0 | Small (30) | Gray `#64748b` | Desktop GUI |
| Orange | 8.0 | 2.5 | Small (25) | Gray `#64748b` | Desktop GUI |
| scikit-learn | 7.0 | 4.0 | Large (80) | Blue `#2563eb` | Python Library |
| pandas + Jupyter | 6.5 | 3.5 | Large (85) | Blue `#2563eb` | Python Library |
| KNIME | 7.5 | 5.5 | Medium (55) | Green `#16a34a` | Workflow Platform |
| RapidMiner | 7.0 | 5.0 | Medium (45) | Green `#16a34a` | Workflow Platform |
| Spark MLlib | 3.5 | 9.0 | Large (75) | Orange `#ea580c` | Distributed Engine |
| Dask-ML | 4.5 | 7.0 | Small (30) | Orange `#ea580c` | Distributed Engine |
| AWS SageMaker | 6.5 | 9.5 | Large (90) | Red `#dc2626` | Cloud Platform |
| Google Vertex AI | 6.0 | 9.0 | Medium (70) | Red `#dc2626` | Cloud Platform |
| Azure ML | 6.0 | 8.5 | Medium (65) | Red `#dc2626` | Cloud Platform |
| Dataiku | 8.0 | 7.0 | Medium (60) | Purple `#9333ea` | Enterprise Platform |
| H2O.ai | 5.5 | 7.5 | Medium (50) | Purple `#9333ea` | Enterprise Platform |

### Visual Details

- **Bubble opacity**: 0.7 fill, 1.0 border (same color but darker shade).
- **Bubble border**: 1.5 pt solid stroke.
- **Label placement**: Use a smart offset strategy — place labels to the right of the bubble by default; if two bubbles are close, alternate above/below to avoid overlap. Connect label to bubble center with a thin 0.5 pt gray leader line if the offset is large.
- **Legend**: A horizontal legend strip at the bottom showing category colors:
  `● Desktop GUI   ● Python Library   ● Workflow Platform   ● Distributed Engine   ● Cloud Platform   ● Enterprise Platform`
- **Size legend**: Three reference circles (small / medium / large) in the bottom-right corner with labels indicating approximate community scale.

### Annotations

- Draw a **curved arrow** from the "ENTRY-LEVEL TOOLS" quadrant toward the "ALL-IN-ONE PLATFORMS" quadrant, labeled: *"Typical adoption path"* in 9 pt italic gray — suggesting that teams often start with simple tools and migrate upward.
- Optionally, highlight the **"sweet spot" region** (ease ≥ 6, scalability ≥ 6) with a very faint green `#16a34a` at 5% opacity rectangle, labeled "Production-Ready Zone".

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for any CJK text.
- **Color palette**: Book standard `COLORS` — `blue #2563eb`, `red #dc2626`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Gridlines**: 0.3 pt, `#f1f5f9`.
- **Axis lines**: 1 pt, `#cbd5e1`.
- Aesthetic: clean, data-dense scatter plot in the style of a Gartner Magic Quadrant or Forrester Wave, adapted for a textbook — informative but not cluttered.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_1_03_platform_positioning_bubble.png`
- Place in `public/figures/ch12/`
