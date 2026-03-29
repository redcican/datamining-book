# Figure 12.6.1 — Tool Evaluation Radar Charts (Four Comparative Groups)

## Prompt

Create a **professional, publication-quality 2×2 grid of radar charts** comparing data mining tools across seven evaluation dimensions, with each quadrant showing a different comparative grouping.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **2×2 grid** with 16 px gaps between quadrants. Each quadrant ~780 × 430 px.
- **Quadrant labels** in bold at top-left of each: "(a) Python Ecosystem", "(b) Visual Tools", "(c) Code vs Visual", "(d) Local vs Cloud".

### Radar Chart Specifications (all four)

Each radar chart has **7 axes** radiating from center, labeled clockwise:
1. Functional Coverage (功能覆盖)
2. Scalability (可扩展性)
3. Usability (易用性)
4. Ecosystem Maturity (生态成熟度)
5. Reproducibility (可复现性)
6. Deployment (部署能力)
7. Cost (成本)

Scale: 1–5, with concentric pentagon rings at 1, 2, 3, 4, 5. Grid lines in light gray `#e2e8f0`, axis labels in 8 pt dark gray.

### Quadrant (a) — Python Ecosystem Comparison

Three overlapping radar polygons:
| Tool | Color | Line Style | Data [FC, SC, US, EM, RE, DE, CO] |
|------|-------|-----------|-----------------------------------|
| scikit-learn | Blue `#2563eb` | Solid 2pt | [3, 2, 4, 5, 5, 3, 5] |
| XGBoost | Green `#16a34a` | Solid 2pt | [2, 3, 3, 4, 5, 3, 5] |
| Spark MLlib | Orange `#ea580c` | Solid 2pt | [3, 5, 2, 4, 3, 3, 3] |

Fill each polygon with its color at 15% opacity. Legend in top-right corner.

### Quadrant (b) — Visual Tools Comparison

Three overlapping radar polygons:
| Tool | Color | Line Style | Data |
|------|-------|-----------|------|
| KNIME | Green `#16a34a` | Solid 2pt | [4, 2, 5, 4, 4, 3, 4] |
| Orange | Purple `#9333ea` | Solid 2pt | [2, 1, 5, 2, 3, 1, 5] |
| RapidMiner | Orange `#ea580c` | Solid 2pt | [4, 2, 4, 3, 3, 3, 2] |

### Quadrant (c) — Code vs Visual

Two overlapping radar polygons:
| Tool | Color | Line Style | Data |
|------|-------|-----------|------|
| scikit-learn | Blue `#2563eb` | Solid 2pt | [3, 2, 4, 5, 5, 3, 5] |
| KNIME | Green `#16a34a` | Dashed 2pt | [4, 2, 5, 4, 4, 3, 4] |

Add annotation arrows pointing to where each tool excels: scikit-learn → "Ecosystem & Reproducibility", KNIME → "Usability & Coverage".

### Quadrant (d) — Local vs Cloud

Two overlapping radar polygons:
| Tool | Color | Line Style | Data |
|------|-------|-----------|------|
| scikit-learn + Spark | Blue `#2563eb` | Solid 2pt | [3, 4, 3, 5, 4, 3, 4] (averaged) |
| SageMaker | Red `#dc2626` | Solid 2pt | [5, 5, 3, 4, 4, 5, 2] |

Highlight the "Deployment" and "Scalability" axes where cloud dominates, and the "Cost" axis where local dominates.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese axis labels (bilingual: English above, Chinese below in 7pt gray).
- **Color palette**: Standard book COLORS per tool.
- **Radar grid**: 0.5 pt `#e2e8f0` gray pentagons, axis lines 0.5 pt.
- **Polygon borders**: 2 pt with corresponding color, fill at 15% opacity.
- **Data points**: Small filled circles (4 px) at each vertex.
- **Background**: Pure white `#ffffff`, each quadrant with very subtle `#fafafa` fill.
- **Legends**: Small, top-right of each quadrant, with colored line samples.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_6_01_evaluation_radar.png`
- Place in `public/figures/ch12/`
