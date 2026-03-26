# Figure 12.2.1 — Python Data Science Ecosystem Layered Architecture

## Prompt

Create a **professional, publication-quality layered architecture diagram** depicting the four-layer Python data science technology stack and its inter-layer data flow.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Three-column layout**:
  - **Left column (~12%)**: Workflow stage labels.
  - **Center column (~60%)**: Four-layer stack (hero element).
  - **Right column (~28%)**: Mapping to the 5-layer platform architecture from §12.1.

### Center — Four-Layer Stack

Four **rounded rectangles** stacked vertically with 10 px gaps, ordered bottom-to-top. Each layer contains library name badges (small rounded pill shapes) arranged horizontally inside:

| Layer | Label | Color Fill | Libraries (as pills inside) |
|-------|-------|-----------|----------------------------|
| 1 (bottom) | Foundation | Blue `#2563eb` at 12% opacity, border `#2563eb` | **NumPy** (bold, largest pill) · SciPy |
| 2 | Data Manipulation | Green `#16a34a` at 12% opacity, border `#16a34a` | **pandas** (bold, largest) · Polars |
| 3 (widest, ~15% wider) | Modeling & Evaluation | Orange `#ea580c` at 12% opacity, border `#ea580c` | **scikit-learn** (bold, center, largest) · XGBoost · LightGBM · | · PyTorch · TensorFlow |
| 4 (top) | Visualization & Interaction | Purple `#9333ea` at 12% opacity, border `#9333ea` | **matplotlib** · seaborn · plotly · **Jupyter** |

- Each library pill has a white background with a thin colored border matching its layer.
- The **Modeling & Evaluation** layer is split by a thin dashed vertical line, with "Classical ML" on the left side and "Deep Learning" on the right side, labeled in 8 pt gray italic.

### Inter-Layer Data Flow

Between each pair of adjacent layers, draw a **dashed arrow** pointing upward, with a small label `"ndarray"` in a rounded badge (gray `#64748b` background, white text, 8 pt). This emphasizes NumPy ndarray as the universal data exchange format.

### Left Column — Workflow Stages

Draw **bracket lines** connecting typical workflow stages to the corresponding layers:
- "Data Acquisition" → Data Manipulation layer
- "Preprocessing & Feature Eng." → Data Manipulation + Modeling layers
- "Model Training & Evaluation" → Modeling layer
- "Exploration & Reporting" → Visualization layer

Use dashed gray lines with small circle endpoints. Stage labels in 9 pt, dark gray `#475569`.

### Right Column — Platform Architecture Mapping

Show the 5-layer platform architecture from Def 12.1 as five thin horizontal bars (matching the colors from Fig 12.1.2):
- Data Layer (blue)
- Processing Layer (green)
- Modeling Layer (orange)
- Evaluation Layer (purple)
- Deployment Layer (red)

Draw **dotted connector lines** from each Python ecosystem layer to the platform architecture layers it covers. For example:
- pandas → Data Layer + Processing Layer
- scikit-learn → Processing + Modeling + Evaluation
- A small note at the bottom: "Deployment layer requires additional tools (Flask, Docker, MLflow)"

### Bottom — Timeline Annotation

A thin horizontal timeline bar at the very bottom showing first-release years of key libraries:
`NumPy 2006 · pandas 2008 · scikit-learn 2010 · Jupyter 2014 · PyTorch 2016 · Polars 2021`

Use small circles on the timeline with year labels below.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for CJK text.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Layer borders**: 1.5 pt solid, corners radius 12 px.
- **Library pills**: 1 pt border, corner radius 8 px, white fill, 9 pt bold text.
- **Shadows**: Subtle (`2px 2px 6px rgba(0,0,0,0.06)`) on layer boxes.
- Aesthetic: clean layered architecture diagram in the style of Fig 12.1.2, adapted for a software ecosystem view.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_2_01_python_ecosystem_layers.png`
- Place in `public/figures/ch12/`
