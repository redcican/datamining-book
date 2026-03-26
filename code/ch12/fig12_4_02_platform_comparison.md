# Figure 12.4.2 — Four Visual Data Mining Platforms: Interface Comparison

## Prompt

Create a **professional, publication-quality 2×2 grid diagram** showing stylized interface mockups of the four major open-source visual data mining platforms: KNIME, RapidMiner, Orange, and WEKA. Each quadrant should capture the tool's distinctive visual identity and interaction paradigm.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **2×2 grid** with 12 px gaps between quadrants. Each quadrant ~780 × 430 px.
- **Quadrant labels**: Bold platform name + one-line positioning in the top-left corner of each.
- **Thin gray border** (`#e2e8f0`, 1 pt) around each quadrant with corner radius 8 px.

### Quadrant (a) — KNIME (Top-Left)

- **Label**: "KNIME Analytics Platform — Enterprise Workflow Engine"
- **Content**: A stylized workflow canvas showing 7 nodes connected in a branching pattern:
  - Row 1: "CSV Reader" → "Missing Value" → "Normalizer"
  - Branch: "Normalizer" splits into "Random Forest" and "Gradient Boosting"
  - Both model nodes connect to "Model Comparison" → "ROC Curve"
- Nodes use KNIME's characteristic rounded-square shape with colored top bands (yellow for data, blue for manipulation, green for mining, red for visualization).
- Bottom area: a simplified "Node Configuration" panel with parameter fields.
- Status indicators: traffic-light circles (green/yellow/red) on each node.

### Quadrant (b) — RapidMiner (Top-Right)

- **Label**: "RapidMiner — AutoML + Visual Analytics"
- **Content**: Similar workflow layout but with RapidMiner's characteristic flat rectangular operators.
- Highlight two distinctive panels:
  - "Turbo Prep" panel on the left showing automated data quality recommendations (3 bullet items with checkmarks).
  - "Auto Model" panel at bottom showing model ranking (3 bars: best/good/fair).
- Operators connected by RapidMiner's straight-line edges with port indicators (circles).

### Quadrant (c) — Orange (Bottom-Left)

- **Label**: "Orange Data Mining — Lightweight & Interactive"
- **Content**: Orange's characteristic compact widget canvas with rounded elliptical widgets:
  - "File" → "Data Table" → "Select Columns" → "Test & Score" with 3 learner widgets ("Tree", "RF", "LR") connected.
- Right side: An embedded interactive scatter plot showing Iris data with 3 colored clusters and a selection lasso tool active.
- Emphasize the interactive visualization linkage: an arrow from a selection in the scatter plot back to the data table, labeled "Interactive Brushing".

### Quadrant (d) — WEKA (Bottom-Right)

- **Label**: "WEKA Explorer — Algorithm Laboratory"
- **Content**: WEKA's classic tabbed interface (not a workflow):
  - Tab bar: "Preprocess | Classify | Cluster | Associate | Select attributes | Visualize" with "Classify" tab active.
  - Left panel: Classifier selector tree (showing "trees > RandomForest" highlighted).
  - Center panel: Test options (radio buttons: "Cross-validation 10 folds" selected).
  - Right panel: Result list showing 3 classifier results with accuracy percentages.
- The non-workflow nature should be visually apparent — this is a form-based UI, not a canvas.

### Style & Typography

- **Font**: Inter or Helvetica Neue; code/labels in monospace (JetBrains Mono).
- **Color palette**:
  - KNIME quadrant: Yellow/blue tones (KNIME brand)
  - RapidMiner quadrant: Blue/orange tones
  - Orange quadrant: Orange/white tones (Orange brand)
  - WEKA quadrant: Gray/blue classic Java Swing tones
- **Borders**: 1 pt `#e2e8f0` gray, corner radius 8 px for each quadrant.
- **Background**: Pure white `#ffffff` for the overall image; each quadrant has a slightly tinted background matching its brand.
- **Shadows**: Very subtle on each quadrant panel.
- **Important**: These should be stylized mockups/illustrations, NOT screenshots. Use simplified shapes and text that capture the essence of each interface without trying to be pixel-accurate reproductions.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_4_02_platform_comparison.png`
- Place in `public/figures/ch12/`
