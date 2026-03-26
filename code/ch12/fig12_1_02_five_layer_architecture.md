# Figure 12.1.2 — Five-Layer Architecture of Data Mining Platforms

## Prompt

Create a **professional, publication-quality architectural diagram** illustrating the five-layer functional architecture of a data mining platform, with CRISP-DM mapping and platform coverage comparison.

### Layout & Composition

- **Orientation**: Portrait-leaning (roughly 4:5 aspect ratio, ~1200 × 1500 px at 150 DPI).
- **Three-column layout**:
  - **Left column (~15%)**: CRISP-DM phase labels.
  - **Center column (~50%)**: The five-layer stack (hero element).
  - **Right column (~35%)**: Three platform coverage brackets.

### Center — Five-Layer Stack

Five **rounded rectangles** stacked vertically with 8 px gaps, ordered bottom-to-top:

| Layer | Label | Color Fill | Keywords Inside |
|-------|-------|-----------|-----------------|
| 1 (bottom) | Data Layer | Blue `#2563eb` at 15% opacity, border `#2563eb` | Databases · File Systems · Streaming · Data Catalog |
| 2 | Processing Layer | Green `#16a34a` at 15% opacity, border `#16a34a` | Cleaning · Missing Values · Feature Engineering · Transformations |
| 3 | Modeling Layer | Orange `#ea580c` at 15% opacity, border `#ea580c` | Classification · Regression · Clustering · Association · Deep Learning |
| 4 | Evaluation Layer | Purple `#9333ea` at 15% opacity, border `#9333ea` | Cross-Validation · Hyperparameter Tuning · Model Comparison · Visual Diagnostics |
| 5 (top) | Deployment Layer | Red `#dc2626` at 15% opacity, border `#dc2626` | Model Serving · REST API · Batch Inference · A/B Testing · Monitoring |

- The **Modeling Layer** (layer 3) should be slightly wider than the others (~10% wider) to emphasize it as the core.
- Each layer rectangle has a **bold layer name** (left-aligned inside) and **keywords** (right-aligned, smaller, gray `#64748b`).
- Add subtle **connecting arrows** (thin, gray) between adjacent layers to show data flow direction (upward).

### Left Column — CRISP-DM Mapping

Draw **bracket lines** connecting CRISP-DM phases to the corresponding layers:
- "Data Understanding" → Data Layer
- "Data Preparation" → Processing Layer
- "Modeling" → Modeling Layer
- "Evaluation" → Evaluation Layer
- "Deployment" → Deployment Layer

Use dashed gray lines with a small circle endpoint. Phase labels in italic, 10 pt, dark gray.

### Right Column — Platform Coverage

Three **vertical dashed-border rectangles**, each spanning a different height range of the stack, representing which layers each platform covers:

| Platform | Covered Layers | Border Color | Bracket Label |
|----------|---------------|-------------|---------------|
| scikit-learn | Layers 2–4 (Processing through Evaluation) | Blue `#2563eb` | "scikit-learn" + "Algorithm Library" subtitle |
| Spark MLlib | Layers 1–4 (Data through Evaluation) | Green `#16a34a` | "Spark MLlib" + "Big Data Engine" subtitle |
| AWS SageMaker | Layers 1–5 (full stack) | Orange `#ea580c` | "AWS SageMaker" + "Full-Stack Cloud" subtitle |

- Each coverage bracket is a **tall rounded dashed rectangle** positioned to the right, with horizontal connector lines pointing into the stack layers it covers.
- Use the bracket height difference to visually communicate "specialized tool vs. full-stack platform."
- Add a small legend/annotation: `Specialized ←→ Full-Stack` with an arrow.

### Style & Typography

- **Font**: Inter or Helvetica Neue (Latin), Noto Sans SC (CJK if needed).
- **Color palette**: Book standard — `blue #2563eb`, `red #dc2626`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`.
- **Background**: Pure white `#ffffff`.
- **Borders**: 1.5 pt solid for layer boxes, 1 pt dashed for coverage brackets.
- **Shadows**: Very subtle (`2px 2px 6px rgba(0,0,0,0.08)`) on the layer boxes for depth.
- Aesthetic: clean enterprise architecture diagram — think AWS Well-Architected or Confluent-style layered views, but adapted for a textbook audience.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_1_02_five_layer_architecture.png`
- Place in `public/figures/ch12/`
