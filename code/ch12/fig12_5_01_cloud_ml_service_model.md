# Figure 12.5.1 — Cloud ML Service Model & Platform Positioning

## Prompt

Create a **professional, publication-quality two-part diagram** showing (top) the three-layer MLaaS service model as a pyramid, and (bottom) a four-column comparison of major cloud ML platforms.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Top section (~50%)**: Three-layer MLaaS pyramid with annotation arrows.
- **Bottom section (~50%)**: Four-column platform comparison grid.
- Thin horizontal separator between sections.

### Top Section — MLaaS Pyramid

A **centered trapezoid/pyramid** with three horizontal layers, widest at bottom:

| Layer | Label | Description (inside) | Color Fill | Example Products (right side) |
|-------|-------|---------------------|-----------|------------------------------|
| Top (narrowest) | ML Application Layer (SaaS) | "Pre-trained API: Vision, Speech, NLP" | Red `#fecaca`, border `#dc2626` | "AWS Rekognition, Google Vision AI, Azure Cognitive Services" |
| Middle | ML Platform Layer (PaaS) | "Managed Training, Tuning, Deployment" | Orange `#ffedd5`, border `#ea580c` | "SageMaker, Vertex AI, Azure ML" |
| Bottom (widest) | ML Infrastructure Layer (IaaS) | "GPU/TPU Instances, Storage, Network" | Blue `#dbeafe`, border `#2563eb` | "EC2 GPU, Cloud TPU, Azure NC-series" |

Two vertical annotation arrows on the left side of the pyramid:
- **Upward arrow** labeled "Managed Degree ↑" (green `#16a34a`)
- **Downward arrow** labeled "Flexibility ↑" (blue `#2563eb`)

A horizontal bracket on the right connecting to text: "Most used for data mining" pointing at the middle (PaaS) layer, highlighted with a star icon.

### Bottom Section — Four-Column Platform Grid

Four equal-width columns, each representing a cloud platform:

| Column | Platform | Brand Color | Key Components (as stacked icons) |
|--------|----------|------------|----------------------------------|
| 1 | AWS SageMaker | Orange `#ff9900` | Studio · Built-in Algorithms · Autopilot · Endpoints · Model Monitor |
| 2 | Google Vertex AI | Blue `#4285f4` | Workbench · AutoML · Pipelines · Feature Store · TPU |
| 3 | Azure ML | Purple `#7b2ff7` | Designer · AutoML · Pipelines · Responsible AI · Power BI |
| 4 | Databricks | Red `#ff3621` | Lakehouse · MLflow · Notebooks · Spark · Workflows |

Each column:
- Platform logo/name at top in bold with brand color background
- 5 component items stacked vertically as rounded pill badges
- Component maturity indicated by fill opacity (darker = more mature)
- A one-line positioning tagline at the bottom in italic

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese labels.
- **Color palette**: AWS orange `#ff9900`, Google blue `#4285f4`, Azure purple `#7b2ff7`, Databricks red `#ff3621`. Pyramid layers use standard book colors.
- **Pyramid borders**: 1.5 pt, corner radius 0 (sharp trapezoid edges).
- **Column borders**: 1 pt `#e2e8f0`, corner radius 8 px.
- **Background**: Pure white `#ffffff` with ≥ 40 px margins.
- **Shadows**: Subtle on pyramid layers and column cards.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_5_01_cloud_ml_service_model.png`
- Place in `public/figures/ch12/`
