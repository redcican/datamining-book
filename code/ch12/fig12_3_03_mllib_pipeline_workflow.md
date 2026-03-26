# Figure 12.3.3 — Spark MLlib Distributed Pipeline Workflow

## Prompt

Create a **professional, publication-quality workflow diagram** showing a complete Spark MLlib machine learning pipeline — from distributed data ingestion through feature engineering, model training with cross-validation, to model persistence — emphasizing the distributed nature at every stage.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Main flow**: A left-to-right horizontal pipeline occupying ~70% of the vertical space.
- **Bottom annotation bar**: A mapping strip showing scikit-learn equivalents (~20% height).
- **Top-right badge**: "Distributed Execution" indicator (~10% height).

### Main Pipeline Flow (Left to Right)

Seven stage boxes connected by thick directional arrows. Each box is a rounded rectangle with an icon in the top-left corner and a "Distributed" badge (small orange circle with "⚡" or a cluster icon) in the bottom-right corner.

| Stage | Box Label | Icon | Color Fill | Details Inside |
|-------|-----------|------|-----------|----------------|
| 1 | Data Source | database icon | Gray `#f1f5f9`, border `#64748b` | Three source labels stacked: "HDFS", "S3", "Parquet/CSV". Arrow labeled `spark.read` |
| 2 | SparkSession | spark icon | Orange `#fff7ed`, border `#ea580c` | "SparkSession.builder" text. Badge: "Driver Program" |
| 3 | DataFrame | table icon | Orange `#ffedd5`, border `#ea580c` | Show a mini table (3 cols × 4 rows) with header "age_group \| views \| label". Badge: "N partitions" |
| 4 | Pipeline Stages | gear icon | Orange `#ffedd5`, border `#ea580c` | **Expanded sub-flow** (see below) |
| 5 | CrossValidator | grid-search icon | Purple `#f3e8ff`, border `#9333ea` | Show a 2×2 parameter grid: "numTrees: [100, 200]" × "maxDepth: [5, 10]". Badge: "parallelism=4" with 4 small parallel arrows |
| 6 | Best Model | trophy icon | Green `#dcfce7`, border `#16a34a` | "AUC = 0.87" in bold. "RandomForest (numTrees=200, maxDepth=10)" below |
| 7 | model.save | disk icon | Blue `#dbeafe`, border `#2563eb` | "hdfs:///models/" path. "Serialized Pipeline" label |

**Stage 4 — Pipeline Stages (Expanded)**:

Inside the Stage 4 box, show a horizontal sub-pipeline with 4 smaller internal boxes connected by thin arrows:

| Sub-stage | Label | Type Badge |
|-----------|-------|-----------|
| 4a | StringIndexer | Transformer |
| 4b | VectorAssembler | Transformer |
| 4c | StandardScaler | Transformer |
| 4d | RandomForestClassifier | Estimator |

The Transformer boxes have green `#16a34a` top borders; the Estimator box has orange `#ea580c` top border. A small label above: "pipeline = Pipeline(stages=[...])".

### Arrows Between Stages

- Stages 1→2: labeled `spark.read.parquet()`
- Stages 2→3: labeled `DataFrame`
- Stages 3→4: labeled `fit()` (bold, with a training icon)
- Stages 4→5: labeled `Pipeline`
- Stages 5→6: labeled `bestModel`
- Stages 6→7: labeled `.save()`

Arrow style: 2 pt solid, dark gray `#334155`, with filled arrowheads. Labels in 8 pt italic along the arrows.

### Bottom Annotation Bar — scikit-learn Mapping

A thin horizontal strip across the bottom with light gray `#f8fafc` background, containing a mapping table:

| MLlib Component | ↔ | scikit-learn Equivalent |
|----------------|---|----------------------|
| StringIndexer | ↔ | LabelEncoder |
| VectorAssembler | ↔ | ColumnTransformer |
| StandardScaler | ↔ | StandardScaler |
| CrossValidator | ↔ | GridSearchCV |
| Pipeline | ↔ | Pipeline |

Each pair connected by a small double-headed arrow. MLlib names in orange `#ea580c`, scikit-learn names in blue `#2563eb`. Header: "API 对应关系 (§12.2 → §12.3)" in bold gray.

### Top-Right Badge

A floating badge with orange gradient background:
- Text: "All stages execute across cluster"
- Subtext: "Data never leaves distributed storage"
- Small cluster icon (3 interconnected nodes)

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese labels.
- **Color palette**: Spark orange (`#ea580c`, `#ffedd5`, `#fff7ed`), CrossValidator purple (`#9333ea`, `#f3e8ff`), Best Model green (`#16a34a`, `#dcfce7`), Persist blue (`#2563eb`, `#dbeafe`), neutral gray (`#64748b`, `#f1f5f9`).
- **Stage boxes**: 1.5 pt border, corner radius 12 px, consistent height (~120 px).
- **Sub-pipeline boxes** (inside Stage 4): 1 pt border, corner radius 8 px, ~60 px height.
- **Distributed badges**: Small orange circles (16 px diameter) with white "⚡" icon, positioned at bottom-right of each stage box.
- **Shadows**: Subtle (`2px 2px 6px rgba(0,0,0,0.06)`) on all stage boxes.
- **Background**: Pure white `#ffffff` with ≥ 40 px margins.
- Aesthetic: resembles the end-to-end workflow style of Fig 12.2.3, but with Spark's orange color scheme and distributed execution emphasis throughout.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_3_03_mllib_pipeline_workflow.png`
- Place in `public/figures/ch12/`
