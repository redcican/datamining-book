# Figure 12.2.3 — Python Data Mining End-to-End Workflow

## Prompt

Create a **professional, publication-quality workflow diagram** illustrating the complete end-to-end Python data mining pipeline, from data acquisition to model deployment, with tool annotations at each stage.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Primary flow**: Left-to-right horizontal pipeline with 7 stages.
- **Two environment zones**: A dashed boundary divides the diagram into "Exploration Environment" (left ~60%) and "Production Environment" (right ~40%).

### Pipeline Stages

Seven **rounded rectangles** connected by thick directional arrows (2 pt, `#94a3b8` gray), flowing left to right:

| Stage | Label | Color | Icon/Visual | Key Operations (small text inside) |
|-------|-------|-------|-------------|-----------------------------------|
| 1 | Data Sources | Gray `#64748b` | Database + CSV + API icons | Databases · CSV files · REST APIs · Streaming |
| 2 | Data Loading & Cleaning | Green `#16a34a` | pandas logo silhouette | `pd.read_csv` · `dropna` · `astype` · `merge` |
| 3 | EDA & Visualization | Purple `#9333ea` | Chart icon | `df.describe` · `sns.pairplot` · `plt.hist` |
| 4 | Feature Engineering | Green `#16a34a` | Gear/transform icon | `ColumnTransformer` · `StandardScaler` · `OneHotEncoder` |
| 5 | Model Training & Selection | Orange `#ea580c` | Brain/model icon | `Pipeline.fit` · `cross_val_score` · `GridSearchCV` |
| 6 | Experiment Tracking | Blue `#2563eb` | Log/chart icon | `mlflow.log_param` · `mlflow.log_metric` · Model Registry |
| 7 | Deployment | Red `#dc2626` | Rocket/server icon | Flask API · ONNX Runtime · Docker · Monitoring |

### Stage Box Design

Each stage box:
- **Width**: ~180 px, **Height**: ~120 px.
- **Top section** (30 px): Colored header bar with stage number and label in white bold text.
- **Middle section**: Small monochrome icon (centered).
- **Bottom section**: 2–3 key operations in 8 pt `#475569` gray text.
- **Border**: 1.5 pt solid in the stage's color.
- **Corner radius**: 10 px.
- **Shadow**: `2px 2px 6px rgba(0,0,0,0.06)`.

### Arrows Between Stages

- **Forward arrows**: 2 pt solid `#94a3b8`, with arrowheads.
- **Feedback arrow**: A curved dashed arrow from Stage 5 (Model Training) back to Stage 4 (Feature Engineering), labeled "Iterate" in 8 pt italic. This shows the iterative nature of the modeling process.
- **Second feedback arrow**: A curved dashed arrow from Stage 5 back to Stage 3 (EDA), labeled "Re-explore" in 8 pt italic.

### Environment Zones

Two dashed boundary boxes overlaid on the pipeline:

| Zone | Coverage | Border Color | Label |
|------|----------|-------------|-------|
| Exploration | Stages 1–5 | Purple `#9333ea` dashed (1 pt) | "Jupyter Notebook 交互环境" (top-left corner, 10 pt) |
| Production | Stages 6–7 | Red `#dc2626` dashed (1 pt) | "生产环境" (top-left corner, 10 pt) |

The zones slightly overlap at Stage 5–6 boundary to show the transition point.

### Tool Badges

Below each stage, place a **tool badge strip** — small rounded pills showing the primary Python library used:
- Stage 2: `pandas` pill (green)
- Stage 3: `matplotlib` + `seaborn` pills (purple)
- Stage 4: `scikit-learn` pill (orange)
- Stage 5: `scikit-learn` + `XGBoost` pills (orange)
- Stage 6: `MLflow` pill (blue)
- Stage 7: `Flask` + `ONNX` pills (red)

Each pill: 8 pt bold text, white fill, 1 pt colored border, corner radius 6 px.

### Bottom — Data Flow Annotation

A thin horizontal bar at the very bottom showing the data format transformation:
`Raw Files → DataFrame → ndarray → Model Object → Serialized Model → REST API`

Use small arrows between each format label. Color transitions from green (data) through orange (model) to red (deployment).

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for CJK text.
- **Color palette**: Book standard — `blue #2563eb`, `red #dc2626`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Gridlines**: None (clean workflow diagram).
- Aesthetic: clean process flow diagram in the style of a DevOps/MLOps pipeline visualization (think MLflow docs or Databricks architecture diagrams), adapted for a textbook audience.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_2_03_end_to_end_workflow.png`
- Place in `public/figures/ch12/`
