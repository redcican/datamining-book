# Figure 12.6.3 — Data Mining Tool Ecosystem Landscape & Evolution Trends

## Prompt

Create a **professional, publication-quality ecosystem landscape diagram** showing all major data mining tools organized around a central CRISP-DM workflow ring, with four evolution trends annotated in the corners.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Center (~60%)**: Circular CRISP-DM workflow ring with tools arranged around it.
- **Four corners**: Trend annotation boxes with arrows pointing to affected areas.

### Center — CRISP-DM Workflow Ring

A circular ring divided into **6 arc segments** (matching CRISP-DM phases), each segment a different color:

| Segment | Phase | Color | Position |
|---------|-------|-------|----------|
| 1 | Business Understanding | Gray `#64748b` | 12 o'clock |
| 2 | Data Understanding | Blue `#2563eb` | 2 o'clock |
| 3 | Data Preparation | Green `#16a34a` | 4 o'clock |
| 4 | Modeling | Orange `#ea580c` | 6 o'clock |
| 5 | Evaluation | Purple `#9333ea` | 8 o'clock |
| 6 | Deployment | Red `#dc2626` | 10 o'clock |

Phase labels inside each arc segment in white bold text.

### Tools Around the Ring

Tools are placed as **small pill-shaped badges** around the outside of the ring, positioned near the phase(s) they support. Group them by category using background tint:

**Python Ecosystem** (blue-tinted pills):
- pandas → near Data Preparation
- scikit-learn → spanning Modeling + Evaluation
- XGBoost / LightGBM → near Modeling
- matplotlib / seaborn → near Data Understanding + Evaluation
- PyTorch / TensorFlow → near Modeling

**Distributed Platforms** (orange-tinted pills):
- Spark MLlib → spanning Data Preparation + Modeling
- Hadoop / HDFS → near Data Preparation
- Dask → near Data Preparation + Modeling

**Visual Tools** (green-tinted pills):
- KNIME → spanning Data Preparation + Modeling + Evaluation
- Orange → near Data Understanding + Modeling
- RapidMiner → spanning Modeling + Evaluation

**Cloud Services** (purple-tinted pills):
- SageMaker → spanning Modeling + Deployment
- Vertex AI → spanning Modeling + Deployment
- Databricks → spanning Data Preparation + Modeling
- MLflow → near Evaluation + Deployment

**Deployment Tools** (red-tinted pills):
- Docker → near Deployment
- BentoML / Seldon → near Deployment
- ONNX → between Modeling and Deployment

### Four Corner Trend Boxes

Each corner has a **rounded rectangle** with a trend title, description, and dashed arrow pointing to the affected ring segments:

| Corner | Trend | Description | Arrow Target |
|--------|-------|-------------|-------------|
| Top-left | 🤖 AI-Assisted DS | "LLM agents automate code generation and data exploration" | → Data Understanding, Data Preparation |
| Top-right | 🏗️ Unified Platforms | "Lakehouse architecture merges data engineering + ML" | → Data Preparation, Modeling |
| Bottom-left | 📱 Edge Inference | "TFLite, ONNX Runtime enable on-device prediction" | → Deployment |
| Bottom-right | ⚖️ Responsible AI | "Fairness, explainability, privacy become standard" | → Evaluation |

Each trend box:
- 200 × 80 px, rounded corners
- Light colored fill matching the trend theme
- Bold title (10 pt) + one-line description (8 pt)
- Dashed arrow (2 pt, trend color) curving toward the relevant ring segment

### Center Text

Inside the ring:
- "Data Mining" in large bold (20 pt)
- "Tool Ecosystem" below (14 pt)
- "2025" in small gray (10 pt)

### Legend

Small legend box at bottom-center:
- Four colored squares: "Python Ecosystem" (blue) | "Distributed" (orange) | "Visual" (green) | "Cloud & MLOps" (purple)

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese.
- **Color palette**: Standard book COLORS — blue `#2563eb`, green `#16a34a`, orange `#ea580c`, purple `#9333ea`, red `#dc2626`, gray `#64748b`.
- **Ring**: 30 px width arc segments, with 2 px white gaps between segments.
- **Tool pills**: 8 pt bold text, white background, 1 pt colored border, corner radius 10 px.
- **Trend arrows**: 2 pt dashed, with arrowheads.
- **Background**: Pure white `#ffffff`.
- **Shadows**: Very subtle on ring and trend boxes.
- **Overall feel**: A simplified "landscape map" — not as dense as CNCF Landscape, but captures the major players and their positioning. Should feel authoritative and comprehensive.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_6_03_ecosystem_landscape.png`
- Place in `public/figures/ch12/`
