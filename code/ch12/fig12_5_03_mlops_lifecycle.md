# Figure 12.5.3 — MLOps Full Lifecycle Loop

## Prompt

Create a **professional, publication-quality circular lifecycle diagram** showing the six stages of MLOps as a continuous closed loop, with tool annotations and a maturity level inset.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Center (~70%)**: Circular loop of 6 stages.
- **Right inset (~25%)**: MLOps maturity level table.
- **Center of circle**: "MLOps Closed Loop" label with a circular arrow icon.

### Circular Loop — 6 Stages

Six stage nodes arranged in a clockwise circle (like a clock face), connected by thick curved arrows:

| Position | Stage | Label | Icon | Color | Tool Annotation |
|----------|-------|-------|------|-------|----------------|
| 12 o'clock | 1 | Data Preparation | database icon | Blue `#2563eb` on `#dbeafe` | "Great Expectations, Feast" |
| 2 o'clock | 2 | Model Development | flask/beaker icon | Green `#16a34a` on `#dcfce7` | "Jupyter, MLflow Tracking" |
| 4 o'clock | 3 | Model Training | GPU/chip icon | Orange `#ea580c` on `#ffedd5` | "SageMaker, Vertex AI" |
| 6 o'clock | 4 | Evaluation & Registry | clipboard-check icon | Purple `#9333ea` on `#f3e8ff` | "MLflow Model Registry" |
| 8 o'clock | 5 | Deployment & Serving | rocket icon | Red `#dc2626` on `#fecaca` | "SageMaker Endpoints, Seldon" |
| 10 o'clock | 6 | Monitoring & Feedback | dashboard icon | Gray `#64748b` on `#f1f5f9` | "Evidently AI, CloudWatch" |

Each stage node:
- Rounded rectangle, 120 × 80 px
- Icon centered at top, label below
- Thin tool annotation on the outside of the circle in italic gray text

**Connecting arrows**:
- Thick curved arrows (3 pt) flowing clockwise between stages
- Color gradient from one stage's color to the next
- Between Stage 6 and Stage 1 (the "feedback" arrow): **dashed red** arrow with a special label badge: "Data Drift → Trigger Retraining" in red text on white background

**Center label**:
- "MLOps" in large bold text (24 pt)
- "闭环" (Closed Loop) below in 14 pt
- A subtle circular arrow icon behind the text

### Right Inset — Maturity Level Table

A small table card with rounded corners and light gray background:

**Title**: "MLOps Maturity Levels"

| Level | Name | Key Feature | Color Bar |
|-------|------|-------------|-----------|
| Level 0 | Manual | Manual training & deployment | Red bar (short) |
| Level 1 | Pipeline | Automated training, manual trigger | Yellow bar (medium) |
| Level 2 | CI/CD | Fully automated, drift-triggered retraining | Green bar (full) |

Each row has a small colored bar on the right showing relative maturity (like a progress bar). Below the table: "Most enterprises: Level 0–1" in italic gray.

### Additional Annotations

**Inner ring labels** (between center text and stage nodes):
- Between stages 1→2→3: "Experiment Phase" bracket in blue dashed line
- Between stages 4→5→6: "Production Phase" bracket in red dashed line

**Bottom-left corner**: A small legend showing:
- Solid arrow = data/model flow
- Dashed arrow = feedback/trigger

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese labels.
- **Color palette**: Each stage uses its assigned color from the book's standard COLORS dict: blue `#2563eb`, green `#16a34a`, orange `#ea580c`, purple `#9333ea`, red `#dc2626`, gray `#64748b`.
- **Stage nodes**: 1.5 pt border, corner radius 12 px.
- **Arrows**: 3 pt curved, with filled arrowheads.
- **Background**: Pure white `#ffffff`.
- **Shadows**: Subtle on stage nodes and maturity table card.
- **Overall feel**: Clean, modern DevOps-style lifecycle diagram adapted for ML context. Should feel like it belongs in a cloud platform's documentation.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_5_03_mlops_lifecycle.png`
- Place in `public/figures/ch12/`
