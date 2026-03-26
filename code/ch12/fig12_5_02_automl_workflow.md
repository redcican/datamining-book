# Figure 12.5.2 — AutoML Workflow & CASH Search Space

## Prompt

Create a **professional, publication-quality two-panel diagram** showing (a) the AutoML end-to-end pipeline, and (b) the hierarchical CASH search space structure.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Panel (a)** (top ~55%): "AutoML Pipeline" — horizontal flow.
- **Panel (b)** (bottom ~45%): "CASH Search Space" — hierarchical tree + Bayesian optimization curve.
- Panel labels in bold with "(a)" and "(b)" prefixes in gray circles.

### Panel (a) — AutoML Pipeline

A left-to-right flow of 5 stages, each as a rounded rectangle with an icon:

| Stage | Label | Icon | Color Fill | Details |
|-------|-------|------|-----------|---------|
| 1 | Raw Data | table icon | Gray `#f1f5f9`, border `#64748b` | CSV/DB icon, "N rows × M features" |
| 2 | Auto Feature Engineering | gear+sparkle icon | Blue `#dbeafe`, border `#2563eb` | 3 branching sub-arrows inside: "Polynomial", "Encoding", "Interaction" |
| 3 | Algorithm Selection | grid icon | Orange `#ffedd5`, border `#ea580c` | 6 small algorithm badges in 2×3 grid: LR, RF, XGB, SVM, KNN, MLP. Top 3 highlighted with golden border |
| 4 | Hyperparameter Optimization | chart icon | Purple `#f3e8ff`, border `#9333ea` | Small Bayesian optimization curve (see below for detail) |
| 5 | Final Ensemble | trophy icon | Green `#dcfce7`, border `#16a34a` | 3 models with weight percentages (e.g., "XGB 45%, RF 30%, LR 25%") merging into one |

Arrows between stages: 2 pt solid, dark gray. Between stages 3 and 4, add a bidirectional arrow labeled "CASH (Eq 12.7)" in a red badge.

**Bayesian Optimization Mini-Chart** (inside stage 4):
- X-axis: "Iteration" (1–20)
- Y-axis: "Validation Loss"
- A curve that starts high, drops rapidly in early iterations, then plateaus
- Two regions labeled: "Exploration" (early, blue shaded) and "Exploitation" (later, orange shaded)
- Star marker at the best point

### Panel (b) — CASH Search Space Tree

A **tree diagram** showing the hierarchical structure:

**Root node**: "CASH Search Space" (gray)

**Level 1 — Algorithm Selection** (3 branches):
- Branch 1: "Random Forest" (green node)
- Branch 2: "XGBoost" (orange node)
- Branch 3: "SVM" (blue node)

**Level 2 — Hyperparameter Spaces** (each algorithm expands into its own parameters):

| Algorithm | Parameters (as leaf nodes) |
|-----------|--------------------------|
| Random Forest | `n_estimators ∈ [50, 500]` · `max_depth ∈ [3, 20]` · `min_samples_split ∈ [2, 20]` |
| XGBoost | `learning_rate ∈ [0.01, 0.3]` · `n_estimators ∈ [100, 1000]` · `max_depth ∈ [3, 10]` · `subsample ∈ [0.5, 1.0]` |
| SVM | `C ∈ [0.1, 100]` · `kernel ∈ {rbf, poly, linear}` · `gamma ∈ [0.001, 1.0]` |

Each parameter node: small rounded rectangle with parameter name and range. Continuous parameters shown with a mini slider/range bar, categorical parameters shown with option pills.

**Search space size annotation**: At the bottom, a bar showing approximate search space size for each algorithm (RF: ~10⁴, XGB: ~10⁶, SVM: ~10⁴), with total "Combined: ~10⁶+" in bold.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese. Parameter names in monospace (JetBrains Mono).
- **Color palette**: Blue `#2563eb` (feature eng), Orange `#ea580c` (algorithm), Purple `#9333ea` (optimization), Green `#16a34a` (result), Gray `#64748b`.
- **Stage boxes**: 1.5 pt border, corner radius 12 px.
- **Tree nodes**: 1 pt border, corner radius 8 px.
- **Tree edges**: 1 pt solid, gray.
- **Background**: Pure white `#ffffff`.
- **Shadows**: Subtle on stage boxes.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_5_02_automl_workflow.png`
- Place in `public/figures/ch12/`
