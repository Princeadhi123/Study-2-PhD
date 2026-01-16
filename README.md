# Study 2 – Narrative Embeddings and Student Knowledge Trajectories

This repository centres on a transformer-based sentence-embedding narrative pipeline built on top of a numeric clustering baseline. It uses encoder-only sentence-transformer models (e.g. `all-mpnet-base-v2`, `all-MiniLM-L6-v2`) to embed per-student narratives; these are not full generative Large Language Models (LLMs), but specialised embedding models. It contains code to:

- Derive per-student behavioural features from item-level response data.
- Run multiple clustering algorithms over these features (GMM, KMeans, etc.).
- Build per-student narratives from numeric features (Templates A/B/C) and embed them with encoder-only transformers.
- Cluster narrative embeddings with GMM (using BIC and AICc for model selection) and compare narrative clusters to numeric GMM clusters.
- Link both numeric and narrative cluster memberships to subject-wise marks and visualise subject profiles by cluster (z-mean heatmaps).

The analysis is therefore organised in two layers:

- **Narrative-embedding pipeline (main)**: `narrative_embedding_clustering/` (scripts `01_build_narratives.py`–`07_plot_metrics.py`).
- **Numeric baseline pipeline**: `cluster_knowledge_trajectories.py` and `subjectwise_by_cluster.py`.

---

## 1. Repository structure

At the top level of `Study-2` you will typically have:

- **`narrative_embedding_clustering/`**  
  Main subproject. Builds per-student narratives from numeric features, computes transformer embeddings, runs narrative GMM clustering (BIC/AICc), compares narrative vs numeric clusters (generating subject-wise heatmaps), and generates cross-template metrics and ANOVA plots.

- **`cluster_knowledge_trajectories.py`**  
  Numeric baseline pipeline. Reads an itemwise CSV, computes per-student features, runs clustering, evaluates models (including AICc for GMM), and writes diagnostics + figures that feed into the narrative pipeline.

- **`subjectwise_by_cluster.py`**  
  Post-hoc numeric analysis. Merges student cluster labels with an Excel file of subject-wise marks and produces cluster-by-subject profiles; also feeds marks into Template C narratives and narrative ANOVA.

- **`compare_models_final.py`**  
  Aggregates metrics from various template/model combinations to identify the best overall performing model based on a composite score of internal validity, external agreement (ARI), and predictive power (ANOVA).

- **`data/`**  
  - `DigiArvi_25_itemwise.csv` – default input for `cluster_knowledge_trajectories.py` (item-level responses).  
  - `EQTd_DAi_25_cleaned 3_1 for Prince.xlsx` – default input for `subjectwise_by_cluster.py` (subject marks).

- **`diagnostics/`** *(created by the numeric baseline + subjectwise scripts; reused by the narrative pipeline)*  
  - `cluster input features/` – derived per-student features and merged marks+clusters.  
  - `student cluster labels/` – cluster labels per student for each clustering method.  
  - `model results/` – numeric diagnostics for clustering model sweeps (e.g. silhouette vs K, GMM BIC/AIC/AICc grid).  
  - `cluster validity/` – internal and external cluster validity metrics.

- **`figures/`** *(created by the numeric baseline + subjectwise scripts; reused by the narrative pipeline)*  
  Subfolders for each algorithm and analysis, for example:  
  - `gmm/BIC/`, `gmm/AIC/`  
  - `subjectwise by cluster/`

- **`.gitignore`** – Git ignore rules.
- **`README.md`** – this file.

---

## 2. Dependencies

Tested with **Python 3.10+**.

Core Python packages:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `umap-learn` *(optional, used only if installed for UMAP plots)*
- `openpyxl` *(or another Excel engine; used by `pandas.read_excel`)*
- `sentence-transformers` *(for narrative text embeddings such as all-mpnet-base-v2 and all-MiniLM-L6-v2)*

### 2.1. Install with `pip`

From the `Study-2` directory, create an environment and install dependencies, for example:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows

pip install numpy pandas scipy scikit-learn matplotlib seaborn umap-learn openpyxl sentence-transformers
```

---

## 3. Input data formats

### 3.1. Itemwise response file (`DigiArvi_25_itemwise.csv`)

The clustering pipeline expects a **long-format** CSV with at least:

- **`IDCode`** – student identifier.
- **`orig_order`** – original item order *within each student*. Used for streak/consecutive features.
- **`response`** – binary correctness indicator (0/1).
- **`response_time_sec`** – response time in **seconds**.

Optional columns:
- **`sex`** – used for external validity checks.

### 3.2. Subject-wise marks file (`EQTd_DAi_25_cleaned 3_1 for Prince.xlsx`)

`subjectwise_by_cluster.py` expects an Excel file with:
- A student ID column (auto-detected).
- One or more subject columns (e.g. `s1`, `s2`...).
- Optionally a total/overall column.

---

## 4. Running the clustering pipeline

### 4.1. Default run (using repository data)

From the `Study-2` directory:

```bash
python cluster_knowledge_trajectories.py
```

It will:
1. Compute per-student features (accuracy, response times, streaks, etc.).
2. Standardise features.
3. Run clustering: KMeans, Agglomerative, Birch, GMM (with BIC, AIC, and **AICc** selection), DBSCAN.
4. Compute internal/external validity metrics.
5. Generate figures (PCA, silhouettes, heatmaps).

### 4.2. Custom input path

```bash
python cluster_knowledge_trajectories.py path\to\your_itemwise.csv
```

---

## 5. Outputs from `cluster_knowledge_trajectories.py`

### 5.1. Derived features and cluster labels

Under `diagnostics/`:
- **`student cluster labels/student_clusters.csv`**:
  - `gmm_bic_best_label`
  - `gmm_aic_best_label`
  - **`gmm_aicc_best_label`** (Corrected AIC, often preferred for smaller samples)
  - ...and others.

### 5.2. Model sweeps and diagnostics

- **`gmm_bic_aic.csv`**: Contains BIC, AIC, and AICc values for all K and covariance types.

### 5.3. Figures

- **GMM AICc**: The pipeline generates specific plots for the AICc-selected model if distinct, including PCA scatter plots.
- **Cluster profiles**: `gmm/BIC/gmm_bic_feature_zmean_by_cluster.png` (and similarly for AICc models).

---

## 6. Subject-wise analysis by cluster

After running the numeric clustering, link to subject marks:

```bash
python subjectwise_by_cluster.py
```

Outputs in `figures/subjectwise by cluster/`:
- **`subject_zmean_by_cluster.png`**: Heatmap of z-standardised subject scores by cluster (defaults to using `gmm_aicc_best_label` logic where applicable).
- **`subject_radar_all_clusters.png`**: Radar plot comparing subject profiles.

---

## 7. Narrative-embedding clustering (`narrative_embedding_clustering/`)

### 7.1. Aim

Use encoder-only transformer models to embed per-student narratives, cluster them with GMM (selecting via BIC/AICc), and compare these narrative clusters to the numeric GMM baselines and subject marks.

### 7.2. Scripts

- **`01_build_narratives.py`**  
  Builds text narratives (Template A/B/C) from derived features.

- **`02_compute_embeddings.py`**  
  Encodes narratives into dense embeddings.

- **`03_cluster_embeddings.py`**  
  Fits GMMs over a range of K. Computes **BIC, AIC, and AICc**. Selects the best models and saves labels to `narrative_clusters.csv`.

- **`04_compare_with_gmm_bic.py`**  
  The core comparison script.
  - Merges narrative clusters with numeric clusters (`gmm_bic_best_label`, `gmm_aicc_best_label`).
  - Computes ARI (Adjusted Rand Index) between narrative and numeric clusters.
  - **Generates Subject-wise Z-Mean Heatmap**: Automatically produces `subject_zmean_by_narrative_cluster.png`, showing how narrative clusters map to actual subject performance (standardised).
  - Runs ANOVAs and saves effect sizes.

- **`05_visualize_embeddings.py`**  
  Produces PCA/t-SNE/UMAP plots of the embeddings, coloured by cluster.

- **`06_run_all_models.py`**  
  Batch driver to run the full pipeline (01–05) for all templates (`A`, `B`, `C`) and models (`all-mpnet-base-v2`, `all-MiniLM-L6-v2`).

- **`07_plot_metrics.py`**  
  Aggregates metrics across all runs to visualise performance comparisons (ARI, Silhouette, ANOVA Eta-squared).

### 7.3. Running the narrative pipeline

```bash
cd narrative_embedding_clustering

# Run everything for all templates and models
python 06_run_all_models.py

# Or run step-by-step for a specific configuration
python 01_build_narratives.py
python 02_compute_embeddings.py
python 03_cluster_embeddings.py
python 04_compare_with_gmm_bic.py
python 05_visualize_embeddings.py
```

Outputs location: `narrative_embedding_clustering/outputs/template_<T>/<embedding_id>/`.
- **Figures**: `figures/subject_zmean_by_narrative_cluster.png` (Subject Heatmap), PCA plots, etc.
- **Metrics**: `gmm_vs_narrative_metrics.csv`.

---

## 8. Final Model Comparison

**`compare_models_final.py`** (in the root) can be run to read all the generated metrics from the narrative outputs and rank the models/templates.

```bash
python compare_models_final.py
```

This helps identify which template and embedding model combination yields the best balance of internal structure, agreement with numeric baselines, and predictive power for subject grades.
