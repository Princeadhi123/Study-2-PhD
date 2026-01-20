import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "narrative_embedding_clustering" / "outputs"
MARKS_PATH = PROJECT_ROOT / "diagnostics" / "cluster input features" / "marks_with_clusters.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

MODELS = [
    ("Template A", "MPNet", OUTPUT_DIR / "template_A/all_mpnet_base_v2"),
    ("Template A", "MiniLM", OUTPUT_DIR / "template_A/all_MiniLM_L6_v2"),
    ("Template B", "MPNet", OUTPUT_DIR / "template_B/all_mpnet_base_v2"),
    ("Template B", "MiniLM", OUTPUT_DIR / "template_B/all_MiniLM_L6_v2"),
    ("Template C", "MPNet", OUTPUT_DIR / "template_C/all_mpnet_base_v2"),
    ("Template C", "MiniLM", OUTPUT_DIR / "template_C/all_MiniLM_L6_v2"),
]

def calculate_eta_squared(df, group_col, value_cols):
    etas = []
    for col in value_cols:
        # ANOVA
        groups = [d[col].dropna() for _, d in df.groupby(group_col)]
        if len(groups) < 2:
            continue
        
        # Flatten
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        ss_total = np.sum((all_values - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        
        if ss_total > 0:
            eta = ss_between / ss_total
            etas.append(eta)
            
    return np.mean(etas) if etas else 0.0

def get_comparison_data():
    results = []
    
    # Load Marks
    if not MARKS_PATH.exists():
        print(f"Error: Marks file not found at {MARKS_PATH}")
        return pd.DataFrame()

    marks_df = pd.read_csv(MARKS_PATH)
    numeric_cols = [c for c in marks_df.columns if c not in ["IDCode", "student_id"] and pd.api.types.is_numeric_dtype(marks_df[c])]
    mark_cols = [c for c in numeric_cols if c.startswith("S") or c in ["FinalMark", "Total"]]
    if not mark_cols:
        mark_cols = numeric_cols 
        
    print(f"Calculating metrics...")

    for template, model, path in MODELS:
        # 1. Get Internal Metrics
        if "template_B" in str(path): suffix = "_B"
        elif "template_C" in str(path): suffix = "_C"
        else: suffix = ""
            
        metrics_file = path / f"gmm_vs_narrative_metrics{suffix}.csv"
        
        sil = np.nan
        ch = np.nan
        db = np.nan
        ari = np.nan
        
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            for _, row in metrics_df.iterrows():
                m = row["metric"]
                b = row["baseline"]
                v = row["value"]
                
                if b == "narrative_best_label":
                    if m == "silhouette_cosine": sil = v
                    elif m == "calinski_harabasz": ch = v
                    elif m == "davies_bouldin": db = v
                elif m == "adjusted_rand_index" and b == "gmm_aicc_best_label":
                    ari = v

        # 2. Get External Metrics (Eta Squared)
        clusters_file = path / f"narrative_clusters{suffix}.csv"
        mean_eta = 0.0
        
        if clusters_file.exists():
            clusters_df = pd.read_csv(clusters_file)
            label_col = "narrative_gmm_aicc_best_label"
            if label_col in clusters_df.columns:
                id_col = "IDCode" if "IDCode" in clusters_df.columns else "id"
                merged = pd.merge(clusters_df, marks_df, left_on=id_col, right_on="IDCode", how="inner")
                mean_eta = calculate_eta_squared(merged, label_col, mark_cols)
        
        results.append({
            "Template": template,
            "Model": model,
            "Silhouette (Cosine)\nScore": sil,
            "Calinski-Harabasz\nScore": ch,
            "Davies-Bouldin\nScore": db,
            "ARI (vs Numeric)\nScore": ari,
            "Mean Eta^2\nScore": mean_eta
        })

    return pd.DataFrame(results)

def plot_heatmap(df):
    if df.empty:
        print("No data to plot.")
        return

    # Normalize metrics to 0-1 score for the Decision Matrix
    df_norm = df.copy()
    
    # Handle NaNs
    df_norm = df_norm.fillna(0)

    # Normalize (Max-based normalization)
    # We overwrite the columns with their normalized versions, but keep the original names
    eta_max = df_norm["Mean Eta^2\nScore"].max()
    sil_max = df_norm["Silhouette (Cosine)\nScore"].max()
    ch_max = df_norm["Calinski-Harabasz\nScore"].max()
    ari_max = df_norm["ARI (vs Numeric)\nScore"].max()

    df_norm["Mean Eta^2\nScore"] = df_norm["Mean Eta^2\nScore"] / eta_max if eta_max else 0
    df_norm["Silhouette (Cosine)\nScore"] = df_norm["Silhouette (Cosine)\nScore"] / sil_max if sil_max else 0
    df_norm["Calinski-Harabasz\nScore"] = df_norm["Calinski-Harabasz\nScore"] / ch_max if ch_max else 0

    # DB is lower-is-better
    db_series = df_norm["Davies-Bouldin\nScore"].replace(0, np.nan)
    db_min = db_series.min(skipna=True)
    df_norm["Davies-Bouldin\nScore"] = (db_min / db_series) if (db_min and not np.isnan(db_min)) else 0
    df_norm["Davies-Bouldin\nScore"] = df_norm["Davies-Bouldin\nScore"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df_norm["ARI (vs Numeric)\nScore"] = df_norm["ARI (vs Numeric)\nScore"] / ari_max if ari_max else 0

    # Combine internal metrics into a single score (equal weights)
    df_norm["Internal Score"] = (
        df_norm["Silhouette (Cosine)\nScore"]
        + df_norm["Calinski-Harabasz\nScore"]
        + df_norm["Davies-Bouldin\nScore"]
    ) / 3
    
    # Composite Score calculation
    df_norm["Composite"] = (0.5 * df_norm["Mean Eta^2\nScore"]) + \
                           (0.4 * df_norm["Internal Score"]) + \
                           (0.1 * df_norm["ARI (vs Numeric)\nScore"])
    
    # Prepare data for heatmap
    df_norm["Model Name"] = df_norm["Template"] + " + " + df_norm["Model"]
    
    # Sort by Template (Ascending) and Model (Descending)
    # 'MPNet' < 'MiniLM' is True.
    # We want MiniLM first. So we want Descending order for Model (MiniLM, MPNet).
    df_norm = df_norm.sort_values(['Template', 'Model'], ascending=[True, False])
    
    df_norm = df_norm.set_index("Model Name")
    
    # Select columns to plot - Matching Raw Table Order + Composite
    cols_to_plot = [
        "Internal Score",
        "ARI (vs Numeric)\nScore", 
        "Mean Eta^2\nScore", 
        "Composite"
    ]
    plot_data = df_norm[cols_to_plot]
    
    # Plot - Compact Size for Paper
    plt.figure(figsize=(9, 3.5))
    sns.set_context("paper", font_scale=1.0)
    
    # Custom colormap (Blue = Lower, Red = Higher)
    cmap = "coolwarm"
    
    ax = sns.heatmap(plot_data, annot=True, fmt=".4f", cmap=cmap, linewidths=.5, cbar_kws={'label': 'Normalized Score', 'shrink': 1.0})
    
    plt.title("Model Decision Matrix (Normalized Scores)", fontsize=11, fontweight='bold', pad=10)
    plt.ylabel("") # Hide "Label" label
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    internal_note = "Internal Score = mean(Silhouette_norm, Calinski-Harabasz_norm, Davies-Bouldin_norm)."
    plt.figtext(0.5, 0.01, internal_note, ha='center', fontsize=8)
    plt.tight_layout(pad=0.5, rect=[0, 0.06, 1, 1])
    
    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir(exist_ok=True)
        
    output_path = FIGURES_DIR / "model_decision_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    print(internal_note)

if __name__ == "__main__":
    df = get_comparison_data()
    plot_heatmap(df)
