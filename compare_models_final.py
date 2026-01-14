import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import f as f_dist

# Configuration
OUTPUT_DIR = Path("narrative_embedding_clustering/outputs")
MARKS_PATH = Path("diagnostics/cluster input features/marks_with_clusters.csv")

MODELS = [
    ("Template A", "MPNet", OUTPUT_DIR / "template_A/all_mpnet_base_v2"),
    ("Template A", "MiniLM", OUTPUT_DIR / "template_A/all_MiniLM_L6_v2"),
    ("Template B", "MPNet", OUTPUT_DIR / "template_B/all_mpnet_base_v2"),
    ("Template B", "MiniLM", OUTPUT_DIR / "template_B/all_MiniLM_L6_v2"),
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
            
    return np.mean(etas) if etas else 0.0, etas

def main():
    results = []
    
    # Load Marks
    marks_df = pd.read_csv(MARKS_PATH)
    # Identify numeric mark columns (excluding IDs and existing cluster labels)
    numeric_cols = [c for c in marks_df.columns if c not in ["IDCode", "student_id"] and pd.api.types.is_numeric_dtype(marks_df[c])]
    # Filter for relevant outcome marks if known, or use all. The user mentioned S1, S2, S3...
    # Let's try to detect "S" columns or use the ones from previous ANOVAs
    mark_cols = [c for c in numeric_cols if c.startswith("S") or c in ["FinalMark", "Total"]]
    if not mark_cols:
        mark_cols = numeric_cols # Fallback
        
    print(f"Calculating predictive power (Mean Eta^2) across {len(mark_cols)} outcomes: {mark_cols}")

    for template, model, path in MODELS:
        print(f"Processing {template} - {model}...")
        
        # 1. Get Internal Metrics for Best AICc Model (Cosine/Raw)
        # We use gmm_vs_narrative_metrics.csv which has the Cosine Silhouette used in plots
        suffix = "_B" if "template_B" in str(path) else ""
        metrics_file = path / f"gmm_vs_narrative_metrics{suffix}.csv"
        
        sil = np.nan
        ch = np.nan
        db = np.nan
        
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            # Filter for narrative_best_label rows
            # This file contains metrics for the *chosen* best model (which is now AICc based)
            subset = metrics_df[metrics_df["baseline"] == "narrative_best_label"]
            
            for _, row in subset.iterrows():
                m = row["metric"]
                v = row["value"]
                if m == "silhouette_cosine":
                    sil = v
                elif m == "calinski_harabasz":
                    ch = v
                elif m == "davies_bouldin":
                    db = v
        else:
            print(f"  Missing metrics file: {metrics_file}")

        # Get K and Cov from model_results just for info
        model_results_file = path / f"model_results_narrative{suffix}.csv"
        k_val = 0
        cov_val = "unknown"
        if model_results_file.exists():
             model_res = pd.read_csv(model_results_file)
             valid_aicc = model_res[model_res["aicc"] != np.inf]
             if not valid_aicc.empty:
                 best_row = valid_aicc.loc[valid_aicc["aicc"].idxmin()]
                 k_val = int(best_row["K"])
                 cov_val = best_row["covariance_type"]

        # 2. Get External Metrics (Eta Squared)
        clusters_file = path / f"narrative_clusters{suffix}.csv"
        if not clusters_file.exists():
            print(f"  Missing clusters file: {clusters_file}")
            continue
            
        clusters_df = pd.read_csv(clusters_file)
        
        # We specifically want the AICc label column
        label_col = "narrative_gmm_aicc_best_label"
        if label_col not in clusters_df.columns:
            print(f"  Missing {label_col} in {clusters_file}")
            continue
            
        # Merge with marks
        # Detect ID col in clusters
        id_col = "IDCode" if "IDCode" in clusters_df.columns else "id"
        merged = pd.merge(clusters_df, marks_df, left_on=id_col, right_on="IDCode", how="inner")
        
        mean_eta, all_etas = calculate_eta_squared(merged, label_col, mark_cols)
        
        results.append({
            "Template": template,
            "Model": model,
            "Winner K": k_val,
            "Winner Cov": cov_val,
            "Silhouette (Cosine)": sil,
            "Calinski-Harabasz": ch,
            "Davies-Bouldin": db,
            "Mean Eta^2": mean_eta
        })

    # Create Summary Table
    final_df = pd.DataFrame(results)
    
    # Formatting
    final_df = final_df.sort_values("Mean Eta^2", ascending=False)
    
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON (DEFLATION CORRECTED / AICc)")
    print("="*80)
    print(final_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    
    print("\n" + "="*80)
    print("DECISION MATRIX")
    print("="*80)
    
    # Decision Logic
    # Normalize metrics to 0-1 score
    # Eta: Higher is better
    # Sil: Higher is better
    # DB: Lower is better -> Invert
    
    df_norm = final_df.copy()
    df_norm["Eta_Score"] = df_norm["Mean Eta^2"] / df_norm["Mean Eta^2"].max()
    df_norm["Sil_Score"] = df_norm["Silhouette (Cosine)"] / df_norm["Silhouette (Cosine)"].max()
    df_norm["DB_Score"] = df_norm["Davies-Bouldin"].min() / df_norm["Davies-Bouldin"] 
    
    # Composite Score: 60% External (Eta), 40% Internal (Average of Sil and DB)
    df_norm["Composite"] = (0.6 * df_norm["Eta_Score"]) + (0.2 * df_norm["Sil_Score"]) + (0.2 * df_norm["DB_Score"])
    
    df_norm = df_norm.sort_values("Composite", ascending=False)
    print("Scoring Weights: 60% Predictive Power (Eta^2), 40% Cluster Quality (Sil + DB)")
    print(df_norm[["Template", "Model", "Composite", "Eta_Score", "Sil_Score", "DB_Score"]].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    
    winner = df_norm.iloc[0]
    print(f"\n🏆 THE WINNER IS: {winner['Template']} + {winner['Model']} 🏆")
    print(f"Reason: Best balance of predictive power (Eta={winner['Eta_Score']:.2f} relative score) and structural quality.")

if __name__ == "__main__":
    main()
