import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Configuration
INPUT_FILE = Path("figures/final_model_comparison.csv")
OUTPUT_FILE = Path("figures/final_model_comparison_heatmap.png")

def normalize_column(series, invert=False):
    """Normalize a pandas series to 0-1 range. 
    If invert is True, lower values get higher scores (closer to 1)."""
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    
    if invert:
        # Lower is better (1.0)
        return (max_val - series) / (max_val - min_val)
    else:
        # Higher is better (1.0)
        return (series - min_val) / (max_val - min_val)

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Sort by Template (Ascending) and Model (Descending: MiniLM > MPNet, so MiniLM first)
    df = df.sort_values(['Template', 'Model'], ascending=[True, False])

    # Create descriptive index
    df['Label'] = df['Template'] + " + " + df['Model']
    df = df.set_index('Label')
    
    # Rename columns to match desired output
    df = df.rename(columns={
        'Silhouette (Cosine)': 'Silhouette (Cosine)\nScore',
        'Calinski-Harabasz': 'Calinski-Harabasz\nScore',
        'Davies-Bouldin': 'Davies-Bouldin\nScore',
        'ARI (vs Numeric)': 'ARI (vs Numeric)\nScore',
        'Mean Eta^2': 'Mean Eta^2\nScore'
    })
    
    # Select numeric metrics for the heatmap
    metrics = [
        'Silhouette (Cosine)\nScore', 
        'Calinski-Harabasz\nScore', 
        'Davies-Bouldin\nScore', 
        'ARI (vs Numeric)\nScore', 
        'Mean Eta^2\nScore'
    ]
    
    # Create annotation dataframe (Raw Values)
    annot_df = df[metrics]
    
    # Create color dataframe (Normalized 0-1)
    color_df = pd.DataFrame(index=annot_df.index, columns=annot_df.columns)
    
    # Apply normalization logic
    for col in metrics:
        invert = (col == 'Davies-Bouldin')
        color_df[col] = normalize_column(annot_df[col], invert=invert)

    # Plotting - Compact Size for Paper (e.g., 1 column width)
    # 8 inches wide is roughly a full page width, 3.5 inches high is compact
    plt.figure(figsize=(9, 3.5)) 
    sns.set_context("paper", font_scale=1.0) # Use 'paper' context for smaller fonts
    
    # Create heatmap
    ax = sns.heatmap(
        data=color_df, 
        annot=annot_df, 
        fmt=".4f", 
        cmap="coolwarm", 
        linewidths=.5,
        cbar_kws={'label': 'Relative Performance', 'shrink': 1.0}
    )
    
    # Compact Title
    plt.title("Model Comparison", fontsize=11, fontweight='bold', pad=10)
    plt.ylabel("") # Hide "Label" label
    
    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
