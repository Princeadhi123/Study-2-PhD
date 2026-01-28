import matplotlib.pyplot as plt
from pathlib import Path

# Paths to the two images
img1_path = Path(r"C:\Users\pdaadh\Desktop\Study-2\figures\gmm\AICc\gmm_aicc_best_pca.png")
img2_path = Path(r"C:\Users\pdaadh\Desktop\Study-2\narrative_embedding_clustering\outputs\template_A\all_MiniLM_L6_v2\figures\embeddings_pca_PC1_vs_PC3_final_hulls.png")

# Output path
output_path = Path(r"C:\Users\pdaadh\Desktop\Study-2\figures\combined_pca_hull_comparison.png")

# Load images
img1 = plt.imread(img1_path)
img2 = plt.imread(img2_path)

# Create side-by-side plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot first image (a)
axs[0].imshow(img1)
axs[0].axis('off')
axs[0].set_title('(a) Numeric Baseline (GMM-AICc, K=4)', fontsize=12, fontweight='bold')

# Plot second image (b)
axs[1].imshow(img2)
axs[1].axis('off')
axs[1].set_title('(b) Narrative Clustering (Strategy C + MiniLM, K=9)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined figure saved to {output_path}")
