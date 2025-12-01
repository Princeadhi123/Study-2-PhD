import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from config import OUTPUT_DIR, STUDENT_CLUSTERS_PATH


def _load_embeddings():
    emb_path = OUTPUT_DIR / "embeddings.npy"
    index_path = OUTPUT_DIR / "embeddings_index.csv"
    if not emb_path.exists() or not index_path.exists():
        raise SystemExit("Embeddings or index not found. Run 02_compute_embeddings.py first.")
    X = np.load(emb_path)
    index_df = pd.read_csv(index_path)
    return X, index_df


def _load_clusters():
    narrative_clusters_path = OUTPUT_DIR / "narrative_clusters.csv"
    if not narrative_clusters_path.exists():
        raise SystemExit("Missing narrative_clusters.csv. Run 03_cluster_embeddings.py first.")
    nar_clusters = pd.read_csv(narrative_clusters_path)

    stud_clusters = pd.read_csv(STUDENT_CLUSTERS_PATH)

    return nar_clusters, stud_clusters


def _prepare_coords(index_df: pd.DataFrame, nar_clusters: pd.DataFrame, stud_clusters: pd.DataFrame) -> pd.DataFrame:
    """Merge embedding index with narrative and numeric cluster labels.

    This does *not* compute any projection; projections are computed separately
    and added as columns when plotting.
    """
    coords = index_df.merge(nar_clusters[["IDCode", "narrative_best_label"]], on="IDCode", how="left")
    coords = coords.merge(
        stud_clusters[["IDCode", "gmm_bic_best_label", "gmm_aic_best_label"]], on="IDCode", how="left"
    )
    return coords


def _plot_scatter(
    coords: pd.DataFrame,
    color_col: str,
    title: str,
    filename: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it with:\n"
            "pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 6))

    labels = sorted(coords[color_col].dropna().unique())
    scatter = ax.scatter(
        coords[x_col],
        coords[y_col],
        c=coords[color_col],
        cmap="tab10",
        s=15,
        alpha=0.8,
    )

    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [f"{color_col} = {int(l)}" for l in labels]
    ax.legend(handles, legend_labels, title=color_col, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {title} plot to {out_path}")


def main() -> None:
    X, index_df = _load_embeddings()
    nar_clusters, stud_clusters = _load_clusters()
    coords = _prepare_coords(index_df, nar_clusters, stud_clusters)

    # --- PCA projection (baseline, as before) ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    coords_pca = coords.copy()
    coords_pca["dim1"] = X_pca[:, 0]
    coords_pca["dim2"] = X_pca[:, 1]

    _plot_scatter(
        coords_pca,
        color_col="narrative_best_label",
        title="Narrative GMM-BIC clusters (embedding PCA)",
        filename="embeddings_pca_narrative_clusters.png",
        x_col="dim1",
        y_col="dim2",
        x_label="PC1 (narrative embeddings)",
        y_label="PC2 (narrative embeddings)",
    )

    _plot_scatter(
        coords_pca,
        color_col="gmm_bic_best_label",
        title="Numeric GMM-BIC clusters (embedding PCA)",
        filename="embeddings_pca_gmm_bic_clusters.png",
        x_col="dim1",
        y_col="dim2",
        x_label="PC1 (narrative embeddings)",
        y_label="PC2 (narrative embeddings)",
    )

    _plot_scatter(
        coords_pca,
        color_col="gmm_aic_best_label",
        title="Numeric GMM-AIC clusters (embedding PCA)",
        filename="embeddings_pca_gmm_aic_clusters.png",
        x_col="dim1",
        y_col="dim2",
        x_label="PC1 (narrative embeddings)",
        y_label="PC2 (narrative embeddings)",
    )

    # --- UMAP projection (narrative clusters only) ---
    try:
        import umap

        umap_model = umap.UMAP(n_components=2, random_state=42)
        X_umap = umap_model.fit_transform(X)
        coords_umap = coords.copy()
        coords_umap["dim1"] = X_umap[:, 0]
        coords_umap["dim2"] = X_umap[:, 1]

        _plot_scatter(
            coords_umap,
            color_col="narrative_best_label",
            title="Narrative GMM-BIC clusters (embedding UMAP)",
            filename="embeddings_umap_narrative_clusters.png",
            x_col="dim1",
            y_col="dim2",
            x_label="UMAP1 (narrative embeddings)",
            y_label="UMAP2 (narrative embeddings)",
        )
    except ImportError:
        print(
            "umap-learn is not installed; skipping UMAP plot. "
            "Install it with: pip install umap-learn"
        )

    # --- t-SNE projection (narrative clusters only) ---
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)
    coords_tsne = coords.copy()
    coords_tsne["dim1"] = X_tsne[:, 0]
    coords_tsne["dim2"] = X_tsne[:, 1]

    _plot_scatter(
        coords_tsne,
        color_col="narrative_best_label",
        title="Narrative GMM-BIC clusters (embedding t-SNE)",
        filename="embeddings_tsne_narrative_clusters.png",
        x_col="dim1",
        y_col="dim2",
        x_label="t-SNE1 (narrative embeddings)",
        y_label="t-SNE2 (narrative embeddings)",
    )

    # --- LDA projection (supervised by narrative clusters) ---
    lda = LinearDiscriminantAnalysis(n_components=2)
    y = coords["narrative_best_label"].to_numpy()
    X_lda = lda.fit_transform(X, y)
    coords_lda = coords.copy()
    coords_lda["dim1"] = X_lda[:, 0]
    coords_lda["dim2"] = X_lda[:, 1]

    _plot_scatter(
        coords_lda,
        color_col="narrative_best_label",
        title="Narrative GMM-BIC clusters (embedding LDA)",
        filename="embeddings_lda_narrative_clusters.png",
        x_col="dim1",
        y_col="dim2",
        x_label="LDA1 (narrative embeddings)",
        y_label="LDA2 (narrative embeddings)",
    )


if __name__ == "__main__":
    main()
