import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from config import K_RANGE, NARRATIVE_TEMPLATE_VERSION, OUTPUT_DIR, make_versioned_filename


def main() -> None:
    emb_filename = make_versioned_filename("embeddings.npy")
    index_filename = make_versioned_filename("embeddings_index.csv")
    emb_path = OUTPUT_DIR / emb_filename
    index_path = OUTPUT_DIR / index_filename
    if not emb_path.exists() or not index_path.exists():
        raise SystemExit("Embeddings or index not found. Run 02_compute_embeddings.py first.")
    X = np.load(emb_path)
    index_df = pd.read_csv(index_path)

    best_model_info: dict | None = None
    best_bic = np.inf
    best_labels: np.ndarray | None = None
    rows: list[dict] = []

    covariance_types = ("full", "diag", "tied", "spherical")

    for k in K_RANGE:
        for cov in covariance_types:
            try:
                gm = GaussianMixture(n_components=int(k), covariance_type=cov, random_state=42, n_init=5)
                gm.fit(X)
                bic_val = float(gm.bic(X))
                rows.append({"K": int(k), "covariance_type": cov, "bic": bic_val})
                if bic_val < best_bic:
                    best_bic = bic_val
                    best_model_info = {"k": int(k), "covariance_type": cov}
                    best_labels = gm.predict(X)
            except Exception:
                continue

    if not rows:
        raise SystemExit("Failed to fit any GMM models on embeddings.")

    results_df = pd.DataFrame(rows)
    model_results_filename = make_versioned_filename("model_results_narrative.csv")
    model_results_path = OUTPUT_DIR / model_results_filename
    results_df.to_csv(model_results_path, index=False)

    if best_labels is None or best_model_info is None:
        raise SystemExit("Failed to obtain a valid GMM solution for any K / covariance_type.")

    clusters_df = index_df.copy()
    clusters_df["narrative_gmm_bic_best_label"] = best_labels
    clusters_df["narrative_best_label"] = best_labels
    out_filename = make_versioned_filename("narrative_clusters.csv")
    out_path = OUTPUT_DIR / out_filename
    clusters_df.to_csv(out_path, index=False)

    print(
        f"Saved narrative clustering results (template {NARRATIVE_TEMPLATE_VERSION.upper()}) to {out_path}"
    )
    print(
        "Best GMM by BIC: k={k}, cov={cov}, BIC={bic:.2f}".format(
            k=best_model_info["k"], cov=best_model_info["covariance_type"], bic=best_bic
        )
    )


if __name__ == "__main__":
    main()
