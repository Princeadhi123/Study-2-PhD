import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from config import (
    MARKS_WITH_CLUSTERS_PATH,
    NARRATIVE_TEMPLATE_VERSION,
    OUTPUT_DIR,
    STUDENT_CLUSTERS_PATH,
    make_versioned_filename,
)


def _detect_id_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"idcode", "id", "studentid", "student_id"}]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if c.lower().startswith("id"):
            return c
    raise ValueError("Could not find an ID column in marks_with_clusters.csv")


def main() -> None:
    narrative_clusters_path = OUTPUT_DIR / make_versioned_filename("narrative_clusters.csv")
    narratives_path = OUTPUT_DIR / make_versioned_filename("narratives.csv")
    if not narrative_clusters_path.exists():
        raise SystemExit(f"Missing narrative_clusters.csv at {narrative_clusters_path}")
    if not narratives_path.exists():
        raise SystemExit(f"Missing narratives.csv at {narratives_path}")

    stud_clusters = pd.read_csv(STUDENT_CLUSTERS_PATH)
    nar_clusters = pd.read_csv(narrative_clusters_path)
    narratives = pd.read_csv(narratives_path)

    # Include both BIC- and AIC-based numeric GMM labels for comparison.
    base = stud_clusters[["IDCode", "gmm_bic_best_label", "gmm_aic_best_label"]].merge(
        nar_clusters[["IDCode", "narrative_best_label"]], on="IDCode", how="inner"
    )
    base = base.merge(narratives[["IDCode", "narrative_text"]], on="IDCode", how="left")

    if base.empty:
        raise SystemExit("No overlapping students between numeric and narrative clustering.")

    # Overlap and ARI for BIC-based numeric GMM vs narrative GMM.
    overlap_bic = pd.crosstab(base["gmm_bic_best_label"], base["narrative_best_label"])
    overlap_bic_filename = make_versioned_filename("gmm_bic_vs_narrative_overlap.csv")
    overlap_bic_path = OUTPUT_DIR / overlap_bic_filename
    overlap_bic.to_csv(overlap_bic_path)

    ari_bic = float(adjusted_rand_score(base["gmm_bic_best_label"], base["narrative_best_label"]))

    # Overlap and ARI for AIC-based numeric GMM vs narrative GMM.
    overlap_aic = pd.crosstab(base["gmm_aic_best_label"], base["narrative_best_label"])
    overlap_aic_filename = make_versioned_filename("gmm_aic_vs_narrative_overlap.csv")
    overlap_aic_path = OUTPUT_DIR / overlap_aic_filename
    overlap_aic.to_csv(overlap_aic_path)

    ari_aic = float(adjusted_rand_score(base["gmm_aic_best_label"], base["narrative_best_label"]))

    # Silhouette score for narrative clusters based on embeddings (internal validity).
    silhouette_value = None
    emb_filename = make_versioned_filename("embeddings.npy")
    emb_path = OUTPUT_DIR / emb_filename
    if emb_path.exists():
        try:
            X = np.load(emb_path)
            labels = nar_clusters["narrative_best_label"].to_numpy()
            label_counts = pd.Series(labels).value_counts()
            if len(label_counts) >= 2 and (label_counts > 1).all():
                silhouette_value = float(silhouette_score(X, labels, metric="cosine"))
        except Exception:
            silhouette_value = None

    metrics_rows = [
        {
            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
            "baseline": "gmm_bic_best_label",
            "metric": "adjusted_rand_index",
            "value": ari_bic,
        },
        {
            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
            "baseline": "gmm_aic_best_label",
            "metric": "adjusted_rand_index",
            "value": ari_aic,
        },
    ]

    if silhouette_value is not None:
        metrics_rows.append(
            {
                "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                "baseline": "narrative_best_label",
                "metric": "silhouette_cosine",
                "value": silhouette_value,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    ari_filename = make_versioned_filename("gmm_vs_narrative_metrics.csv")
    ari_path = OUTPUT_DIR / ari_filename
    metrics_df.to_csv(ari_path, index=False)

    if MARKS_WITH_CLUSTERS_PATH.exists():
        marks_df = pd.read_csv(MARKS_WITH_CLUSTERS_PATH)
        id_col = _detect_id_col(marks_df)
        # Treat cluster-label columns from the marks file as labels, not as marks to be averaged.
        # Exclude them from numeric_cols so that the base gmm_bic_best_label column is preserved
        # and not duplicated/renamed during the merge.
        label_cols = {"gmm_bic_best_label", "narrative_best_label"}
        numeric_cols = [
            c
            for c in marks_df.columns
            if c != id_col and c not in label_cols and pd.api.types.is_numeric_dtype(marks_df[c])
        ]
        merged = base.merge(marks_df[[id_col] + numeric_cols], left_on="IDCode", right_on=id_col, how="left")

        if numeric_cols:
            gmm_group = merged.groupby("gmm_bic_best_label")[numeric_cols].mean().reset_index()
            narrative_group = merged.groupby("narrative_best_label")[numeric_cols].mean().reset_index()

            gmm_marks_filename = make_versioned_filename("marks_by_gmm_bic.csv")
            narrative_marks_filename = make_versioned_filename("marks_by_narrative.csv")
            gmm_marks_path = OUTPUT_DIR / gmm_marks_filename
            narrative_marks_path = OUTPUT_DIR / narrative_marks_filename
            gmm_group.to_csv(gmm_marks_path, index=False)
            narrative_group.to_csv(narrative_marks_path, index=False)

    examples = []
    for _, group in base.groupby("narrative_best_label"):
        # Take up to 5 example students per narrative cluster; the
        # 'narrative_best_label' column is already present in 'base'.
        sample = group.head(5).copy()
        examples.append(sample)

    if examples:
        examples_df = pd.concat(examples, ignore_index=True)
        examples_filename = make_versioned_filename("example_profiles.csv")
        examples_path = OUTPUT_DIR / examples_filename
        examples_df.to_csv(examples_path, index=False)

    print(f"Saved BIC overlap table to {overlap_bic_path}")
    print(f"Saved AIC overlap table to {overlap_aic_path}")
    print(f"Saved ARI metrics to {ari_path}")
    if MARKS_WITH_CLUSTERS_PATH.exists():
        print("Saved marks summaries by cluster if numeric marks were found.")
    print("Saved example profiles for narrative clusters.")


if __name__ == "__main__":
    main()
