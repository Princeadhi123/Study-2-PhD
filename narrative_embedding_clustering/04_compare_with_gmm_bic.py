import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from scipy.stats import f as f_dist

from config import (
    MARKS_WITH_CLUSTERS_PATH,
    NARRATIVE_TEMPLATE_VERSION,
    OUTPUT_DIR,
    OUTPUT_ROOT,
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


def save_cluster_keywords(df: pd.DataFrame, text_col: str, cluster_col: str, top_n: int = 4) -> None:
    # Group text by cluster
    clusters = sorted(df[cluster_col].unique())
    grouped_text = [" ".join(df[df[cluster_col] == c][text_col].astype(str)) for c in clusters]

    # Define template-specific boilerplate words to ignore
    template_stopwords = {
        "student", "answered", "items", "accuracy", "proportion", "correct",
        "responses", "average", "mean", "response", "time", "seconds",
        "timing", "variance", "coefficient", "variation",
        "longest", "streak", "row", "incorrect",
        "consecutive", "answers", "rate",
        "subject", "marks", "overall", "missingness",
        "values", "unknown",
        # Fluff words added by user request
        "highly", "moderately", "occasional", "variable", "stable", "long", "short"
    }
    # Union with standard English stop words
    custom_stop_words = list(ENGLISH_STOP_WORDS.union(template_stopwords))

    # Compute TF-IDF
    # token_pattern=r'(?u)\b[a-zA-Z]{2,}\b' ensures we only match words with 2+ letters, excluding numbers.
    tfidf = TfidfVectorizer(
        stop_words=custom_stop_words, 
        max_features=1000,
        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
    )
    X_tfidf = tfidf.fit_transform(grouped_text)
    feature_names = np.array(tfidf.get_feature_names_out())

    # Extract top words per cluster
    rows = []
    for i, cluster_label in enumerate(clusters):
        # Get sorting indices for the i-th cluster row
        sorted_ids = np.argsort(X_tfidf[i].toarray().flatten())[::-1][:top_n]
        top_words = feature_names[sorted_ids]
        scores = X_tfidf[i].toarray().flatten()[sorted_ids]
        
        row = {"Cluster": cluster_label, "Keywords": ", ".join([f"{w} ({s:.2f})" for w, s in zip(top_words, scores)])}
        rows.append(row)

    # Save
    out_path = OUTPUT_DIR / make_versioned_filename("cluster_keywords_tfidf.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved cluster keywords to {out_path}")


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

    # Extract and save keywords for narrative clusters
    save_cluster_keywords(base, "narrative_text", "narrative_best_label")

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

    metric_descriptions = {
        "adjusted_rand_index": (
            "Agreement between baseline cluster labels and narrative_best_label "
            "(1 = perfect match, 0 ≈ random)."
        ),
        "silhouette_cosine": (
            "Cluster separation in narrative embedding space using cosine distance "
            "(higher is better, max = 1)."
        ),
        "calinski_harabasz": (
            "Calinski-Harabasz index in narrative embedding space (higher is better)."
        ),
        "davies_bouldin": (
            "Davies-Bouldin index in narrative embedding space (lower is better)."
        ),
    }

    metrics_rows = [
        {
            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
            "baseline": "gmm_bic_best_label",
            "metric": "adjusted_rand_index",
            "value": ari_bic,
            "description": metric_descriptions["adjusted_rand_index"],
        },
        {
            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
            "baseline": "gmm_aic_best_label",
            "metric": "adjusted_rand_index",
            "value": ari_aic,
            "description": metric_descriptions["adjusted_rand_index"],
        },
    ]

    # Embedding-space internal indices for each clustering (numeric BIC, numeric AIC, narrative).
    emb_filename = make_versioned_filename("embeddings.npy")
    emb_index_filename = make_versioned_filename("embeddings_index.csv")
    emb_path = OUTPUT_DIR / emb_filename
    emb_index_path = OUTPUT_DIR / emb_index_filename

    # These will be reused later to select central example profiles.
    X_all: np.ndarray | None = None
    emb_base: pd.DataFrame | None = None

    if emb_path.exists() and emb_index_path.exists():
        try:
            X_all = np.load(emb_path)
            index_df = pd.read_csv(emb_index_path)

            # Restrict to students present in both embeddings and base (i.e., have all labels).
            emb_base = index_df.merge(
                base[[
                    "IDCode",
                    "gmm_bic_best_label",
                    "gmm_aic_best_label",
                    "narrative_best_label",
                ]],
                on="IDCode",
                how="inner",
            )

            if not emb_base.empty:
                indices = emb_base["index"].to_numpy(dtype=int)
                X = X_all[indices]

                label_sets = {
                    "narrative_best_label": emb_base["narrative_best_label"].to_numpy(),
                    "gmm_bic_best_label": emb_base["gmm_bic_best_label"].to_numpy(),
                    "gmm_aic_best_label": emb_base["gmm_aic_best_label"].to_numpy(),
                }

                for baseline_name, labels in label_sets.items():
                    label_counts = pd.Series(labels).value_counts()
                    # Need at least 2 clusters, each with at least 2 members, for stable indices.
                    if len(label_counts) < 2 or not (label_counts > 1).all():
                        continue

                    sil_val = None
                    ch_val = None
                    db_val = None

                    try:
                        sil_val = float(silhouette_score(X, labels, metric="cosine"))
                    except Exception:
                        pass

                    try:
                        ch_val = float(calinski_harabasz_score(X, labels))
                    except Exception:
                        pass

                    try:
                        db_val = float(davies_bouldin_score(X, labels))
                    except Exception:
                        pass

                    if sil_val is not None:
                        metrics_rows.append(
                            {
                                "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                                "baseline": baseline_name,
                                "metric": "silhouette_cosine",
                                "value": sil_val,
                                "description": metric_descriptions["silhouette_cosine"],
                            }
                        )

                    if ch_val is not None:
                        metrics_rows.append(
                            {
                                "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                                "baseline": baseline_name,
                                "metric": "calinski_harabasz",
                                "value": ch_val,
                                "description": metric_descriptions["calinski_harabasz"],
                            }
                        )

                    if db_val is not None:
                        metrics_rows.append(
                            {
                                "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                                "baseline": baseline_name,
                                "metric": "davies_bouldin",
                                "value": db_val,
                                "description": metric_descriptions["davies_bouldin"],
                            }
                        )
        except Exception:
            # If anything goes wrong with embedding-based metrics, skip them silently
            # so that ARI results are still produced.
            X_all = None
            emb_base = None

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

            # One-way ANOVA and effect sizes for marks by cluster label.
            anova_rows: list[dict] = []
            cluster_labels = ("gmm_bic_best_label", "gmm_aic_best_label", "narrative_best_label")

            for cluster_col in cluster_labels:
                if cluster_col not in merged.columns:
                    continue

                for col in numeric_cols:
                    # Skip ANOVA for the overall missing-work metric; it will still be
                    # included in the descriptive marks_by_* summaries.
                    if col.lower() == "missing_total":
                        continue

                    sub = merged[[cluster_col, col]].dropna()
                    if sub.empty:
                        continue

                    grouped = list(sub.groupby(cluster_col)[col])
                    if len(grouped) < 2:
                        continue

                    groups = [g.values for _, g in grouped]
                    sizes = [len(g) for g in groups]
                    # Require at least 2 observations per group for stability.
                    if any(n < 2 for n in sizes):
                        continue

                    n_total = sum(sizes)
                    if n_total <= len(groups):
                        continue

                    grand_mean = float(sub[col].mean())
                    ss_between = float(
                        sum(n * (float(g.mean()) - grand_mean) ** 2 for n, g in zip(sizes, groups))
                    )
                    ss_within = float(sum(((g - float(g.mean())) ** 2).sum() for g in groups))
                    df_between = len(groups) - 1
                    df_within = n_total - len(groups)
                    if df_within <= 0 or ss_within <= 0.0:
                        continue

                    ms_between = ss_between / df_between
                    ms_within = ss_within / df_within
                    f_stat = ms_between / ms_within

                    try:
                        p_value = float(1.0 - f_dist.cdf(f_stat, df_between, df_within))
                    except Exception:
                        p_value = float("nan")

                    ss_total = ss_between + ss_within
                    eta_squared = ss_between / ss_total if ss_total > 0.0 else float("nan")

                    anova_rows.append(
                        {
                            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                            "cluster_label": cluster_col,
                            "outcome": col,
                            "f_statistic": f_stat,
                            "df_between": df_between,
                            "df_within": df_within,
                            "p_value": p_value,
                            "eta_squared": eta_squared,
                        }
                    )

            if anova_rows:
                anova_df = pd.DataFrame(anova_rows)

                # --- Split ANOVA outputs into narrative (template-specific) and
                # numeric (template-independent) parts.
                numeric_mask = anova_df["cluster_label"].isin(
                    ["gmm_bic_best_label", "gmm_aic_best_label"]
                )
                narrative_mask = anova_df["cluster_label"] == "narrative_best_label"

                # Narrative ANOVA: keep only narrative_best_label rows in each
                # template/model directory so these files truly reflect the
                # template-specific narrative clustering.
                if narrative_mask.any():
                    narrative_df = anova_df[narrative_mask].copy()
                    anova_filename = make_versioned_filename("marks_anova_by_cluster.csv")
                    anova_path = OUTPUT_DIR / anova_filename
                    narrative_df.to_csv(anova_path, index=False)

                # Numeric ANOVA (BIC/AIC): write a single global file under the
                # outputs/ root, marking template_version as GLOBAL so it is
                # clear these results are not tied to any particular narrative
                # template.
                if numeric_mask.any():
                    numeric_df = anova_df[numeric_mask].copy()
                    numeric_df["template_version"] = "GLOBAL"
                    numeric_anova_path = OUTPUT_ROOT / "numeric_marks_anova_by_cluster.csv"
                    numeric_df.to_csv(numeric_anova_path, index=False)

    # --- Example profiles: choose central students per narrative cluster.
    # If we have embeddings and an alignment between embeddings and labels
    # (emb_base), pick the most central students in embedding space for each
    # narrative cluster. Otherwise, fall back to taking the first 5.
    examples: list[pd.DataFrame] = []

    if X_all is not None and emb_base is not None and not emb_base.empty:
        # Map base by ID for easy lookup in the desired order.
        base_by_id = base.set_index("IDCode")

        for label_val, group in emb_base.groupby("narrative_best_label"):
            # Embedding indices for this narrative cluster.
            idx_array = group["index"].to_numpy(dtype=int)
            if idx_array.size == 0:
                continue
            X_group = X_all[idx_array]
            # Cluster centroid in embedding space.
            center = X_group.mean(axis=0)
            # Euclidean distance to the centroid.
            dists = np.linalg.norm(X_group - center, axis=1)
            order = np.argsort(dists)[:5]
            central_ids = group.iloc[order]["IDCode"].tolist()

            # Recover the corresponding rows (including narrative_text) in
            # the order of increasing distance to the centroid.
            try:
                sample = base_by_id.loc[central_ids].reset_index()
            except Exception:
                # If anything goes wrong, skip this cluster.
                continue

            examples.append(sample)
    else:
        # Fallback: simple head(5) per narrative cluster as before.
        for _, group in base.groupby("narrative_best_label"):
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
