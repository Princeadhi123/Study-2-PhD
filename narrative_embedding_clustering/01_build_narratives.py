import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DERIVED_FEATURES_PATH,
    MARKS_WITH_CLUSTERS_PATH,
    NARRATIVE_TEMPLATE_VERSION,
    OUTPUT_DIR,
    make_versioned_filename,
)


def _compute_thresholds(series: pd.Series, method: str = "quantile", abs_low: float | None = None, abs_high: float | None = None) -> tuple[float, float]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0, 0.0
    
    if method == "absolute" and abs_low is not None and abs_high is not None:
        return abs_low, abs_high
        
    # Default to quantiles (0.33, 0.67)
    q_low, q_high = 0.33, 0.67
    low = float(s.quantile(q_low))
    high = float(s.quantile(q_high))
    if math.isclose(low, high):
        high = float(s.max())
    return low, high


def _categorise(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if math.isnan(value):
        return "unknown"
    if value <= low:
        return low_label
    if value >= high:
        return high_label
    return mid_label


def build_narratives(df: pd.DataFrame, marks_df: pd.DataFrame | None = None) -> pd.DataFrame:
    feats = df.copy()
    if marks_df is not None:
        mark_cols = [c for c in ["S1", "S2", "S3", "S5", "S6", "Missing_total"] if c in marks_df.columns]
        if mark_cols:
            marks_sel = marks_df[["IDCode"] + mark_cols].copy()
            feats = feats.merge(marks_sel, on="IDCode", how="left")

    # Ensure we have derived columns needed for narratives.
    # Current derived_features.csv has total_correct but not accuracy, and no rt_cv.
    # Compute accuracy as proportion correct, and rt_cv as (std_rt / mean_rt).
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["accuracy"] = feats["total_correct"] / feats["n_items"].replace(0, np.nan)
        std_rt = np.sqrt(feats["var_rt"].clip(lower=0.0))
        feats["rt_cv"] = std_rt / feats["avg_rt"].replace(0, np.nan)

    # Impute missing values to avoid "unknown" clusters (Issue 6)
    # Using median imputation for numeric features
    impute_cols = ["accuracy", "avg_rt", "var_rt", "rt_cv", "longest_correct_streak", "longest_incorrect_streak", "consecutive_correct_rate"]
    for col in impute_cols:
        if col in feats.columns:
            median_val = feats[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            feats[col] = feats[col].fillna(median_val)

    # Use absolute thresholds for accuracy to avoid forced tertiles (Issue 3)
    # accuracy: < 0.60 low, > 0.85 high
    acc_low, acc_high = _compute_thresholds(feats["accuracy"], method="absolute", abs_low=0.60, abs_high=0.85)
    
    # For others, use quantiles but include numerical values in text to fix quantization bottleneck (Issue 1)
    rt_low, rt_high = _compute_thresholds(feats["avg_rt"])
    var_low, var_high = _compute_thresholds(feats["var_rt"])
    lc_low, lc_high = _compute_thresholds(feats["longest_correct_streak"])
    li_low, li_high = _compute_thresholds(feats["longest_incorrect_streak"])
    cc_low, cc_high = _compute_thresholds(feats["consecutive_correct_rate"])

    template = NARRATIVE_TEMPLATE_VERSION.upper()

    narratives: list[str] = []

    for _, row in feats.iterrows():
        acc = float(row["accuracy"])
        avg_rt = float(row["avg_rt"])
        var_rt = float(row["var_rt"])
        rt_cv = float(row["rt_cv"])
        lc = float(row["longest_correct_streak"])
        li = float(row["longest_incorrect_streak"])
        cc = float(row["consecutive_correct_rate"])
        n_items = int(row["n_items"])

        acc_cat = _categorise(acc, acc_low, acc_high, "low", "medium", "high")
        speed_cat = _categorise(avg_rt, rt_low, rt_high, "fast", "moderate", "slow")
        var_cat = _categorise(var_rt, var_low, var_high, "stable", "moderately variable", "highly variable")
        streak_correct_cat = _categorise(lc, lc_low, lc_high, "short", "moderate", "long")
        streak_incorrect_cat = _categorise(li, li_low, li_high, "short", "moderate", "long")
        cc_cat = _categorise(cc, cc_low, cc_high, "rare", "occasional", "frequent")

        if template == "B":
            narrative = (
                f"Accuracy {acc_cat} ({acc:.2%}); "
                f"speed {speed_cat} ({avg_rt:.2f}s); "
                f"timing {var_cat} (var={var_rt:.2f}); "
                f"correct streak {streak_correct_cat} ({int(lc)}); "
                f"incorrect streak {streak_incorrect_cat} ({int(li)}); "
                f"consecutive correct {cc_cat} ({cc:.2f})."
            )
        else:
            parts: list[str] = []
            parts.append(
                f"The student answered {n_items} items with {acc_cat} accuracy ({acc:.2%} proportion correct)."
            )
            parts.append(
                f"Their responses are {speed_cat} on average (mean response time {avg_rt:.2f} seconds) and {var_cat} in timing (response-time variance {var_rt:.2f}, coefficient of variation {rt_cv:.2f})."
            )
            parts.append(
                f"They have a {streak_correct_cat} longest correct streak ({int(lc)} in a row) and a {streak_incorrect_cat} longest incorrect streak ({int(li)} in a row)."
            )
            parts.append(
                f"Consecutive correct answers are {cc_cat} (consecutive-correct rate {cc:.2f})."
            )

            if template == "C":
                mark_values: list[str] = []
                for col in ("S1", "S2", "S3", "S5", "S6"):
                    if col in row and not pd.isna(row[col]):
                        mark_values.append(f"{col}={int(row[col])}")
                missing_sentence = ""
                if "Missing_total" in row and not pd.isna(row["Missing_total"]):
                    missing_sentence = f"overall missingness {float(row['Missing_total']):.2f}%"
                if mark_values or missing_sentence:
                    details: list[str] = []
                    if mark_values:
                        details.append("subject marks: " + ", ".join(mark_values))
                    if missing_sentence:
                        details.append(missing_sentence)
                    parts.append("They have " + " and ".join(details) + ".")

            narrative = " ".join(parts)
        narratives.append(narrative)

    out = feats[["IDCode"]].copy()
    out["narrative_text"] = narratives
    return out


def main() -> None:
    df = pd.read_csv(DERIVED_FEATURES_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    marks_df: pd.DataFrame | None = None
    if NARRATIVE_TEMPLATE_VERSION.upper() == "C" and MARKS_WITH_CLUSTERS_PATH.exists():
        marks_df = pd.read_csv(MARKS_WITH_CLUSTERS_PATH)
    narratives = build_narratives(df, marks_df=marks_df)
    out_filename = make_versioned_filename("narratives.csv")
    out_path = OUTPUT_DIR / out_filename
    narratives.to_csv(out_path, index=False)
    print(f"Written narratives (template {NARRATIVE_TEMPLATE_VERSION.upper()}) to {out_path}")


if __name__ == "__main__":
    main()
