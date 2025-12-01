import math
import argparse

import numpy as np
import pandas as pd

from config import DERIVED_FEATURES_PATH, MARKS_WITH_CLUSTERS_PATH, OUTPUT_DIR


def _compute_quantile_thresholds(series: pd.Series, q_low: float = 0.33, q_high: float = 0.67) -> tuple[float, float]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0, 0.0
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


def _build_template_a(
    n_items: int,
    total_correct: float,
    avg_rt: float,
    var_rt: float,
    lc: float,
    li: float,
    cc: float,
    correct_cat: str,
    speed_cat: str,
    var_cat: str,
    streak_correct_cat: str,
    streak_incorrect_cat: str,
    cc_cat: str,
    extra_parts: list[str] | None = None,
) -> str:
    """Current 4-sentence behavioural template (Version A).

    Optionally appends extra_parts (e.g. marks sentences) at the end.
    """
    parts: list[str] = []
    parts.append(
        f"The student answered {n_items} items with a {correct_cat} number of correct answers ({total_correct:.0f} correct)."
    )
    parts.append(
        f"Their responses are {speed_cat} on average (mean response time {avg_rt:.2f} seconds) and {var_cat} in timing (response-time variance {var_rt:.2f})."
    )
    parts.append(
        f"They have a {streak_correct_cat} longest correct streak ({int(lc)} in a row) and a {streak_incorrect_cat} longest incorrect streak ({int(li)} in a row)."
    )
    parts.append(
        f"Consecutive correct answers are {cc_cat} (consecutive-correct rate {cc:.2f})."
    )
    if extra_parts:
        parts.extend(extra_parts)
    return " ".join(parts)


def _build_template_b(
    correct_cat: str,
    speed_cat: str,
    var_cat: str,
    streak_correct_cat: str,
    streak_incorrect_cat: str,
    cc_cat: str,
) -> str:
    """Shorter, more categorical template (Version B).

    Focuses on labels rather than numbers, using the total number of correct
    answers instead of accuracy.
    """
    return (
        "Behaviour summary: correct_level={correct}, speed={speed}, timing_variability={var}, "
        "correct_streak={cs}, incorrect_streak={is_}, consecutive_correct={cc}."
    ).format(
        correct=correct_cat,
        speed=speed_cat,
        var=var_cat,
        cs=streak_correct_cat,
        is_=streak_incorrect_cat,
        cc=cc_cat,
    )


def _build_template_c_marks_parts(
    marks_row: pd.Series | None,
    mark_low: float,
    mark_high: float,
    miss_low: float,
    miss_high: float,
) -> list[str]:
    """Build extra sentences about marks / missingness for Version C.

    Uses mean of S1, S2, S3, S5, S6 and Missing_total.
    """
    if marks_row is None:
        return []

    subject_cols = [c for c in ["S1", "S2", "S3", "S5", "S6"] if c in marks_row.index]
    if not subject_cols:
        return []

    mean_mark = float(marks_row[subject_cols].mean())
    missing_total = float(marks_row.get("Missing_total", float("nan")))

    ach_cat = _categorise(mean_mark, mark_low, mark_high, "low", "medium", "high")
    miss_cat = _categorise(missing_total, miss_low, miss_high, "low", "moderate", "high")

    parts: list[str] = []
    parts.append(
        f"Overall subject performance is {ach_cat} (mean mark {mean_mark:.2f} across core subjects)."
    )
    if not math.isnan(missing_total):
        parts.append(
            f"The level of missing responses is {miss_cat} (Missing_total {missing_total:.2f})."
        )
    return parts


def build_narratives(
    df: pd.DataFrame,
    template: str = "A",
    marks_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build narratives for all students using the specified template.

    template:
      - "A": current 4-sentence behavioural description.
      - "B": shorter categorical summary.
      - "C": version A plus extra sentences about marks/missingness.
    """
    feats = df.copy()

    # We base narratives on the same variables used in the numeric pipeline:
    # total_correct, avg_rt, var_rt, longest_correct_streak,
    # longest_incorrect_streak, consecutive_correct_rate, response_variance.
    correct_low, correct_high = _compute_quantile_thresholds(feats["total_correct"])
    rt_low, rt_high = _compute_quantile_thresholds(feats["avg_rt"])
    var_low, var_high = _compute_quantile_thresholds(feats["var_rt"])
    lc_low, lc_high = _compute_quantile_thresholds(feats["longest_correct_streak"])
    li_low, li_high = _compute_quantile_thresholds(feats["longest_incorrect_streak"])
    cc_low, cc_high = _compute_quantile_thresholds(feats["consecutive_correct_rate"])

    # If marks are provided (Version C), pre-compute thresholds for mean marks and missingness.
    marks_by_id: dict[str, pd.Series] | None = None
    mark_low = mark_high = miss_low = miss_high = 0.0
    if template.upper() == "C" and marks_df is not None:
        subject_cols = [c for c in ["S1", "S2", "S3", "S5", "S6"] if c in marks_df.columns]
        if subject_cols:
            marks_df = marks_df.copy()
            marks_df["mean_mark"] = marks_df[subject_cols].mean(axis=1)
            mark_low, mark_high = _compute_quantile_thresholds(marks_df["mean_mark"])
        if "Missing_total" in marks_df.columns:
            miss_low, miss_high = _compute_quantile_thresholds(marks_df["Missing_total"])
        marks_by_id = {str(row["IDCode"]): row for _, row in marks_df.iterrows()}

    narratives: list[str] = []

    for _, row in feats.iterrows():
        total_correct = float(row["total_correct"])
        avg_rt = float(row["avg_rt"])
        var_rt = float(row["var_rt"])
        lc = float(row["longest_correct_streak"])
        li = float(row["longest_incorrect_streak"])
        cc = float(row["consecutive_correct_rate"])
        n_items = int(row["n_items"])

        correct_cat = _categorise(total_correct, correct_low, correct_high, "low", "medium", "high")
        speed_cat = _categorise(avg_rt, rt_low, rt_high, "fast", "moderate", "slow")
        var_cat = _categorise(var_rt, var_low, var_high, "stable", "moderately variable", "highly variable")
        streak_correct_cat = _categorise(lc, lc_low, lc_high, "short", "moderate", "long")
        streak_incorrect_cat = _categorise(li, li_low, li_high, "short", "moderate", "long")
        cc_cat = _categorise(cc, cc_low, cc_high, "rare", "occasional", "frequent")

        id_code = str(row["IDCode"])
        narrative: str

        if template.upper() == "B":
            narrative = _build_template_b(
                correct_cat,
                speed_cat,
                var_cat,
                streak_correct_cat,
                streak_incorrect_cat,
                cc_cat,
            )
        else:
            extra_parts: list[str] | None = None
            if template.upper() == "C" and marks_by_id is not None:
                marks_row = marks_by_id.get(id_code)
                extra_parts = _build_template_c_marks_parts(marks_row, mark_low, mark_high, miss_low, miss_high)
            narrative = _build_template_a(
                n_items,
                total_correct,
                avg_rt,
                var_rt,
                lc,
                li,
                cc,
                correct_cat,
                speed_cat,
                var_cat,
                streak_correct_cat,
                streak_incorrect_cat,
                cc_cat,
                extra_parts=extra_parts,
            )

        narratives.append(narrative)

    out = feats[["IDCode"]].copy()
    out["narrative_text"] = narratives
    return out

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build narrative templates from derived features.")
    parser.add_argument(
        "--template",
        "-t",
        choices=["A", "B", "C"],
        default="A",
        help="Narrative template version to use (A=current, B=short categorical, C=detailed with marks)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(DERIVED_FEATURES_PATH)

    marks_df: pd.DataFrame | None = None
    if args.template.upper() == "C":
        if not MARKS_WITH_CLUSTERS_PATH.exists():
            raise SystemExit(f"Template C requires marks file, but not found at {MARKS_WITH_CLUSTERS_PATH}")
        marks_df = pd.read_csv(MARKS_WITH_CLUSTERS_PATH)

    template_dir = OUTPUT_DIR / f"template_{args.template.upper()}"
    template_dir.mkdir(parents=True, exist_ok=True)

    narratives = build_narratives(df, template=args.template.upper(), marks_df=marks_df)
    out_path = template_dir / "narratives.csv"
    narratives.to_csv(out_path, index=False)
    print(f"Written narratives for template {args.template.upper()} to {out_path}")


if __name__ == "__main__":
    main()
