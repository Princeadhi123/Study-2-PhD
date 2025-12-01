import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import DERIVED_FEATURES_PATH, OUTPUT_DIR


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


def build_narratives(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()

    # Ensure we have derived columns needed for narratives.
    # Current derived_features.csv has total_correct but not accuracy, and no rt_cv.
    # Compute accuracy as proportion correct, and rt_cv as (std_rt / mean_rt).
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["accuracy"] = feats["total_correct"] / feats["n_items"].replace(0, np.nan)
        std_rt = np.sqrt(feats["var_rt"].clip(lower=0.0))
        feats["rt_cv"] = std_rt / feats["avg_rt"].replace(0, np.nan)

    acc_low, acc_high = _compute_quantile_thresholds(feats["accuracy"])
    rt_low, rt_high = _compute_quantile_thresholds(feats["avg_rt"])
    var_low, var_high = _compute_quantile_thresholds(feats["var_rt"])
    lc_low, lc_high = _compute_quantile_thresholds(feats["longest_correct_streak"])
    li_low, li_high = _compute_quantile_thresholds(feats["longest_incorrect_streak"])
    cc_low, cc_high = _compute_quantile_thresholds(feats["consecutive_correct_rate"])

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

        parts: list[str] = []
        parts.append(
            f"The student answered {n_items} items with {acc_cat} accuracy ({acc:.2f} proportion correct)."
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

        narrative = " ".join(parts)
        narratives.append(narrative)

    out = feats[["IDCode"]].copy()
    out["narrative_text"] = narratives
    return out


def main() -> None:
    df = pd.read_csv(DERIVED_FEATURES_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    narratives = build_narratives(df)
    out_path = OUTPUT_DIR / "narratives.csv"
    narratives.to_csv(out_path, index=False)
    print(f"Written narratives to {out_path}")


if __name__ == "__main__":
    main()
