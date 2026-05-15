from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from smoke_source_fusion_topics import weighted_mean


PROJECT = Path(__file__).resolve().parents[2]
ROOT_OUT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
DEMAND = ROOT_OUT / "demand_residual_full_2025"
PANEL = DEMAND / "panel_with_demand.csv"
PREDS = DEMAND / "baseline_ladder_predictions.csv"
TARGET_COLS = {"long_arrival_delay": "arr_delay60_count", "cancellation": "cancel_count"}
BASELINE = "calendar_weather_schedule_demand"
FUSION = "schedule_fused_state"


def build_scores() -> pd.DataFrame:
    panel = pd.read_csv(PANEL, parse_dates=["utc_hour"])
    preds = pd.read_csv(PREDS, parse_dates=["utc_hour"])
    keep = preds[preds["model"].isin([BASELINE, FUSION])].copy()
    wide = keep.pivot_table(
        index=["airport", "utc_hour", "month", "target"],
        columns="model",
        values="pred_prob",
        aggfunc="first",
    ).reset_index()
    merged = panel.merge(wide, on=["airport", "utc_hour", "month"], how="inner")
    merged["baseline_pred"] = merged[BASELINE]
    merged["fusion_pred"] = merged[FUSION]
    merged["residual_score"] = merged["fusion_pred"] - merged["baseline_pred"]
    rows = []
    for target, success_col in TARGET_COLS.items():
        use = merged[merged["target"] == target].copy()
        use["success_count"] = use[success_col]
        use["obs_rate"] = use["success_count"] / use["arrivals"]
        use["obs_residual"] = use["obs_rate"] - use["baseline_pred"]
        rows.append(use)
    return pd.concat(rows, ignore_index=True)


def decile_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in scores.groupby("target"):
        group = group[group["arrivals"] > 0].copy()
        group["score_decile"] = pd.qcut(group["residual_score"], 10, labels=False, duplicates="drop") + 1
        for decile, g in group.groupby("score_decile"):
            rows.append(
                {
                    "target": target,
                    "score_decile": int(decile),
                    "airport_hours": len(g),
                    "arrivals": int(g["arrivals"].sum()),
                    "mean_score": weighted_mean(g, "residual_score", "arrivals"),
                    "baseline_pred": weighted_mean(g, "baseline_pred", "arrivals"),
                    "fusion_pred": weighted_mean(g, "fusion_pred", "arrivals"),
                    "observed_rate": weighted_mean(g, "obs_rate", "arrivals"),
                    "observed_residual": weighted_mean(g, "obs_residual", "arrivals"),
                }
            )
    return pd.DataFrame(rows)


def summary_table(deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in deciles.groupby("target"):
        group = group.sort_values("score_decile")
        bottom = group.iloc[0]
        top = group.iloc[-1]
        rate_diffs = group["observed_rate"].diff().dropna()
        resid_diffs = group["observed_residual"].diff().dropna()
        rows.append(
            {
                "target": target,
                "bottom_decile_rate": bottom["observed_rate"],
                "top_decile_rate": top["observed_rate"],
                "top_bottom_ratio": top["observed_rate"] / bottom["observed_rate"] if bottom["observed_rate"] > 0 else np.nan,
                "top_bottom_diff": top["observed_rate"] - bottom["observed_rate"],
                "positive_rate_steps": int((rate_diffs > 0).sum()),
                "positive_residual_steps": int((resid_diffs > 0).sum()),
                "total_adjacent_steps": int(len(rate_diffs)),
                "spearman_decile_rate": group[["score_decile", "observed_rate"]].corr(method="spearman").iloc[0, 1],
                "spearman_decile_residual": group[["score_decile", "observed_residual"]].corr(method="spearman").iloc[0, 1],
            }
        )
    return pd.DataFrame(rows)


def calibration_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in scores.groupby("target"):
        group = group[group["arrivals"] > 0].copy()
        group["fusion_risk_decile"] = pd.qcut(group["fusion_pred"], 10, labels=False, duplicates="drop") + 1
        for decile, g in group.groupby("fusion_risk_decile"):
            rows.append(
                {
                    "target": target,
                    "risk_decile": int(decile),
                    "arrivals": int(g["arrivals"].sum()),
                    "predicted_rate": weighted_mean(g, "fusion_pred", "arrivals"),
                    "observed_rate": weighted_mean(g, "obs_rate", "arrivals"),
                    "calibration_error": weighted_mean(g, "obs_rate", "arrivals") - weighted_mean(g, "fusion_pred", "arrivals"),
                }
            )
    return pd.DataFrame(rows)


def run(output_name: str) -> None:
    out = ROOT_OUT / output_name
    out.mkdir(parents=True, exist_ok=True)
    scores = build_scores()
    deciles = decile_table(scores)
    summary = summary_table(deciles)
    calibration = calibration_table(scores)
    scores.to_csv(out / "residual_state_scores.csv", index=False)
    deciles.to_csv(out / "residual_score_deciles.csv", index=False)
    summary.to_csv(out / "residual_score_summary.csv", index=False)
    calibration.to_csv(out / "residual_score_calibration.csv", index=False)
    chosen = summary[summary["target"] == "long_arrival_delay"].iloc[0]
    accepted = chosen["top_bottom_ratio"] >= 3 and chosen["positive_residual_steps"] >= 8
    lines = ["# Residual score assessment", "", f"Assessment: {'accepted' if accepted else 'diagnostic'}.", ""]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.target}: top/bottom ratio {row.top_bottom_ratio:.2f}, "
            f"positive residual steps {row.positive_residual_steps}/{row.total_adjacent_steps}, "
            f"residual Spearman {row.spearman_decile_residual:.3f}."
        )
    (out / "residual_score_assessment.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="residual_score_full_2025")
    args = parser.parse_args()
    run(args.output_name)


if __name__ == "__main__":
    main()
