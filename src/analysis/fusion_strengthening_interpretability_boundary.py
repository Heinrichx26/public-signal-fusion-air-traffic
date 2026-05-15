from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smoke_source_fusion_topics import weighted_mean


PROJECT = Path(__file__).resolve().parents[2]
ROOT_OUT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
DEMAND = ROOT_OUT / "demand_residual_full_2025"
PANEL = DEMAND / "panel_with_demand.csv"
PREDS = DEMAND / "baseline_ladder_predictions.csv"
NOAA = PROJECT / "results" / "experiments" / "supplemental_validation" / "noaa_storm_filter" / "base10_2025_panel_with_storm_flag.csv"
MATCHED = PROJECT / "results" / "experiments" / "atcscc_full_year_windows" / "matched_control_pairs.csv"


def baseline_predictions() -> pd.DataFrame:
    preds = pd.read_csv(PREDS, parse_dates=["utc_hour"])
    return preds[
        (preds["model"] == "calendar_weather_schedule_demand")
        & (preds["target"] == "long_arrival_delay")
    ][["airport", "utc_hour", "month", "pred_prob"]]


def residual_diff(panel: pd.DataFrame, pred: pd.DataFrame, filter_name: str) -> list[dict]:
    use = panel.merge(pred, on=["airport", "utc_hour", "month"], how="inner")
    use = use[(use["arrivals"] > 0) & (use["mild_weather_abs"] == 1.0)].copy()
    use["obs_rate"] = use["arr_delay60_count"] / use["arrivals"]
    use["residual"] = use["obs_rate"] - use["pred_prob"]
    rows = []
    for window, strong_col in [("active", "active_strong"), ("post_3h", "post_3h_strong")]:
        treated = use[use[strong_col] == 1.0]
        control = use[use[strong_col] == 0.0]
        rows.append(
            {
                "check": filter_name,
                "window": window,
                "treated_arrivals": int(treated["arrivals"].sum()),
                "control_arrivals": int(control["arrivals"].sum()),
                "treated_residual": weighted_mean(treated, "residual", "arrivals"),
                "control_residual": weighted_mean(control, "residual", "arrivals"),
                "residual_diff": weighted_mean(treated, "residual", "arrivals") - weighted_mean(control, "residual", "arrivals"),
            }
        )
    return rows


def matched_residual(pred: pd.DataFrame) -> list[dict]:
    matched = pd.read_csv(MATCHED, parse_dates=["utc_hour"])
    pred = pred.rename(columns={"pred_prob": "treated_pred"})
    matched = matched.merge(pred[["airport", "utc_hour", "month", "treated_pred"]], on=["airport", "utc_hour", "month"], how="left")
    matched["treated_residual"] = matched["treated_arr_delay60_rate"] - matched["treated_pred"]
    matched["control_residual"] = matched["control_arr_delay60_rate"] - matched["treated_pred"]
    rows = []
    for scope, g in matched.groupby("scope"):
        rows.append(
            {
                "check": "matched residual",
                "window": scope,
                "treated_arrivals": int(g["treated_arrivals"].sum()),
                "control_arrivals": int(g["control_arrivals"].sum()),
                "treated_residual": weighted_mean(g, "treated_residual", "treated_arrivals"),
                "control_residual": weighted_mean(g, "control_residual", "control_arrivals"),
                "residual_diff": weighted_mean(g, "delta_arr_delay60_rate", "treated_arrivals"),
            }
        )
    return rows


def run(output_name: str) -> None:
    out = ROOT_OUT / output_name
    out.mkdir(parents=True, exist_ok=True)
    pred = baseline_predictions()
    panel = pd.read_csv(PANEL, parse_dates=["utc_hour"])
    noaa = pd.read_csv(NOAA, parse_dates=["utc_hour"])
    noaa = noaa[noaa["nearby_storm_event"] == 0.0].copy()
    rows = []
    rows.extend(residual_diff(panel, pred, "mild-weather residual"))
    rows.extend(residual_diff(noaa, pred, "storm-filter residual"))
    rows.extend(matched_residual(pred))
    score = pd.DataFrame(rows)
    score.to_csv(out / "advisory_only_interpretability_boundary.csv", index=False)
    accepted = score[score["check"].isin(["mild-weather residual", "storm-filter residual"])]["residual_diff"].min() >= 0.10
    lines = ["# Advisory-only interpretability boundary", "", f"Assessment: {'accepted' if accepted else 'diagnostic'}.", ""]
    for row in score.itertuples(index=False):
        lines.append(f"- {row.check}, {row.window}: residual diff {row.residual_diff:+.3f}.")
    (out / "advisory_only_interpretability_boundary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="interpretability_boundary_full_2025")
    args = parser.parse_args()
    run(args.output_name)


if __name__ == "__main__":
    main()
