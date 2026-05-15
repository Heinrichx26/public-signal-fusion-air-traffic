from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT_OUT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
OUT = ROOT_OUT / "airport_hour_fusion_benchmark"
DEMAND = ROOT_OUT / "demand_residual_full_2025"
TEMPORAL = ROOT_OUT / "temporal_availability_full_2025"
EVENT = ROOT_OUT / "event_level_full_2025_30airports"
SCORE = ROOT_OUT / "residual_score_full_2025"
DIAGNOSTICS = ROOT_OUT / "prediction_diagnostics_full"


def write_schema() -> None:
    rows = [
        ("airport", "Airport identifier", "categorical key"),
        ("utc_hour", "Coordinated Universal Time airport-hour", "temporal key"),
        ("arrivals", "Scheduled arrival count", "outcome denominator"),
        ("arr_delay60_count", "Arrivals delayed at least 60 min", "long-delay outcome"),
        ("cancel_count", "Cancelled arrivals", "cancellation outcome"),
        ("weather_score", "Surface-weather severity score", "weather feature"),
        ("mild_weather_abs", "Mild local surface-weather indicator", "state feature"),
        ("scheduled_arrivals", "Scheduled arrivals in airport-hour", "demand feature"),
        ("scheduled_departures", "Scheduled departures in airport-hour", "demand feature"),
        ("arrival_bank_intensity", "Arrival-bank intensity relative to airport-month mean", "demand feature"),
        ("departure_bank_intensity", "Departure-bank intensity relative to airport-month mean", "demand feature"),
        ("active_minutes", "GDP or GS overlap minutes during active interval", "advisory feature"),
        ("post_3h_minutes", "GDP or GS overlap minutes in active plus 3 h window", "advisory feature"),
        ("residual_score", "Fusion predicted risk minus weather-demand baseline risk", "state score"),
    ]
    pd.DataFrame(rows, columns=["field", "definition", "role"]).to_csv(OUT / "benchmark_field_dictionary.csv", index=False)


def write_tasks() -> None:
    rows = [
        ("Task 1", "Residual state detection", "Rank airport-hours by residual operational state score; evaluate decile lift, residual monotonicity, and top-decile risk."),
        ("Task 2", "Long-delay prediction", "Predict the probability that scheduled arrivals in an airport-hour include arrivals delayed at least 60 min; evaluate AUC, PR-AUC, calibration, Brier score, and lift."),
        ("Task 3", "Cancellation prediction", "Predict the probability that scheduled arrivals in an airport-hour include cancellations; evaluate AUC, PR-AUC, calibration, Brier score, and lift."),
        ("Task 4", "Post-advisory persistence", "Detect elevated disruption in post-advisory and clean-lag windows after advisory end."),
        ("Task 5", "Event-level validation", "Aggregate airport-hour outcomes to GDP or GS advisory events and compare with matched non-advisory windows."),
    ]
    pd.DataFrame(rows, columns=["task_id", "task_name", "definition"]).to_csv(OUT / "benchmark_task_definitions.csv", index=False)


def write_splits() -> None:
    rows = [
        ("leave_one_month_2025", "Train on 11 months of 2025 and test on the held-out month.", "Main temporal generalization split"),
        ("external_year_2024_to_2025", "Use 2024 as external-year validation for the 2025 main pattern.", "Cross-year validation"),
        ("leave_one_airport_2025", "Hold out one airport and evaluate airport-to-airport stability.", "Airport generalization"),
        ("event_bootstrap_2025", "Resample GDP or GS advisory events with replacement.", "Event-level uncertainty"),
    ]
    pd.DataFrame(rows, columns=["split", "definition", "role"]).to_csv(OUT / "benchmark_split_definitions.csv", index=False)


def write_scores() -> None:
    ablation = pd.read_csv(DEMAND / "direct_ablation_matrix.csv")
    temporal = pd.read_csv(TEMPORAL / "temporal_availability_gain_summary.csv")
    event = pd.read_csv(EVENT / "event_level_scorecard.csv")
    score = pd.read_csv(SCORE / "residual_score_summary.csv")
    diag = pd.read_csv(DIAGNOSTICS / "pr_auc_calibration_gains.csv")
    loo = pd.read_csv(DIAGNOSTICS / "leave_one_airport_gain_summary.csv")
    pretrend = pd.read_csv(DIAGNOSTICS / "event_pretrend_contrast.csv")
    rows = []
    for r in ablation.itertuples(index=False):
        rows.append(("long_delay_or_cancel_prediction", r.target, r.model, r.auc, r.auc_gain_vs_weather, "direct_ablation_matrix.csv"))
    for r in temporal.itertuples(index=False):
        rows.append(("temporal_availability_prediction", r.target, r.setting, r.auc, r.auc_gain, "temporal_availability_gain_summary.csv"))
    for r in event.itertuples(index=False):
        rows.append(("event_level_validation", "long_arrival_delay", r.window, r.delay_diff, r.delay_ci_low, "event_level_scorecard.csv"))
    for r in score.itertuples(index=False):
        rows.append(("residual_score_lift", r.target, "top_vs_bottom_decile", r.top_bottom_ratio, r.spearman_decile_residual, "residual_score_summary.csv"))
    for r in diag[diag["model"] == "schedule_fused_state"].itertuples(index=False):
        rows.append(("class_imbalance_prediction", r.target, "fusion_state_pr_auc_gain", r.pr_auc_gain_vs_primary, r.top_decile_lift, "pr_auc_calibration_gains.csv"))
        rows.append(("calibration", r.target, "fusion_state_calibration", r.calibration_slope, r.expected_calibration_error, "pr_auc_calibration_gains.csv"))
    for target, group in loo.groupby("target"):
        rows.append(("leave_one_airport_transfer", target, "positive_auc_airports", int((group["auc_gain_vs_primary"] > 0).sum()), len(group), "leave_one_airport_gain_summary.csv"))
        rows.append(("leave_one_airport_transfer", target, "mean_pr_auc_gain", group["pr_auc_gain_vs_primary"].mean(), int((group["pr_auc_gain_vs_primary"] > 0).sum()), "leave_one_airport_gain_summary.csv"))
    active = pretrend[pretrend["window"] == "active"].iloc[0]
    rows.append(("event_pretrend", "long_arrival_delay", "active_minus_pre_mean", active["delay_diff_minus_pre_mean"], active["delay_ci_low"], "event_pretrend_contrast.csv"))
    pd.DataFrame(rows, columns=["benchmark_task", "target", "baseline_or_setting", "primary_value", "secondary_value", "source_table"]).to_csv(
        OUT / "benchmark_baseline_scores.csv", index=False
    )


def write_readme() -> None:
    text = """# Airport-hour fusion benchmark

This benchmark defines reusable airport-hour tasks for multi-source information fusion in air traffic disruptions. It uses flight outcomes, surface weather, schedule-derived demand, ATCSCC advisory signals, and realized outcomes.

Core tasks:
- residual state detection;
- long-delay prediction;
- cancellation prediction;
- post-advisory persistence detection;
- event-level advisory validation.

The benchmark tables define fields, task targets, split rules, and baseline scores. Plotting and reported tables read these result tables directly.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_schema()
    write_tasks()
    write_splits()
    write_scores()
    write_readme()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
