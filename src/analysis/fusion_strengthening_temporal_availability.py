from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fusion_prediction_increment import evaluate, feature_levels, fit_grouped_logit, make_design, sigmoid, train_stats
from fusion_strengthening_common import ROOT_OUT, prepare_base_panel
from smoke_source_fusion_topics import weighted_mean


ISSUE = ROOT_OUT / "issue_time_full_2025" / "advisory_issue_times.csv"
TARGETS = {"long_arrival_delay": "arr_delay60_count", "cancellation": "cancel_count"}
CALENDAR = ["month_sin", "month_cos"]
WEATHER = ["weather_score", "mild_weather_abs", "wind_speed_mps", "visibility_km", "ceiling_m", "temperature_c"]
DEMAND = ["scheduled_arrivals", "scheduled_departures", "arrival_bank_intensity", "departure_bank_intensity", "arrival_carrier_hhi", "departure_carrier_hhi"]
CATS = ["airport", "local_hour", "day_of_week"]
SETTINGS = {
    "issue_before_hour_active": ("active_before_minutes", "active_before_mild"),
    "issue_within_hour_active": ("active_within_minutes", "active_within_mild"),
    "post_1h_known": ("post_1h_known_minutes", "post_1h_known_mild"),
    "post_3h_known": ("post_3h_known_minutes", "post_3h_known_mild"),
}


def parse_months(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        elif part:
            out.append(int(part))
    return sorted({m for m in out if 1 <= m <= 12})


def add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for col, _ in SETTINGS.values():
        panel[col] = 0.0
    issues = pd.read_csv(ISSUE)
    issues = issues[issues["airport"].isin(panel["airport"].unique())].copy()
    for col in ["issue_utc", "effective_start_utc", "effective_end_utc"]:
        issues[col] = pd.to_datetime(issues[col], utc=True).dt.tz_localize(None)
    hour_start = panel["utc_hour"]
    hour_end = hour_start + pd.Timedelta(hours=1)
    for ev in issues.itertuples(index=False):
        airport_mask = panel["airport"].eq(ev.airport)
        intervals = {
            "active_before_minutes": (ev.effective_start_utc, ev.effective_end_utc, ev.issue_utc <= hour_start),
            "active_within_minutes": (ev.effective_start_utc, ev.effective_end_utc, ev.issue_utc < hour_end),
            "post_1h_known_minutes": (ev.effective_start_utc, ev.effective_end_utc + pd.Timedelta(hours=1), ev.issue_utc <= hour_start),
            "post_3h_known_minutes": (ev.effective_start_utc, ev.effective_end_utc + pd.Timedelta(hours=3), ev.issue_utc <= hour_start),
        }
        for name, (start, end, known_mask) in intervals.items():
            mask = airport_mask & known_mask & (hour_start < end) & (hour_end > start)
            if not mask.any():
                continue
            overlap_start = hour_start[mask].map(lambda x: max(x, start))
            overlap_end = hour_end[mask].map(lambda x: min(x, end))
            minutes = (overlap_end - overlap_start).dt.total_seconds() / 60.0
            panel.loc[mask, name] += minutes.to_numpy()
    for minute_col, mild_col in SETTINGS.values():
        panel[mild_col] = ((panel[minute_col] >= 45) & (panel["mild_weather_abs"] == 1.0)).astype(float)
        panel[minute_col] = (panel[minute_col] / 60).clip(0, 8)
    return panel


def fit_setting(panel: pd.DataFrame, target: str, success_col: str, setting: str, months: list[int]) -> list[dict]:
    minute_col, mild_col = SETTINGS[setting]
    specs = {
        "baseline": CALENDAR + WEATHER + DEMAND,
        setting: CALENDAR + WEATHER + DEMAND + [minute_col, mild_col],
    }
    rows = []
    for model, numeric in specs.items():
        levels = feature_levels(panel, CATS)
        fold_frames = []
        for month in months:
            train = panel[panel["month"] != month].copy()
            test = panel[panel["month"] == month].copy()
            stats = train_stats(train, numeric)
            x_train = make_design(train, numeric, CATS, stats, levels)
            x_test = make_design(test, numeric, CATS, stats, levels)
            beta = fit_grouped_logit(x_train, train[success_col].to_numpy(float), train["arrivals"].to_numpy(float))
            prob = sigmoid(x_test @ beta)
            fold = test[["arrivals", success_col]].copy()
            fold["pred_prob"] = prob
            fold_frames.append(fold)
        pred = pd.concat(fold_frames, ignore_index=True)
        rows.append(evaluate(pred[success_col].to_numpy(float), pred["arrivals"].to_numpy(float), pred["pred_prob"].to_numpy(float)) | {"target": target, "setting": setting, "model": model})
    return rows


def descriptive(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, (minute_col, mild_col) in SETTINGS.items():
        strong = panel[(panel[mild_col] == 1.0) & (panel["mild_weather_abs"] == 1.0)]
        base = panel[(panel[mild_col] == 0.0) & (panel["mild_weather_abs"] == 1.0)]
        rows.append(
            {
                "setting": setting,
                "conflict_arrivals": int(strong["arrivals"].sum()),
                "baseline_arrivals": int(base["arrivals"].sum()),
                "delay_diff": weighted_mean(strong, "arr_delay60_rate", "arrivals") - weighted_mean(base, "arr_delay60_rate", "arrivals"),
                "cancel_diff": weighted_mean(strong, "cancel_rate", "arrivals") - weighted_mean(base, "cancel_rate", "arrivals"),
            }
        )
    return pd.DataFrame(rows)


def run(args) -> None:
    out = ROOT_OUT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    panel = add_temporal_features(prepare_base_panel(2025, months))
    metric_rows = []
    for target, success_col in TARGETS.items():
        for setting in SETTINGS:
            metric_rows.extend(fit_setting(panel, target, success_col, setting, months))
    metrics = pd.DataFrame(metric_rows)
    base = metrics[metrics["model"] == "baseline"][["target", "setting", "auc", "log_loss", "brier"]]
    gains = metrics.merge(base, on=["target", "setting"], suffixes=("", "_base"))
    gains = gains[gains["model"] != "baseline"].copy()
    gains["auc_gain"] = gains["auc"] - gains["auc_base"]
    gains["log_loss_gain"] = gains["log_loss_base"] - gains["log_loss"]
    gains["brier_gain"] = gains["brier_base"] - gains["brier"]
    desc = descriptive(panel)
    panel.to_csv(out / "temporal_availability_panel.csv", index=False)
    metrics.to_csv(out / "temporal_availability_metrics.csv", index=False)
    gains.to_csv(out / "temporal_availability_gain_summary.csv", index=False)
    desc.to_csv(out / "temporal_availability_state_diffs.csv", index=False)
    accepted = (gains[gains["target"] == "long_arrival_delay"]["auc_gain"] >= 0.03).any() and desc["delay_diff"].max() >= 0.15
    lines = ["# Temporal availability assessment", "", f"Assessment: {'accepted' if accepted else 'diagnostic'}.", ""]
    for row in gains.itertuples(index=False):
        lines.append(f"- {row.target}, {row.setting}: AUC gain {row.auc_gain:+.3f}.")
    for row in desc.itertuples(index=False):
        lines.append(f"- {row.setting}: delay diff {row.delay_diff:+.3f}; cancel diff {row.cancel_diff:+.3f}.")
    (out / "temporal_availability_assessment.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--output-name", default="temporal_availability_smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
