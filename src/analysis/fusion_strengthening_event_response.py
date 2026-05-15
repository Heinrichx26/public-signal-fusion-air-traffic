from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fusion_strengthening_event_level import (
    OUT_ROOT,
    bootstrap_summary,
    candidate_control,
    load_events,
    load_panel,
    parse_months,
    summarize_hours,
)
from smoke_source_fusion_topics import weighted_mean


WINDOW_ORDER = ["pre_3h", "pre_1h", "active", "post_1h", "post_3h", "clean_lag_6h"]


def response_bounds(row: pd.Series, window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if window == "pre_3h":
        return row.start_utc - pd.Timedelta(hours=3), row.start_utc - pd.Timedelta(hours=1)
    if window == "pre_1h":
        return row.start_utc - pd.Timedelta(hours=1), row.start_utc
    if window == "active":
        return row.start_utc, row.end_utc
    if window == "post_1h":
        return row.end_utc, row.end_utc + pd.Timedelta(hours=1)
    if window == "post_3h":
        return row.end_utc + pd.Timedelta(hours=1), row.end_utc + pd.Timedelta(hours=3)
    if window == "clean_lag_6h":
        return row.end_utc + pd.Timedelta(hours=3), row.end_utc + pd.Timedelta(hours=6)
    raise ValueError(window)


def treated_hours(panel: pd.DataFrame, event: pd.Series, window: str) -> pd.DataFrame:
    start, end = response_bounds(event, window)
    hours = panel[
        (panel["utc_hour"] < end)
        & (panel["utc_hour"] + pd.Timedelta(hours=1) > start)
        & (panel["mild_weather_abs"] == 1.0)
    ].copy()
    if window != "active":
        hours = hours[hours["active_minutes"] < 1].copy()
    return hours


def build_response_table(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    panel = panel.sort_values(["airport", "utc_hour"]).copy()
    for idx, event in enumerate(events.itertuples(index=False), start=1):
        event = pd.Series(event._asdict() | {"event_id": idx})
        airport_panel = panel[panel["airport"] == event.airport]
        for window in WINDOW_ORDER:
            hours = treated_hours(airport_panel, event, window)
            if hours.empty or hours["arrivals"].sum() <= 0:
                continue
            control = candidate_control(airport_panel, hours)
            if control.empty or control["arrivals"].sum() <= 0:
                continue
            treated = summarize_hours(hours)
            baseline = summarize_hours(control)
            rows.append(
                {
                    "event_id": idx,
                    "airport": event.airport,
                    "month": int(event.month),
                    "tmi_type": event.tmi_type,
                    "window": window,
                    "window_order": WINDOW_ORDER.index(window) + 1,
                    "treated_arrivals": int(treated["arrivals"]),
                    "control_arrivals": int(baseline["arrivals"]),
                    "treated_delay": treated["delay_rate"],
                    "control_delay": baseline["delay_rate"],
                    "delay_diff": treated["delay_rate"] - baseline["delay_rate"],
                    "treated_cancel": treated["cancel_rate"],
                    "control_cancel": baseline["cancel_rate"],
                    "cancel_diff": treated["cancel_rate"] - baseline["cancel_rate"],
                    "treated_weather_score": treated["weather_score"],
                    "control_weather_score": baseline["weather_score"],
                }
            )
    return pd.DataFrame(rows)


def summarize_response(table: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    summary = bootstrap_summary(table, reps, seed)
    summary["window_order"] = summary["window"].map({w: i + 1 for i, w in enumerate(WINDOW_ORDER)})
    summary = summary.sort_values("window_order").reset_index(drop=True)
    summary["delay_diff_vs_pre_1h"] = summary["delay_diff"] - float(
        summary.loc[summary["window"] == "pre_1h", "delay_diff"].iloc[0]
    )
    return summary


def write_assessment(summary: pd.DataFrame, path) -> None:
    lookup = summary.set_index("window")
    pre_ok = abs(float(lookup.loc["pre_3h", "delay_diff"])) <= 0.05
    active_ok = float(lookup.loc["active", "delay_ci_low"]) > 0.15
    post_ok = float(lookup.loc["post_3h", "delay_ci_low"]) > 0.10
    lag_ok = float(lookup.loc["clean_lag_6h", "delay_diff"]) > 0
    status = "accepted" if pre_ok and active_ok and post_ok and lag_ok else "diagnostic"
    lines = ["# Event response assessment", "", f"Assessment: {status}.", ""]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.window}: events {row.events}, delay diff {row.delay_diff:+.3f}, "
            f"95% CI [{row.delay_ci_low:+.3f}, {row.delay_ci_high:+.3f}]."
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    out = OUT_ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    panel = load_panel(args.scope, months)
    events = load_events(args.scope, months)
    table = build_response_table(panel, events)
    summary = summarize_response(table, args.reps, args.seed)
    table.to_csv(out / "event_response_matched_windows.csv", index=False)
    summary.to_csv(out / "event_response_curve.csv", index=False)
    write_assessment(summary, out / "event_response_assessment.md")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["10", "30"], default="30")
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--output-name", default="event_response_smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
