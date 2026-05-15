from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from smoke_source_fusion_topics import AIRPORT_TZ, weighted_mean


PROJECT = Path(__file__).resolve().parents[2]
RAW_EVENTS = PROJECT / "data" / "raw" / "faa_atcscc_advisories" / "faa_atcscc_gdp_gs_reparsed_2025_v2.csv"
PANEL_10 = PROJECT / "results" / "experiments" / "atcscc_full_year_windows" / "airport_hour_panel_with_windows.csv"
PANEL_30 = (
    PROJECT
    / "results"
    / "experiments"
    / "supplemental_validation"
    / "extended_airports_2025"
    / "full_30_airports_2025_airport_hour_panel.csv"
)
OUT_ROOT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
AIRPORTS_30 = "ATL CLT DEN DFW EWR JFK LAX LGA ORD SFO BOS BWI DCA DTW FLL HNL IAD IAH LAS MCO MDW MIA MSP PHL PHX RDU SAN SEA SLC TPA".split()


def load_panel(scope: str, months: list[int]) -> pd.DataFrame:
    path = PANEL_30 if scope == "30" else PANEL_10
    panel = pd.read_csv(path, parse_dates=["utc_hour"])
    panel = panel[(panel["month"].isin(months)) & (panel["arrivals"] > 0)].copy()
    panel = panel[panel["weather_score"].notna()].copy()
    return panel


def load_events(scope: str, months: list[int]) -> pd.DataFrame:
    events = pd.read_csv(RAW_EVENTS)
    airports = AIRPORTS_30 if scope == "30" else list(AIRPORT_TZ)
    events = events[events["airport"].isin(airports)].copy()
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True).dt.tz_localize(None)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True).dt.tz_localize(None)
    events["month"] = events["start_utc"].dt.month
    events = events[events["month"].isin(months)].copy()
    return events.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc", "source_url"]).reset_index(drop=True)


def window_bounds(row, window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if window == "active":
        return row.start_utc, row.end_utc
    if window == "post_3h":
        return row.start_utc, row.end_utc + pd.Timedelta(hours=3)
    if window == "clean_lag_3h":
        return row.end_utc, row.end_utc + pd.Timedelta(hours=3)
    raise ValueError(window)


def summarize_hours(hours: pd.DataFrame) -> dict[str, float]:
    return {
        "hours": len(hours),
        "arrivals": float(hours["arrivals"].sum()),
        "delay_rate": weighted_mean(hours, "arr_delay60_rate", "arrivals"),
        "cancel_rate": weighted_mean(hours, "cancel_rate", "arrivals"),
        "mild_share": weighted_mean(hours, "mild_weather_abs", "arrivals"),
        "weather_score": weighted_mean(hours, "weather_score", "arrivals"),
    }


def candidate_control(panel: pd.DataFrame, treated_hours: pd.DataFrame) -> pd.DataFrame:
    selected = []
    used: set[pd.Timestamp] = set()
    base = panel[
        (panel["mild_weather_abs"] == 1.0)
        & (panel["active_minutes"] < 1)
        & (panel["post_3h_minutes"] < 1)
        & (panel["arrivals"] > 0)
    ].copy()
    for hour in treated_hours.itertuples(index=False):
        controls = base[(base["month"] == hour.month) & (base["local_hour"] == hour.local_hour)].copy()
        if controls.empty:
            controls = base[base["month"] == hour.month].copy()
        controls["weather_distance"] = (controls["weather_score"] - hour.weather_score).abs()
        controls["used"] = controls["utc_hour"].isin(used)
        controls = controls.sort_values(["used", "weather_distance", "utc_hour"])
        if controls.empty:
            continue
        chosen = controls.iloc[0]
        used.add(chosen["utc_hour"])
        selected.append(chosen)
    if not selected:
        return base.iloc[0:0].copy()
    return pd.DataFrame(selected)


def build_event_table(panel: pd.DataFrame, events: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    rows = []
    panel = panel.sort_values(["airport", "utc_hour"]).copy()
    for idx, event in enumerate(events.itertuples(index=False), start=1):
        event = event._asdict() | {"event_id": idx}
        event = pd.Series(event)
        airport_panel = panel[panel["airport"] == event.airport]
        for window in windows:
            start, end = window_bounds(event, window)
            hours = airport_panel[
                (airport_panel["utc_hour"] < end)
                & (airport_panel["utc_hour"] + pd.Timedelta(hours=1) > start)
                & (airport_panel["mild_weather_abs"] == 1.0)
            ].copy()
            if window == "clean_lag_3h":
                hours = hours[hours["active_minutes"] < 1].copy()
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
                    "start_hour_utc": int(event.start_utc.hour),
                    "tmi_type": event.tmi_type,
                    "window": window,
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


def bootstrap_summary(events: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for window, group in events.groupby("window"):
        values = group[["delay_diff", "cancel_diff", "treated_arrivals", "control_arrivals"]].to_numpy(float)
        n = len(values)
        delay_draws, cancel_draws = [], []
        for _ in range(reps):
            sample = values[rng.integers(0, n, size=n)]
            w = sample[:, 2] + sample[:, 3]
            delay_draws.append(np.average(sample[:, 0], weights=w))
            cancel_draws.append(np.average(sample[:, 1], weights=w))
        rows.append(
            {
                "window": window,
                "events": n,
                "treated_arrivals": int(group["treated_arrivals"].sum()),
                "control_arrivals": int(group["control_arrivals"].sum()),
                "treated_delay": weighted_mean(group, "treated_delay", "treated_arrivals"),
                "control_delay": weighted_mean(group, "control_delay", "control_arrivals"),
                "delay_diff": np.average(group["delay_diff"], weights=group["treated_arrivals"] + group["control_arrivals"]),
                "delay_ci_low": float(np.quantile(delay_draws, 0.025)),
                "delay_ci_high": float(np.quantile(delay_draws, 0.975)),
                "cancel_diff": np.average(group["cancel_diff"], weights=group["treated_arrivals"] + group["control_arrivals"]),
                "cancel_ci_low": float(np.quantile(cancel_draws, 0.025)),
                "cancel_ci_high": float(np.quantile(cancel_draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


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


def run(args) -> None:
    out = OUT_ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    panel = load_panel(args.scope, months)
    events = load_events(args.scope, months)
    table = build_event_table(panel, events, args.windows.split(","))
    summary = bootstrap_summary(table, args.reps, args.seed)
    table.to_csv(out / "event_level_matched_windows.csv", index=False)
    summary.to_csv(out / "event_level_scorecard.csv", index=False)
    status = "accepted" if (summary["delay_ci_low"] > 0).all() else "diagnostic"
    lines = [f"# Event-level validation", "", f"Scope: {args.scope} airports; months {args.months}.", f"Assessment: {status}.", ""]
    for row in summary.itertuples(index=False):
        lines.append(f"- {row.window}: events {row.events}, delay diff {row.delay_diff:+.3f}, 95% CI [{row.delay_ci_low:+.3f}, {row.delay_ci_high:+.3f}].")
    (out / "event_level_assessment.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["10", "30"], default="30")
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--windows", default="active,post_3h")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--output-name", default="event_level_smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
