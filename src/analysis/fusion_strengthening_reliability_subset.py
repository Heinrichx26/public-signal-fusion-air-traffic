from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fusion_strengthening_event_level import bootstrap_summary, build_event_table, load_events, parse_months
from fusion_strengthening_residual_score import summary_table
from fusion_strengthening_common import ROOT_OUT


STRENGTH = ROOT_OUT
SCORES = STRENGTH / "residual_score_full_2025" / "residual_state_scores.csv"
PANEL = STRENGTH / "demand_residual_full_2025" / "panel_with_demand.csv"
NEIGHBOR = (
    ROOT_OUT.parents[0]
    / "supplemental_validation"
    / "neighbor_weather"
    / "base10_2025_panel_with_neighbor_weather.csv"
)


def add_reliability_flags(df: pd.DataFrame) -> pd.DataFrame:
    neighbor = pd.read_csv(
        NEIGHBOR,
        parse_dates=["utc_hour"],
        usecols=["airport", "utc_hour", "neighbor_stations", "mild_agree"],
    )
    out = df.merge(neighbor, on=["airport", "utc_hour"], how="left")
    out["q_weather"] = ((out["neighbor_stations"] >= 2) & (out["mild_agree"] == 1.0)).astype(float)
    out["q_advisory"] = 1.0
    out["q_demand"] = (out["scheduled_arrivals"].fillna(0) > 0).astype(float)
    out["q_high"] = ((out["q_weather"] == 1.0) & (out["q_advisory"] == 1.0) & (out["q_demand"] == 1.0)).astype(float)
    return out


def residual_lift_table() -> pd.DataFrame:
    scores = pd.read_csv(SCORES, parse_dates=["utc_hour"])
    scores = add_reliability_flags(scores)
    rows = []
    for subset, group in [("all", scores), ("high_reliability", scores[scores["q_high"] == 1.0])]:
        summary = summary_table_for_group(group)
        summary["subset"] = subset
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def summary_table_for_group(group: pd.DataFrame) -> pd.DataFrame:
    decile_rows = []
    for target, target_group in group.groupby("target"):
        use = target_group[target_group["arrivals"] > 0].copy()
        use["score_decile"] = pd.qcut(use["residual_score"], 10, labels=False, duplicates="drop") + 1
        for decile, g in use.groupby("score_decile"):
            decile_rows.append(
                {
                    "target": target,
                    "score_decile": int(decile),
                    "airport_hours": len(g),
                    "arrivals": int(g["arrivals"].sum()),
                    "observed_rate": np.average(g["obs_rate"], weights=g["arrivals"]),
                    "observed_residual": np.average(g["obs_residual"], weights=g["arrivals"]),
                }
            )
    return summary_table(pd.DataFrame(decile_rows))


def panel_with_reliability(months: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(PANEL, parse_dates=["utc_hour"])
    panel = panel[(panel["month"].isin(months)) & (panel["arrivals"] > 0)].copy()
    panel = add_reliability_flags(panel)
    keep_cols = [
        "airport",
        "utc_hour",
        "month",
        "arrivals",
        "arr_delay60_rate",
        "cancel_rate",
        "weather_score",
        "mild_weather_abs",
        "active_minutes",
        "post_3h_minutes",
        "local_hour",
    ]
    all_panel = panel[keep_cols].copy()
    high_panel = panel[panel["q_high"] == 1.0][keep_cols].copy()
    return all_panel, high_panel


def event_summary(months: list[int], reps: int, seed: int) -> pd.DataFrame:
    events = load_events("10", months)
    rows = []
    for subset, panel in zip(["all", "high_reliability"], panel_with_reliability(months)):
        table = build_event_table(panel, events, ["active", "post_3h"])
        summary = bootstrap_summary(table, reps, seed)
        summary["subset"] = subset
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def combine(lift: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_wide = events.pivot_table(index="subset", columns="window", values=["events", "delay_diff", "delay_ci_low", "delay_ci_high"], aggfunc="first")
    event_wide.columns = [f"{metric}_{window}" for metric, window in event_wide.columns]
    event_wide = event_wide.reset_index()
    rows = []
    for _, r in lift.iterrows():
        e = event_wide[event_wide["subset"] == r["subset"]].iloc[0]
        rows.append(
            {
                "subset": r["subset"],
                "target": r["target"],
                "residual_top_bottom_ratio": r["top_bottom_ratio"],
                "residual_positive_steps": f"{int(r['positive_residual_steps'])}/{int(r['total_adjacent_steps'])}",
                "residual_spearman": r["spearman_decile_residual"],
                "active_events": int(e["events_active"]),
                "active_event_diff": e["delay_diff_active"],
                "active_ci_low": e["delay_ci_low_active"],
                "active_ci_high": e["delay_ci_high_active"],
                "post_events": int(e["events_post_3h"]),
                "post_event_diff": e["delay_diff_post_3h"],
                "post_ci_low": e["delay_ci_low_post_3h"],
                "post_ci_high": e["delay_ci_high_post_3h"],
            }
        )
    return pd.DataFrame(rows)


def run(args) -> None:
    out = STRENGTH / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    lift = residual_lift_table()
    events = event_summary(months, args.reps, args.seed)
    scorecard = combine(lift, events)
    lift.to_csv(out / "reliability_residual_lift.csv", index=False)
    events.to_csv(out / "reliability_event_diff.csv", index=False)
    scorecard.to_csv(out / "reliability_subset_scorecard.csv", index=False)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1-12")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--output-name", default="reliability_subset_full_2025")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
