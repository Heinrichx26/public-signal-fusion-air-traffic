from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_strengthening_event_level import load_events, window_bounds
from fusion_strengthening_residual_score import summary_table
from smoke_source_fusion_topics import weighted_mean


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
SCORES = ROOT / "residual_score_full_2025" / "residual_state_scores.csv"
NEIGHBOR = (
    PROJECT
    / "results"
    / "experiments"
    / "supplemental_validation"
    / "neighbor_weather"
    / "base10_2025_panel_with_neighbor_weather.csv"
)
EVENT_TABLE = ROOT / "event_level_full_2025_base10" / "event_level_matched_windows.csv"


def add_quality(scores: pd.DataFrame) -> pd.DataFrame:
    neighbor = pd.read_csv(
        NEIGHBOR,
        parse_dates=["utc_hour"],
        usecols=["airport", "utc_hour", "neighbor_stations", "mild_agree"],
    )
    out = scores.merge(neighbor, on=["airport", "utc_hour"], how="left")
    out["q_weather"] = np.where(
        (out["neighbor_stations"] >= 2) & (out["mild_agree"].eq(1.0)),
        1.0,
        np.where(out["neighbor_stations"] >= 2, 0.75, 0.50),
    )
    out["q_advisory"] = 1.0
    out["q_demand"] = (out["scheduled_arrivals"].fillna(0) > 0).astype(float)
    out["source_reliability_q"] = out["q_weather"] * out["q_advisory"] * out["q_demand"]
    out["reliability_adjusted_score"] = out["source_reliability_q"] * out["residual_score"]
    return out


def deciles_for(scores: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows = []
    for target, group in scores.groupby("target"):
        use = group[group["arrivals"] > 0].copy()
        use["score_decile"] = pd.qcut(use[score_col], 10, labels=False, duplicates="drop") + 1
        for decile, g in use.groupby("score_decile"):
            rows.append(
                {
                    "ranking_score": label,
                    "target": target,
                    "score_decile": int(decile),
                    "airport_hours": len(g),
                    "arrivals": int(g["arrivals"].sum()),
                    "observed_rate": weighted_mean(g, "obs_rate", "arrivals"),
                    "observed_residual": weighted_mean(g, "obs_residual", "arrivals"),
                    "mean_r": weighted_mean(g, "residual_score", "arrivals"),
                    "mean_q": weighted_mean(g, "source_reliability_q", "arrivals"),
                    "mean_qr": weighted_mean(g, "reliability_adjusted_score", "arrivals"),
                }
            )
    return pd.DataFrame(rows)


def score_summaries(deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in deciles.groupby("ranking_score"):
        summary = summary_table(group.rename(columns={"score_decile": "score_decile"}))
        summary["ranking_score"] = label
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)[
        [
            "ranking_score",
            "target",
            "bottom_decile_rate",
            "top_decile_rate",
            "top_bottom_ratio",
            "positive_residual_steps",
            "total_adjacent_steps",
            "spearman_decile_residual",
        ]
    ]


def event_scores(scores: pd.DataFrame) -> pd.DataFrame:
    events = load_events("10", list(range(1, 13)))
    score_panel = scores[scores["target"].isin(["long_arrival_delay", "cancellation"])].copy()
    rows = []
    for idx, event in enumerate(events.itertuples(index=False), start=1):
        ap = score_panel[score_panel["airport"].eq(event.airport)]
        if ap.empty:
            continue
        for window in ["active", "post_3h"]:
            start, end = window_bounds(event, window)
            hours = ap[
                (ap["utc_hour"] < end)
                & (ap["utc_hour"] + pd.Timedelta(hours=1) > start)
                & ap["mild_weather_abs"].eq(1.0)
            ].copy()
            if hours.empty or hours["arrivals"].sum() <= 0:
                continue
            for target, g in hours.groupby("target"):
                rows.append(
                    {
                        "event_id": idx,
                        "window": window,
                        "target": target,
                        "event_r": weighted_mean(g, "residual_score", "arrivals"),
                        "event_qr": weighted_mean(g, "reliability_adjusted_score", "arrivals"),
                    }
                )
    return pd.DataFrame(rows)


def weighted_event_diff(group: pd.DataFrame, outcome_col: str) -> float:
    weights = group["treated_arrivals"] + group["control_arrivals"]
    return float(np.average(group[outcome_col], weights=weights))


def event_diff_table(scores: pd.DataFrame) -> pd.DataFrame:
    matched = pd.read_csv(EVENT_TABLE)
    event_score = event_scores(scores)
    merged = matched.merge(event_score, on=["event_id", "window"], how="inner")
    rows = []
    settings = [("R", "event_r"), ("QxR", "event_qr")]
    outcome = {"long_arrival_delay": "delay_diff", "cancellation": "cancel_diff"}
    for (label, score_col), target in [(s, t) for s in settings for t in outcome]:
        use = merged[merged["target"].eq(target)].copy()
        for window, wgroup in use.groupby("window"):
            threshold = wgroup[score_col].quantile(0.90)
            top = wgroup[wgroup[score_col] >= threshold].copy()
            rows.append(
                {
                    "ranking_score": label,
                    "target": target,
                    "window": window,
                    "events_top_decile": len(top),
                    "event_diff_top_decile": weighted_event_diff(top, outcome[target]),
                    "event_score_threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def run(output_name: str) -> None:
    out = ROOT / output_name
    out.mkdir(parents=True, exist_ok=True)
    scores = add_quality(pd.read_csv(SCORES, parse_dates=["utc_hour"]))
    deciles = pd.concat(
        [
            deciles_for(scores, "residual_score", "R"),
            deciles_for(scores, "reliability_adjusted_score", "QxR"),
        ],
        ignore_index=True,
    )
    summary = score_summaries(deciles)
    events = event_diff_table(scores)
    event_wide = events.pivot_table(
        index=["ranking_score", "target"],
        columns="window",
        values=["events_top_decile", "event_diff_top_decile"],
        aggfunc="first",
    ).reset_index()
    event_wide.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in event_wide.columns]
    scorecard = summary.merge(event_wide, on=["ranking_score", "target"], how="left")
    deciles.to_csv(out / "reliability_adjusted_score_deciles.csv", index=False)
    summary.to_csv(out / "reliability_adjusted_score_summary.csv", index=False)
    events.to_csv(out / "reliability_adjusted_event_diff.csv", index=False)
    scorecard.to_csv(out / "reliability_adjusted_scorecard.csv", index=False)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="reliability_adjusted_score_full_2025")
    run(parser.parse_args().output_name)


if __name__ == "__main__":
    main()
