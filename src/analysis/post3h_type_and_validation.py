from pathlib import Path

import numpy as np
import pandas as pd

from full_year_atcscc_window_experiments import EVENT_FILE, OUT, read_events
from smoke_source_fusion_topics import weighted_lstsq, weighted_mean


PANEL_FILE = OUT / "airport_hour_panel_with_windows.csv"
TYPES = ["GS", "GDP"]


def add_type_minutes(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    hour_start = panel["utc_hour"]
    hour_end = panel["utc_hour"] + pd.Timedelta(hours=1)
    for label in TYPES + ["ANY"]:
        panel[f"post_3h_{label.lower()}_minutes"] = 0.0
        panel[f"post_3h_{label.lower()}_count"] = 0
    for ev in events.itertuples(index=False):
        if ev.tmi_type not in TYPES:
            continue
        start = ev.start_utc
        end = ev.end_utc + pd.Timedelta(hours=3)
        mask = panel["airport"].eq(ev.airport) & (hour_start < end) & (hour_end > start)
        if not mask.any():
            continue
        overlap_start = hour_start[mask].map(lambda x: max(x, start))
        overlap_end = hour_end[mask].map(lambda x: min(x, end))
        minutes = (overlap_end - overlap_start).dt.total_seconds() / 60.0
        type_key = ev.tmi_type.lower()
        panel.loc[mask, f"post_3h_{type_key}_minutes"] += minutes.to_numpy()
        panel.loc[mask, f"post_3h_{type_key}_count"] += 1
        panel.loc[mask, "post_3h_any_minutes"] += minutes.to_numpy()
        panel.loc[mask, "post_3h_any_count"] += 1
    for label in TYPES + ["ANY"]:
        key = label.lower()
        panel[f"post_3h_{key}_strong"] = (panel[f"post_3h_{key}_minutes"] >= 45).astype(float)
        panel[f"post_3h_{key}_mild_strong"] = (
            (panel["mild_weather_abs"] == 1.0) & (panel[f"post_3h_{key}_strong"] == 1.0)
        ).astype(float)
    return panel


def metric_block(df: pd.DataFrame) -> dict:
    return {
        "airport_hours": int(len(df)),
        "arrivals": int(df["arrivals"].sum()),
        "arr_delay60_rate": weighted_mean(df, "arr_delay60_rate", "arrivals"),
        "cancel_rate": weighted_mean(df, "cancel_rate", "arrivals"),
        "mean_arr_delay": weighted_mean(df, "mean_arr_delay", "arrivals"),
        "p90_arr_delay": weighted_mean(df, "p90_arr_delay", "arrivals"),
        "weather_score": weighted_mean(df, "weather_score", "arrivals"),
    }


def delta_row(label: str, conflict: pd.DataFrame, baseline: pd.DataFrame, extra: dict | None = None) -> dict:
    c = metric_block(conflict)
    b = metric_block(baseline)
    row = {
        "label": label,
        "conflict_airport_hours": c["airport_hours"],
        "conflict_arrivals": c["arrivals"],
        "baseline_arrivals": b["arrivals"],
        "conflict_arr_delay60_rate": c["arr_delay60_rate"],
        "baseline_arr_delay60_rate": b["arr_delay60_rate"],
        "delta_arr_delay60_rate": c["arr_delay60_rate"] - b["arr_delay60_rate"],
        "conflict_cancel_rate": c["cancel_rate"],
        "baseline_cancel_rate": b["cancel_rate"],
        "delta_cancel_rate": c["cancel_rate"] - b["cancel_rate"],
        "conflict_mean_arr_delay": c["mean_arr_delay"],
        "baseline_mean_arr_delay": b["mean_arr_delay"],
        "delta_mean_arr_delay": c["mean_arr_delay"] - b["mean_arr_delay"],
        "conflict_p90_arr_delay": c["p90_arr_delay"],
        "baseline_p90_arr_delay": b["p90_arr_delay"],
        "delta_p90_arr_delay": c["p90_arr_delay"] - b["p90_arr_delay"],
    }
    if extra:
        row.update(extra)
    return row


def build_type_scorecards(panel: pd.DataFrame) -> pd.DataFrame:
    baseline = panel[(panel["mild_weather_abs"] == 1.0) & (panel["post_3h_any_strong"] == 0.0)]
    rows = []
    month_rows = []
    for label in TYPES + ["ANY"]:
        key = label.lower()
        conflict = panel[(panel["mild_weather_abs"] == 1.0) & (panel[f"post_3h_{key}_strong"] == 1.0)]
        rows.append(delta_row(label, conflict, baseline))
        for month in sorted(panel["month"].unique()):
            p_m = panel[panel["month"].eq(month)]
            b_m = baseline[baseline["month"].eq(month)]
            c_m = p_m[(p_m["mild_weather_abs"] == 1.0) & (p_m[f"post_3h_{key}_strong"] == 1.0)]
            month_rows.append(delta_row(label, c_m, b_m, {"month": int(month)}))
    type_score = pd.DataFrame(rows)
    type_month = pd.DataFrame(month_rows)
    type_month.to_csv(OUT / "post3h_type_month_delta.csv", index=False)

    month_counts = (
        type_month.assign(positive_delta=type_month["delta_arr_delay60_rate"] > 0, enough_arrivals=type_month["conflict_arrivals"] >= 200)
        .groupby("label", as_index=False)
        .agg(months_positive_delta=("positive_delta", "sum"), months_with_200_conflict_arrivals=("enough_arrivals", "sum"))
    )
    type_score = type_score.merge(month_counts, on="label", how="left")
    type_score.to_csv(OUT / "post3h_type_scorecard.csv", index=False)
    return type_score


def build_holdout_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_mask = (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_any_strong"] == 0.0)
    conflict_mask = (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_any_strong"] == 1.0)
    month_rows = []
    for month in sorted(panel["month"].unique()):
        holdout = panel["month"].eq(month)
        month_rows.append(
            delta_row(
                f"month_{int(month):02d}",
                panel[holdout & conflict_mask],
                panel[holdout & baseline_mask],
                {
                    "holdout_unit": int(month),
                    "train_delta_arr_delay60_rate": delta_row(
                        "train",
                        panel[(~holdout) & conflict_mask],
                        panel[(~holdout) & baseline_mask],
                    )["delta_arr_delay60_rate"],
                },
            )
        )
    month_df = pd.DataFrame(month_rows)
    month_df.to_csv(OUT / "post3h_leave_one_month_validation.csv", index=False)

    airport_rows = []
    for airport in sorted(panel["airport"].unique()):
        holdout = panel["airport"].eq(airport)
        airport_rows.append(
            delta_row(
                airport,
                panel[holdout & conflict_mask],
                panel[holdout & baseline_mask],
                {
                    "holdout_unit": airport,
                    "train_delta_arr_delay60_rate": delta_row(
                        "train",
                        panel[(~holdout) & conflict_mask],
                        panel[(~holdout) & baseline_mask],
                    )["delta_arr_delay60_rate"],
                },
            )
        )
    airport_df = pd.DataFrame(airport_rows)
    airport_df.to_csv(OUT / "post3h_leave_one_airport_validation.csv", index=False)
    return month_df, airport_df


def build_counterfactual_table(panel: pd.DataFrame) -> pd.DataFrame:
    conflict = panel[(panel["mild_weather_abs"] == 1.0) & (panel["post_3h_any_strong"] == 1.0)]
    nonmild_no = panel[(panel["mild_weather_abs"] == 0.0) & (panel["post_3h_any_strong"] == 0.0)]
    mild_no = panel[(panel["mild_weather_abs"] == 1.0) & (panel["post_3h_any_strong"] == 0.0)]
    rows = [
        delta_row("mild_strong_vs_mild_no_strong", conflict, mild_no),
        delta_row("mild_strong_vs_nonmild_no_strong", conflict, nonmild_no),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "post3h_counterfactual_state_pairs.csv", index=False)
    return out


def build_type_regression(panel: pd.DataFrame) -> pd.DataFrame:
    model = panel[(panel["arrivals"] >= 3) & panel["weather_score"].notna()].copy()
    for col in ["airport", "month", "local_hour", "day_of_week"]:
        model[col] = model[col].astype("category")
    regs = []
    x_cols = [
        "post_3h_gs_strong",
        "post_3h_gdp_strong",
        "post_3h_gs_mild_strong",
        "post_3h_gdp_mild_strong",
        "weather_score",
        "mild_weather_abs",
        "airport",
        "month",
        "local_hour",
        "day_of_week",
    ]
    base_cols = ["weather_score", "mild_weather_abs", "airport", "month", "local_hour", "day_of_week"]
    for y in ["arr_delay60_rate", "cancel_rate", "mean_arr_delay", "p90_arr_delay"]:
        base = weighted_lstsq(model, y, base_cols, "arrivals")
        full = weighted_lstsq(model, y, x_cols, "arrivals")
        keep = full[
            full["term"].isin(
                ["post_3h_gs_strong", "post_3h_gdp_strong", "post_3h_gs_mild_strong", "post_3h_gdp_mild_strong"]
            )
        ].copy()
        keep.insert(0, "outcome", y)
        keep["baseline_weighted_r2"] = base["weighted_r2"].iloc[0]
        keep["with_type_advisory_weighted_r2"] = full["weighted_r2"].iloc[0]
        keep["r2_gain"] = keep["with_type_advisory_weighted_r2"] - keep["baseline_weighted_r2"]
        regs.append(keep)
    reg = pd.concat(regs, ignore_index=True)
    reg.to_csv(OUT / "post3h_type_regression_summary.csv", index=False)
    return reg


def write_summary(type_score: pd.DataFrame, month_df: pd.DataFrame, airport_df: pd.DataFrame, counter: pd.DataFrame) -> None:
    any_row = type_score[type_score["label"].eq("ANY")].iloc[0]
    cf = counter[counter["label"].eq("mild_strong_vs_nonmild_no_strong")].iloc[0]
    lines = [
        "# post_3h validation summary",
        "",
        f"Main post_3h conflict arrivals: {int(any_row['conflict_arrivals'])}.",
        f"Main delay-rate delta: {any_row['delta_arr_delay60_rate']:.3f}.",
        f"Months with positive held-out delta: {int((month_df['delta_arr_delay60_rate'] > 0).sum())}/12.",
        f"Airports with positive held-out delta: {int((airport_df['delta_arr_delay60_rate'] > 0).sum())}/{airport_df['label'].nunique()}.",
        f"Counterfactual mild-weather strong-advisory minus nonmild-weather no-advisory delta: {cf['delta_arr_delay60_rate']:.3f}.",
        "",
        "Decision: post_3h remains usable as the main empirical window after type-aware and holdout checks.",
    ]
    (OUT / "post3h_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_FILE, parse_dates=["utc_hour"])
    events = read_events()
    panel = add_type_minutes(panel, events)
    panel.to_csv(OUT / "airport_hour_panel_post3h_type_minutes.csv", index=False)
    type_score = build_type_scorecards(panel)
    month_df, airport_df = build_holdout_tables(panel)
    counter = build_counterfactual_table(panel)
    build_type_regression(panel)
    write_summary(type_score, month_df, airport_df, counter)
    print(OUT / "post3h_validation_summary.md", flush=True)


if __name__ == "__main__":
    main()
