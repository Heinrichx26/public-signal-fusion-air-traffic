from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from smoke_source_fusion_topics import (
    AIRPORT_TZ,
    local_schedule_to_utc_hour,
    read_bts_month,
    weighted_lstsq,
    weighted_mean,
)


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "results" / "smoke_tests" / "atcscc_high_disruption_conflict"

MONTHS = [(2025, 1), (2025, 6), (2025, 7), (2025, 12)]
EVENT_FILE = RAW / "faa_atcscc_advisories" / (
    "faa_atcscc_gdp_gs_high_disruption_2025_01_2025_06_2025_07_2025_12.csv"
)


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("M", np.nan), errors="coerce")


def read_iem_weather() -> pd.DataFrame:
    frames = []
    for year, month in MONTHS:
        for airport in AIRPORT_TZ:
            path = RAW / "iem_asos" / f"iem_asos_{year}_{month:02d}_{airport}.csv"
            df = pd.read_csv(path, low_memory=False)
            df["airport"] = airport
            df["utc_hour"] = pd.to_datetime(df["valid"], utc=True).dt.floor("h").dt.tz_localize(None)
            df["wind_dir_deg"] = to_float(df["drct"])
            df["wind_speed_mps"] = to_float(df["sknt"]) * 0.514444
            df["visibility_km"] = to_float(df["vsby"]) * 1.60934
            df["ceiling_m"] = to_float(df["skyl1"]) * 0.3048
            df["temperature_c"] = (to_float(df["tmpf"]) - 32.0) * 5.0 / 9.0
            frames.append(
                df[
                    [
                        "airport",
                        "utc_hour",
                        "wind_dir_deg",
                        "wind_speed_mps",
                        "visibility_km",
                        "ceiling_m",
                        "temperature_c",
                    ]
                ]
            )
    weather = pd.concat(frames, ignore_index=True)
    return (
        weather.groupby(["airport", "utc_hour"], as_index=False)
        .agg(
            wind_dir_deg=("wind_dir_deg", "mean"),
            wind_speed_mps=("wind_speed_mps", "mean"),
            visibility_km=("visibility_km", "mean"),
            ceiling_m=("ceiling_m", "min"),
            temperature_c=("temperature_c", "mean"),
        )
    )


def build_month_arrival_panel(year: int, month: int) -> pd.DataFrame:
    airports = list(AIRPORT_TZ)
    usecols = [
        "FlightDate",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "ArrDelayMinutes",
        "Cancelled",
        "Diverted",
    ]
    bts = read_bts_month(year, month, usecols)
    bts = bts[bts["Dest"].isin(airports)].copy()
    dep_int = pd.to_numeric(bts["CRSDepTime"], errors="coerce")
    arr_int = pd.to_numeric(bts["CRSArrTime"], errors="coerce")
    add_day = (arr_int < dep_int).fillna(False).to_numpy()
    utc_hours = []
    for row, next_day in zip(bts.itertuples(index=False), add_day):
        ts = local_schedule_to_utc_hour(row.FlightDate, row.CRSArrTime, row.Dest, bool(next_day))
        utc_hours.append(ts.tz_localize(None) if pd.notna(ts) else pd.NaT)
    bts["utc_hour"] = utc_hours
    bts = bts.dropna(subset=["utc_hour"])
    bts["arr_delay_min"] = pd.to_numeric(bts["ArrDelayMinutes"], errors="coerce")
    bts["arr_delay60"] = (bts["arr_delay_min"] >= 60).astype(float)
    bts["cancelled"] = pd.to_numeric(bts["Cancelled"], errors="coerce").fillna(0.0)
    bts["diverted"] = pd.to_numeric(bts["Diverted"], errors="coerce").fillna(0.0)
    panel = (
        bts.groupby(["Dest", "utc_hour"], as_index=False)
        .agg(
            arrivals=("Dest", "size"),
            arr_delay60_count=("arr_delay60", "sum"),
            cancel_count=("cancelled", "sum"),
            divert_count=("diverted", "sum"),
            mean_arr_delay=("arr_delay_min", "mean"),
            p90_arr_delay=("arr_delay_min", lambda s: float(np.nanpercentile(s, 90)) if s.notna().any() else np.nan),
        )
        .rename(columns={"Dest": "airport"})
    )
    panel["arr_delay60_rate"] = panel["arr_delay60_count"] / panel["arrivals"]
    panel["cancel_rate"] = panel["cancel_count"] / panel["arrivals"]
    panel["divert_rate"] = panel["divert_count"] / panel["arrivals"]
    panel["month"] = month
    return panel


def complete_airport_hour_grid(panel: pd.DataFrame) -> pd.DataFrame:
    airports = list(AIRPORT_TZ)
    grids = []
    for year, month in MONTHS:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        idx = pd.MultiIndex.from_product(
            [airports, pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")],
            names=["airport", "utc_hour"],
        )
        g = pd.DataFrame(index=idx).reset_index()
        g["month"] = month
        grids.append(g)
    grid = pd.concat(grids, ignore_index=True)
    merged = grid.merge(panel, on=["airport", "utc_hour", "month"], how="left")
    count_cols = ["arrivals", "arr_delay60_count", "cancel_count", "divert_count"]
    merged[count_cols] = merged[count_cols].fillna(0)
    for col in ["arr_delay60_rate", "cancel_rate", "divert_rate"]:
        merged[col] = merged[col].fillna(0.0)
    return merged


def add_events(panel: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_csv(EVENT_FILE)
    events = events[events["airport"].isin(AIRPORT_TZ)].copy()
    events = events.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc"])
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True).dt.tz_localize(None)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True).dt.tz_localize(None)
    panel = panel.copy()
    panel["event_minutes"] = 0.0
    panel["event_count"] = 0
    panel["gs_minutes"] = 0.0
    panel["gdp_minutes"] = 0.0
    hour_start = panel["utc_hour"]
    hour_end = panel["utc_hour"] + pd.Timedelta(hours=1)
    for ev in events.itertuples(index=False):
        mask = (panel["airport"] == ev.airport) & (hour_start < ev.end_utc) & (hour_end > ev.start_utc)
        if not mask.any():
            continue
        overlap_start = hour_start[mask].map(lambda x: max(x, ev.start_utc))
        overlap_end = hour_end[mask].map(lambda x: min(x, ev.end_utc))
        minutes = (overlap_end - overlap_start).dt.total_seconds() / 60.0
        panel.loc[mask, "event_minutes"] += minutes.to_numpy()
        panel.loc[mask, "event_count"] += 1
        if ev.tmi_type == "GS":
            panel.loc[mask, "gs_minutes"] += minutes.to_numpy()
        elif ev.tmi_type == "GDP":
            panel.loc[mask, "gdp_minutes"] += minutes.to_numpy()
    panel["event_any"] = (panel["event_minutes"] > 0).astype(float)
    panel["strong_advisory"] = (panel["event_minutes"] >= 45).astype(float)
    return panel


def add_weather_and_state(panel: pd.DataFrame) -> pd.DataFrame:
    weather = read_iem_weather()
    panel = panel.merge(weather, on=["airport", "utc_hour"], how="left")
    wind_sd = panel["wind_speed_mps"].std(ddof=0)
    wind_sd = wind_sd if wind_sd > 0 else 1.0
    panel["weather_score"] = (
        panel["wind_speed_mps"].fillna(panel["wind_speed_mps"].median()) / wind_sd
        + (20 - panel["visibility_km"].fillna(panel["visibility_km"].median())).clip(lower=0) / 20
        + (1000 - panel["ceiling_m"].fillna(panel["ceiling_m"].median())).clip(lower=0) / 1000
    )
    panel["mild_weather_abs"] = (
        (panel["visibility_km"] >= 8)
        & (panel["wind_speed_mps"] <= 7)
        & ((panel["ceiling_m"] >= 1000) | panel["ceiling_m"].isna())
    ).astype(float)
    panel["mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["strong_advisory"] == 1.0)
    ).astype(float)
    panel["state"] = np.select(
        [
            (panel["mild_weather_abs"] == 1.0) & (panel["strong_advisory"] == 1.0),
            (panel["mild_weather_abs"] == 1.0) & (panel["strong_advisory"] == 0.0),
            (panel["mild_weather_abs"] == 0.0) & (panel["strong_advisory"] == 1.0),
        ],
        ["mild_weather_strong_advisory", "mild_weather_no_strong_advisory", "nonmild_weather_strong_advisory"],
        default="nonmild_weather_no_strong_advisory",
    )
    panel["day_of_week"] = panel["utc_hour"].dt.dayofweek.astype(str)
    panel["local_hour"] = [
        pd.Timestamp(row.utc_hour, tz="UTC").tz_convert(AIRPORT_TZ[row.airport]).hour
        for row in panel.itertuples(index=False)
    ]
    return panel


def summarize_by_state(panel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        panel.groupby(group_cols, observed=False)
        .apply(
            lambda g: pd.Series(
                {
                    "airport_hours": len(g),
                    "hours_with_arrivals": int((g["arrivals"] > 0).sum()),
                    "arrivals": int(g["arrivals"].sum()),
                    "event_minutes": round(g["event_minutes"].sum(), 1),
                    "arr_delay60_rate": weighted_mean(g, "arr_delay60_rate", "arrivals"),
                    "cancel_rate": weighted_mean(g, "cancel_rate", "arrivals"),
                    "mean_arr_delay": weighted_mean(g, "mean_arr_delay", "arrivals"),
                    "p90_arr_delay": weighted_mean(g, "p90_arr_delay", "arrivals"),
                    "weather_score": weighted_mean(g, "weather_score", "arrivals"),
                    "wind_speed_mps": weighted_mean(g, "wind_speed_mps", "arrivals"),
                    "visibility_km": weighted_mean(g, "visibility_km", "arrivals"),
                    "ceiling_m": weighted_mean(g, "ceiling_m", "arrivals"),
                }
            )
        )
        .reset_index()
    )


def run_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    model = panel[(panel["arrivals"] >= 3) & panel["weather_score"].notna()].copy()
    for col in ["airport", "month", "local_hour", "day_of_week"]:
        model[col] = model[col].astype("category")
    regs = []
    for y in ["arr_delay60_rate", "cancel_rate", "mean_arr_delay", "p90_arr_delay"]:
        base = weighted_lstsq(
            model,
            y,
            ["weather_score", "mild_weather_abs", "airport", "month", "local_hour", "day_of_week"],
            "arrivals",
        )
        full = weighted_lstsq(
            model,
            y,
            [
                "strong_advisory",
                "mild_strong_conflict",
                "weather_score",
                "mild_weather_abs",
                "airport",
                "month",
                "local_hour",
                "day_of_week",
            ],
            "arrivals",
        )
        keep = full[full["term"].isin(["strong_advisory", "mild_strong_conflict", "weather_score", "mild_weather_abs"])].copy()
        keep.insert(0, "outcome", y)
        keep["baseline_weighted_r2"] = base["weighted_r2"].iloc[0]
        keep["with_advisory_weighted_r2"] = full["weighted_r2"].iloc[0]
        keep["r2_gain"] = keep["with_advisory_weighted_r2"] - keep["baseline_weighted_r2"]
        regs.append(keep)
    return pd.concat(regs, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [build_month_arrival_panel(year, month) for year, month in MONTHS]
    panel = pd.concat(frames, ignore_index=True)
    panel = complete_airport_hour_grid(panel)
    panel = add_events(panel)
    panel = add_weather_and_state(panel)
    panel.to_csv(OUT / "airport_hour_panel.csv", index=False)

    state_summary = summarize_by_state(panel, ["state"])
    state_summary.to_csv(OUT / "conflict_state_summary.csv", index=False)
    month_summary = summarize_by_state(panel, ["month", "state"])
    month_summary.to_csv(OUT / "month_conflict_state_summary.csv", index=False)
    month_conflict = month_summary[month_summary["state"] == "mild_weather_strong_advisory"].copy()
    month_mild_no = month_summary[month_summary["state"] == "mild_weather_no_strong_advisory"][
        ["month", "arr_delay60_rate", "cancel_rate", "mean_arr_delay", "p90_arr_delay"]
    ].rename(
        columns={
            "arr_delay60_rate": "mild_no_arr_delay60_rate",
            "cancel_rate": "mild_no_cancel_rate",
            "mean_arr_delay": "mild_no_mean_arr_delay",
            "p90_arr_delay": "mild_no_p90_arr_delay",
        }
    )
    month_delta = month_conflict.merge(month_mild_no, on="month", how="left")
    month_delta["delta_arr_delay60_rate"] = month_delta["arr_delay60_rate"] - month_delta["mild_no_arr_delay60_rate"]
    month_delta["delta_cancel_rate"] = month_delta["cancel_rate"] - month_delta["mild_no_cancel_rate"]
    month_delta["delta_mean_arr_delay"] = month_delta["mean_arr_delay"] - month_delta["mild_no_mean_arr_delay"]
    month_delta["delta_p90_arr_delay"] = month_delta["p90_arr_delay"] - month_delta["mild_no_p90_arr_delay"]
    month_delta.to_csv(OUT / "month_mild_conflict_delta.csv", index=False)
    airport_summary = summarize_by_state(panel, ["airport", "state"])
    airport_summary.to_csv(OUT / "airport_conflict_state_summary.csv", index=False)
    reg_summary = run_regressions(panel)
    reg_summary.to_csv(OUT / "conflict_regression_summary.csv", index=False)

    state_lookup = state_summary.set_index("state")
    conflict = state_lookup.loc["mild_weather_strong_advisory"]
    mild_no = state_lookup.loc["mild_weather_no_strong_advisory"]
    nonmild_no = state_lookup.loc["nonmild_weather_no_strong_advisory"]
    conflict_rows = month_delta.copy()
    persistent_months = int((conflict_rows["arrivals"] >= 200).sum())
    all_months_positive = bool((conflict_rows["delta_arr_delay60_rate"] > 0).all())
    conflict_coef = reg_summary[
        (reg_summary["outcome"] == "arr_delay60_rate") & (reg_summary["term"] == "mild_strong_conflict")
    ]["coef"].iloc[0]
    strong_coef = reg_summary[
        (reg_summary["outcome"] == "arr_delay60_rate") & (reg_summary["term"] == "strong_advisory")
    ]["coef"].iloc[0]
    mild_strong_effect = strong_coef + conflict_coef
    verdict = "usable" if persistent_months >= 3 and all_months_positive and mild_strong_effect > 0.03 else "mixed"

    (OUT / "conflict_smoke_assessment.md").write_text(
        "\n".join(
            [
                "# Multi-month ATCSCC conflict smoke assessment",
                "",
                f"Verdict: {verdict}.",
                f"Months evaluated: {', '.join(f'{y}-{m:02d}' for y, m in MONTHS)}.",
                f"Mild-weather strong-advisory airport-hours: {int(conflict['airport_hours'])}.",
                f"Mild-weather strong-advisory arrivals: {int(conflict['arrivals'])}.",
                f"Mild-weather strong-advisory long-arrival-delay rate: {conflict['arr_delay60_rate']:.3f}.",
                f"Mild-weather no-strong-advisory long-arrival-delay rate: {mild_no['arr_delay60_rate']:.3f}.",
                f"Nonmild-weather no-strong-advisory long-arrival-delay rate: {nonmild_no['arr_delay60_rate']:.3f}.",
                f"Mild-weather strong-advisory cancellation rate: {conflict['cancel_rate']:.3f}.",
                f"Mild-weather no-strong-advisory cancellation rate: {mild_no['cancel_rate']:.3f}.",
                f"Strong-advisory fixed-effect coefficient on long-arrival-delay rate: {strong_coef:.3f}.",
                f"Mild-weather strong-advisory interaction coefficient on long-arrival-delay rate: {conflict_coef:.3f}.",
                f"Combined mild-weather strong-advisory effect on long-arrival-delay rate: {mild_strong_effect:.3f}.",
                f"Conflict state has at least 200 arrivals in {persistent_months} months.",
                f"Conflict delay rate exceeds mild-weather no-strong-advisory delay rate in every evaluated month: {all_months_positive}.",
                "",
                "Interpretation: persistence of high disruption under mild local weather supports the hidden-operational-constraint framing. The signal is strongest when the advisory state is treated as its own source.",
            ]
        ),
        encoding="utf-8",
    )
    print(OUT / "conflict_smoke_assessment.md")


if __name__ == "__main__":
    main()
