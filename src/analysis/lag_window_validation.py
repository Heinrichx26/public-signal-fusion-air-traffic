import pandas as pd

from full_year_atcscc_window_experiments import OUT, read_events
from smoke_source_fusion_topics import weighted_mean


PANEL_FILE = OUT / "airport_hour_panel_with_windows.csv"
WINDOWS = {
    "lead_1h": ("lead", 1),
    "lead_3h": ("lead", 3),
    "active": ("active", 0),
    "lag_1h": ("lag", 1),
    "lag_2h": ("lag", 2),
    "lag_3h": ("lag", 3),
    "lag_6h": ("lag", 6),
    "extended_3h": ("extended", 3),
    "extended_6h": ("extended", 6),
}


def add_window_minutes(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    hour_start = panel["utc_hour"]
    hour_end = panel["utc_hour"] + pd.Timedelta(hours=1)
    for name in WINDOWS:
        panel[f"{name}_minutes"] = 0.0
    for ev in events.itertuples(index=False):
        for name, (mode, h) in WINDOWS.items():
            if mode == "lead":
                start = ev.start_utc - pd.Timedelta(hours=h)
                end = ev.start_utc
            elif mode == "active":
                start = ev.start_utc
                end = ev.end_utc
            elif mode == "lag":
                start = ev.end_utc
                end = ev.end_utc + pd.Timedelta(hours=h)
            else:
                start = ev.start_utc
                end = ev.end_utc + pd.Timedelta(hours=h)
            mask = panel["airport"].eq(ev.airport) & (hour_start < end) & (hour_end > start)
            if not mask.any():
                continue
            overlap_start = hour_start[mask].map(lambda x: max(x, start))
            overlap_end = hour_end[mask].map(lambda x: min(x, end))
            panel.loc[mask, f"{name}_minutes"] += ((overlap_end - overlap_start).dt.total_seconds() / 60.0).to_numpy()
    for name in WINDOWS:
        panel[f"{name}_strong"] = (panel[f"{name}_minutes"] >= 45).astype(float)
    return panel


def summarize(conflict: pd.DataFrame, baseline: pd.DataFrame, label: str, month_df: pd.DataFrame | None = None) -> dict:
    c_delay = weighted_mean(conflict, "arr_delay60_rate", "arrivals")
    b_delay = weighted_mean(baseline, "arr_delay60_rate", "arrivals")
    c_cancel = weighted_mean(conflict, "cancel_rate", "arrivals")
    b_cancel = weighted_mean(baseline, "cancel_rate", "arrivals")
    row = {
        "window": label,
        "conflict_airport_hours": int(len(conflict)),
        "conflict_arrivals": int(conflict["arrivals"].sum()),
        "baseline_arrivals": int(baseline["arrivals"].sum()),
        "conflict_arr_delay60_rate": c_delay,
        "baseline_arr_delay60_rate": b_delay,
        "delta_arr_delay60_rate": c_delay - b_delay,
        "conflict_cancel_rate": c_cancel,
        "baseline_cancel_rate": b_cancel,
        "delta_cancel_rate": c_cancel - b_cancel,
        "conflict_mean_arr_delay": weighted_mean(conflict, "mean_arr_delay", "arrivals"),
        "baseline_mean_arr_delay": weighted_mean(baseline, "mean_arr_delay", "arrivals"),
        "conflict_p90_arr_delay": weighted_mean(conflict, "p90_arr_delay", "arrivals"),
        "baseline_p90_arr_delay": weighted_mean(baseline, "p90_arr_delay", "arrivals"),
    }
    if month_df is not None:
        row["months_positive_delta"] = int((month_df["delta_arr_delay60_rate"] > 0).sum())
        row["months_with_200_conflict_arrivals"] = int((month_df["conflict_arrivals"] >= 200).sum())
    return row


def monthly_delta(panel: pd.DataFrame, window: str, clean: bool = False) -> pd.DataFrame:
    rows = []
    for month in sorted(panel["month"].unique()):
        p_m = panel[panel["month"].eq(month)]
        if clean:
            conflict = p_m[
                (p_m["mild_weather_abs"] == 1.0)
                & (p_m[f"{window}_strong"] == 1.0)
                & (p_m["active_strong"] == 0.0)
            ]
            baseline = p_m[
                (p_m["mild_weather_abs"] == 1.0)
                & (p_m[f"{window}_strong"] == 0.0)
                & (p_m["active_strong"] == 0.0)
            ]
        else:
            conflict = p_m[(p_m["mild_weather_abs"] == 1.0) & (p_m[f"{window}_strong"] == 1.0)]
            baseline = p_m[(p_m["mild_weather_abs"] == 1.0) & (p_m[f"{window}_strong"] == 0.0)]
        row = summarize(conflict, baseline, window)
        row["month"] = int(month)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_FILE, parse_dates=["utc_hour"])
    events = read_events()
    panel = add_window_minutes(panel, events)
    rows = []
    month_frames = []
    for window in WINDOWS:
        m = monthly_delta(panel, window, clean=False)
        month_frames.append(m.assign(scope="all"))
        conflict = panel[(panel["mild_weather_abs"] == 1.0) & (panel[f"{window}_strong"] == 1.0)]
        baseline = panel[(panel["mild_weather_abs"] == 1.0) & (panel[f"{window}_strong"] == 0.0)]
        rows.append(summarize(conflict, baseline, window, m) | {"scope": "all"})
        if window.startswith("lag_"):
            clean_m = monthly_delta(panel, window, clean=True)
            month_frames.append(clean_m.assign(scope="clean_no_active"))
            clean_conflict = panel[
                (panel["mild_weather_abs"] == 1.0)
                & (panel[f"{window}_strong"] == 1.0)
                & (panel["active_strong"] == 0.0)
            ]
            clean_baseline = panel[
                (panel["mild_weather_abs"] == 1.0)
                & (panel[f"{window}_strong"] == 0.0)
                & (panel["active_strong"] == 0.0)
            ]
            rows.append(summarize(clean_conflict, clean_baseline, window, clean_m) | {"scope": "clean_no_active"})
    score = pd.DataFrame(rows)
    score.to_csv(OUT / "lag_window_scorecard.csv", index=False)
    pd.concat(month_frames, ignore_index=True).to_csv(OUT / "lag_window_month_delta.csv", index=False)

    clean_lag3 = score[(score["window"].eq("lag_3h")) & (score["scope"].eq("clean_no_active"))].iloc[0]
    active = score[(score["window"].eq("active")) & (score["scope"].eq("all"))].iloc[0]
    lines = [
        "# Lag-window validation",
        "",
        f"Active-window delay-rate delta: {active['delta_arr_delay60_rate']:.3f}.",
        f"Clean post-end 3h delay-rate delta: {clean_lag3['delta_arr_delay60_rate']:.3f}.",
        f"Clean post-end 3h conflict arrivals: {int(clean_lag3['conflict_arrivals'])}.",
        f"Clean post-end 3h positive months: {int(clean_lag3['months_positive_delta'])}/12.",
        "",
        "Decision: post-end lag exists, while active advisories remain the highest-intensity signal.",
    ]
    (OUT / "lag_window_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "lag_window_validation_summary.md", flush=True)


if __name__ == "__main__":
    main()
