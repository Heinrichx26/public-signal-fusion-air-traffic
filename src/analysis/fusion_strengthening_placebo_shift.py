from __future__ import annotations

import argparse

import pandas as pd

from fusion_strengthening_common import PANEL_FILE, ROOT_OUT
from smoke_source_fusion_topics import weighted_mean


SHIFTS = [-14, -7, 7, 14]
WINDOWS = ["active", "post_3h"]
OUTCOMES = {
    "long_arrival_delay": "arr_delay60_rate",
    "cancellation": "cancel_rate",
}


def parse_shifts(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def prepare_panel(year_months: list[int] | None = None) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["utc_hour"])
    panel = panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy()
    if year_months:
        panel = panel[panel["month"].isin(year_months)].copy()
    for window in WINDOWS:
        panel[f"{window}_strong"] = (panel[f"{window}_minutes"] >= 45).astype(float)
    return panel.sort_values(["airport", "utc_hour"]).copy()


def shifted_exposure(panel: pd.DataFrame, window: str, shift_days: int) -> pd.Series:
    exposure = panel[["airport", "utc_hour", f"{window}_strong"]].copy()
    exposure["utc_hour"] = exposure["utc_hour"] + pd.Timedelta(days=shift_days)
    exposure = exposure.rename(columns={f"{window}_strong": "shifted_strong"})
    merged = panel[["airport", "utc_hour"]].merge(exposure, on=["airport", "utc_hour"], how="left")
    return merged["shifted_strong"].fillna(0.0)


def state_delta(panel: pd.DataFrame, window: str, exposure_col: str, label: str) -> list[dict]:
    rows = []
    conflict_mask = (panel["mild_weather_abs"] == 1.0) & (panel[exposure_col] == 1.0)
    baseline_mask = (panel["mild_weather_abs"] == 1.0) & (panel[exposure_col] == 0.0)
    conflict = panel[conflict_mask].copy()
    baseline = panel[baseline_mask].copy()
    for outcome, value_col in OUTCOMES.items():
        c_rate = weighted_mean(conflict, value_col, "arrivals")
        b_rate = weighted_mean(baseline, value_col, "arrivals")
        rows.append(
            {
                "label": label,
                "window": window,
                "outcome": outcome,
                "conflict_arrivals": int(conflict["arrivals"].sum()),
                "baseline_arrivals": int(baseline["arrivals"].sum()),
                "conflict_rate": c_rate,
                "baseline_rate": b_rate,
                "delta": c_rate - b_rate,
            }
        )
    return rows


def run(panel: pd.DataFrame, shifts: list[int]) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        rows.extend(state_delta(panel, window, f"{window}_strong", "observed"))
        for shift in shifts:
            col = f"{window}_shift_{shift:+d}d"
            panel[col] = shifted_exposure(panel, window, shift)
            rows.extend(state_delta(panel, window, col, f"shift_{shift:+d}d"))
    return pd.DataFrame(rows)


def write_assessment(result: pd.DataFrame, path) -> None:
    lines = ["# Shifted-advisory placebo assessment", ""]
    for window in WINDOWS:
        for outcome in OUTCOMES:
            use = result[(result["window"] == window) & (result["outcome"] == outcome)]
            observed = float(use[use["label"] == "observed"]["delta"].iloc[0])
            placebo_max = float(use[use["label"] != "observed"]["delta"].max())
            lines.append(
                f"- {window}, {outcome}: observed {observed:+.3f}; "
                f"maximum shifted placebo {placebo_max:+.3f}."
            )
    lines.append("")
    lines.append("Assessment: observed deltas should exceed shifted-placebo deltas before this enters the reported analysis.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts", default="-14,-7,7,14")
    parser.add_argument("--output-name", default="placebo_shift_full_2025")
    args = parser.parse_args()

    out_dir = ROOT_OUT / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(prepare_panel(), parse_shifts(args.shifts))
    result.to_csv(out_dir / "placebo_shift_test.csv", index=False)
    write_assessment(result, out_dir / "placebo_shift_assessment.md")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
