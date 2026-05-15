from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
DEFAULT_RESPONSE = ROOT / "event_response_full_2025_30airports" / "event_response_matched_windows.csv"
PLACEBO = ROOT / "placebo_shift_full_2025" / "placebo_shift_test.csv"
WINDOWS = ["pre_3h", "pre_1h", "active", "post_1h", "post_3h", "clean_lag_6h"]
OUTCOMES = {"long_arrival_delay": "delay_diff", "cancellation": "cancel_diff"}


def load_response(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data[data["window"].isin(WINDOWS)].copy()


def event_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["event_id", "airport", "month", "tmi_type"]
    for key, group in data.groupby(keys):
        values = group.set_index("window")
        if any(w not in values.index for w in WINDOWS):
            continue
        weight = float((group["treated_arrivals"] + group["control_arrivals"]).sum())
        for outcome, col in OUTCOMES.items():
            pre = 0.5 * (float(values.loc["pre_3h", col]) + float(values.loc["pre_1h", col]))
            active = float(values.loc["active", col])
            post1 = float(values.loc["post_1h", col])
            post3 = float(values.loc["post_3h", col])
            lag6 = float(values.loc["clean_lag_6h", col])
            peak = active - pre
            tail = max(post1 - pre, 0.0) + 2 * max(post3 - pre, 0.0) + 3 * max(lag6 - pre, 0.0)
            event_aohi = tail / peak if peak > 1e-6 else np.nan
            rows.append(
                {
                    "event_id": key[0],
                    "airport": key[1],
                    "month": key[2],
                    "tmi_type": key[3],
                    "outcome": outcome,
                    "pre_diff": pre,
                    "active_diff": active,
                    "post_1h_diff": post1,
                    "post_3h_diff": post3,
                    "clean_lag_6h_diff": lag6,
                    "peak": peak,
                    "tail_area": tail,
                    "event_aohi": event_aohi,
                    "event_weight": weight,
                }
            )
    return pd.DataFrame(rows)


def bootstrap(metrics: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for outcome, group in metrics.groupby("outcome"):
        values = group[["peak", "tail_area", "event_aohi", "event_weight"]].to_numpy(float)
        n = len(values)
        draws = []
        for _ in range(reps):
            sample = values[rng.integers(0, n, n)]
            draws.append(
                {
                    "mean_peak": float(sample[:, 0].mean()),
                    "mean_tail_area": float(sample[:, 1].mean()),
                    "aggregate_aohi": float(sample[:, 1].mean() / max(sample[:, 0].mean(), 1e-6)),
                    "median_event_aohi": float(np.nanmedian(sample[:, 2])),
                    "weighted_aggregate_aohi": float(
                        np.average(sample[:, 1], weights=np.maximum(sample[:, 3], 1.0))
                        / max(np.average(sample[:, 0], weights=np.maximum(sample[:, 3], 1.0)), 1e-6)
                    ),
                }
            )
        draw = pd.DataFrame(draws)
        base = {
            "outcome": outcome,
            "events": n,
            "mean_peak": float(group["peak"].mean()),
            "mean_tail_area": float(group["tail_area"].mean()),
            "aggregate_aohi": float(group["tail_area"].mean() / max(group["peak"].mean(), 1e-6)),
            "median_event_aohi": float(group["event_aohi"].median()),
            "weighted_aggregate_aohi": float(
                np.average(group["tail_area"], weights=np.maximum(group["event_weight"], 1.0))
                / max(np.average(group["peak"], weights=np.maximum(group["event_weight"], 1.0)), 1e-6)
            ),
            "positive_peak_share": float((group["peak"] > 0).mean()),
            "positive_tail_share": float((group["tail_area"] > 0).mean()),
        }
        for col in ["mean_peak", "mean_tail_area", "aggregate_aohi", "median_event_aohi", "weighted_aggregate_aohi"]:
            base[f"{col}_ci_low"] = float(draw[col].quantile(0.025))
            base[f"{col}_ci_high"] = float(draw[col].quantile(0.975))
        rows.append(base)
    return pd.DataFrame(rows)


def stability(metrics: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (outcome, value), group in metrics.groupby(["outcome", group_col]):
        rows.append(
            {
                "outcome": outcome,
                "grouping": group_col,
                "group": value,
                "events": len(group),
                "mean_peak": float(group["peak"].mean()),
                "mean_tail_area": float(group["tail_area"].mean()),
                "aggregate_aohi": float(group["tail_area"].mean() / max(group["peak"].mean(), 1e-6)),
                "positive_peak_share": float((group["peak"] > 0).mean()),
                "positive_tail_share": float((group["tail_area"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def placebo_summary() -> pd.DataFrame:
    if not PLACEBO.exists():
        return pd.DataFrame()
    data = pd.read_csv(PLACEBO)
    rows = []
    for outcome in data["outcome"].unique():
        use = data[(data["outcome"] == outcome) & (data["window"].isin(["active", "post_3h"]))]
        observed = use[use["label"] == "observed"].set_index("window")["delta"]
        shifted = use[use["label"] != "observed"]
        rows.append(
            {
                "outcome": outcome,
                "observed_active": float(observed.get("active", np.nan)),
                "observed_post_3h": float(observed.get("post_3h", np.nan)),
                "max_shifted_active": float(shifted[shifted["window"] == "active"]["delta"].max()),
                "max_shifted_post_3h": float(shifted[shifted["window"] == "post_3h"]["delta"].max()),
            }
        )
    return pd.DataFrame(rows)


def write_assessment(summary: pd.DataFrame, placebo: pd.DataFrame, path: Path) -> None:
    lines = ["# Advisory-outcome hysteresis assessment", ""]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.outcome}: aggregate AOHI {row.aggregate_aohi:.2f} "
            f"[{row.aggregate_aohi_ci_low:.2f}, {row.aggregate_aohi_ci_high:.2f}], "
            f"tail area {row.mean_tail_area:.3f}."
        )
    if not placebo.empty:
        delay = placebo[placebo["outcome"].eq("long_arrival_delay")].iloc[0]
        lines.append(
            f"- Shifted placebo: observed active/post {delay.observed_active:+.3f}/{delay.observed_post_3h:+.3f}; "
            f"maximum shifted active/post {delay.max_shifted_active:+.3f}/{delay.max_shifted_post_3h:+.3f}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-file", default=str(DEFAULT_RESPONSE))
    parser.add_argument("--output-name", default="hysteresis_full_2025_30airports")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    out = ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    metrics = event_metrics(load_response(Path(args.response_file)))
    summary = bootstrap(metrics, args.reps, args.seed)
    stable = pd.concat([stability(metrics, "month"), stability(metrics, "airport")], ignore_index=True)
    placebo = placebo_summary()
    metrics.to_csv(out / "aohi_event_metrics.csv", index=False)
    summary.to_csv(out / "aohi_bootstrap_summary.csv", index=False)
    stable.to_csv(out / "aohi_stability.csv", index=False)
    placebo.to_csv(out / "aohi_placebo_context.csv", index=False)
    write_assessment(summary, placebo, out / "aohi_assessment.md")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
