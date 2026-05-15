from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_strengthening_common import MAIN, ROOT_OUT


TARGETS = {
    "long_arrival_delay": "arr_delay60_count",
    "cancellation": "cancel_count",
}
WINDOWS = {
    "active": "active_minutes",
    "post_3h": "post_3h_minutes",
}


def parse_months(text: str) -> list[int]:
    months: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            months.extend(range(start, end + 1))
        else:
            months.append(int(part))
    return sorted({m for m in months if 1 <= m <= 12})


def load_panel(months: list[int]) -> pd.DataFrame:
    panel = pd.read_csv(MAIN / "airport_hour_panel_with_windows.csv", parse_dates=["utc_hour"])
    panel = panel[(panel["month"].isin(months)) & (panel["arrivals"] > 0)].copy()
    panel = panel[panel["weather_score"].notna()].copy()
    panel["airport_day"] = panel["airport"].astype(str) + "_" + panel["utc_hour"].dt.strftime("%Y-%m-%d")
    return panel


def block_arrays(panel: pd.DataFrame, target_col: str, window_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    use = panel[panel["mild_weather_abs"] == 1.0].copy()
    use["state"] = np.where(use[window_col] >= 45, "conflict", "baseline")
    grouped = (
        use.groupby(["airport_day", "state"], as_index=False)
        .agg(success=(target_col, "sum"), arrivals=("arrivals", "sum"))
    )
    wide = grouped.pivot(index="airport_day", columns="state", values=["success", "arrivals"]).fillna(0.0)
    wide.columns = [f"{metric}_{state}" for metric, state in wide.columns]
    for col in ["success_conflict", "arrivals_conflict", "success_baseline", "arrivals_baseline"]:
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide.reset_index()
    observed = diff_from_sums(wide)
    return wide, observed


def diff_from_sums(frame: pd.DataFrame) -> dict[str, float]:
    c_success = frame["success_conflict"].sum()
    c_arrivals = frame["arrivals_conflict"].sum()
    b_success = frame["success_baseline"].sum()
    b_arrivals = frame["arrivals_baseline"].sum()
    c_rate = c_success / c_arrivals if c_arrivals else np.nan
    b_rate = b_success / b_arrivals if b_arrivals else np.nan
    return {
        "conflict_arrivals": c_arrivals,
        "baseline_arrivals": b_arrivals,
        "conflict_rate": c_rate,
        "baseline_rate": b_rate,
        "diff": c_rate - b_rate,
    }


def bootstrap_ci(wide: pd.DataFrame, reps: int, seed: int) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(wide)
    cols = ["success_conflict", "arrivals_conflict", "success_baseline", "arrivals_baseline"]
    values = wide[cols].to_numpy(float)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = values[rng.integers(0, n, size=n)]
        c_rate = sample[:, 0].sum() / sample[:, 1].sum()
        b_rate = sample[:, 2].sum() / sample[:, 3].sum()
        draws[i] = c_rate - b_rate
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), draws


def leave_one_airport(panel: pd.DataFrame, target_col: str, window_col: str) -> dict[str, float]:
    diffs = []
    for airport in sorted(panel["airport"].unique()):
        wide, obs = block_arrays(panel[panel["airport"] != airport], target_col, window_col)
        if obs["conflict_arrivals"] > 0 and obs["baseline_arrivals"] > 0:
            diffs.append(obs["diff"])
    return {"loo_min": float(np.min(diffs)), "loo_max": float(np.max(diffs)), "loo_positive": int(np.sum(np.array(diffs) > 0))}


def run(months: list[int], reps: int, seed: int, output_name: str) -> None:
    out_dir = ROOT_OUT / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(months)
    rows = []
    for target, target_col in TARGETS.items():
        for window, window_col in WINDOWS.items():
            wide, obs = block_arrays(panel, target_col, window_col)
            low, high, draws = bootstrap_ci(wide, reps, seed + len(rows))
            loo = leave_one_airport(panel, target_col, window_col)
            rows.append(
                {
                    "target": target,
                    "window": window,
                    "observed_diff": obs["diff"],
                    "ci_low": low,
                    "ci_high": high,
                    "conflict_arrivals": int(obs["conflict_arrivals"]),
                    "baseline_arrivals": int(obs["baseline_arrivals"]),
                    "blocks": len(wide),
                    "bootstrap_reps": reps,
                    "loo_min": loo["loo_min"],
                    "loo_max": loo["loo_max"],
                    "loo_positive_airports": loo["loo_positive"],
                }
            )
            pd.DataFrame({"bootstrap_diff": draws}).to_csv(out_dir / f"bootstrap_draws_{target}_{window}.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "block_bootstrap_ci.csv", index=False)
    accepted = (result["ci_low"] > 0).all()
    lines = [
        "# Block bootstrap assessment",
        "",
        f"Scope months: {','.join(str(m) for m in months)}.",
        f"Airport-day bootstrap blocks: {int(result['blocks'].max())}; repetitions: {reps}.",
        f"Assessment: {'accepted' if accepted else 'diagnostic only'} for reported use.",
        "",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"- {row.target}, {row.window}: observed {row.observed_diff:+.3f}, "
            f"95% CI [{row.ci_low:+.3f}, {row.ci_high:+.3f}], "
            f"leave-one-airport range [{row.loo_min:+.3f}, {row.loo_max:+.3f}]."
        )
    (out_dir / "block_bootstrap_assessment.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--output-name", default="block_bootstrap_smoke")
    args = parser.parse_args()
    run(parse_months(args.months), args.reps, args.seed, args.output_name)


if __name__ == "__main__":
    main()
