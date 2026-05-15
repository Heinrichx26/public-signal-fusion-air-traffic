from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    "README.md",
    "requirements.txt",
    "results/benchmark/benchmark_field_dictionary.csv",
    "results/benchmark/benchmark_task_definitions.csv",
    "results/benchmark/benchmark_split_definitions.csv",
    "results/benchmark/benchmark_baseline_scores.csv",
    "results/scorecards/paper_scorecard.csv",
    "results/scorecards/acrf_model_metrics.csv",
    "results/scorecards/acrf_fold_metrics.csv",
    "results/scorecards/acrf_eta_selection.csv",
    "results/scorecards/acrf_state_score_diffs.csv",
    "results/scorecards/acrf_full_2025_assessment.md",
    "results/scorecards/reliability_scorecard.csv",
    "results/scorecards/reliability_adjusted_scorecard.csv",
    "results/scorecards/pr_auc_calibration_gains.csv",
    "results/scorecards/leave_one_airport_gain_summary.csv",
    "results/scorecards/event_pretrend_contrast.csv",
    "results/scorecards/aohi_bootstrap_summary.csv",
    "results/scorecards/aohi_stability.csv",
    "results/scorecards/aohi_placebo_context.csv",
]


def row(path: str) -> dict[str, object]:
    full = ROOT / path
    return {
        "path": path,
        "exists": full.exists(),
        "bytes": full.stat().st_size if full.exists() else 0,
    }


def table_summary(path: str) -> dict[str, object]:
    full = ROOT / path
    if not full.exists() or full.suffix.lower() != ".csv":
        return {"path": path, "rows": "", "columns": ""}
    data = pd.read_csv(full)
    return {"path": path, "rows": len(data), "columns": len(data.columns)}


def main() -> None:
    audit = pd.DataFrame([row(path) for path in REQUIRED])
    missing = audit[~audit["exists"]]
    out = ROOT / "results" / "release_package_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)

    summaries = pd.DataFrame([table_summary(path) for path in REQUIRED])
    summaries.to_csv(ROOT / "results" / "release_table_summary.csv", index=False)

    if not missing.empty:
        raise SystemExit("Missing required files:\n" + "\n".join(missing["path"].tolist()))
    print("Release package audit passed.")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
