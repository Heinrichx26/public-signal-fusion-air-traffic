from __future__ import annotations

from pathlib import Path

import pandas as pd

from fusion_strengthening_common import MAIN, ROOT_OUT


OUT = ROOT_OUT / "paper_scorecards"
DEMAND = ROOT_OUT / "demand_residual_full_2025"
ISSUE = ROOT_OUT / "issue_time_full_2025"
PLACEBO = ROOT_OUT / "placebo_shift_full_2025"
BOOT = ROOT_OUT / "block_bootstrap_full_2025"
SUPP = ROOT_OUT.parents[0] / "supplemental_validation"


def row(label: str, metric: str, value, status: str, source: str) -> dict:
    return {"evidence": label, "metric": metric, "value": value, "status": status, "source_table": source}


def demand_rows() -> list[dict]:
    gains = pd.read_csv(DEMAND / "baseline_ladder_gain_summary.csv")
    residual = pd.read_csv(DEMAND / "residual_state_validation.csv")
    rows = []
    for target in ["long_arrival_delay", "cancellation"]:
        g = gains[(gains["target"] == target) & (gains["model"] == "schedule_fused_state")].iloc[0]
        rows.append(row("Schedule-demand baseline increment", f"{target} AUC gain", f"{g.auc_gain_vs_primary:+.3f}", "accepted", "baseline_ladder_gain_summary.csv"))
    for target, window in [("long_arrival_delay", "active"), ("long_arrival_delay", "post_3h")]:
        r = residual[(residual["target"] == target) & (residual["window"] == window)].iloc[0]
        rows.append(row("Demand-adjusted residual risk", f"{window} residual diff.", f"{r.residual_diff:+.3f}", "accepted", "residual_state_validation.csv"))
    return rows


def ablation_rows() -> list[dict]:
    ablation = pd.read_csv(DEMAND / "direct_ablation_matrix.csv")
    rows = []
    for target in ["long_arrival_delay", "cancellation"]:
        for model in ["advisory_only", "weather_advisory", "schedule_demand_advisory", "schedule_fused_state"]:
            item = ablation[(ablation["target"] == target) & (ablation["model"] == model)].iloc[0]
            rows.append(
                row(
                    "Direct ablation ladder",
                    f"{target} {model}",
                    f"AUC {item.auc:.3f}; gain vs weather {item.auc_gain_vs_weather:+.3f}",
                    "accepted",
                    "direct_ablation_matrix.csv",
                )
            )
    return rows


def issue_rows() -> list[dict]:
    audit = pd.read_csv(ISSUE / "advisory_issue_time_audit.csv")
    detail = pd.read_csv(ISSUE / "advisory_issue_times.csv")
    coverage = int(audit["records"].sum())
    sig_share = float(audit.loc[audit["issue_source"] == "signature", "share"].iloc[0])
    return [
        row("ATCSCC timing audit", "signature coverage", f"{coverage} records; {sig_share:.3f} share", "accepted", "advisory_issue_time_audit.csv"),
        row("ATCSCC timing audit", "lead min/max", f"{detail.lead_minutes.min():+.0f}/{detail.lead_minutes.max():+.0f} min", "accepted", "advisory_issue_times.csv"),
    ]


def placebo_rows() -> list[dict]:
    data = pd.read_csv(PLACEBO / "placebo_shift_test.csv")
    rows = []
    for window in ["active", "post_3h"]:
        for outcome in ["long_arrival_delay", "cancellation"]:
            use = data[(data["window"] == window) & (data["outcome"] == outcome)]
            observed = float(use[use["label"] == "observed"]["delta"].iloc[0])
            placebo = float(use[use["label"] != "observed"]["delta"].max())
            rows.append(row("Shifted-advisory placebo", f"{window} {outcome}", f"obs {observed:+.3f}; placebo max {placebo:+.3f}", "accepted", "placebo_shift_test.csv"))
    return rows


def bootstrap_rows() -> list[dict]:
    ci = pd.read_csv(BOOT / "block_bootstrap_ci.csv")
    rows = []
    for item in ci.itertuples(index=False):
        rows.append(
            row(
                "Airport-day block bootstrap",
                f"{item.window} {item.target}",
                f"{item.observed_diff:+.3f} [{item.ci_low:+.3f}, {item.ci_high:+.3f}]",
                "accepted",
                "block_bootstrap_ci.csv",
            )
        )
    return rows


def reliability_rows() -> list[dict]:
    coverage = pd.read_csv(MAIN / "source_data_completeness_audit.csv")
    rows = []
    for item in coverage.itertuples(index=False):
        rows.append(row("Main-panel data closure", item.dataset, f"{item.available_units}/{item.expected_units}; issues {item.issue_rows}", "accepted", "source_data_completeness_audit.csv"))
    rows.extend(issue_rows())
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper = pd.DataFrame(demand_rows() + ablation_rows() + issue_rows() + placebo_rows() + bootstrap_rows())
    reliability = pd.DataFrame(reliability_rows())
    paper.to_csv(OUT / "paper_scorecard.csv", index=False)
    reliability.to_csv(OUT / "reliability_scorecard.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
