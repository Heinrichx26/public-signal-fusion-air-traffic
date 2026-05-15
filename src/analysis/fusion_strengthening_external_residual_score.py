from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_prediction_increment import feature_levels, fit_grouped_logit, make_design, sigmoid, train_stats
from fusion_strengthening_common import MAIN, ROOT_OUT, add_demand_features
from fusion_strengthening_demand_residual import CATS, MODEL_SPECS, TARGETS
from smoke_source_fusion_topics import weighted_mean


PANEL_2024 = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "experiments"
    / "supplemental_validation"
    / "cross_year_2024"
    / "full_2024_airport_hour_panel.csv"
)
PANEL_2025 = MAIN / "airport_hour_panel_with_windows.csv"
BASELINE = "calendar_weather_schedule_demand"
FUSION = "schedule_fused_state"


def parse_months(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        elif part:
            out.append(int(part))
    return sorted({m for m in out if 1 <= m <= 12})


def prepare_panel(path: Path, year: int, months: list[int]) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["utc_hour"])
    panel = panel[panel["month"].isin(months) & (panel["arrivals"] > 0)].copy()
    panel = panel[panel["weather_score"].notna()].copy()
    panel["active_strong"] = (panel["active_minutes"] >= 45).astype(float)
    panel["post_3h_strong"] = (panel["post_3h_minutes"] >= 45).astype(float)
    panel["active_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["active_strong"] == 1.0)
    ).astype(float)
    panel["post_3h_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_strong"] == 1.0)
    ).astype(float)
    panel["active_hours_capped"] = (panel["active_minutes"] / 60).clip(0, 8)
    panel["post_3h_hours_capped"] = (panel["post_3h_minutes"] / 60).clip(0, 8)
    angle = 2 * np.pi * (panel["month"].astype(float) - 1) / 12.0
    panel["month_sin"] = np.sin(angle)
    panel["month_cos"] = np.cos(angle)
    for col in CATS:
        panel[col] = panel[col].astype(str)
    return add_demand_features(panel, year, months)


def score_models(train: pd.DataFrame, test: pd.DataFrame, target: str, success_col: str) -> pd.DataFrame:
    frames = []
    for model_name in [BASELINE, FUSION]:
        spec = MODEL_SPECS[model_name]
        levels = feature_levels(pd.concat([train, test], ignore_index=True), spec["categorical"])
        stats = train_stats(train, spec["numeric"])
        x_train = make_design(train, spec["numeric"], spec["categorical"], stats, levels)
        x_test = make_design(test, spec["numeric"], spec["categorical"], stats, levels)
        beta = fit_grouped_logit(x_train, train[success_col].to_numpy(float), train["arrivals"].to_numpy(float))
        out = test[["airport", "utc_hour", "month", "arrivals", success_col]].copy()
        out["target"] = target
        out["model"] = model_name
        out["pred_prob"] = sigmoid(x_test @ beta)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def build_scores(preds: pd.DataFrame) -> pd.DataFrame:
    wide = preds.pivot_table(
        index=["airport", "utc_hour", "month", "arrivals", "target"],
        columns="model",
        values="pred_prob",
        aggfunc="first",
    ).reset_index()
    rows = []
    for target, success_col in TARGETS.items():
        truth = preds[preds["target"] == target][["airport", "utc_hour", success_col]].drop_duplicates()
        use = wide[wide["target"] == target].merge(truth, on=["airport", "utc_hour"], how="left")
        use["baseline_pred"] = use[BASELINE]
        use["fusion_pred"] = use[FUSION]
        use["residual_score"] = use["fusion_pred"] - use["baseline_pred"]
        use["obs_rate"] = use[success_col] / use["arrivals"]
        use["obs_residual"] = use["obs_rate"] - use["baseline_pred"]
        rows.append(use)
    return pd.concat(rows, ignore_index=True)


def deciles(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail, summary = [], []
    for target, group in scores.groupby("target"):
        group = group[group["arrivals"] > 0].copy()
        group["score_decile"] = pd.qcut(group["residual_score"], 10, labels=False, duplicates="drop") + 1
        rows = []
        for decile, g in group.groupby("score_decile"):
            rows.append(
                {
                    "target": target,
                    "score_decile": int(decile),
                    "arrivals": int(g["arrivals"].sum()),
                    "mean_score": weighted_mean(g, "residual_score", "arrivals"),
                    "observed_rate": weighted_mean(g, "obs_rate", "arrivals"),
                    "observed_residual": weighted_mean(g, "obs_residual", "arrivals"),
                }
            )
        table = pd.DataFrame(rows).sort_values("score_decile")
        detail.append(table)
        bottom, top = table.iloc[0], table.iloc[-1]
        summary.append(
            {
                "target": target,
                "top_bottom_ratio": top["observed_rate"] / bottom["observed_rate"],
                "positive_residual_steps": int((table["observed_residual"].diff().dropna() > 0).sum()),
                "total_adjacent_steps": len(table) - 1,
                "spearman_decile_rate": table[["score_decile", "observed_rate"]].corr(method="spearman").iloc[0, 1],
                "spearman_decile_residual": table[["score_decile", "observed_residual"]].corr(method="spearman").iloc[0, 1],
            }
        )
    return pd.concat(detail, ignore_index=True), pd.DataFrame(summary)


def run(args) -> None:
    months = parse_months(args.months)
    out = ROOT_OUT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    train = prepare_panel(PANEL_2024, 2024, months)
    test = prepare_panel(PANEL_2025, 2025, months)
    preds = pd.concat([score_models(train, test, t, c) for t, c in TARGETS.items()], ignore_index=True)
    scores = build_scores(preds)
    decile_table, summary = deciles(scores)
    preds.to_csv(out / "external_residual_predictions.csv", index=False)
    scores.to_csv(out / "external_residual_scores.csv", index=False)
    decile_table.to_csv(out / "external_residual_deciles.csv", index=False)
    summary.to_csv(out / "external_residual_summary.csv", index=False)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--output-name", default="external_residual_score_smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
