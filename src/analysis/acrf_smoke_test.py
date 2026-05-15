from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_prediction_increment import (
    evaluate,
    feature_levels,
    fit_grouped_logit,
    make_design,
    sigmoid,
    train_stats,
)
from fusion_strengthening_common import ROOT_OUT
from smoke_source_fusion_topics import weighted_mean


PROJECT = Path(__file__).resolve().parents[2]
TEMPORAL_PANEL = (
    ROOT_OUT
    / "temporal_availability_full_2025"
    / "temporal_availability_panel.csv"
)
OUT_DIR = ROOT_OUT / "acrf_smoke_ewr_ord_2025_06_07"

TARGETS = {
    "long_arrival_delay": "arr_delay60_count",
    "cancellation": "cancel_count",
}

CALENDAR = ["month_sin", "month_cos"]
WEATHER = [
    "weather_score",
    "mild_weather_abs",
    "wind_speed_mps",
    "visibility_km",
    "ceiling_m",
    "temperature_c",
]
DEMAND = [
    "scheduled_arrivals",
    "scheduled_departures",
    "arrival_bank_intensity",
    "departure_bank_intensity",
    "arrival_carrier_hhi",
    "departure_carrier_hhi",
]
ADVISORY = [
    "active_strong",
    "post_3h_strong",
    "active_hours_capped",
    "post_3h_hours_capped",
    "active_mild_strong_conflict",
    "post_3h_mild_strong_conflict",
    "active_before_hours",
    "post_3h_known_hours",
]
CATS = ["airport", "local_hour", "day_of_week"]

STATE_BITS = {
    "N": 1,  # normal or low operational pressure
    "L": 2,  # local physical pressure
    "R": 4,  # residual management/network pressure
    "H": 8,  # joint local and residual pressure
}
OMEGA = 15
N = STATE_BITS["N"]
L = STATE_BITS["L"]
R = STATE_BITS["R"]
H = STATE_BITS["H"]
NL = N | L
LH = L | H
RH = R | H


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_str_list(text: str) -> list[str]:
    return [part.strip().upper() for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return sorted(set(out))


def load_panel(airports: list[str], months: list[int]) -> pd.DataFrame:
    panel = pd.read_csv(TEMPORAL_PANEL, parse_dates=["utc_hour"])
    panel = panel[panel["airport"].isin(airports) & panel["month"].isin(months)].copy()
    panel = panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy()
    for col in CATS:
        panel[col] = panel[col].astype(str)
    for col in ["active_minutes", "post_1h_minutes", "post_2h_minutes", "post_3h_minutes"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
    panel["active_hours_capped"] = (panel["active_minutes"] / 60.0).clip(0, 8)
    panel["post_1h_hours_capped"] = (panel["post_1h_minutes"] / 60.0).clip(0, 8)
    panel["post_2h_hours_capped"] = (panel["post_2h_minutes"] / 60.0).clip(0, 8)
    panel["post_3h_hours_capped"] = (panel["post_3h_minutes"] / 60.0).clip(0, 8)
    panel["active_strong"] = (panel["active_minutes"] >= 45).astype(float)
    panel["post_3h_strong"] = (panel["post_3h_minutes"] >= 45).astype(float)
    panel["active_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["active_strong"] == 1.0)
    ).astype(float)
    panel["post_3h_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_strong"] == 1.0)
    ).astype(float)
    panel["active_before_hours"] = pd.to_numeric(
        panel.get("active_before_minutes", 0.0), errors="coerce"
    ).fillna(0.0)
    panel["post_3h_known_hours"] = pd.to_numeric(
        panel.get("post_3h_known_minutes", 0.0), errors="coerce"
    ).fillna(0.0)
    return panel


def weather_confidence(df: pd.DataFrame, reliability: bool) -> np.ndarray:
    visibility = pd.to_numeric(df["visibility_km"], errors="coerce")
    wind = pd.to_numeric(df["wind_speed_mps"], errors="coerce")
    ceiling = pd.to_numeric(df["ceiling_m"], errors="coerce")
    mild = pd.to_numeric(df["mild_weather_abs"], errors="coerce").fillna(0).to_numpy(float)
    mild_margin = np.minimum.reduce(
        [
            ((visibility.fillna(8.0) - 8.0) / 8.0).clip(0, 1).to_numpy(float),
            ((7.0 - wind.fillna(7.0)) / 7.0).clip(0, 1).to_numpy(float),
            ((ceiling.fillna(2000.0) - 1000.0) / 1000.0).clip(0, 1).to_numpy(float),
        ]
    )
    severe_margin = np.maximum.reduce(
        [
            ((8.0 - visibility.fillna(8.0)) / 8.0).clip(0, 1).to_numpy(float),
            ((wind.fillna(7.0) - 7.0) / 7.0).clip(0, 1).to_numpy(float),
            ((1000.0 - ceiling.fillna(1000.0)) / 1000.0).clip(0, 1).to_numpy(float),
        ]
    )
    margin = np.where(mild == 1.0, mild_margin, severe_margin)
    confidence = 0.60 + 0.35 * margin
    if reliability:
        complete = visibility.notna() & wind.notna() & ceiling.notna()
        q_w = np.where(complete.to_numpy(), 1.0, 0.65)
        confidence *= q_w
    return np.clip(confidence, 0.05, 0.95)


def advisory_strength(df: pd.DataFrame, async_kernel: bool, reliability: bool) -> np.ndarray:
    active = pd.to_numeric(df["active_hours_capped"], errors="coerce").fillna(0.0).clip(0, 1)
    post_tail = (
        pd.to_numeric(df["post_3h_hours_capped"], errors="coerce").fillna(0.0)
        - pd.to_numeric(df["active_hours_capped"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0, upper=3.0)
    issue_before = pd.to_numeric(df["active_before_hours"], errors="coerce").fillna(0.0).clip(0, 1)
    post_known = pd.to_numeric(df["post_3h_known_hours"], errors="coerce").fillna(0.0).clip(0, 4)
    if async_kernel:
        strength = np.maximum.reduce(
            [
                active.to_numpy(float),
                0.35 * issue_before.to_numpy(float),
                0.60 * (post_tail.to_numpy(float) / 3.0),
                0.25 * (post_known.to_numpy(float) / 4.0),
            ]
        )
    else:
        strength = (
            (pd.to_numeric(df["active_strong"], errors="coerce").fillna(0.0) == 1.0)
            | (pd.to_numeric(df["post_3h_strong"], errors="coerce").fillna(0.0) == 1.0)
        ).astype(float).to_numpy()
    q_a = 1.0
    if reliability:
        q_a = np.where(post_known.to_numpy(float) > 0, 1.0, 0.90)
    return np.clip(strength * q_a, 0.0, 0.95)


def no_advisory_confidence(df: pd.DataFrame, reliability: bool) -> np.ndarray:
    active = pd.to_numeric(df["active_strong"], errors="coerce").fillna(0.0)
    post = pd.to_numeric(df["post_3h_strong"], errors="coerce").fillna(0.0)
    confidence = np.where((active == 0.0) & (post == 0.0), 0.82, 0.0)
    if reliability:
        confidence *= 0.95
    return confidence.astype(float)


def demand_mass_strength(p_base: np.ndarray, train_base_rate: float, reliability: bool, df: pd.DataFrame) -> np.ndarray:
    scale = max(train_base_rate * 3.0, 0.03)
    strength = np.clip(p_base / scale, 0.0, 1.0)
    confidence = 0.35 + 0.45 * strength
    if reliability:
        has_demand = pd.to_numeric(df["scheduled_arrivals"], errors="coerce").fillna(0.0).to_numpy(float) > 0
        confidence *= np.where(has_demand, 1.0, 0.50)
    return np.clip(confidence, 0.0, 0.90)


def mass_weather(row_mild: float, c_w: float) -> dict[int, float]:
    if row_mild == 1.0:
        return {N: c_w, OMEGA: 1.0 - c_w}
    return {LH: c_w, OMEGA: 1.0 - c_w}


def mass_action(a_strength: float, no_action_c: float) -> dict[int, float]:
    if a_strength > 0:
        return {RH: a_strength, OMEGA: 1.0 - a_strength}
    return {NL: no_action_c, OMEGA: 1.0 - no_action_c}


def mass_demand(p_base: float, c_d: float, train_base_rate: float) -> dict[int, float]:
    if p_base >= train_base_rate:
        return {RH: c_d, OMEGA: 1.0 - c_d}
    return {NL: c_d, OMEGA: 1.0 - c_d}


def combine_masses(m1: dict[int, float], m2: dict[int, float], rule: str) -> tuple[dict[int, float], float]:
    out: dict[int, float] = {}
    conflict = 0.0
    for set1, mass1 in m1.items():
        for set2, mass2 in m2.items():
            inter = set1 & set2
            value = mass1 * mass2
            if inter == 0:
                conflict += value
            else:
                out[inter] = out.get(inter, 0.0) + value
    if rule == "ds":
        denom = max(1.0 - conflict, 1e-8)
        out = {key: value / denom for key, value in out.items()}
    elif rule == "yager":
        out[OMEGA] = out.get(OMEGA, 0.0) + conflict
    else:
        raise ValueError(f"unknown mass-combination rule: {rule}")
    total = sum(out.values())
    if total > 0:
        out = {key: value / total for key, value in out.items()}
    return out, conflict


def risk_belief(mass: dict[int, float]) -> float:
    score = 0.0
    for states, value in mass.items():
        size = int(states.bit_count())
        pressure = int((states & (R | H)).bit_count())
        score += value * pressure / size
    return float(score)


def residual_belief(mass: dict[int, float]) -> float:
    score = 0.0
    for states, value in mass.items():
        if states & R:
            score += value / int(states.bit_count())
    return float(score)


def credal_scores(
    df: pd.DataFrame,
    p_base: np.ndarray,
    train_base_rate: float,
    model: str,
    eta: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model not in {
        "ds_fusion",
        "yager_fusion",
        "acrf_no_async",
        "acrf_no_reliability",
        "full_acrf",
        "full_acrf_eta",
    }:
        raise ValueError(model)
    rule = "ds" if model == "ds_fusion" else "yager"
    use_async = model in {"acrf_no_reliability", "full_acrf", "full_acrf_eta"}
    use_reliability = model != "acrf_no_reliability"
    redistribute = model in {"acrf_no_async", "acrf_no_reliability", "full_acrf", "full_acrf_eta"}

    c_w = weather_confidence(df, reliability=use_reliability)
    a_strength = advisory_strength(df, async_kernel=use_async, reliability=use_reliability)
    no_action_c = no_advisory_confidence(df, reliability=use_reliability)
    c_d = demand_mass_strength(p_base, train_base_rate, reliability=use_reliability, df=df)
    mild = pd.to_numeric(df["mild_weather_abs"], errors="coerce").fillna(0.0).to_numpy(float)
    risk = np.zeros(len(df), dtype=float)
    residual = np.zeros(len(df), dtype=float)
    conflict_mass = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        mw = mass_weather(mild[i], float(c_w[i]))
        ma = mass_action(float(a_strength[i]), float(no_action_c[i]))
        md = mass_demand(float(p_base[i]), float(c_d[i]), train_base_rate)
        mwa, conflict = combine_masses(mw, ma, rule=rule)
        phys_action_conflict = 0.0
        if mild[i] == 1.0 and a_strength[i] > 0:
            phys_action_conflict = float(c_w[i]) * float(a_strength[i])
        if redistribute and phys_action_conflict > 0:
            demand_gate = 0.50 + 0.50 * min(float(p_base[i]) / max(train_base_rate * 2.0, 1e-6), 1.0)
            async_gate = max(float(a_strength[i]), 0.35) if use_async else 1.0
            transfer = min(mwa.get(OMEGA, 0.0), eta * phys_action_conflict * demand_gate * async_gate)
            mwa[OMEGA] = mwa.get(OMEGA, 0.0) - transfer
            mwa[R] = mwa.get(R, 0.0) + transfer
        mf, conflict2 = combine_masses(mwa, md, rule="yager")
        risk[i] = risk_belief(mf)
        residual[i] = residual_belief(mf)
        conflict_mass[i] = conflict + conflict2
    return risk, residual, conflict_mass


def fit_predict_logit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    success_col: str,
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    stats = train_stats(train, numeric)
    x_train = make_design(train, numeric, CATS, stats, levels)
    x_test = make_design(test, numeric, CATS, stats, levels)
    beta = fit_grouped_logit(
        x_train,
        train[success_col].to_numpy(float),
        train["arrivals"].to_numpy(float),
    )
    return sigmoid(x_train @ beta), sigmoid(x_test @ beta)


def calibrate_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train: pd.DataFrame,
    success_col: str,
) -> np.ndarray:
    x_train = np.column_stack([np.ones(len(train_features)), train_features])
    beta = fit_grouped_logit(
        x_train,
        train[success_col].to_numpy(float),
        train["arrivals"].to_numpy(float),
        ridge=1e-4,
    )
    x_test = np.column_stack([np.ones(len(test_features)), test_features])
    return sigmoid(x_test @ beta)


def choose_inner_validation_month(train_months: list[int], heldout_month: int) -> int | None:
    if len(train_months) < 2:
        return None
    earlier = [m for m in train_months if m < heldout_month]
    if earlier:
        return max(earlier)
    return min(train_months)


def build_credal_features(
    df: pd.DataFrame,
    p_base: np.ndarray,
    train_base_rate: float,
    model: str,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score, residual, conflict = credal_scores(df, p_base, train_base_rate, model, eta=eta)
    features = np.column_stack([p_base, score, residual, conflict])
    return features, score, residual, conflict


def learn_eta(
    train: pd.DataFrame,
    p_base_train: np.ndarray,
    train_base_rate: float,
    success_col: str,
    heldout_month: int,
    eta_grid: list[float],
    eta_objective: str,
) -> tuple[float, pd.DataFrame]:
    train_months = sorted(train["month"].astype(int).unique().tolist())
    valid_month = choose_inner_validation_month(train_months, heldout_month)
    if valid_month is None:
        return 0.85, pd.DataFrame(
            [
                {
                    "outer_fold_month": heldout_month,
                    "inner_valid_month": np.nan,
                    "eta": 0.85,
                    "log_loss": np.nan,
                    "brier": np.nan,
                    "auc": np.nan,
                    "selected": True,
                    "eta_objective": eta_objective,
                    "note": "default_eta_no_inner_split",
                }
            ]
        )
    inner_valid_mask = train["month"].astype(int).eq(valid_month).to_numpy()
    inner_train_mask = ~inner_valid_mask
    inner_train = train.loc[inner_train_mask].copy()
    inner_valid = train.loc[inner_valid_mask].copy()
    rows = []
    for eta in eta_grid:
        inner_train_features, _, _, _ = build_credal_features(
            inner_train,
            p_base_train[inner_train_mask],
            train_base_rate,
            "full_acrf_eta",
            eta,
        )
        inner_valid_features, _, _, _ = build_credal_features(
            inner_valid,
            p_base_train[inner_valid_mask],
            train_base_rate,
            "full_acrf_eta",
            eta,
        )
        valid_prob = calibrate_features(inner_train_features, inner_valid_features, inner_train, success_col)
        metric = evaluate(
            inner_valid[success_col].to_numpy(float),
            inner_valid["arrivals"].to_numpy(float),
            valid_prob,
        )
        metric["pr_auc"] = grouped_pr_auc(
            inner_valid[success_col].to_numpy(float),
            inner_valid["arrivals"].to_numpy(float),
            valid_prob,
        )
        rows.append(
            metric
            | {
                "outer_fold_month": heldout_month,
                "inner_valid_month": valid_month,
                "eta": eta,
                "selected": False,
                "eta_objective": eta_objective,
                "note": "grid_search",
            }
        )
    result = pd.DataFrame(rows)
    if eta_objective == "auc":
        result = result.sort_values(["auc", "log_loss", "eta"], ascending=[False, True, True]).reset_index(drop=True)
    elif eta_objective == "brier":
        result = result.sort_values(["brier", "log_loss", "eta"], ascending=[True, True, True]).reset_index(drop=True)
    else:
        result = result.sort_values(["log_loss", "brier", "eta"], ascending=[True, True, True]).reset_index(drop=True)
    result.loc[0, "selected"] = True
    return float(result.loc[0, "eta"]), result


def grouped_pr_auc(successes: np.ndarray, totals: np.ndarray, score: np.ndarray) -> float:
    positives = successes.astype(float)
    total_pos = positives.sum()
    if total_pos <= 0:
        return np.nan
    order = np.argsort(-score)
    sorted_pos = positives[order]
    sorted_total = totals.astype(float)[order]
    cum_pos = np.cumsum(sorted_pos)
    cum_total = np.cumsum(sorted_total)
    precision = np.divide(cum_pos, cum_total, out=np.zeros_like(cum_pos), where=cum_total > 0)
    return float(np.sum((sorted_pos / total_pos) * precision))


def top_decile_lift(successes: np.ndarray, totals: np.ndarray, score: np.ndarray) -> float:
    total = totals.astype(float).sum()
    total_events = successes.astype(float).sum()
    if total <= 0 or total_events <= 0:
        return np.nan
    order = np.argsort(-score)
    cutoff = 0.10 * total
    chosen_total = 0.0
    chosen_events = 0.0
    for idx in order:
        if chosen_total >= cutoff:
            break
        chosen_total += float(totals[idx])
        chosen_events += float(successes[idx])
    if chosen_total <= 0:
        return np.nan
    return float((chosen_events / chosen_total) / (total_events / total))


def metric_with_extras(frame: pd.DataFrame, success_col: str) -> dict:
    successes = frame[success_col].to_numpy(float)
    totals = frame["arrivals"].to_numpy(float)
    prob = frame["pred_prob"].to_numpy(float)
    metrics = evaluate(successes, totals, prob)
    metrics["pr_auc"] = grouped_pr_auc(successes, totals, prob)
    metrics["top_decile_lift"] = top_decile_lift(successes, totals, prob)
    return metrics


def state_score_diffs(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    pred = predictions[predictions["target"].eq(target)].copy()
    for model, model_df in pred.groupby("model"):
        for window, strong_col in [("active", "active_strong"), ("post_3h", "post_3h_strong")]:
            conflict = model_df[(model_df["mild_weather_abs"] == 1.0) & (model_df[strong_col] == 1.0)]
            baseline = model_df[(model_df["mild_weather_abs"] == 1.0) & (model_df[strong_col] == 0.0)]
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "window": window,
                    "conflict_arrivals": int(conflict["arrivals"].sum()),
                    "baseline_arrivals": int(baseline["arrivals"].sum()),
                    "risk_score_diff": weighted_mean(conflict, "pred_prob", "arrivals")
                    - weighted_mean(baseline, "pred_prob", "arrivals"),
                    "residual_belief_diff": weighted_mean(conflict, "residual_belief", "arrivals")
                    - weighted_mean(baseline, "residual_belief", "arrivals"),
                    "observed_delay_diff": weighted_mean(conflict, "arr_delay60_rate", "arrivals")
                    - weighted_mean(baseline, "arr_delay60_rate", "arrivals"),
                    "mean_conflict_mass": weighted_mean(conflict, "conflict_mass", "arrivals"),
                }
            )
    return pd.DataFrame(rows)


def run_target(
    panel: pd.DataFrame,
    target: str,
    success_col: str,
    months: list[int],
    eta_grid: list[float],
    eta_objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_specs = {
        "calendar_weather_demand": CALENDAR + WEATHER + DEMAND,
        "early_fusion_wad": CALENDAR + WEATHER + DEMAND + ADVISORY,
        "residual_logistic_fusion": CALENDAR + WEATHER + DEMAND + ADVISORY + [
            "active_mild_strong_conflict",
            "post_3h_mild_strong_conflict",
        ],
    }
    credal_models = [
        "ds_fusion",
        "yager_fusion",
        "acrf_no_async",
        "acrf_no_reliability",
        "full_acrf",
        "full_acrf_eta",
    ]
    levels = feature_levels(panel, CATS)
    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    eta_frames: list[pd.DataFrame] = []
    for month in months:
        train = panel[panel["month"] != month].copy()
        test = panel[panel["month"] == month].copy()
        p_base_train, p_base_test = fit_predict_logit(
            train, test, model_specs["calendar_weather_demand"], success_col, levels
        )
        train_base_rate = float(train[success_col].sum() / train["arrivals"].sum())
        learned_eta, eta_table = learn_eta(
            train,
            p_base_train,
            train_base_rate,
            success_col,
            int(month),
            eta_grid,
            eta_objective,
        )
        eta_table["target"] = target
        eta_frames.append(eta_table)
        for model, numeric in model_specs.items():
            train_prob, test_prob = fit_predict_logit(train, test, numeric, success_col, levels)
            fold = test.copy()
            fold["target"] = target
            fold["model"] = model
            fold["pred_prob"] = test_prob
            fold["residual_belief"] = test_prob - p_base_test
            fold["conflict_mass"] = 0.0
            fold["eta"] = np.nan
            pred_frames.append(fold)
            metric_rows.append(metric_with_extras(fold, success_col) | {"target": target, "model": model, "fold_month": month})
        for model in credal_models:
            eta = learned_eta if model == "full_acrf_eta" else 0.85
            train_features, train_score, train_residual, train_conflict = build_credal_features(
                train,
                p_base_train,
                train_base_rate,
                model,
                eta,
            )
            test_features, test_score, test_residual, test_conflict = build_credal_features(
                test,
                p_base_test,
                train_base_rate,
                model,
                eta,
            )
            test_prob = calibrate_features(train_features, test_features, train, success_col)
            fold = test.copy()
            fold["target"] = target
            fold["model"] = model
            fold["pred_prob"] = test_prob
            fold["raw_risk_belief"] = test_score
            fold["residual_belief"] = test_residual
            fold["conflict_mass"] = test_conflict
            fold["eta"] = eta
            pred_frames.append(fold)
            metric_rows.append(metric_with_extras(fold, success_col) | {"target": target, "model": model, "fold_month": month})
    predictions = pd.concat(pred_frames, ignore_index=True)
    for model, model_df in predictions.groupby("model"):
        metric_rows.append(metric_with_extras(model_df, success_col) | {"target": target, "model": model, "fold_month": "all"})
    return pd.DataFrame(metric_rows), predictions, pd.concat(eta_frames, ignore_index=True)


def add_gains(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[metrics["fold_month"].astype(str).eq("all")].copy()
    base = overall[overall["model"].eq("calendar_weather_demand")][
        ["target", "auc", "pr_auc", "brier", "log_loss", "top_decile_lift"]
    ].rename(
        columns={
            "auc": "base_auc",
            "pr_auc": "base_pr_auc",
            "brier": "base_brier",
            "log_loss": "base_log_loss",
            "top_decile_lift": "base_top_decile_lift",
        }
    )
    out = overall.merge(base, on="target", how="left")
    out["auc_gain_vs_base"] = out["auc"] - out["base_auc"]
    out["pr_auc_gain_vs_base"] = out["pr_auc"] - out["base_pr_auc"]
    out["brier_gain_vs_base"] = out["base_brier"] - out["brier"]
    out["log_loss_gain_vs_base"] = out["base_log_loss"] - out["log_loss"]
    out["top_lift_gain_vs_base"] = out["top_decile_lift"] - out["base_top_decile_lift"]
    return out.drop(columns=[col for col in out.columns if col.startswith("base_")])


def write_assessment(gains: pd.DataFrame, diffs: pd.DataFrame, out_dir: Path, airports: list[str], months: list[int]) -> None:
    lines = [
        "# ACRF smoke-test assessment",
        "",
        f"Scope: airports {','.join(airports)}; 2025 months {','.join(str(m) for m in months)}.",
        "",
        "Smoke gate: learned-eta Full ACRF should beat D-S and Yager on long-delay AUC, keep positive post-window residual belief, and avoid a Brier loss larger than 0.002 versus the calendar-weather-demand baseline.",
        "",
    ]
    delay = gains[gains["target"].eq("long_arrival_delay")].set_index("model")
    cancel = gains[gains["target"].eq("cancellation")].set_index("model")
    full_delay = delay.loc["full_acrf"]
    full_cancel = cancel.loc["full_acrf"]
    eta_delay = delay.loc["full_acrf_eta"]
    eta_cancel = cancel.loc["full_acrf_eta"]
    early_delay = delay.loc["early_fusion_wad"]
    early_cancel = cancel.loc["early_fusion_wad"]
    ds_delay_auc = float(delay.loc["ds_fusion", "auc"])
    yager_delay_auc = float(delay.loc["yager_fusion", "auc"])
    post_full = diffs[
        (diffs["target"].eq("long_arrival_delay"))
        & (diffs["model"].eq("full_acrf_eta"))
        & (diffs["window"].eq("post_3h"))
    ].iloc[0]
    passes = (
        float(eta_delay["auc"]) > max(ds_delay_auc, yager_delay_auc)
        and float(post_full["residual_belief_diff"]) > 0
        and float(eta_delay["brier_gain_vs_base"]) > -0.002
    )
    lines.append(f"Assessment: {'usable for full-year expansion' if passes else 'needs redesign before full-year expansion'}.")
    lines.append("")
    lines.append(
        f"- Learned-eta Full ACRF long-delay: AUC {eta_delay['auc']:.3f} "
        f"({eta_delay['auc_gain_vs_base']:+.3f} vs baseline), PR-AUC {eta_delay['pr_auc']:.3f}, "
        f"Brier gain {eta_delay['brier_gain_vs_base']:+.4f}, top-decile lift {eta_delay['top_decile_lift']:.2f}; "
        f"gap to early fusion AUC {eta_delay['auc'] - early_delay['auc']:+.3f}."
    )
    lines.append(
        f"- Learned-eta Full ACRF cancellation: AUC {eta_cancel['auc']:.3f} "
        f"({eta_cancel['auc_gain_vs_base']:+.3f} vs baseline), PR-AUC {eta_cancel['pr_auc']:.3f}, "
        f"Brier gain {eta_cancel['brier_gain_vs_base']:+.4f}, top-decile lift {eta_cancel['top_decile_lift']:.2f}; "
        f"gap to early fusion AUC {eta_cancel['auc'] - early_cancel['auc']:+.3f}."
    )
    lines.append(
        f"- D-S/Yager long-delay AUC: {ds_delay_auc:.3f}/{yager_delay_auc:.3f}; "
        f"learned-eta Full ACRF post residual-belief diff: {post_full['residual_belief_diff']:+.3f}."
    )
    lines.append(
        f"- Fixed-eta Full ACRF long-delay AUC {full_delay['auc']:.3f}; "
        f"cancellation AUC {full_cancel['auc']:.3f}."
    )
    lines.append("")
    lines.append("Model ranking by long-delay AUC:")
    ranked = delay.sort_values("auc", ascending=False)
    for row in ranked.itertuples():
        lines.append(
            f"- {row.Index}: AUC {row.auc:.3f}; PR-AUC {row.pr_auc:.3f}; "
            f"Brier gain {row.brier_gain_vs_base:+.4f}; top lift {row.top_decile_lift:.2f}."
        )
    (out_dir / "acrf_smoke_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    airports: list[str],
    months: list[int],
    out_dir: Path,
    eta_grid: list[float],
    eta_objective: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(airports, months)
    all_metrics, all_predictions, all_eta = [], [], []
    for target, success_col in TARGETS.items():
        metrics, predictions, eta_table = run_target(
            panel,
            target,
            success_col,
            months,
            eta_grid,
            eta_objective,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_eta.append(eta_table)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    eta_selection = pd.concat(all_eta, ignore_index=True)
    gains = add_gains(metrics)
    diffs = pd.concat(
        [state_score_diffs(predictions, target) for target in TARGETS],
        ignore_index=True,
    )
    panel.to_csv(out_dir / "acrf_smoke_panel.csv", index=False)
    metrics.to_csv(out_dir / "acrf_fold_metrics.csv", index=False)
    gains.to_csv(out_dir / "acrf_model_metrics.csv", index=False)
    eta_selection.to_csv(out_dir / "acrf_eta_selection.csv", index=False)
    diffs.to_csv(out_dir / "acrf_state_score_diffs.csv", index=False)
    predictions[
        [
            "airport",
            "utc_hour",
            "month",
            "arrivals",
            "target",
            "model",
            "pred_prob",
            "residual_belief",
            "conflict_mass",
            "eta",
            "arr_delay60_rate",
            "cancel_rate",
            "active_strong",
            "post_3h_strong",
            "mild_weather_abs",
        ]
    ].to_csv(out_dir / "acrf_predictions.csv", index=False)
    write_assessment(gains, diffs, out_dir, airports, months)
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airports", default="EWR,ORD")
    parser.add_argument("--months", default="6,7")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--eta-grid", default="0,0.25,0.5,0.75,1,1.25,1.5,1.75,2")
    parser.add_argument("--eta-objective", choices=["log_loss", "brier", "auc"], default="log_loss")
    args = parser.parse_args()
    run(
        parse_str_list(args.airports),
        parse_int_list(args.months),
        Path(args.output_dir),
        parse_float_list(args.eta_grid),
        args.eta_objective,
    )


if __name__ == "__main__":
    main()
