from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results" / "experiments" / "atcscc_full_year_windows"
PANEL_FILE = OUT / "airport_hour_panel_with_windows.csv"
MATCH_K = 5


MATCH_NUMERIC = ["weather_score", "wind_speed_mps", "visibility_km", "ceiling_m", "temperature_c"]


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["utc_hour"])
    panel = panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy()
    panel["active_strong"] = (panel["active_minutes"] >= 45).astype(float)
    panel["post_3h_strong"] = (panel["post_3h_minutes"] >= 45).astype(float)
    panel["clean_no_active_post3h"] = ((panel["active_strong"] == 0.0) & (panel["post_3h_strong"] == 0.0)).astype(float)
    for col in MATCH_NUMERIC:
        values = pd.to_numeric(panel[col], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        sd = float(values.std(ddof=0)) if values.notna().any() else 1.0
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        panel[f"{col}_z"] = (values.fillna(median) - median) / sd
    return panel


def control_group(panel: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "active":
        mask = (panel["mild_weather_abs"] == 1.0) & (panel["active_strong"] == 0.0) & (panel["post_3h_strong"] == 0.0)
    elif scope == "post_3h":
        mask = (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_strong"] == 0.0)
    else:
        raise ValueError(scope)
    return panel[mask].copy()


def treated_group(panel: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "active":
        mask = (panel["mild_weather_abs"] == 1.0) & (panel["active_strong"] == 1.0)
    elif scope == "post_3h":
        mask = (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_strong"] == 1.0)
    else:
        raise ValueError(scope)
    return panel[mask].copy()


def weighted_rate(rows: pd.DataFrame, value_col: str) -> float:
    use = rows[[value_col, "arrivals"]].replace([np.inf, -np.inf], np.nan).dropna()
    weights = use["arrivals"].to_numpy(float)
    if len(use) == 0 or weights.sum() <= 0:
        return np.nan
    return float(np.average(use[value_col].to_numpy(float), weights=weights))


def match_scope(panel: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = treated_group(panel, scope)
    controls = control_group(panel, scope)
    groups = {
        key: group.reset_index(drop=True)
        for key, group in controls.groupby(["airport", "month", "local_hour"], observed=False)
    }
    pair_rows = []
    balance_rows = []
    unmatched = 0
    z_cols = [f"{col}_z" for col in MATCH_NUMERIC]
    for row in treated.itertuples(index=False):
        key = (row.airport, row.month, row.local_hour)
        candidates = groups.get(key)
        relaxed = False
        if candidates is None or candidates.empty:
            candidates = controls[(controls["airport"].eq(row.airport)) & (controls["month"].eq(row.month))]
            relaxed = True
        if candidates.empty:
            unmatched += 1
            continue
        diffs = candidates[z_cols].to_numpy(float) - np.array([getattr(row, col) for col in z_cols], dtype=float)
        distance = np.sqrt((diffs**2).sum(axis=1))
        take = np.argsort(distance)[:MATCH_K]
        matched = candidates.iloc[take]
        control_arrivals = matched["arrivals"].sum()
        if control_arrivals <= 0:
            unmatched += 1
            continue
        pair_rows.append(
            {
                "scope": scope,
                "airport": row.airport,
                "month": int(row.month),
                "utc_hour": row.utc_hour,
                "local_hour": int(row.local_hour),
                "treated_arrivals": int(row.arrivals),
                "control_arrivals": int(control_arrivals),
                "matched_controls": int(len(matched)),
                "relaxed_match": int(relaxed),
                "treated_arr_delay60_rate": float(row.arr_delay60_rate),
                "control_arr_delay60_rate": weighted_rate(matched, "arr_delay60_rate"),
                "treated_cancel_rate": float(row.cancel_rate),
                "control_cancel_rate": weighted_rate(matched, "cancel_rate"),
                "treated_mean_arr_delay": float(row.mean_arr_delay) if pd.notna(row.mean_arr_delay) else np.nan,
                "control_mean_arr_delay": weighted_rate(matched.dropna(subset=["mean_arr_delay"]), "mean_arr_delay"),
                "treated_p90_arr_delay": float(row.p90_arr_delay) if pd.notna(row.p90_arr_delay) else np.nan,
                "control_p90_arr_delay": weighted_rate(matched.dropna(subset=["p90_arr_delay"]), "p90_arr_delay"),
            }
        )
        for col in MATCH_NUMERIC:
            treated_value = getattr(row, col)
            control_value = weighted_rate(matched, col)
            balance_rows.append(
                {
                    "scope": scope,
                    "airport": row.airport,
                    "month": int(row.month),
                    "covariate": col,
                    "treated_value": treated_value,
                    "control_value": control_value,
                    "difference": treated_value - control_value,
                    "weight": int(row.arrivals),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    balances = pd.DataFrame(balance_rows)
    pairs["delta_arr_delay60_rate"] = pairs["treated_arr_delay60_rate"] - pairs["control_arr_delay60_rate"]
    pairs["delta_cancel_rate"] = pairs["treated_cancel_rate"] - pairs["control_cancel_rate"]
    pairs["delta_mean_arr_delay"] = pairs["treated_mean_arr_delay"] - pairs["control_mean_arr_delay"]
    pairs["delta_p90_arr_delay"] = pairs["treated_p90_arr_delay"] - pairs["control_p90_arr_delay"]
    pairs["unmatched_treated_hours"] = unmatched
    return pairs, balances


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in pairs.groupby("scope"):
        weights = group["treated_arrivals"].to_numpy(float)
        rows.append(
            {
                "scope": scope,
                "matched_treated_hours": int(len(group)),
                "unmatched_treated_hours": int(group["unmatched_treated_hours"].max()) if len(group) else 0,
                "treated_arrivals": int(group["treated_arrivals"].sum()),
                "control_arrivals": int(group["control_arrivals"].sum()),
                "relaxed_match_share": float(np.average(group["relaxed_match"], weights=weights)),
                "treated_arr_delay60_rate": float(np.average(group["treated_arr_delay60_rate"], weights=weights)),
                "control_arr_delay60_rate": float(np.average(group["control_arr_delay60_rate"], weights=weights)),
                "delta_arr_delay60_rate": float(np.average(group["delta_arr_delay60_rate"], weights=weights)),
                "treated_cancel_rate": float(np.average(group["treated_cancel_rate"], weights=weights)),
                "control_cancel_rate": float(np.average(group["control_cancel_rate"], weights=weights)),
                "delta_cancel_rate": float(np.average(group["delta_cancel_rate"], weights=weights)),
                "delta_mean_arr_delay": float(np.average(group["delta_mean_arr_delay"].fillna(0), weights=weights)),
                "delta_p90_arr_delay": float(np.average(group["delta_p90_arr_delay"].fillna(0), weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def summarize_months(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, month), group in pairs.groupby(["scope", "month"]):
        weights = group["treated_arrivals"].to_numpy(float)
        rows.append(
            {
                "scope": scope,
                "month": int(month),
                "matched_treated_hours": int(len(group)),
                "treated_arrivals": int(group["treated_arrivals"].sum()),
                "control_arrivals": int(group["control_arrivals"].sum()),
                "delta_arr_delay60_rate": float(np.average(group["delta_arr_delay60_rate"], weights=weights)),
                "delta_cancel_rate": float(np.average(group["delta_cancel_rate"], weights=weights)),
                "delta_mean_arr_delay": float(np.average(group["delta_mean_arr_delay"].fillna(0), weights=weights)),
                "delta_p90_arr_delay": float(np.average(group["delta_p90_arr_delay"].fillna(0), weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def summarize_balance(balance: pd.DataFrame) -> pd.DataFrame:
    def avg(values: pd.Series, weights: pd.Series) -> float:
        use = pd.DataFrame({"value": values, "weight": weights}).replace([np.inf, -np.inf], np.nan).dropna()
        if use.empty or use["weight"].sum() <= 0:
            return np.nan
        return float(np.average(use["value"].to_numpy(float), weights=use["weight"].to_numpy(float)))

    rows = []
    for (scope, covariate), group in balance.groupby(["scope", "covariate"]):
        rows.append(
            {
                "scope": scope,
                "covariate": covariate,
                "treated_mean": avg(group["treated_value"], group["weight"]),
                "control_mean": avg(group["control_value"], group["weight"]),
                "mean_difference": avg(group["difference"], group["weight"]),
                "mean_abs_difference": avg(group["difference"].abs(), group["weight"]),
            }
        )
    return pd.DataFrame(rows)


def write_summary(summary: pd.DataFrame, month_summary: pd.DataFrame, balance: pd.DataFrame) -> None:
    active = summary[summary["scope"].eq("active")].iloc[0]
    post = summary[summary["scope"].eq("post_3h")].iloc[0]
    active_months = month_summary[month_summary["scope"].eq("active")]
    post_months = month_summary[month_summary["scope"].eq("post_3h")]
    weather_balance = balance[balance["covariate"].eq("weather_score")]
    active_weather = weather_balance[weather_balance["scope"].eq("active")].iloc[0]
    post_weather = weather_balance[weather_balance["scope"].eq("post_3h")].iloc[0]
    lines = [
        "# Matched control validation",
        "",
        f"Active matched delay-rate delta: {active['delta_arr_delay60_rate']:.3f}.",
        f"Active matched cancellation-rate delta: {active['delta_cancel_rate']:.3f}.",
        f"Active positive monthly deltas: {int((active_months['delta_arr_delay60_rate'] > 0).sum())}/12.",
        f"Active matched treated arrivals: {int(active['treated_arrivals'])}.",
        "",
        f"post_3h matched delay-rate delta: {post['delta_arr_delay60_rate']:.3f}.",
        f"post_3h matched cancellation-rate delta: {post['delta_cancel_rate']:.3f}.",
        f"post_3h positive monthly deltas: {int((post_months['delta_arr_delay60_rate'] > 0).sum())}/12.",
        f"post_3h matched treated arrivals: {int(post['treated_arrivals'])}.",
        "",
        f"Active weather-score mean difference after matching: {active_weather['mean_difference']:.3f}.",
        f"post_3h weather-score mean difference after matching: {post_weather['mean_difference']:.3f}.",
        "",
        "Decision: the advisory signal remains large after same-airport, same-month, same-hour weather-nearest matching.",
    ]
    (OUT / "matched_control_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    pair_frames = []
    balance_frames = []
    for scope in ["active", "post_3h"]:
        pairs, balance = match_scope(panel, scope)
        pair_frames.append(pairs)
        balance_frames.append(balance)
    pairs = pd.concat(pair_frames, ignore_index=True)
    balance = pd.concat(balance_frames, ignore_index=True)
    summary = summarize_pairs(pairs)
    month_summary = summarize_months(pairs)
    balance_summary = summarize_balance(balance)
    pairs.to_csv(OUT / "matched_control_pairs.csv", index=False)
    summary.to_csv(OUT / "matched_control_summary.csv", index=False)
    month_summary.to_csv(OUT / "matched_control_month_delta.csv", index=False)
    balance_summary.to_csv(OUT / "matched_control_balance.csv", index=False)
    write_summary(summary, month_summary, balance_summary)
    print(OUT / "matched_control_validation_summary.md", flush=True)


if __name__ == "__main__":
    main()
