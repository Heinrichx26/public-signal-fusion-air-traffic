from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


STRENGTH = _first_existing([
    ROOT / "results" / "experiments" / "fusion_framework_strengthening",
    ROOT.parent.parent.parent / "results" / "experiments" / "fusion_framework_strengthening",
])
SRC = STRENGTH / "residual_score_full_2025"
EVENT_RESPONSE = STRENGTH / "event_response_full_2025_30airports"
DEMAND = STRENGTH / "demand_residual_full_2025"
RELIABILITY = STRENGTH / "reliability_adjusted_score_full_2025"
OUT = ROOT / "results" / "figures"


def label_target(value: str) -> str:
    return "Long delay" if value == "long_arrival_delay" else "Cancellation"


def short_model(value: str) -> str:
    labels = {
        "weather_only": "Weather",
        "advisory_only": "Advisory",
        "weather_advisory": "Weather\n+ advisory",
        "schedule_demand_advisory": "Weather + advisory\n+ demand",
        "schedule_fused_state": "Fusion-state\ninteraction",
    }
    return labels.get(value, value)


def annotate_point(ax, x, y, label, *, dx=0, dy=8, color="#111111", ha="center", va="bottom") -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=7.2,
        color=color,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
        clip_on=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
    deciles = pd.read_csv(SRC / "residual_score_deciles.csv")
    calib = pd.read_csv(SRC / "residual_score_calibration.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axes[0]
    for target, group in deciles.groupby("target"):
        group = group.sort_values("score_decile")
        ax.plot(group["score_decile"], group["observed_residual"], marker="o", linewidth=1.6, label=label_target(target))
        if target == "long_arrival_delay":
            top = group[group["score_decile"] == 10].iloc[0]
            annotate_point(ax, top["score_decile"], top["observed_residual"], f"+{top['observed_residual']:.3f}", dx=5, dy=4, color="#D95F02", ha="left")
        elif target == "cancellation":
            top = group[group["score_decile"] == 10].iloc[0]
            annotate_point(ax, top["score_decile"], top["observed_residual"], f"+{top['observed_residual']:.3f}", dx=5, dy=4, color="#1F77B4", ha="left")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlim(0.8, 12.15)
    ax.set_ylim(-0.095, 0.205)
    ax.set_xlabel("Residual-score decile")
    ax.set_ylabel("Observed residual rate")
    ax.set_title("(a)Residual decile", loc="center", pad=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    ax = axes[1]
    use = calib[calib["target"] == "long_arrival_delay"].sort_values("risk_decile")
    ax.plot(use["predicted_rate"], use["observed_rate"], marker="o", linewidth=1.6, color="#C45A11")
    lo = min(use["predicted_rate"].min(), use["observed_rate"].min())
    hi = max(use["predicted_rate"].max(), use["observed_rate"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#777777", linewidth=0.9)
    upper = hi * 1.48
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    top = use[use["risk_decile"] == 10].iloc[0]
    annotate_point(ax, top["predicted_rate"], top["observed_rate"], f"{top['observed_rate']:.3f}", dx=5, dy=4, color="#C45A11", ha="left")
    ax.set_xlabel("Predicted long-delay rate")
    ax.set_ylabel("Observed long-delay rate")
    ax.set_title("(b)Risk calibration", loc="center", pad=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.09))

    fig.savefig(OUT / "fig7_residual_score_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    response = pd.read_csv(EVENT_RESPONSE / "event_response_curve.csv").sort_values("window_order")
    labels = ["Pre -3 to -1 h", "Pre -1 to 0 h", "Active", "Post 0 to 1 h", "Post 1 to 3 h", "Clean lag 3 to 6 h"]
    x = range(len(response))
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.errorbar(
        x,
        response["delay_diff"],
        yerr=[
            response["delay_diff"] - response["delay_ci_low"],
            response["delay_ci_high"] - response["delay_diff"],
        ],
        marker="o",
        linewidth=1.7,
        capsize=3,
        color="#1F5C99",
    )
    ax.set_xlim(-0.35, len(response) - 0.10)
    ax.set_ylim(-0.015, max(response["delay_ci_high"]) * 1.20)
    for idx, row in response.reset_index(drop=True).iterrows():
        label = f"{row['delay_diff']:+.3f}"
        if row["window"] == "pre_1h":
            annotate_point(ax, idx, row["delay_diff"], label, dx=7, dy=0, color="#111111", ha="left", va="center")
        elif row["window"] == "post_1h":
            annotate_point(ax, idx, row["delay_diff"], label, dx=5, dy=-15, color="#111111", ha="left", va="top")
        else:
            annotate_point(ax, idx, row["delay_diff"], label, dx=5, dy=4, color="#111111", ha="left")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylabel("Matched long-delay difference")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    fig.savefig(OUT / "fig8_event_response_function.pdf", bbox_inches="tight")
    plt.close(fig)

    ablation = pd.read_csv(DEMAND / "direct_ablation_matrix.csv").sort_values(["target", "ablation_order"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True, sharex=True)
    for ax, target in zip(axes, ["long_arrival_delay", "cancellation"]):
        use = ablation[ablation["target"] == target].copy()
        labels = [short_model(v).replace("\n", " ") for v in use["model"]]
        y = range(len(use))
        colors = ["#8FB3D9", "#D97941", "#D97941", "#5A9A68", "#2C6B4F"]
        bars = ax.barh(y, use["auc_gain_vs_weather"], color=colors, height=0.50)
        ax.axvline(0, color="#777777", linewidth=0.8)
        prefix = "(a)" if target == "long_arrival_delay" else "(b)"
        ax.set_title(f"{prefix}{label_target(target)}", loc="center", pad=8)
        ax.set_yticks(list(y), labels)
        ax.set_xlabel("AUC gain vs. weather-only")
        ax.set_xlim(0, 0.100)
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.45)
        for bar, val in zip(bars, use["auc_gain_vs_weather"]):
            if val > 0:
                ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"+{val:.3f}", ha="left", va="center", fontsize=7)
        ax.invert_yaxis()
    fig.savefig(OUT / "fig9_ablation_ladder.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = pd.read_csv(RELIABILITY / "reliability_adjusted_score_summary.csv")
    event = pd.read_csv(RELIABILITY / "reliability_adjusted_event_diff.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    ax = axes[0]
    x = [0, 1]
    width = 0.34
    for offset, target, color in [(-width / 2, "long_arrival_delay", "#1F5C99"), (width / 2, "cancellation", "#2C6B4F")]:
        vals = []
        for score in ["R", "QxR"]:
            vals.append(float(summary[(summary["ranking_score"] == score) & (summary["target"] == target)]["top_bottom_ratio"].iloc[0]))
        ax.bar([i + offset for i in x], vals, width=width, label=label_target(target), color=color)
        for i, val in zip(x, vals):
            ax.text(i + offset, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, ["R", r"$Q \times R$"])
    ax.set_ylim(0, max(summary["top_bottom_ratio"]) * 1.28)
    ax.set_ylabel("Top/bottom decile lift")
    ax.set_title("(a)Ranking lift", loc="center", pad=8)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)

    ax = axes[1]
    use = event[event["target"] == "long_arrival_delay"].copy()
    labels = ["R active", "R post", r"$Q \times R$ active", r"$Q \times R$ post"]
    vals = [
        float(use[(use["ranking_score"] == "R") & (use["window"] == "active")]["event_diff_top_decile"].iloc[0]),
        float(use[(use["ranking_score"] == "R") & (use["window"] == "post_3h")]["event_diff_top_decile"].iloc[0]),
        float(use[(use["ranking_score"] == "QxR") & (use["window"] == "active")]["event_diff_top_decile"].iloc[0]),
        float(use[(use["ranking_score"] == "QxR") & (use["window"] == "post_3h")]["event_diff_top_decile"].iloc[0]),
    ]
    bars = ax.bar(range(4), vals, color=["#8FB3D9", "#8FB3D9", "#1F5C99", "#1F5C99"], width=0.68)
    ax.set_ylim(0, max(vals) * 1.30)
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_ylabel("Top-decile event diff.")
    ax.set_title("(b)Long-delay event separation", loc="center", pad=8)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"+{val:.3f}", ha="center", va="bottom", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(OUT / "fig10_reliability_adjusted_ranking.pdf", bbox_inches="tight")
    plt.close(fig)
    print(OUT / "fig7_residual_score_calibration.pdf")
    print(OUT / "fig8_event_response_function.pdf")
    print(OUT / "fig9_ablation_ladder.pdf")
    print(OUT / "fig10_reliability_adjusted_ranking.pdf")


if __name__ == "__main__":
    main()
