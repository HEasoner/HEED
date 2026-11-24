import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "heed_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pointbiserialr, t

from _paths import PROJECT_ROOT
from intervention_formal import (
    FORMAL_ROOT,
    PILOT_ROOT,
    PRIMARY_ALPHA,
    PRIMARY_RATIO,
    SETTINGS,
)


AGGREGATE_COLUMNS = [
    "setting",
    "model",
    "dataset",
    "n",
    "baseline_hallucination_rate",
    "intervened_hallucination_rate",
    "delta_hallucination_rate",
    "rescue_rate",
    "damage_rate",
    "net_transition",
    "rescue_count",
    "damage_count",
]
COLORS = {
    "hallucination_rate": "#b24c63",
    "damage_rate": "#9a5a36",
    "rescue_rate": "#3978a8",
}


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(["true", "1"])


def setting_dir(root: Path, setting: str) -> Path:
    return root / setting


def raw_path_for(root: Path, setting: str) -> Path:
    if SETTINGS[setting].get("reuse_pilot") == "true":
        return PILOT_ROOT / "raw_results.csv"
    return setting_dir(root, setting) / "raw_results.csv"


def load_support_raw(root: Path, setting: str) -> pd.DataFrame:
    path = raw_path_for(root, setting)
    if not path.exists():
        raise FileNotFoundError(f"Missing support sweep for {setting}: {path}")
    frame = pd.read_csv(path)
    frame = frame[frame["direction_type"] == "support"].copy()
    frame["baseline_correct"] = bool_series(frame["baseline_correct"])
    frame["intervention_correct"] = bool_series(frame["intervention_correct"])
    frame["setting"] = setting
    frame["model"] = SETTINGS[setting]["model_name"]
    frame["dataset"] = SETTINGS[setting]["dataset_name"]
    return frame


def transition_metrics(group: pd.DataFrame) -> Dict[str, float]:
    baseline = group["baseline_correct"].to_numpy(dtype=bool)
    intervention = group["intervention_correct"].to_numpy(dtype=bool)
    baseline_hall = ~baseline
    baseline_know = baseline
    rescue_count = int(np.sum(baseline_hall & intervention))
    damage_count = int(np.sum(baseline_know & ~intervention))
    rescue = float(rescue_count / max(int(baseline_hall.sum()), 1))
    damage = float(damage_count / max(int(baseline_know.sum()), 1))
    return {
        "n": int(len(group)),
        "baseline_hallucination_rate": float(baseline_hall.mean()),
        "intervened_hallucination_rate": float((~intervention).mean()),
        "delta_hallucination_rate": float((~intervention).mean() - baseline_hall.mean()),
        "rescue_rate": rescue,
        "damage_rate": damage,
        "net_transition": rescue - damage,
        "rescue_count": rescue_count,
        "damage_count": damage_count,
    }


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[
        np.isclose(frame["requested_layer_ratio"].astype(float), PRIMARY_RATIO)
        & np.isclose(frame["alpha"].astype(float), PRIMARY_ALPHA)
    ].copy()
    if rows["pt_index"].duplicated().any():
        raise RuntimeError("Primary configuration contains duplicate sample rows.")
    return rows


def bootstrap_delta(
    baseline: np.ndarray,
    intervention: np.ndarray,
    seed: int,
    resamples: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    per_sample = (~intervention).astype(float) - (~baseline).astype(float)
    samples = rng.choice(per_sample, size=(resamples, len(per_sample)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def exact_mcnemar(baseline: np.ndarray, intervention: np.ndarray) -> Tuple[int, int, float]:
    damage = int(np.sum(baseline & ~intervention))
    rescue = int(np.sum(~baseline & intervention))
    discordant = damage + rescue
    p_value = 1.0 if discordant == 0 else float(binomtest(damage, discordant, 0.5).pvalue)
    return damage, rescue, p_value


def primary_and_statistics(
    raw_by_setting: Dict[str, pd.DataFrame], bootstrap_resamples: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary_output = []
    statistical_output = []
    for setting in raw_by_setting:
        rows = primary_rows(raw_by_setting[setting])
        metrics = transition_metrics(rows)
        base = rows["baseline_correct"].to_numpy(dtype=bool)
        intervention = rows["intervention_correct"].to_numpy(dtype=bool)
        ci_low, ci_high = bootstrap_delta(
            base, intervention, seed + list(SETTINGS).index(setting), bootstrap_resamples
        )
        damage, rescue, p_value = exact_mcnemar(base, intervention)
        common = {
            "setting": setting,
            "model": SETTINGS[setting]["model_name"],
            "dataset": SETTINGS[setting]["dataset_name"],
        }
        primary_output.append({**common, **metrics})
        statistical_output.append(
            {
                **common,
                "n": len(rows),
                "delta_hallucination_rate": metrics["delta_hallucination_rate"],
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "mcnemar_damage_discordant": damage,
                "mcnemar_rescue_discordant": rescue,
                "mcnemar_exact_p": p_value,
                "bootstrap_resamples": bootstrap_resamples,
            }
        )
    return pd.DataFrame(primary_output)[AGGREGATE_COLUMNS], pd.DataFrame(statistical_output)


def control_results(root: Path, primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary_by_setting = primary.set_index("setting")
    for setting in primary["setting"]:
        path = setting_dir(root, setting) / "control_raw_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing controls for {setting}: {path}")
        frame = pd.read_csv(path)
        frame["baseline_correct"] = bool_series(frame["baseline_correct"])
        frame["intervention_correct"] = bool_series(frame["intervention_correct"])
        per_direction = []
        for (direction_type, direction_id), group in frame.groupby(
            ["direction_type", "direction_id"]
        ):
            per_direction.append(
                {
                    "direction_type": direction_type,
                    "direction_id": int(direction_id),
                    **transition_metrics(group),
                }
            )
        directions = pd.DataFrame(per_direction)
        support_damage = float(primary_by_setting.loc[setting, "damage_rate"])
        support_delta = float(primary_by_setting.loc[setting, "delta_hallucination_rate"])
        output = {
            "setting": setting,
            "model": SETTINGS[setting]["model_name"],
            "dataset": SETTINGS[setting]["dataset_name"],
            "support_delta_hallucination_rate": support_delta,
            "support_damage_rate": support_damage,
        }
        for direction_type in ["random", "orthogonal"]:
            values = directions[directions["direction_type"] == direction_type]
            if len(values) != 10:
                raise RuntimeError(
                    f"Expected 10 {direction_type} directions for {setting}, got {len(values)}."
                )
            damage_values = values["damage_rate"].to_numpy(dtype=float)
            delta_values = values["delta_hallucination_rate"].to_numpy(dtype=float)
            output.update(
                {
                    f"{direction_type}_damage_mean": float(damage_values.mean()),
                    f"{direction_type}_damage_std": float(damage_values.std(ddof=1)),
                    f"{direction_type}_delta_hall_mean": float(delta_values.mean()),
                    f"{direction_type}_delta_hall_std": float(delta_values.std(ddof=1)),
                    f"support_minus_{direction_type}_damage": float(
                        support_damage - damage_values.mean()
                    ),
                    f"support_damage_empirical_p_vs_{direction_type}": float(
                        (1 + np.sum(damage_values >= support_damage)) / (len(values) + 1)
                    ),
                }
            )
        rows.append(output)
    return pd.DataFrame(rows)


def collect_support_change(root: Path, settings: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for setting in settings:
        path = setting_dir(root, setting) / "support_change.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing support analysis for {setting}: {path}")
        frame = pd.read_csv(path)
        frame["setting"] = setting
        frame["model"] = SETTINGS[setting]["model_name"]
        frame["dataset"] = SETTINGS[setting]["dataset_name"]
        frame["baseline_correct"] = bool_series(frame["baseline_correct"])
        frame["intervention_correct"] = bool_series(frame["intervention_correct"])
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    summaries = []
    for setting, frame in all_rows.groupby("setting", sort=False):
        binary_hall = (~frame["intervention_correct"].to_numpy(dtype=bool)).astype(int)
        delta = frame["delta_support_score"].to_numpy(dtype=float)
        correlation, p_value = pointbiserialr(binary_hall, delta)
        item = {
            "setting": setting,
            "model": frame["model"].iloc[0],
            "dataset": frame["dataset"].iloc[0],
            "n": len(frame),
            "mean_delta_support": float(delta.mean()),
            "median_delta_support": float(np.median(delta)),
            "fraction_delta_support_negative": float(np.mean(delta < 0)),
            "mean_delta_projection": float(frame["delta_support_projection"].mean()),
            "point_biserial_delta_support_vs_hall": float(correlation),
            "point_biserial_p": float(p_value),
        }
        for transition in ["KNOW -> KNOW", "KNOW -> HALL", "HALL -> HALL", "HALL -> KNOW"]:
            values = frame.loc[
                frame["transition_type"] == transition, "delta_support_score"
            ].to_numpy(dtype=float)
            key = transition.lower().replace(" -> ", "_to_")
            item[f"{key}_n"] = int(len(values))
            item[f"{key}_mean_delta_support"] = (
                float(values.mean()) if len(values) else float("nan")
            )
        summaries.append(item)
    return all_rows, pd.DataFrame(summaries)


def collect_generation_quality(root: Path, settings: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for setting in settings:
        path = setting_dir(root, setting) / "generation_quality_raw.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing generation quality for {setting}: {path}")
        frame = pd.read_csv(path)
        frame["setting"] = setting
        frame["model"] = SETTINGS[setting]["model_name"]
        frame["dataset"] = SETTINGS[setting]["dataset_name"]
        for column in [
            "intervention_correct",
            "empty_response",
            "malformed_response",
            "repetition_indicator",
        ]:
            frame[column] = bool_series(frame[column])
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    output = []
    for (setting, alpha), frame in all_rows.groupby(["setting", "alpha"], sort=False):
        output.append(
            {
                "setting": setting,
                "model": frame["model"].iloc[0],
                "dataset": frame["dataset"].iloc[0],
                "alpha": float(alpha),
                "n_baseline_know": len(frame),
                "accuracy": float(frame["intervention_correct"].mean()),
                "mean_token_length": float(frame["token_length"].mean()),
                "empty_response_rate": float(frame["empty_response"].mean()),
                "malformed_response_rate": float(frame["malformed_response"].mean()),
                "repetition_rate": float(frame["repetition_indicator"].mean()),
                "mean_generated_token_log_probability": float(
                    frame["average_generated_token_log_probability"].mean()
                ),
            }
        )
    return all_rows, pd.DataFrame(output)


def setting_sweep_metrics(raw_by_setting: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for setting, frame in raw_by_setting.items():
        for (layer, ratio, alpha), group in frame.groupby(
            ["layer", "requested_layer_ratio", "alpha"]
        ):
            rows.append(
                {
                    "setting": setting,
                    "model": SETTINGS[setting]["model_name"],
                    "dataset": SETTINGS[setting]["dataset_name"],
                    "layer": int(layer),
                    "layer_ratio": float(ratio),
                    "alpha": float(alpha),
                    **transition_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dose_response_plot(
    sweep: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    primary_layer = sweep[np.isclose(sweep["layer_ratio"], PRIMARY_RATIO)].copy()
    rows = []
    for setting, frame in primary_layer.groupby("setting"):
        baseline = float(frame.loc[np.isclose(frame["alpha"], 0.0), metric].iloc[0])
        for row in frame.itertuples():
            rows.append(
                {"setting": setting, "alpha": row.alpha, "delta": getattr(row, metric) - baseline}
            )
    deltas = pd.DataFrame(rows)
    alphas = sorted(deltas["alpha"].unique())
    summary = [mean_ci(deltas.loc[np.isclose(deltas["alpha"], alpha), "delta"]) for alpha in alphas]
    means = np.asarray([item[0] for item in summary])
    lows = np.asarray([item[1] for item in summary])
    highs = np.asarray([item[2] for item in summary])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.plot(alphas, means, color=COLORS[metric], marker="o", linewidth=1.8)
    ax.fill_between(alphas, lows, highs, color=COLORS[metric], alpha=0.18)
    ax.set_xlabel(r"Intervention strength $\alpha$")
    ax.set_ylabel(ylabel + " relative to alpha=0")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_stem)


def support_change_plot(rows: pd.DataFrame, output_stem: Path) -> None:
    groups = ["KNOW -> KNOW", "KNOW -> HALL", "HALL -> HALL", "HALL -> KNOW"]
    values = [
        rows.loc[rows["transition_type"] == group, "delta_support_score"].to_numpy(dtype=float)
        for group in groups
    ]
    present = [(group, value) for group, value in zip(groups, values) if len(value)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    boxes = ax.boxplot(
        [value for _, value in present],
        labels=[group.replace(" -> ", "\n→\n") for group, _ in present],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )
    palette = ["#6f9e78", "#b24c63", "#8b7eb8", "#4f86a6"]
    for patch, color in zip(boxes["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"$\Delta$ relative factual-support score")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_stem)


def layer_robustness_plot(sweep: pd.DataFrame, output_stem: Path) -> pd.DataFrame:
    rows = sweep[np.isclose(sweep["alpha"], -1.0)].copy()
    output = []
    for ratio, group in rows.groupby("layer_ratio"):
        values = group["delta_hallucination_rate"].to_numpy(dtype=float)
        mean, low, high = mean_ci(values)
        output.append(
            {
                "layer_ratio": ratio,
                "settings": len(values),
                "mean_delta_hallucination_rate": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    frame = pd.DataFrame(output).sort_values("layer_ratio")
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    errors = np.vstack(
        [
            frame["mean_delta_hallucination_rate"] - frame["ci_low"],
            frame["ci_high"] - frame["mean_delta_hallucination_rate"],
        ]
    )
    ax.errorbar(
        frame["layer_ratio"],
        frame["mean_delta_hallucination_rate"],
        yerr=errors,
        color="#6f4e7c",
        marker="o",
        capsize=4,
        linewidth=1.7,
    )
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer ratio")
    ax.set_ylabel(r"Mean $\Delta$ hallucination rate")
    ax.set_xticks(frame["layer_ratio"])
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_stem)
    return frame


def latex_escape(value: object) -> str:
    text = str(value)
    for source, target in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        text = text.replace(source, target)
    return text


def write_primary_tables(primary: pd.DataFrame, aggregate_dir: Path) -> None:
    mean_values = primary.select_dtypes(include=[np.number]).mean()
    md = [
        "| Model | Dataset | Base Hall. | -Support Hall. | Delta Hall. | Rescue | Damage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Primary intervention results at layer $0.8L$ and $\alpha=-1.0$.}",
        r"\label{tab:latent-support-primary}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Dataset & Base Hall. & $-$Support Hall. & $\Delta$Hall. & Rescue & Damage \\",
        r"\midrule",
    ]
    for row in primary.itertuples():
        values = [
            row.baseline_hallucination_rate,
            row.intervened_hallucination_rate,
            row.delta_hallucination_rate,
            row.rescue_rate,
            row.damage_rate,
        ]
        md.append(
            f"| {row.model} | {row.dataset} | " + " | ".join(f"{value:.3f}" for value in values) + " |"
        )
        latex.append(
            f"{latex_escape(row.model)} & {latex_escape(row.dataset)} & "
            + " & ".join(f"{value:.3f}" for value in values)
            + r" \\"
        )
    mean_row = [
        mean_values["baseline_hallucination_rate"],
        mean_values["intervened_hallucination_rate"],
        mean_values["delta_hallucination_rate"],
        mean_values["rescue_rate"],
        mean_values["damage_rate"],
    ]
    increased = int((primary["delta_hallucination_rate"] > 0).sum())
    md.extend(
        [
            "| **Mean** | - | " + " | ".join(f"{value:.3f}" for value in mean_row) + " |",
            "",
            f"Hallucination rate increased in **{increased}/{len(primary)}** completed settings.",
        ]
    )
    latex.extend(
        [
            r"\midrule",
            r"Mean & -- & " + " & ".join(f"{value:.3f}" for value in mean_row) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            f"% Hallucination rate increased in {increased}/{len(primary)} completed settings.",
        ]
    )
    (aggregate_dir / "primary_results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (aggregate_dir / "primary_results_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def write_control_tables(control: pd.DataFrame, aggregate_dir: Path) -> None:
    md = [
        "| Model | Dataset | Support Damage | Random Damage | Orthogonal Damage |",
        "|---|---|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Direction-specificity controls at layer $0.8L$ and unit intervention magnitude.}",
        r"\label{tab:latent-support-controls}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Dataset & Support Damage & Random Damage & Orthogonal Damage \\",
        r"\midrule",
    ]
    for row in control.itertuples():
        random_text = f"{row.random_damage_mean:.3f} ± {row.random_damage_std:.3f}"
        orthogonal_text = f"{row.orthogonal_damage_mean:.3f} ± {row.orthogonal_damage_std:.3f}"
        md.append(
            f"| {row.model} | {row.dataset} | {row.support_damage_rate:.3f} | "
            f"{random_text} | {orthogonal_text} |"
        )
        latex.append(
            f"{latex_escape(row.model)} & {latex_escape(row.dataset)} & "
            f"{row.support_damage_rate:.3f} & "
            f"${row.random_damage_mean:.3f} \\pm {row.random_damage_std:.3f}$ & "
            f"${row.orthogonal_damage_mean:.3f} \\pm {row.orthogonal_damage_std:.3f}$ \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (aggregate_dir / "control_results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (aggregate_dir / "control_results_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def select_examples(
    raw_by_setting: Dict[str, pd.DataFrame], support_rows: pd.DataFrame, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outputs = []
    support_lookup = support_rows.set_index(["setting", "pt_index"])
    for setting in raw_by_setting:
        primary = primary_rows(raw_by_setting[setting])
        for transition, limit in [("KNOW -> HALL", 5), ("HALL -> KNOW", 3)]:
            candidates = primary[primary["transition_type"] == transition].sort_values("pt_index")
            if candidates.empty:
                continue
            selected = candidates.iloc[rng.permutation(len(candidates))[:limit]]
            for row in selected.itertuples():
                support = support_lookup.loc[(setting, int(row.pt_index))]
                outputs.append(
                    {
                        "setting": setting,
                        "model": SETTINGS[setting]["model_name"],
                        "dataset": SETTINGS[setting]["dataset_name"],
                        "selection_rule": f"seed={seed} random among valid {transition}",
                        "transition_type": transition,
                        "sample_id": row.sample_id,
                        "pt_index": int(row.pt_index),
                        "question": row.question,
                        "reference_answers": row.reference_answers,
                        "baseline_answer": row.baseline_answer,
                        "intervention_answer": row.intervened_answer,
                        "baseline_label": row.baseline_label,
                        "intervention_label": row.intervened_label,
                        "layer": int(row.layer),
                        "alpha": float(row.alpha),
                        "support_score_before": float(support["support_score_before"]),
                        "support_score_after": float(support["support_score_after"]),
                        "delta_support_score": float(support["delta_support_score"]),
                    }
                )
    return pd.DataFrame(outputs)


def write_summary(
    primary: pd.DataFrame,
    statistics: pd.DataFrame,
    controls: pd.DataFrame,
    support_summary: pd.DataFrame,
    quality: pd.DataFrame,
    sweep: pd.DataFrame,
    aggregate_dir: Path,
    missing_settings: Sequence[str],
) -> None:
    n_settings = len(primary)
    increased = int((primary["delta_hallucination_rate"] > 0).sum())
    mean_delta = float(primary["delta_hallucination_rate"].mean())
    mean_damage = float(primary["damage_rate"].mean())
    significant_increase = int(
        ((statistics["mcnemar_exact_p"] < 0.05) & (statistics["delta_hallucination_rate"] > 0)).sum()
    )
    ci_positive = int((statistics["bootstrap_95_ci_low"] > 0).sum())
    stronger_random = int((controls["support_minus_random_damage"] > 0).sum())
    stronger_orthogonal = int((controls["support_minus_orthogonal_damage"] > 0).sum())
    support_negative = int((support_summary["mean_delta_support"] < 0).sum())
    know_hall_larger = int(
        (
            support_summary["know_to_hall_mean_delta_support"]
            < support_summary["know_to_know_mean_delta_support"]
        ).fillna(False).sum()
    )

    quality_pivot = quality.pivot(index="setting", columns="alpha")
    empty_change = (
        quality_pivot["empty_response_rate"][-1.0] - quality_pivot["empty_response_rate"][0.0]
    )
    malformed_change = (
        quality_pivot["malformed_response_rate"][-1.0]
        - quality_pivot["malformed_response_rate"][0.0]
    )
    repetition_change = (
        quality_pivot["repetition_rate"][-1.0] - quality_pivot["repetition_rate"][0.0]
    )
    collapse_signal = bool(
        (empty_change.mean() > 0.05)
        or (malformed_change.mean() > 0.05)
        or (repetition_change.mean() > 0.05)
    )

    positive = sweep[
        np.isclose(sweep["layer_ratio"], PRIMARY_RATIO) & (sweep["alpha"] > 0)
    ]
    positive_by_setting = []
    for setting, frame in positive.groupby("setting"):
        positive_by_setting.append(
            float(frame.sort_values("alpha").iloc[-1]["delta_hallucination_rate"])
        )
    positive_improved = int(np.sum(np.asarray(positive_by_setting) < 0))

    systematic = increased >= 5 and mean_delta > 0 and n_settings == 6
    direction_specific = stronger_random >= 4 and stronger_orthogonal >= 4 and n_settings == 6
    support_verified = support_negative >= 5 and n_settings == 6
    claim_supported = systematic and support_verified and not collapse_signal and not missing_settings
    if claim_supported:
        english = (
            "Controlled interventions provide cross-setting functional evidence for an "
            "asymmetric role of latent factual support. At the fixed layer and intervention "
            f"strength, weakening the learned support component increased hallucination in "
            f"{increased} of {n_settings} settings (mean change {mean_delta:.3f}). The intervention "
            f"reduced the measured pre-generation factual-support score in {support_negative} "
            "settings. Positive support injection was not a reliable recovery mechanism, "
            "which is consistent with generic support being unable to supply missing "
            "query-specific factual content."
        )
    else:
        english = (
            "The fixed intervention produced mixed or incomplete evidence for the proposed "
            "functional role of latent factual support. Weakening the support component "
            f"increased hallucination in {increased} of {n_settings} completed settings (mean change "
            f"{mean_delta:.3f}), and reduced the measured support score in "
            f"{support_negative} settings. These results should be reported as a bounded "
            "intervention study rather than as general causal proof of latent voids."
        )

    lines = [
        "# Formal Latent-Support Intervention Summary",
        "",
        "The primary configuration `layer=0.8L, alpha=-1.0` was fixed after the Qwen2.5+PopQA pilot, not preregistered; no per-setting best result is used.",
        f"Completed settings: **{n_settings}/6**. Missing: {', '.join(missing_settings) if missing_settings else 'none'}.",
        "",
        f"1. Hallucination rate increased in **{increased}/{n_settings} completed** settings.",
        f"2. Mean delta hallucination rate across settings was **{mean_delta:+.3f}**.",
        f"3. Mean KNOW-to-HALL damage rate was **{mean_damage:.3f}**.",
        f"4. McNemar was significant in the increasing direction in **{significant_increase}/{n_settings}** settings; the bootstrap CI was strictly above zero in **{ci_positive}/{n_settings}** settings.",
        f"5. Support damage exceeded the random mean in **{stronger_random}/{n_settings}** settings and the orthogonal mean in **{stronger_orthogonal}/{n_settings}** settings.",
        f"6. Mean kNN factual-support score decreased in **{support_negative}/{n_settings}** settings.",
        f"7. KNOW-to-HALL had a more negative mean support change than KNOW-to-KNOW in **{know_hall_larger}/{n_settings}** settings with both groups available.",
        f"8. The fixed collapse indicators show {'a possible aggregate collapse signal' if collapse_signal else 'no marked aggregate generation-collapse signal'}; inspect `generation_quality.csv` for effect sizes.",
        f"9. At alpha=+1.5, hallucination decreased relative to alpha=0 in **{positive_improved}/{n_settings}** settings; positive injection was {'not stable' if positive_improved < 5 else 'mostly consistent'}.",
        f"10. The target asymmetric claim is **{'supported at the tested scope' if claim_supported else 'not fully supported across all required checks'}**.",
        "",
        "## Direction specificity",
        "",
        f"The fixed support effect was stronger than both control means in "
        f"{min(stronger_random, stronger_orthogonal)}/{n_settings} completed settings. Therefore direction "
        f"specificity is {'broadly supported' if direction_specific else 'mixed'}.",
        "",
        "## Recommended paper claim",
        "",
        english,
        "",
        "## Claims that should NOT be made",
        "",
        "- The experiment does not prove that latent voids cause hallucinations.",
        "- Positive support steering should not be described as factual correction unless it consistently rescues answers.",
        "- The sweep must not be used to replace the fixed primary configuration with per-setting best layers or strengths.",
        "- Random or orthogonal controls must not be omitted when discussing direction specificity.",
        "",
        "## Generation-quality definitions",
        "",
        "- Empty: cleaned generated answer contains no non-whitespace characters.",
        "- Malformed: empty or contains no alphanumeric characters.",
        "- Repetition: at least one generated token trigram occurs more than once.",
    ]
    (aggregate_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (aggregate_dir / "paper_summary_en.txt").write_text(english + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the formal latent-support experiment.")
    parser.add_argument("--root", default=str(FORMAL_ROOT))
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Report completed settings and explicitly disclose incomplete settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    aggregate_dir = root / "aggregate"
    figures_dir = root / "figures"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    complete_settings = []
    missing_settings = []
    for setting in SETTINGS:
        sweep_path = raw_path_for(root, setting)
        setting_root = setting_dir(root, setting)
        control_path = setting_root / "control_raw_results.csv"
        support_path = setting_root / "support_change.csv"
        quality_path = setting_root / "generation_quality_raw.csv"
        if not all(path.exists() for path in [sweep_path, control_path, support_path, quality_path]):
            missing_settings.append(setting)
            continue
        expected_rows = {"sweep": 2800, "control": 2000, "support": 100, "quality": 100}
        actual_rows = {
            "sweep": int((pd.read_csv(sweep_path)["direction_type"] == "support").sum()),
            "control": len(pd.read_csv(control_path)),
            "support": len(pd.read_csv(support_path)),
            "quality": len(pd.read_csv(quality_path)),
        }
        if actual_rows != expected_rows:
            missing_settings.append(setting)
        else:
            complete_settings.append(setting)
    if missing_settings and not args.allow_partial:
        raise RuntimeError("Formal aggregate is incomplete: " + ", ".join(missing_settings))
    if not complete_settings:
        raise RuntimeError("No complete setting is available for aggregate analysis.")
    raw_by_setting = {setting: load_support_raw(root, setting) for setting in complete_settings}
    all_raw = pd.concat(raw_by_setting.values(), ignore_index=True)
    all_raw.to_csv(aggregate_dir / "all_raw_results.csv", index=False)
    primary, statistics = primary_and_statistics(
        raw_by_setting, args.bootstrap_resamples, args.seed
    )
    primary.to_csv(aggregate_dir / "primary_results.csv", index=False)
    statistics.to_csv(aggregate_dir / "statistical_tests.csv", index=False)
    controls = control_results(root, primary)
    controls.to_csv(aggregate_dir / "control_results.csv", index=False)
    support_rows, support_summary = collect_support_change(root, complete_settings)
    support_rows.to_csv(aggregate_dir / "support_change.csv", index=False)
    support_summary.to_csv(aggregate_dir / "support_change_summary.csv", index=False)
    quality_rows, quality = collect_generation_quality(root, complete_settings)
    quality_rows.to_csv(aggregate_dir / "generation_quality_raw.csv", index=False)
    quality.to_csv(aggregate_dir / "generation_quality.csv", index=False)
    examples = select_examples(raw_by_setting, support_rows, args.seed)
    examples.to_csv(aggregate_dir / "transition_examples.csv", index=False)
    sweep = setting_sweep_metrics(raw_by_setting)
    sweep.to_csv(aggregate_dir / "sweep_metrics.csv", index=False)

    dose_response_plot(
        sweep,
        "hallucination_rate",
        "Hallucination rate",
        figures_dir / "dose_response_hallucination_rate",
    )
    dose_response_plot(
        sweep,
        "damage_rate",
        "Damage rate",
        figures_dir / "dose_response_damage_rate",
    )
    dose_response_plot(
        sweep,
        "rescue_rate",
        "Rescue rate",
        figures_dir / "dose_response_rescue_rate",
    )
    support_change_plot(support_rows, figures_dir / "support_change_by_transition")
    layer_robustness = layer_robustness_plot(
        sweep, figures_dir / "layer_robustness_delta_hallucination"
    )
    layer_robustness.to_csv(aggregate_dir / "layer_robustness.csv", index=False)

    write_primary_tables(primary, aggregate_dir)
    write_control_tables(controls, aggregate_dir)
    write_summary(
        primary,
        statistics,
        controls,
        support_summary,
        quality,
        sweep,
        aggregate_dir,
        missing_settings,
    )
    print(f">>> Aggregate outputs: {aggregate_dir}")


if __name__ == "__main__":
    main()
