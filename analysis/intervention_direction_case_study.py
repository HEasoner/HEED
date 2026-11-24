import argparse
import gc
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "heed_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from intervention_analyze import bootstrap_delta, exact_mcnemar, load_support_raw
from _paths import PT_DIR, RESULTS_DIR, feature_from_name, resolve_input_path
from intervention_formal import FORMAL_ROOT, PILOT_ROOT, PRIMARY_ALPHA, SETTINGS, rounded_layer_ids
from intervention_steering import (
    build_or_validate_split_manifest,
    choose_balanced_test_subset,
    label_ids,
    load_feature_file,
    load_layer_hidden_cache,
)


OUTPUT_DIR = RESULTS_DIR / "latent_support_intervention" / "case_study_direction"
COLORS = {"KNOW": "#387987", "HALL": "#bf6e40"}


def source_dir(setting: str) -> Path:
    return PILOT_ROOT if SETTINGS[setting].get("reuse_pilot") == "true" else FORMAL_ROOT / setting


def projection_cohens_d(projected: np.ndarray, labels: np.ndarray) -> float:
    know = projected[labels == 0]
    hall = projected[labels == 1]
    pooled = math.sqrt((know.var(ddof=1) + hall.var(ddof=1)) / 2)
    return float((know.mean() - hall.mean()) / pooled) if pooled > 0 else float("nan")


def intervention_metrics(setting: str, layer: int, selected: list[int], labels: np.ndarray) -> dict:
    raw_path = source_dir(setting) / "raw_results.csv"
    if not raw_path.exists():
        return {"delta_hallucination_rate": float("nan"), "damage_rate": float("nan"),
                "rescue_rate": float("nan"), "mcnemar_exact_p": float("nan"),
                "delta_ci_low": float("nan"), "delta_ci_high": float("nan")}
    frame = load_support_raw(FORMAL_ROOT, setting)
    rows = frame[
        (frame["layer"].astype(int) == layer)
        & np.isclose(frame["alpha"].astype(float), PRIMARY_ALPHA)
        & frame["pt_index"].astype(int).isin(selected)
    ].copy()
    if len(rows) != 100 or rows["pt_index"].duplicated().any():
        raise RuntimeError(f"Fixed primary intervention is incomplete at {setting}.")
    baseline = rows["baseline_correct"].to_numpy(dtype=bool)
    after = rows["intervention_correct"].to_numpy(dtype=bool)
    indexed = rows.set_index(rows["pt_index"].astype(int))
    if any(bool(indexed.loc[int(index), "baseline_correct"]) != bool(labels[int(index)] == 0) for index in selected):
        raise RuntimeError(f"Intervention baseline labels differ from PT labels at {setting}.")
    ci_low, ci_high = bootstrap_delta(baseline, after, 42, 2000)
    damage, rescue, p_value = exact_mcnemar(baseline, after)
    return {
        "delta_hallucination_rate": float((~after).mean() - (~baseline).mean()),
        "damage_rate": float((~after[baseline]).mean()),
        "rescue_rate": float(after[~baseline].mean()),
        "damage_count": damage,
        "rescue_count": rescue,
        "mcnemar_exact_p": p_value,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
    }


def analyze_hidden_setting(setting: str) -> tuple[dict, pd.DataFrame]:
    config = SETTINGS[setting]
    feature_path = feature_from_name(config["feature"])
    data = load_feature_file(feature_path)
    labels = label_ids(data)
    root = source_dir(setting)
    manifest = build_or_validate_split_manifest(data, feature_path, root, seed=42)
    selected = choose_balanced_test_subset(manifest, 100)
    train_indices = [int(index) for index in manifest["train_indices"]]
    saved = torch.load(root / "support_directions.pt", map_location="cpu")
    if saved["train_indices"].long().tolist() != train_indices:
        raise RuntimeError(f"Direction train split differs from manifest for {setting}.")
    layer = rounded_layer_ids(int(saved["num_hidden_layers"]), [0.8])[0][0]
    info = saved["layers"][layer]
    direction = info["support_direction"].float().numpy()
    if abs(np.linalg.norm(direction) - 1.0) > 1e-4:
        raise RuntimeError("Support direction is not unit norm.")
    cache = load_layer_hidden_cache(
        root / "cache", layer, int(saved["hidden_size"]), str(manifest["prompt_sha256"])
    )
    required = train_indices + selected
    if set(required) - set(cache):
        raise RuntimeError(f"Prompt hidden cache is missing {len(set(required) - set(cache))} vectors for {setting}.")
    train_hidden = np.stack([cache[index].float().numpy() for index in train_indices])
    test_hidden = np.stack([cache[int(index)].float().numpy() for index in selected])
    train_labels = labels[train_indices]
    test_labels = labels[selected]

    train_proj = train_hidden @ direction
    test_proj = test_hidden @ direction
    train_mean = float(train_proj.mean())
    train_std = float(train_proj.std(ddof=0))
    if train_std <= 0:
        raise RuntimeError("Degenerate train projection standard deviation.")
    normalized_projection = (test_proj - train_mean) / train_std
    cohen_d = projection_cohens_d(normalized_projection, test_labels)
    projection_auc = float(roc_auc_score(test_labels, -test_proj))

    scaler = StandardScaler().fit(train_hidden)
    train_scaled = scaler.transform(train_hidden)
    test_scaled = scaler.transform(test_hidden)
    pca = PCA(n_components=128, svd_solver="randomized", random_state=42).fit(train_scaled)
    train_pca = pca.transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    pca_scaler = StandardScaler().fit(train_pca)
    lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
    lr.fit(pca_scaler.transform(train_pca), train_labels)
    decodability_auc = float(roc_auc_score(test_labels, lr.predict_proba(pca_scaler.transform(test_pca))[:, 1]))

    intervention = intervention_metrics(setting, layer, selected, labels)
    item = {
        "setting": setting,
        "model": config["model_name"],
        "dataset": config["dataset_name"],
        "train_n": len(train_indices),
        "test_n": len(selected),
        "test_know_n": int((test_labels == 0).sum()),
        "test_hall_n": int((test_labels == 1).sum()),
        "layer": layer,
        "sigma_train": train_std,
        "projection_cohens_d_test": cohen_d,
        "projection_auroc_test": projection_auc,
        "linear_decodability_auroc_test": decodability_auc,
        **intervention,
    }
    projected = pd.DataFrame(
        {
            "setting": setting,
            "pt_index": selected,
            "label": np.where(test_labels == 0, "KNOW", "HALL"),
            "projection_train_standardized": normalized_projection,
            "projection_raw": test_proj,
        }
    )
    del data, cache, train_hidden, test_hidden, train_scaled, test_scaled
    gc.collect()
    return item, projected


def save_plot(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_projection_panels(projected: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    all_values = projected["projection_train_standardized"].to_numpy(dtype=float)
    low, high = np.quantile(all_values, [0.0025, 0.9975])
    extent = max(abs(float(low)), abs(float(high)))
    bins = np.linspace(-extent, extent, 38)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.7), sharex=True, sharey=True)
    for ax, setting in zip(axes.flat, SETTINGS):
        frame = projected[projected["setting"] == setting]
        for label in ("KNOW", "HALL"):
            values = frame.loc[frame["label"] == label, "projection_train_standardized"]
            ax.hist(values, bins=bins, density=True, alpha=0.5, color=COLORS[label],
                    edgecolor="none", label="HALLUCINATION" if label == "HALL" else label)
        item = summary.set_index("setting").loc[setting]
        ax.set_title(f"{SETTINGS[setting]['model_name']} / {SETTINGS[setting]['dataset_name']}\n"
                     f"test Cohen's d={item['projection_cohens_d_test']:.2f}", fontsize=10)
        ax.grid(axis="y", alpha=0.15)
    fig.supxlabel(r"Prefill prompt hidden projection onto train KNOW $-$ HALL direction (train-standardized)")
    fig.supylabel("Density")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    save_plot(fig, output_dir / "projection_distributions_2x3")


def plot_scatter(summary: pd.DataFrame, x: str, xlabel: str, stem: Path) -> None:
    valid = summary[summary["delta_hallucination_rate"].notna()]
    fig, ax = plt.subplots(figsize=(7.1, 5.2))
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    palette = {"PopQA": "#387987", "WebQuestions": "#bf6e40"}
    for item in valid.itertuples():
        color = palette[item.dataset]
        y = item.delta_hallucination_rate
        low = y - item.delta_ci_low
        high = item.delta_ci_high - y
        ax.errorbar(getattr(item, x), y, yerr=np.array([[low], [high]]), fmt="o",
                    color=color, capsize=3, markersize=7)
        short = {"Qwen2.5-7B-Instruct": "Qwen2.5", "Llama-3-8B-Instruct": "Llama3",
                 "Gemma-2-9B-it": "Gemma2"}[item.model]
        ax.annotate(f"{short} / {'PopQA' if item.dataset == 'PopQA' else 'WebQ'}",
                    (getattr(item, x), y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Fixed $\alpha=-1$: $\Delta$ hallucination rate")
    ax.grid(axis="y", alpha=0.17)
    ax.text(0.01, 0.01, "Gemma2 / WebQ excluded: exact baseline reproduction failed",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#555555")
    save_plot(fig, stem)


def analyze_matched_controls(output_dir: Path) -> dict | None:
    path = FORMAL_ROOT / "qwen25_popqa" / "control_alpha_minus1.csv"
    if not path.exists():
        return None
    controls = pd.read_csv(path)
    if len(controls) != 2000 or not (controls["alpha"].astype(float) == -1.0).all():
        raise RuntimeError("Matched alpha=-1 controls are incomplete.")
    support = load_support_raw(FORMAL_ROOT, "qwen25_popqa")
    support = support[
        (support["layer"].astype(int) == 22)
        & np.isclose(support["alpha"].astype(float), -1.0)
    ]
    if len(support) != 100:
        raise RuntimeError("Qwen2.5+PopQA fixed support output is incomplete.")
    rows = []
    for (kind, direction_id), group in controls.groupby(["direction_type", "direction_id"]):
        if len(group) != 100:
            raise RuntimeError("Matched control direction does not cover all 100 samples.")
        baseline = group["baseline_correct"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        after = group["intervention_correct"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        rows.append({
            "direction_type": kind,
            "direction_id": int(direction_id),
            "damage_rate": float((~after[baseline]).mean()),
            "rescue_rate": float(after[~baseline].mean()),
            "delta_hallucination_rate": float((~after).mean() - (~baseline).mean()),
        })
    control_summary = pd.DataFrame(rows)
    control_summary.to_csv(output_dir / "matched_control_per_direction.csv", index=False)
    support_damage = float((~support.loc[support["baseline_correct"], "intervention_correct"].to_numpy(dtype=bool)).mean())
    support_net_delta = float((~support["intervention_correct"].to_numpy(dtype=bool)).mean() -
                              (~support["baseline_correct"].to_numpy(dtype=bool)).mean())
    result = {
        "support_damage_rate": support_damage,
        "support_net_delta_hallucination_rate": support_net_delta,
        "pooled_20_direction_empirical_one_sided_p": float(
            (1 + np.sum(control_summary["damage_rate"].to_numpy() >= support_damage)) / 21
        ),
    }
    support_know = support[support["baseline_correct"]].set_index("pt_index")
    support_by_sample = (~support_know["intervention_correct"]).astype(float)
    know_controls = controls[
        controls["baseline_correct"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    know_controls["damaged"] = ~know_controls["intervention_correct"].astype(str).str.lower().isin(["true", "1"])
    rng = np.random.default_rng(42)
    for kind in ("random", "orthogonal"):
        values = control_summary.loc[control_summary["direction_type"] == kind, "damage_rate"].to_numpy()
        result[f"{kind}_damage_mean"] = float(values.mean())
        result[f"{kind}_damage_std"] = float(values.std(ddof=1))
        result[f"{kind}_empirical_one_sided_p"] = float((1 + np.sum(values >= support_damage)) / 11)
        net_values = control_summary.loc[control_summary["direction_type"] == kind, "delta_hallucination_rate"].to_numpy()
        result[f"{kind}_net_delta_mean"] = float(net_values.mean())
        result[f"{kind}_net_delta_std"] = float(net_values.std(ddof=1))
        per_sample = know_controls[know_controls["direction_type"] == kind].groupby("pt_index")["damaged"].mean()
        if set(per_sample.index) != set(support_by_sample.index) or len(per_sample) != 50:
            raise RuntimeError("Matched controls do not cover the same 50 baseline KNOW samples.")
        difference = (support_by_sample - per_sample.reindex(support_by_sample.index)).to_numpy()
        bootstrap = rng.choice(difference, size=(5000, len(difference)), replace=True).mean(axis=1)
        result[f"support_minus_{kind}_damage_mean"] = float(difference.mean())
        result[f"support_minus_{kind}_damage_ci_low"] = float(np.quantile(bootstrap, 0.025))
        result[f"support_minus_{kind}_damage_ci_high"] = float(np.quantile(bootstrap, 0.975))
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    for index, kind in enumerate(("random", "orthogonal")):
        values = control_summary.loc[control_summary["direction_type"] == kind, "damage_rate"].to_numpy()
        ax.scatter(np.full(len(values), index) + np.linspace(-0.12, 0.12, len(values)),
                   values, color="#4269a4" if kind == "random" else "#bf6e40", s=35, alpha=0.85)
        ax.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="s", color="#222222", capsize=5)
    ax.axhline(support_damage, color="#ba4a54", linewidth=1.8, label=f"Support direction ({support_damage:.2f})")
    ax.set_xticks([0, 1], ["10 random", "10 orthogonal"])
    ax.set_ylabel("KNOW → HALL damage rate at alpha=-1")
    ax.set_ylim(-0.02, max(0.30, float(control_summary["damage_rate"].max()) + 0.05))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.16)
    save_plot(fig, output_dir / "matched_controls_alpha_minus1")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanistic projection, decodability, and matched-control analysis.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    projections = []
    for setting in SETTINGS:
        print(f">>> analyzing {setting}", flush=True)
        item, projected = analyze_hidden_setting(setting)
        summaries.append(item)
        projections.append(projected)
    summary = pd.DataFrame(summaries)
    projected = pd.concat(projections, ignore_index=True)
    summary.to_csv(output_dir / "direction_projection_decodability.csv", index=False)
    projected.to_csv(output_dir / "test_projection_values.csv", index=False)
    plot_projection_panels(projected, summary, output_dir)
    plot_scatter(summary, "projection_cohens_d_test", "Test Cohen's d of support-direction projection",
                 output_dir / "projection_d_vs_steerability")
    plot_scatter(summary, "linear_decodability_auroc_test", "Train-fitted PCA128 + LR hidden decodability AUROC",
                 output_dir / "decodability_vs_steerability")
    controls = analyze_matched_controls(output_dir)

    lines = [
        "# Direction case study",
        "",
        "Projection direction, mean and SD are fit on train only; projection distributions and Cohen's d use the same balanced 100-sample test subset. Decodability uses train-fitted StandardScaler -> PCA128 -> StandardScaler -> LR (C=1, liblinear); no test tuning.",
        "Steerability uses the earlier fixed 0.8L, alpha=-1 intervention. The Qwen2.5+PopQA configuration was pilot-informed, not preregistered. Gemma2+WebQ fails exact no-hook reproduction and has no valid intervention delta.",
        "",
        "| Setting | Projection d | 1-D projection AUROC | Hidden LR AUROC | Delta hall @-1 | McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary.itertuples():
        delta = "NA" if math.isnan(item.delta_hallucination_rate) else f"{item.delta_hallucination_rate:+.2f}"
        p = "NA" if math.isnan(item.mcnemar_exact_p) else f"{item.mcnemar_exact_p:.3f}"
        lines.append(f"| {item.setting} | {item.projection_cohens_d_test:.2f} | "
                     f"{item.projection_auroc_test:.3f} | {item.linear_decodability_auroc_test:.3f} | {delta} | {p} |")
    lines += ["", "## Matched alpha=-1 controls", ""]
    valid = summary[summary["delta_hallucination_rate"].notna()]
    projection_r, projection_p = spearmanr(valid["projection_cohens_d_test"], valid["delta_hallucination_rate"])
    decoding_r, decoding_p = spearmanr(valid["linear_decodability_auroc_test"], valid["delta_hallucination_rate"])
    lines += [
        f"Across only {len(valid)} valid intervention settings, Spearman(projection d, delta hall)={projection_r:.3f}, p={projection_p:.3f}; Spearman(hidden LR AUROC, delta hall)={decoding_r:.3f}, p={decoding_p:.3f}. These five-point post-hoc associations are descriptive.",
        "",
    ]
    if controls is None:
        lines.append("Pending: 10 random and 10 orthogonal alpha=-1 directions. Existing +1 controls are not used as the matched comparison.")
    else:
        lines.append(f"Support KNOW damage={controls['support_damage_rate']:.3f}; "
                     f"random={controls['random_damage_mean']:.3f}±{controls['random_damage_std']:.3f}, "
                     f"orthogonal={controls['orthogonal_damage_mean']:.3f}±{controls['orthogonal_damage_std']:.3f}.")
        lines.append(
            f"Net hallucination-rate change: support={controls['support_net_delta_hallucination_rate']:+.3f}; "
            f"random={controls['random_net_delta_mean']:+.3f}±{controls['random_net_delta_std']:.3f}; "
            f"orthogonal={controls['orthogonal_net_delta_mean']:+.3f}±{controls['orthogonal_net_delta_std']:.3f}."
        )
        lines.append(f"One-sided empirical direction p-values: random={controls['random_empirical_one_sided_p']:.3f}, "
                     f"orthogonal={controls['orthogonal_empirical_one_sided_p']:.3f}; pooled 20={controls['pooled_20_direction_empirical_one_sided_p']:.3f}. With 10 directions per family, the minimum possible p is 1/11=0.091; pooled p is a coarse exploratory rank comparison.")
        for kind in ("random", "orthogonal"):
            lines.append(
                f"Paired bootstrap over the same 50 baseline KNOW samples (control directions fixed): support-minus-{kind} damage gap "
                f"{controls[f'support_minus_{kind}_damage_mean']:+.3f}, 95% CI "
                f"[{controls[f'support_minus_{kind}_damage_ci_low']:+.3f}, {controls[f'support_minus_{kind}_damage_ci_high']:+.3f}]."
            )
    lines += [
        "",
        "## Bounded manuscript interpretation",
        "",
        "In the pilot-informed Qwen2.5–PopQA case, weakening a train-derived factual-support direction caused a larger KNOW-to-HALL transition rate than equal-norm random and orthogonal perturbations. Across other settings, hidden-state knowledge-gap information remained decodable, but the same global centroid intervention had weak or inconsistent effects. Projection separation was highest in the responsive case, yet Gemma2–PopQA also had high separation without strong steerability. Thus linear separability is neither sufficient nor proven causal for intervention susceptibility.",
        "",
        "Do not infer linear organization causes steerability from a five-point post-hoc scatter. The control rank p-value is exploratory, and the sample-bootstrap gap for random directions includes zero.",
    ]
    (output_dir / "direction_case_study_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">>> outputs: {output_dir}")


if __name__ == "__main__":
    main()
