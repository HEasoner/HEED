"""k-robustness of the relative factual support s_sup for one data split (run once per seed 42-46,
then analysis/knn_robustness_plot.py aggregates the seeds)."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common import (
    DATASET_COLORS,
    DATASETS,
    FIG_DIR,
    MODELS,
    binary_metrics,
    case_iterator,
    feature_path,
    fit_standardized_pca,
    labels_np,
    load_pt,
    make_split,
    mean_by_label,
    relative_support_scores,
    save_png_pdf,
)
from _paths import RESULTS_DIR, resolve_output_path


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "model",
            "k",
            "know_mean_relative_support",
            "hall_mean_relative_support",
            "relative_factual_support_gap",
            "auroc",
            "aupr",
            "fpr95",
            "pca_components",
            "pca_explained_variance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Appendix kNN robustness for relative factual-support gap.")
    parser.add_argument("--k-values", default="5,10,20,50")
    parser.add_argument("--field", default="probe_target_hidden_mean_answer")
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-stem", default=None, help="default: figures/knn_per_seed/knn_robustness_s<seed>")
    parser.add_argument("--stats-csv", default=None, help="default: outputs/analysis/knn_robustness/s<seed>.csv")
    args = parser.parse_args()
    args.output_stem = args.output_stem or str(FIG_DIR / "knn_per_seed" / f"knn_robustness_s{args.seed}")
    args.stats_csv = args.stats_csv or str(RESULTS_DIR / "knn_robustness" / f"s{args.seed}.csv")

    k_values = [int(item.strip()) for item in args.k_values.split(",") if item.strip()]
    rows = []
    values_by_model = {model: {dataset: [] for dataset, _ in DATASETS} for model, _ in MODELS}

    for dataset, _, model, _ in case_iterator():
        path = feature_path(dataset, model)
        print(f">>> kNN robustness: dataset={dataset} model={model}", flush=True)
        data = load_pt(path)
        labels = labels_np(data)
        train_idx, _, test_idx = make_split(labels, args.seed)
        train_embed, test_embed, pca_info = fit_standardized_pca(
            data,
            args.field,
            train_idx,
            test_idx,
            args.pca_dim,
            args.seed,
        )
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        for k in k_values:
            support = relative_support_scores(train_embed, train_labels, test_embed, k, args.eps)
            rel = support["relative_support"]
            know_mean, hall_mean, gap = mean_by_label(rel, test_labels)
            metrics = binary_metrics(test_labels, -rel)
            row = {
                "dataset": dataset,
                "model": model,
                "k": k,
                "know_mean_relative_support": know_mean,
                "hall_mean_relative_support": hall_mean,
                "relative_factual_support_gap": gap,
                "auroc": metrics["auroc"],
                "aupr": metrics["aupr"],
                "fpr95": metrics["fpr95"],
                "pca_components": pca_info.get("pca_components", 0),
                "pca_explained_variance": pca_info.get("pca_explained_variance_ratio_sum", np.nan),
            }
            rows.append(row)
            values_by_model[model][dataset].append(gap)
        del data

    stats_path = resolve_output_path(args.stats_csv, RESULTS_DIR)
    write_csv(stats_path, rows)

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    for ax, (model, model_name) in zip(axes, MODELS):
        for dataset, dataset_name in DATASETS:
            ax.plot(
                k_values,
                values_by_model[model][dataset],
                marker="o",
                linewidth=1.8,
                color=DATASET_COLORS[dataset],
                label=dataset_name,
            )
        ax.axhline(0.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_title(model_name, fontsize=11)
        ax.grid(axis="y", alpha=0.22)
    for ax in axes[::3]:
        ax.set_ylabel("Relative factual-support gap\nKNOW mean - HALL mean")
    for ax in axes[-3:]:
        ax.set_xlabel("k in kNN reference")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("kNN robustness of relative factual-support gap", y=1.035, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png, pdf = save_png_pdf(fig, resolve_output_path(args.output_stem, FIG_DIR), args.dpi)
    plt.close(fig)
    print(f">>> wrote stats: {stats_path}", flush=True)
    print(f">>> wrote figure: {png}", flush=True)
    print(f">>> wrote figure: {pdf}", flush=True)


if __name__ == "__main__":
    main()
