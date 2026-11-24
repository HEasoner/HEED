import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from _paths import FIG_DIR, PROJECT_ROOT, PT_DIR, feature_file, resolve_input_path, resolve_output_path


DEFAULT_FIELD = "probe_target_hidden_mean_answer"

DATASETS: List[Tuple[str, str]] = [
    ("popqa", "PopQA"),
    ("webquestions", "WebQuestions"),
    ("triviaqa", "TriviaQA"),
    ("mkqa", "MKQA"),
]

MODELS: List[Tuple[str, str]] = [
    ("llama2_7b_chat", "Llama-2-7B-Chat"),
    ("llama3_8b_instruct", "Llama-3-8B-Instruct"),
    ("qwen2_7b_instruct", "Qwen2-7B-Instruct"),
    ("qwen2_5_7b_instruct", "Qwen2.5-7B-Instruct"),
    ("gemma_2_9b_it", "Gemma-2-9B-it"),
    ("deepseek_7b_chat", "DeepSeek-LLM-7B-Chat"),
]

KNOW_COLOR = "#2c7fb8"
HALL_COLOR = "#7e22ce"
VOID_COLOR = "#d81b60"
ENVELOPE_90 = "#266b4a"
ENVELOPE_95 = "#111111"
DEFICIENCY_CMAP = LinearSegmentedColormap.from_list(
    "factual_support_deficiency",
    [
        "#fff7bc",
        "#c7e9b4",
        "#7fcdbb",
        "#41b6c4",
        "#2c7fb8",
        "#253494",
    ],
)


def parse_pairs(value: str, defaults: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not value:
        return defaults
    default_map = {slug: label for slug, label in defaults}
    pairs = []
    for item in value.split(","):
        slug = item.strip()
        if slug:
            pairs.append((slug, default_map.get(slug, slug)))
    return pairs


def make_split(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=labels[temp_idx],
    )
    return train_idx, val_idx, test_idx


def stratified_cap(indices: np.ndarray, labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    if max_per_class <= 0:
        return indices
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        cls_idx = indices[labels[indices] == label]
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        selected.append(cls_idx)
    capped = np.concatenate(selected)
    rng.shuffle(capped)
    return capped


def flatten_subset(hidden: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    idx = torch.from_numpy(indices).long()
    subset = hidden[idx].float()
    return torch.nan_to_num(
        subset.reshape(subset.shape[0], -1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def mean_knn_distance(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    if reference.shape[0] < 1:
        raise ValueError("At least one KNOW reference sample is required.")
    k_eff = min(k, reference.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", algorithm="auto")
    nn.fit(reference)
    distances, _ = nn.kneighbors(query, return_distance=True)
    return distances.mean(axis=1).astype(np.float32)


def leave_one_out_knn_distance(reference: np.ndarray, k: int) -> np.ndarray:
    if reference.shape[0] < 2:
        raise ValueError("At least two KNOW reference samples are required for leave-one-out kNN.")
    k_eff = min(k + 1, reference.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", algorithm="auto")
    nn.fit(reference)
    distances, _ = nn.kneighbors(reference, return_distance=True)
    distances = distances[:, 1:] if distances.shape[1] > 1 else distances
    return distances.mean(axis=1).astype(np.float32)


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr = tp / max(positives, 1.0)
    fpr = fp / max(negatives, 1.0)
    hits = np.where(tpr >= target_tpr)[0]
    if len(hits) == 0:
        return float("nan")
    return float(fpr[hits[0]])


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr_at_tpr(labels, scores),
    }


def project_case(dataset: str, model_slug: str, args: argparse.Namespace) -> Dict[str, object]:
    path = resolve_input_path(feature_file(dataset, model_slug))
    if not path.exists():
        raise FileNotFoundError(path)

    print(f">>> factual support deficiency: dataset={dataset} model={model_slug} file={path.name}", flush=True)
    try:
        data = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        data = torch.load(str(path), map_location="cpu")

    if args.field not in data:
        raise KeyError(f"{args.field} is missing from {path}")

    labels = data["labels"].argmax(dim=1).long().numpy()
    train_idx, _, test_idx = make_split(labels, args.seed)
    test_idx = stratified_cap(test_idx, labels, args.max_test_per_class, args.seed + 29)

    hidden = data[args.field]
    train_hidden = flatten_subset(hidden, train_idx)
    test_hidden = flatten_subset(hidden, test_idx)

    mean = train_hidden.mean(dim=0, keepdim=True)
    std = train_hidden.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_hidden = torch.nan_to_num((train_hidden - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    test_hidden = torch.nan_to_num((test_hidden - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    train_np = train_hidden.numpy().astype(np.float32, copy=False)
    test_np = test_hidden.numpy().astype(np.float32, copy=False)
    pca_dim = min(args.pca_dim, train_np.shape[0] - 1, train_np.shape[1])
    if pca_dim < 2:
        raise ValueError(f"PCA dimension must be at least 2, got {pca_dim}")

    pca128 = PCA(n_components=pca_dim, svd_solver="randomized", random_state=args.seed)
    train_pca128 = pca128.fit_transform(train_np).astype(np.float32)
    test_pca128 = pca128.transform(test_np).astype(np.float32)

    pca2 = PCA(n_components=2, random_state=args.seed)
    train_2d = pca2.fit_transform(train_pca128).astype(np.float32)
    test_2d = pca2.transform(test_pca128).astype(np.float32)

    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    know_ref = train_2d[train_labels == 0]

    all_points = np.concatenate([train_2d, test_2d], axis=0)
    low_q = max(0.0, (1.0 - args.extent_quantile) / 2.0)
    high_q = min(1.0, 1.0 - low_q)
    x_min, x_max = np.quantile(all_points[:, 0], [low_q, high_q])
    y_min, y_max = np.quantile(all_points[:, 1], [low_q, high_q])
    x_pad = max((x_max - x_min) * args.extent_pad, 1e-4)
    y_pad = max((y_max - y_min) * args.extent_pad, 1e-4)
    x_min, x_max = float(x_min - x_pad), float(x_max + x_pad)
    y_min, y_max = float(y_min - y_pad), float(y_max + y_pad)

    x_values = np.linspace(x_min, x_max, args.grid_size, dtype=np.float32)
    y_values = np.linspace(y_min, y_max, args.grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

    d_know_grid = mean_knn_distance(know_ref, grid_points, args.k)
    deficiency_grid = np.log(d_know_grid + args.eps).reshape(grid_x.shape)

    d_know_test = mean_knn_distance(know_ref, test_2d, args.k)
    deficiency_test = np.log(d_know_test + args.eps)

    d_know_train_loo = leave_one_out_knn_distance(know_ref, args.k)
    deficiency_train_know = np.log(d_know_train_loo + args.eps)
    tau90 = float(np.quantile(deficiency_train_know, 0.90))
    tau95 = float(np.quantile(deficiency_train_know, 0.95))

    know_test = test_labels == 0
    hall_test = test_labels == 1
    void_mask_test = deficiency_test >= tau95
    meta = {
        "feature_file": str(path),
        "dataset": dataset,
        "model": model_slug,
        "field": args.field,
        "definition": "D_K=-log(rho_K+eps)=log(d_K+eps). Larger values mean lower KNOW/factual support.",
        "projection": {
            "pca128_components": int(pca_dim),
            "pca128_explained_variance_ratio_sum": float(np.sum(pca128.explained_variance_ratio_)),
            "pca2_explained_variance_ratio_sum_from_pca128": float(np.sum(pca2.explained_variance_ratio_)),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "test": int(len(test_idx)),
        },
        "train_label_counts": {
            "know": int((train_labels == 0).sum()),
            "hallucination": int((train_labels == 1).sum()),
        },
        "test_label_counts": {
            "know": int(know_test.sum()),
            "hallucination": int(hall_test.sum()),
        },
        "support_envelope": {
            "tau90_train_know_deficiency": tau90,
            "tau95_train_know_deficiency": tau95,
            "meaning": "Grid/test locations above a threshold lie outside the corresponding empirical KNOW support envelope.",
        },
        "void_rate_tau95": {
            "know": float(void_mask_test[know_test].mean()) if know_test.any() else float("nan"),
            "hallucination": float(void_mask_test[hall_test].mean()) if hall_test.any() else float("nan"),
        },
        "metrics": {
            "deficiency_as_hallucination_score": score_metrics(test_labels, deficiency_test),
        },
    }

    return {
        "dataset": dataset,
        "model": model_slug,
        "test_2d": test_2d,
        "test_labels": test_labels,
        "deficiency_test": deficiency_test,
        "void_mask_test": void_mask_test,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "deficiency_grid": deficiency_grid,
        "tau90": tau90,
        "tau95": tau95,
        "meta": meta,
    }


def collect_norm(cases: List[Dict[str, object]], color_quantile: float) -> Normalize:
    values = np.concatenate([case["deficiency_grid"].ravel() for case in cases])
    values = values[np.isfinite(values)]
    low = float(np.quantile(values, 1.0 - color_quantile))
    high = float(np.quantile(values, color_quantile))
    if high <= low:
        high = low + 1e-6
    return Normalize(vmin=low, vmax=high)


def panel_title(case: Dict[str, object], panel_mode: str) -> str:
    if panel_mode == "dataset":
        base = dict(DATASETS).get(case["dataset"], case["dataset"])
    elif panel_mode == "model":
        base = dict(MODELS).get(case["model"], case["model"])
    else:
        dataset_name = dict(DATASETS).get(case["dataset"], case["dataset"])
        model_name = dict(MODELS).get(case["model"], case["model"])
        base = f"{dataset_name} / {model_name}"
    rates = case["meta"]["void_rate_tau95"]
    return f"{base}\nHALL void {rates['hallucination'] * 100:.1f}% | KNOW {rates['know'] * 100:.1f}%"


def draw_envelope(ax: plt.Axes, case: Dict[str, object], level: float, color: str, linestyle: str, linewidth: float) -> None:
    grid = case["deficiency_grid"]
    if np.nanmin(grid) <= level <= np.nanmax(grid):
        ax.contour(
            case["grid_x"],
            case["grid_y"],
            grid,
            levels=[level],
            colors=color,
            linestyles=linestyle,
            linewidths=linewidth,
        )


def draw_points(ax: plt.Axes, case: Dict[str, object], show_know: bool) -> None:
    points = case["test_2d"]
    labels = case["test_labels"]
    void_mask = case["void_mask_test"]

    if show_know:
        know = labels == 0
        ax.scatter(
            points[know, 0],
            points[know, 1],
            s=4,
            c=KNOW_COLOR,
            alpha=0.12,
            edgecolors="none",
            rasterized=True,
        )

    hall = labels == 1
    hall_void = hall & void_mask
    hall_nonvoid = hall & ~void_mask
    ax.scatter(
        points[hall_nonvoid, 0],
        points[hall_nonvoid, 1],
        s=6,
        c=HALL_COLOR,
        alpha=0.42,
        edgecolors="none",
        rasterized=True,
    )
    ax.scatter(
        points[hall_void, 0],
        points[hall_void, 1],
        s=11,
        c=VOID_COLOR,
        alpha=0.68,
        edgecolors="white",
        linewidths=0.25,
        rasterized=True,
    )


def place_layout(fig: plt.Figure, nrows: int) -> Tuple[float, List[float]]:
    if nrows == 1:
        top = 0.74
        bottom = 0.17
        legend_y = 0.90
    else:
        top = 0.82
        bottom = 0.08
        legend_y = 0.925
    fig.subplots_adjust(
        left=0.055,
        right=0.875,
        bottom=bottom,
        top=top,
        wspace=0.28,
        hspace=0.42,
    )
    return legend_y, [0.895, bottom, 0.014, top - bottom]


def draw_landscape(
    cases: List[Dict[str, object]],
    output_path: Path,
    layout: Tuple[int, int],
    panel_mode: str,
    title: str,
    args: argparse.Namespace,
) -> None:
    nrows, ncols = layout
    norm = collect_norm(cases, args.color_quantile)
    levels = np.linspace(norm.vmin, norm.vmax, args.contour_levels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.25 * nrows), squeeze=False)
    flat_axes = axes.reshape(-1)

    mappable = None
    for ax, case in zip(flat_axes, cases):
        mappable = ax.contourf(
            case["grid_x"],
            case["grid_y"],
            case["deficiency_grid"],
            levels=levels,
            cmap=DEFICIENCY_CMAP,
            norm=norm,
            extend="both",
        )
        draw_envelope(ax, case, case["tau90"], ENVELOPE_90, "--", 0.95)
        draw_envelope(ax, case, case["tau95"], ENVELOPE_95, "-", 1.05)
        draw_points(ax, case, show_know=args.show_know)
        ax.set_title(panel_title(case, panel_mode), fontsize=10.5)
        ax.set_xlabel("PCA-2 dim 1")
        ax.set_ylabel("PCA-2 dim 2")
        ax.tick_params(labelsize=8)

    for ax in flat_axes[len(cases) :]:
        ax.axis("off")

    handles = []
    if args.show_know:
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=KNOW_COLOR, markersize=5, alpha=0.5, label="KNOW test")
        )
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=HALL_COLOR, markersize=5, label="HALL test"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=VOID_COLOR, markeredgecolor="white", markersize=6, label="HALL outside 95% KNOW envelope"),
            Line2D([0], [0], color=ENVELOPE_90, linestyle="--", linewidth=1.0, label="90% KNOW envelope"),
            Line2D([0], [0], color=ENVELOPE_95, linestyle="-", linewidth=1.1, label="95% KNOW envelope"),
        ]
    )
    legend_y, colorbar_box = place_layout(fig, nrows)
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, legend_y))
    if mappable is not None:
        cax = fig.add_axes(colorbar_box)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"Factual support deficiency $D_K=-\log(\rho_K+\epsilon)$", fontsize=10)
    fig.suptitle(title, y=0.985, fontsize=14)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f">>> saved factual support deficiency landscape: {output_path}", flush=True)


def write_stats(path: Path, stats: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">>> saved stats: {path}", flush=True)


def run_layout(
    name: str,
    case_pairs: List[Tuple[str, str]],
    layout: Tuple[int, int],
    panel_mode: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cases = [project_case(dataset, model_slug, args) for dataset, model_slug in case_pairs]
    output_dir = resolve_output_path(args.output_dir, FIG_DIR)
    output_path = output_dir / f"factual_support_deficiency_{name}_pca2.png"
    draw_landscape(
        cases,
        output_path,
        layout,
        panel_mode,
        "Factual Support Deficiency Landscape",
        args,
    )
    return {
        "name": name,
        "output_png": str(output_path),
        "cases": {f"{case['dataset']}/{case['model']}": case["meta"] for case in cases},
    }


def rank_cases_by_void_gap(args: argparse.Namespace) -> List[Dict[str, object]]:
    dataset_pairs = parse_pairs(args.main_datasets, DATASETS)
    model_pairs = parse_pairs(args.six_models, MODELS)
    cases = []
    for dataset, _ in dataset_pairs:
        for model_slug, _ in model_pairs:
            case = project_case(dataset, model_slug, args)
            rates = case["meta"]["void_rate_tau95"]
            case["void_gap"] = rates["hallucination"] - rates["know"]
            cases.append(case)
    cases.sort(key=lambda item: (item["void_gap"], item["meta"]["void_rate_tau95"]["hallucination"]), reverse=True)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot latent-void candidate regions using only KNOW support rho_K."
    )
    parser.add_argument("--output-dir", default=str(FIG_DIR))
    parser.add_argument("--stats-output", default=str(FIG_DIR / "factual_support_deficiency_stats.json"))
    parser.add_argument("--field", default=DEFAULT_FIELD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--contour-levels", type=int, default=33)
    parser.add_argument("--extent-quantile", type=float, default=0.995)
    parser.add_argument("--extent-pad", type=float, default=0.08)
    parser.add_argument("--color-quantile", type=float, default=0.99)
    parser.add_argument("--max-test-per-class", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-know", action="store_true")
    parser.add_argument(
        "--layouts",
        choices=["all", "main", "six", "top"],
        default="all",
    )
    parser.add_argument("--main-model", default="llama3_8b_instruct")
    parser.add_argument("--main-datasets", default=",".join(slug for slug, _ in DATASETS))
    parser.add_argument("--six-dataset", default="popqa")
    parser.add_argument("--six-models", default=",".join(slug for slug, _ in MODELS))
    parser.add_argument("--top-n", type=int, default=4)
    args = parser.parse_args()

    if not (0.0 < args.extent_quantile <= 1.0):
        raise ValueError("--extent-quantile must be in (0, 1].")
    if not (0.5 < args.color_quantile <= 1.0):
        raise ValueError("--color-quantile must be in (0.5, 1].")

    stats: Dict[str, object] = {
        "definition": {
            "landscape": "Only KNOW support is used: rho_K(z)=1/(d_K(z)+eps), D_K(z)=-log(rho_K(z)+eps)=log(d_K(z)+eps).",
            "projection": "Training split fits mean/std, PCA-128, then PCA-2D. Test samples are transformed only.",
            "support_envelope": "90% and 95% contours are empirical D_K quantiles computed from training KNOW leave-one-out kNN distances.",
            "latent_void_candidate": "A region is a candidate void when factual support deficiency is high and HALL test samples concentrate there.",
            "no_artificial_voids": "No smoothing, masking, or synthetic holes are applied.",
        },
        "seed": int(args.seed),
        "k": int(args.k),
        "runs": [],
    }

    dataset_pairs = parse_pairs(args.main_datasets, DATASETS)
    model_pairs = parse_pairs(args.six_models, MODELS)
    if args.layouts in ["all", "main"]:
        main_case_pairs = [(dataset, args.main_model) for dataset, _ in dataset_pairs]
        stats["runs"].append(
            run_layout(
                f"{args.main_model}_4datasets",
                main_case_pairs,
                (1, len(main_case_pairs)),
                "dataset",
                args,
            )
        )
    if args.layouts in ["all", "six"]:
        six_case_pairs = [(args.six_dataset, model_slug) for model_slug, _ in model_pairs]
        stats["runs"].append(
            run_layout(
                f"{args.six_dataset}_6models",
                six_case_pairs,
                (2, int(math.ceil(len(six_case_pairs) / 2))),
                "model",
                args,
            )
        )
    if args.layouts in ["top"]:
        top_cases = rank_cases_by_void_gap(args)[: args.top_n]
        ncols = min(2, max(1, len(top_cases)))
        nrows = int(math.ceil(len(top_cases) / ncols))
        output_dir = resolve_output_path(args.output_dir, FIG_DIR)
        output_path = output_dir / f"factual_support_deficiency_top{len(top_cases)}_void_gap_pca2.png"
        draw_landscape(
            top_cases,
            output_path,
            (nrows, ncols),
            "mixed",
            "Factual Support Deficiency Landscape: Representative Latent-void Candidates",
            args,
        )
        stats["runs"].append(
            {
                "name": f"top{len(top_cases)}_void_gap",
                "output_png": str(output_path),
                "selection": "Selected by largest HALL void rate minus KNOW void rate across the configured datasets/models.",
                "cases": {f"{case['dataset']}/{case['model']}": case["meta"] for case in top_cases},
            }
        )

    write_stats(resolve_output_path(args.stats_output, FIG_DIR), stats)


if __name__ == "__main__":
    main()
