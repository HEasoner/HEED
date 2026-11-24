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
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from _paths import FIG_DIR, PROJECT_ROOT, PT_DIR, feature_file, resolve_input_path, resolve_output_path


DEFAULT_OUTPUT = FIG_DIR / "hidden_know_manifold_2x3_target_mean_knn.png"
DEFAULT_FIELD = "probe_target_hidden_mean_answer"

DATASETS: List[Tuple[str, str]] = [
    ("popqa", "PopQA"),
    ("webquestions", "WebQ"),
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

COLORS = {
    "KNOW": "#2c7fb8",
    "HALLUCINATION": "#f28e2b",
}


def parse_pairs(value: str, defaults: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not value:
        return defaults
    default_map = {slug: label for slug, label in defaults}
    pairs = []
    for item in value.split(","):
        slug = item.strip()
        if not slug:
            continue
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


def stratified_cap(
    indices: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    seed: int,
) -> np.ndarray:
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


def load_standardized_splits(
    feature_path: Path,
    field: str,
    seed: int,
    max_train_reference_per_class: int,
    max_test_per_class: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, Dict[str, object]]:
    try:
        data = torch.load(str(feature_path), map_location="cpu", mmap=True)
    except TypeError:
        data = torch.load(str(feature_path), map_location="cpu")

    if field not in data:
        raise KeyError(f"{field} is missing from {feature_path}")

    labels = data["labels"].argmax(dim=1).long().numpy()
    train_idx, _, test_idx = make_split(labels, seed)
    ref_idx = stratified_cap(train_idx, labels, max_train_reference_per_class, seed + 17)
    test_idx = stratified_cap(test_idx, labels, max_test_per_class, seed + 31)
    know_ref_idx = ref_idx[labels[ref_idx] == 0]
    hall_ref_idx = ref_idx[labels[ref_idx] == 1]

    hidden = data[field]
    train_hidden = flatten_subset(hidden, train_idx)
    mean = train_hidden.mean(dim=0, keepdim=True)
    std = train_hidden.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_hidden = None

    know_ref_hidden = flatten_subset(hidden, know_ref_idx)
    hall_ref_hidden = flatten_subset(hidden, hall_ref_idx)
    test_hidden = flatten_subset(hidden, test_idx)
    know_ref_hidden = torch.nan_to_num(
        (know_ref_hidden - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    hall_ref_hidden = torch.nan_to_num(
        (hall_ref_hidden - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    test_hidden = torch.nan_to_num((test_hidden - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    meta = {
        "feature_path": str(feature_path),
        "field": field,
        "dataset": data.get("dataset", "unknown"),
        "model_path": data.get("model_path", "unknown"),
        "hidden_shape": list(hidden.shape),
        "feature_dim": int(know_ref_hidden.shape[1]),
        "label_counts_all": {
            "know": int((labels == 0).sum()),
            "hallucination": int((labels == 1).sum()),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "test": int(len(test_idx)),
            "reference": int(len(ref_idx)),
        },
        "reference_label_counts": {
            "know": int(len(know_ref_idx)),
            "hallucination": int(len(hall_ref_idx)),
        },
        "test_label_counts": {
            "know": int((labels[test_idx] == 0).sum()),
            "hallucination": int((labels[test_idx] == 1).sum()),
        },
    }
    return know_ref_hidden, hall_ref_hidden, test_hidden, labels[test_idx], meta


def maybe_pca(
    know_reference: torch.Tensor,
    hall_reference: torch.Tensor,
    test: torch.Tensor,
    pca_dim: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    reference = torch.cat([know_reference, hall_reference], dim=0)
    if pca_dim <= 0 or pca_dim >= reference.shape[1]:
        return know_reference, hall_reference, test, {"enabled": False}

    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
    know_np = know_reference.cpu().numpy().astype(np.float32, copy=False)
    hall_np = hall_reference.cpu().numpy().astype(np.float32, copy=False)
    reference_np = np.concatenate([know_np, hall_np], axis=0)
    test_np = test.cpu().numpy().astype(np.float32, copy=False)
    reference_pca = pca.fit_transform(reference_np).astype(np.float32)
    know_pca = reference_pca[: len(know_np)]
    hall_pca = reference_pca[len(know_np) :]
    test_pca = pca.transform(test_np).astype(np.float32)
    info = {
        "enabled": True,
        "n_components": int(pca_dim),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return torch.from_numpy(know_pca), torch.from_numpy(hall_pca), torch.from_numpy(test_pca), info


@torch.no_grad()
def knn_density_torch(
    reference: torch.Tensor,
    test: torch.Tensor,
    k: int,
    eps: float,
    device: str,
    test_batch_size: int,
    reference_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if reference.shape[0] < 1:
        raise ValueError("At least one reference sample is required.")
    k_eff = min(k, int(reference.shape[0]))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is visible to PyTorch.")
    device_obj = torch.device(device)
    reference = reference.float().to(device_obj)
    ref_norm = (reference * reference).sum(dim=1).view(1, -1)
    densities = []
    distances = []

    for test_start in range(0, test.shape[0], test_batch_size):
        test_batch = test[test_start : test_start + test_batch_size].float().to(device_obj)
        test_norm = (test_batch * test_batch).sum(dim=1, keepdim=True)
        best = torch.full(
            (test_batch.shape[0], k_eff),
            float("inf"),
            dtype=torch.float32,
            device=device_obj,
        )
        for ref_start in range(0, reference.shape[0], reference_batch_size):
            ref_end = min(ref_start + reference_batch_size, reference.shape[0])
            ref_chunk = reference[ref_start:ref_end]
            dist_sq = test_norm + ref_norm[:, ref_start:ref_end] - 2.0 * (
                test_batch @ ref_chunk.t()
            )
            dist = dist_sq.clamp_min_(0.0).sqrt_()
            best = torch.topk(torch.cat([best, dist], dim=1), k=k_eff, largest=False).values

        d_k = best.mean(dim=1)
        rho_k = 1.0 / (d_k + eps)
        distances.append(d_k.cpu())
        densities.append(rho_k.cpu())

    return torch.cat(distances).numpy(), torch.cat(densities).numpy()


def summarize_values(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    return {
        "n": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.min() == labels.max():
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
        "fpr95": fpr_at_tpr(labels, scores, target_tpr=0.95),
    }


def compute_case_density(
    dataset: str,
    model_slug: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    path = resolve_input_path(feature_file(dataset, model_slug))
    if not path.exists():
        raise FileNotFoundError(path)

    print(f">>> hidden geometry: dataset={dataset} model={model_slug} file={path.name}", flush=True)
    know_reference, hall_reference, test, labels, meta = load_standardized_splits(
        path,
        field=args.field,
        seed=args.seed,
        max_train_reference_per_class=args.max_train_reference_per_class,
        max_test_per_class=args.max_test_per_class,
    )
    know_reference, hall_reference, test, pca_info = maybe_pca(
        know_reference,
        hall_reference,
        test,
        args.pca_dim,
        args.seed,
    )
    d_know, rho_know = knn_density_torch(
        reference=know_reference,
        test=test,
        k=args.k,
        eps=args.eps,
        device=args.device,
        test_batch_size=args.test_batch_size,
        reference_batch_size=args.reference_batch_size,
    )
    d_hall, rho_hall = knn_density_torch(
        reference=hall_reference,
        test=test,
        k=args.k,
        eps=args.eps,
        device=args.device,
        test_batch_size=args.test_batch_size,
        reference_batch_size=args.reference_batch_size,
    )

    # Lower values indicate weaker factual support and stronger latent-void tendency.
    relative_factual_support = np.log(rho_know + args.eps) - np.log(rho_hall + args.eps)

    values = {
        "know_distance": {
            "KNOW": d_know[labels == 0],
            "HALLUCINATION": d_know[labels == 1],
        },
        "relative_factual_support": {
            "KNOW": relative_factual_support[labels == 0],
            "HALLUCINATION": relative_factual_support[labels == 1],
        },
    }
    meta.update(
        {
            "k": int(args.k),
            "eps": float(args.eps),
            "pca": pca_info,
            "d_know": {
                "KNOW": summarize_values(d_know[labels == 0]),
                "HALLUCINATION": summarize_values(d_know[labels == 1]),
            },
            "rho_know": {
                "KNOW": summarize_values(rho_know[labels == 0]),
                "HALLUCINATION": summarize_values(rho_know[labels == 1]),
            },
            "d_hall": {
                "KNOW": summarize_values(d_hall[labels == 0]),
                "HALLUCINATION": summarize_values(d_hall[labels == 1]),
            },
            "rho_hall": {
                "KNOW": summarize_values(rho_hall[labels == 0]),
                "HALLUCINATION": summarize_values(rho_hall[labels == 1]),
            },
            "relative_factual_support": {
                "KNOW": summarize_values(values["relative_factual_support"]["KNOW"]),
                "HALLUCINATION": summarize_values(values["relative_factual_support"]["HALLUCINATION"]),
            },
            "metrics": {
                "know_distance": score_metrics(labels, d_know),
                "relative_factual_support_void_score": score_metrics(
                    labels,
                    -relative_factual_support,
                ),
            },
        }
    )
    return {"values": values, "meta": meta}


def draw_violin_box(
    ax: plt.Axes,
    panel_values: Dict[str, Dict[str, np.ndarray]],
    dataset_pairs: List[Tuple[str, str]],
) -> List[np.ndarray]:
    positions = []
    arrays = []
    colors = []
    all_values = []
    for index, (dataset, _) in enumerate(dataset_pairs, start=1):
        for label, offset in [("KNOW", -0.17), ("HALLUCINATION", 0.17)]:
            values = panel_values.get(dataset, {}).get(label)
            if values is None or len(values) == 0:
                continue
            values = np.asarray(values, dtype=np.float64)
            arrays.append(values)
            positions.append(index + offset)
            colors.append(COLORS[label])
            all_values.append(values)

    if not arrays:
        return []

    violins = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.28,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.32)
        body.set_linewidth(0.8)

    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.095,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 0.8},
        capprops={"color": "#444444", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.70)
        patch.set_linewidth(0.8)

    ax.set_xticks(range(1, len(dataset_pairs) + 1))
    ax.set_xticklabels([label for _, label in dataset_pairs], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    return all_values


def plot_grid(
    density: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    model_pairs: List[Tuple[str, str]],
    dataset_pairs: List[Tuple[str, str]],
    args: argparse.Namespace,
    score_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
    zero_bottom: bool,
) -> None:
    n_models = len(model_pairs)
    ncols = 3
    nrows = int(math.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.2, 7.6 if nrows == 2 else 4.2 * nrows),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    global_values = []
    for ax, (model_slug, model_name) in zip(axes, model_pairs):
        panel_values = {
            dataset: density.get(model_slug, {}).get(dataset, {}).get(score_key, {})
            for dataset, _ in dataset_pairs
        }
        values = draw_violin_box(ax, panel_values, dataset_pairs)
        for arr in values:
            global_values.append(arr)
        ax.set_title(model_name, fontsize=11, pad=8)
        ax.set_xlim(0.45, len(dataset_pairs) + 0.55)
        ax.tick_params(axis="both", labelsize=9)

    for ax in axes[n_models:]:
        ax.axis("off")

    if global_values and args.ylim_quantile > 0:
        merged = np.concatenate(global_values)
        low_q = max(0.0, 1.0 - args.ylim_quantile)
        bottom = 0.0 if zero_bottom else float(np.quantile(merged, low_q))
        top = float(np.quantile(merged, args.ylim_quantile))
        pad = max((top - bottom) * 0.08, 1e-8)
        axes[0].set_ylim(bottom - (0.0 if zero_bottom else pad), top + pad)

    handles = [
        Patch(facecolor=COLORS["KNOW"], edgecolor=COLORS["KNOW"], alpha=0.55, label="KNOW"),
        Patch(
            facecolor=COLORS["HALLUCINATION"],
            edgecolor=COLORS["HALLUCINATION"],
            alpha=0.55,
            label="HALLUCINATION",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    title = title + rf" (k={args.k})"
    if args.pca_dim > 0:
        title += f", PCA-{args.pca_dim}"
    fig.suptitle(title, y=1.075, fontsize=14)
    fig.supxlabel("Dataset", y=0.02, fontsize=11)
    fig.supylabel(ylabel, x=0.005, fontsize=11)
    fig.tight_layout(rect=(0.02, 0.04, 1.0, 0.98))

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f">>> saved png: {output_path}", flush=True)


PLOT_SPECS = {
    "know_distance": {
        "title": r"Answer-level distance to the KNOW hidden manifold",
        "ylabel": r"Distance to KNOW manifold ($d_K$)",
        "zero_bottom": True,
    },
    "relative_factual_support": {
        "title": r"Relative factual support density ($\log \rho_K-\log \rho_H$)",
        "ylabel": r"Relative factual support (lower = latent void)",
        "zero_bottom": False,
    },
}

PLOT_ALIASES = {
    "relative_density": "relative_factual_support",
}


def derived_output_path(base_output: Path, plot_key: str, multiple: bool) -> Path:
    if not multiple:
        return base_output
    if plot_key == "know_distance":
        return base_output
    if "know_manifold" in base_output.stem:
        stem = base_output.stem.replace("know_manifold", "relative_factual_support")
    else:
        stem = f"{base_output.stem}_{plot_key}"
    return base_output.with_name(stem + base_output.suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 2x3 model grid of answer-level hidden manifold geometry. "
            "Hidden states are standardized with the training split, and test "
            "samples are scored by kNN distance to class-specific training references."
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stats-output", default=None)
    parser.add_argument(
        "--plots",
        default="know_distance,relative_factual_support",
        help="Comma-separated plot keys: know_distance,relative_factual_support.",
    )
    parser.add_argument("--field", default=DEFAULT_FIELD)
    parser.add_argument("--datasets", default=",".join(slug for slug, _ in DATASETS))
    parser.add_argument("--models", default=",".join(slug for slug, _ in MODELS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for exact chunked kNN distance computation.",
    )
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--reference-batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-train-reference-per-class",
        type=int,
        default=0,
        help="Optional stratified cap for training reference samples. 0 uses all training samples.",
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=0,
        help="Optional stratified cap for plotted/scored test samples. 0 uses all test samples.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="Optional PCA dimension after train-standardization. 0 keeps full hidden space.",
    )
    parser.add_argument(
        "--ylim-quantile",
        type=float,
        default=0.995,
        help="Shared upper y limit by global quantile. Use 0 to disable.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dataset_pairs = parse_pairs(args.datasets, DATASETS)
    model_pairs = parse_pairs(args.models, MODELS)
    plot_keys = [
        PLOT_ALIASES.get(item.strip(), item.strip())
        for item in args.plots.split(",")
        if item.strip()
    ]
    unknown_plots = [item for item in plot_keys if item not in PLOT_SPECS]
    if unknown_plots:
        raise ValueError(f"Unknown plot key(s): {unknown_plots}. Valid keys: {sorted(PLOT_SPECS)}")

    density: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
    stats: Dict[str, object] = {
        "definition": {
            "hidden_representation": args.field,
            "standardization": "Training split mean/std are used for each dataset-model case.",
            "d_know": "Mean L2 distance from each test sample to its k nearest KNOW training reference samples.",
            "rho_know": "1 / (d_know + eps); larger values mean denser local KNOW support.",
            "relative_factual_support": "log(rho_know + eps) - log(rho_hall + eps); lower values mean weaker factual support and stronger latent-void tendency.",
        },
        "seed": int(args.seed),
        "k": int(args.k),
        "eps": float(args.eps),
        "models": [slug for slug, _ in model_pairs],
        "datasets": [slug for slug, _ in dataset_pairs],
        "cases": {},
    }

    for model_slug, _ in model_pairs:
        density[model_slug] = {}
        for dataset, _ in dataset_pairs:
            case = compute_case_density(dataset, model_slug, args)
            density[model_slug][dataset] = case["values"]
            stats["cases"][f"{dataset}/{model_slug}"] = case["meta"]

    output_path = resolve_output_path(args.output, FIG_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_pngs = {}
    for plot_key in plot_keys:
        spec = PLOT_SPECS[plot_key]
        case_output = derived_output_path(output_path, plot_key, multiple=len(plot_keys) > 1)
        plot_grid(
            density,
            model_pairs,
            dataset_pairs,
            args,
            score_key=plot_key,
            output_path=case_output,
            title=spec["title"],
            ylabel=spec["ylabel"],
            zero_bottom=bool(spec["zero_bottom"]),
        )
        output_pngs[plot_key] = str(case_output)

    stats_output = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".json")
    stats_output = resolve_output_path(stats_output, output_path.parent)
    stats["output_pngs"] = output_pngs
    stats_output.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">>> saved stats: {stats_output}", flush=True)


if __name__ == "__main__":
    main()
