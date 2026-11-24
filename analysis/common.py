import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from _paths import FIG_DIR, PROJECT_ROOT, PT_DIR, RESULTS_DIR, feature_file, resolve_input_path



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

DATASET_COLORS = {
    "popqa": "#4e79a7",
    "webquestions": "#f28e2b",
    "triviaqa": "#59a14f",
    "mkqa": "#b07aa1",
}

KNOW_COLOR = "#2c7fb8"
HALL_COLOR = "#d81b60"


def feature_path(dataset: str, model: str) -> Path:
    return feature_file(dataset, model)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pt(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(str(resolve_input_path(path)), map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(str(resolve_input_path(path)), map_location="cpu")


def labels_np(data: Dict[str, Any]) -> np.ndarray:
    labels = data["labels"]
    if isinstance(labels, torch.Tensor):
        if labels.ndim == 2:
            return labels.argmax(dim=1).long().cpu().numpy()
        return labels.long().cpu().numpy()
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return labels.argmax(axis=1).astype(np.int64)
    return labels.astype(np.int64)


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


def tensor_subset_flat(
    tensor: torch.Tensor,
    indices: np.ndarray,
    layer_index: Optional[int] = None,
) -> torch.Tensor:
    idx = torch.from_numpy(indices).long()
    if layer_index is None:
        subset = tensor[idx].float()
    else:
        subset = tensor[idx, layer_index].float()
    subset = subset.reshape(subset.shape[0], -1)
    return torch.nan_to_num(subset, nan=0.0, posinf=0.0, neginf=0.0)


def fit_standardized_pca(
    data: Dict[str, Any],
    field: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_dim: int,
    seed: int,
    layer_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if field not in data:
        raise KeyError(f"Missing hidden field: {field}")
    tensor = data[field]
    train_x = tensor_subset_flat(tensor, train_idx, layer_index=layer_index)
    test_x = tensor_subset_flat(tensor, test_idx, layer_index=layer_index)

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = torch.nan_to_num((train_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    test_x = torch.nan_to_num((test_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    train_np = train_x.numpy().astype(np.float32, copy=False)
    test_np = test_x.numpy().astype(np.float32, copy=False)
    info: Dict[str, Any] = {
        "field": field,
        "input_dim": int(train_np.shape[1]),
        "pca_enabled": False,
    }
    if 0 < pca_dim < train_np.shape[1]:
        n_components = min(int(pca_dim), train_np.shape[0] - 1, train_np.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        train_np = pca.fit_transform(train_np).astype(np.float32, copy=False)
        test_np = pca.transform(test_np).astype(np.float32, copy=False)
        info.update(
            {
                "pca_enabled": True,
                "pca_components": int(n_components),
                "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            }
        )
    return train_np, test_np, info


def mean_knn_distance(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    if len(reference) == 0:
        raise ValueError("Empty kNN reference.")
    n_neighbors = min(int(k), len(reference))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(reference)
    distances, _ = nn.kneighbors(query, return_distance=True)
    return distances.mean(axis=1).astype(np.float64)


def relative_support_scores(
    train_embed: np.ndarray,
    train_labels: np.ndarray,
    test_embed: np.ndarray,
    k: int,
    eps: float,
) -> Dict[str, np.ndarray]:
    know_ref = train_embed[train_labels == 0]
    hall_ref = train_embed[train_labels == 1]
    d_know = mean_knn_distance(know_ref, test_embed, k)
    d_hall = mean_knn_distance(hall_ref, test_embed, k)
    log_rho_know = -np.log(d_know + eps)
    log_rho_hall = -np.log(d_hall + eps)
    relative = log_rho_know - log_rho_hall
    return {
        "d_know": d_know,
        "d_hall": d_hall,
        "log_rho_know": log_rho_know,
        "log_rho_hall": log_rho_hall,
        "relative_support": relative,
        "support_deficiency": -relative,
    }


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    best_fpr = 1.0
    for threshold in thresholds:
        tpr = float(np.mean(pos >= threshold))
        if tpr >= target_tpr:
            fpr = float(np.mean(neg >= threshold))
            best_fpr = min(best_fpr, fpr)
    return best_fpr


def binary_metrics(labels: np.ndarray, hall_scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(np.int64)
    hall_scores = np.asarray(hall_scores).astype(np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, hall_scores)),
        "aupr": float(average_precision_score(labels, hall_scores)),
        "fpr95": fpr_at_tpr(labels, hall_scores, 0.95),
    }


def mean_by_label(values: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    know = values[labels == 0]
    hall = values[labels == 1]
    know_mean = float(np.mean(know))
    hall_mean = float(np.mean(hall))
    return know_mean, hall_mean, know_mean - hall_mean


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    rx = average_ranks(x)
    ry = average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def binned_mean(x: np.ndarray, y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    bx = []
    by = []
    for bin_id in range(bins):
        start = bin_id * len(x) // bins
        end = (bin_id + 1) * len(x) // bins
        if end <= start:
            continue
        idx = order[start:end]
        bx.append(float(np.mean(x[idx])))
        by.append(float(np.mean(y[idx])))
    return np.asarray(bx), np.asarray(by)


def save_png_pdf(fig, output_stem: Path, dpi: int) -> Tuple[Path, Path]:
    ensure_dir(output_stem.parent)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def case_iterator() -> Sequence[Tuple[str, str, str, str]]:
    return [(dataset, dataset_name, model, model_name) for dataset, dataset_name in DATASETS for model, model_name in MODELS]


def short_float(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.4f}"


def save_plot(fig, stem: Path) -> None:
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
