import argparse
import json
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
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from common import save_plot
from _paths import FIG_DIR, PROJECT_ROOT, resolve_input_path
from factual_support import MODELS, feature_file, flatten_subset, make_split


OUTPUT_STEM = FIG_DIR / "latent_void_highdim_label_free_popqa_6models_pca2"
KNOW_COLOR = "#2878a8"
HALL_COLOR = "#c74769"
VOID_COLOR = "#dc7a43"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-free high-dimensional factual-support void candidates.")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--field", default="probe_target_hidden_mean_answer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--know-support-quantile", type=float, default=0.15)
    parser.add_argument("--overall-density-quantile", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=190)
    parser.add_argument("--extent-quantile", type=float, default=0.995)
    parser.add_argument("--extent-pad", type=float, default=0.08)
    parser.add_argument("--visual-occupancy-quantile", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--reference-batch-size", type=int, default=512)
    parser.add_argument("--output-stem", default=str(OUTPUT_STEM))
    return parser.parse_args()


@torch.no_grad()
def knn_mean_distance(
    reference: torch.Tensor,
    query: torch.Tensor,
    k: int,
    device: str,
    query_batch_size: int,
    reference_batch_size: int,
    self_reference_index: np.ndarray | None = None,
) -> np.ndarray:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    reference = reference.float().to(device_obj)
    ref_norm = (reference * reference).sum(dim=1).view(1, -1)
    k_eff = min(k, len(reference) - (1 if self_reference_index is not None else 0))
    if k_eff < 1:
        raise ValueError("Insufficient reference samples for kNN.")
    outputs = []
    for query_start in range(0, len(query), query_batch_size):
        query_end = min(query_start + query_batch_size, len(query))
        batch = query[query_start:query_end].float().to(device_obj)
        batch_norm = (batch * batch).sum(dim=1, keepdim=True)
        best = torch.full((len(batch), k_eff), float("inf"), device=device_obj)
        self_idx = None if self_reference_index is None else self_reference_index[query_start:query_end]
        for ref_start in range(0, len(reference), reference_batch_size):
            ref_end = min(ref_start + reference_batch_size, len(reference))
            ref = reference[ref_start:ref_end]
            dist_sq = batch_norm + ref_norm[:, ref_start:ref_end] - 2.0 * (batch @ ref.t())
            dist = dist_sq.clamp_min_(0.0).sqrt_()
            if self_idx is not None:
                rows = np.where((self_idx >= ref_start) & (self_idx < ref_end))[0]
                if len(rows):
                    cols = self_idx[rows] - ref_start
                    dist[
                        torch.as_tensor(rows, device=device_obj),
                        torch.as_tensor(cols, device=device_obj),
                    ] = float("inf")
            best = torch.topk(torch.cat([best, dist], dim=1), k=k_eff, largest=False).values
        outputs.append(best.mean(dim=1).cpu())
    return torch.cat(outputs).numpy().astype(np.float32)


def visualization_mask(train_2d: np.ndarray, candidate: np.ndarray, grid: np.ndarray, q: float) -> np.ndarray:
    train_nn = NearestNeighbors(n_neighbors=2).fit(train_2d)
    train_dist, _ = train_nn.kneighbors(train_2d)
    occupancy_radius = float(np.quantile(train_dist[:, 1], q))
    grid_nn = NearestNeighbors(n_neighbors=1).fit(train_2d)
    grid_dist, grid_index = grid_nn.kneighbors(grid)
    return candidate[grid_index[:, 0]] & (grid_dist[:, 0] <= occupancy_radius)


def analyze_case(dataset: str, model: str, args: argparse.Namespace) -> dict:
    path = resolve_input_path(feature_file(dataset, model))
    print(f">>> high-dimensional void: dataset={dataset} model={model}", flush=True)
    try:
        data = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        data = torch.load(str(path), map_location="cpu")
    labels = data["labels"].argmax(dim=1).long().numpy()
    train_idx, _, test_idx = make_split(labels, args.seed)
    hidden = data[args.field]
    train = flatten_subset(hidden, train_idx)
    test = flatten_subset(hidden, test_idx)
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train = torch.nan_to_num((train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    test = torch.nan_to_num((test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    know_rows = np.where(train_labels == 0)[0]
    know_ref = train[torch.from_numpy(know_rows).long()]

    common = {
        "k": args.k,
        "device": args.device,
        "query_batch_size": args.query_batch_size,
        "reference_batch_size": args.reference_batch_size,
    }
    d_know_train = knn_mean_distance(
        know_ref, train, self_reference_index=np.where(train_labels == 0, np.searchsorted(know_rows, np.arange(len(train))), -1), **common
    )
    d_all_train = knn_mean_distance(
        train, train, self_reference_index=np.arange(len(train)), **common
    )
    s_know_train = 1.0 / (d_know_train + args.eps)
    p_all_train = 1.0 / (d_all_train + args.eps)
    know_support_threshold = float(
        np.quantile(s_know_train[train_labels == 0], args.know_support_quantile)
    )
    overall_density_threshold = float(
        np.quantile(p_all_train, args.overall_density_quantile)
    )
    train_candidate = (
        (s_know_train < know_support_threshold)
        & (p_all_train > overall_density_threshold)
    )

    d_know_test = knn_mean_distance(know_ref, test, self_reference_index=None, **common)
    d_all_test = knn_mean_distance(train, test, self_reference_index=None, **common)
    s_know_test = 1.0 / (d_know_test + args.eps)
    p_all_test = 1.0 / (d_all_test + args.eps)
    test_candidate = (
        (s_know_test < know_support_threshold)
        & (p_all_test > overall_density_threshold)
    )

    train_np = train.numpy().astype(np.float32, copy=False)
    test_np = test.numpy().astype(np.float32, copy=False)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=args.seed)
    train_2d = pca.fit_transform(train_np).astype(np.float32)
    test_2d = pca.transform(test_np).astype(np.float32)
    all_2d = np.concatenate([train_2d, test_2d])
    tail = (1.0 - args.extent_quantile) / 2.0
    x_min, x_max = np.quantile(all_2d[:, 0], [tail, 1.0 - tail])
    y_min, y_max = np.quantile(all_2d[:, 1], [tail, 1.0 - tail])
    x_pad = (x_max - x_min) * args.extent_pad
    y_pad = (y_max - y_min) * args.extent_pad
    x = np.linspace(x_min - x_pad, x_max + x_pad, args.grid_size)
    y = np.linspace(y_min - y_pad, y_max + y_pad, args.grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_candidate = visualization_mask(
        train_2d, train_candidate, grid, args.visual_occupancy_quantile
    ).reshape(grid_x.shape)

    know_test = test_labels == 0
    hall_test = test_labels == 1
    hall_entry = float(test_candidate[hall_test].mean())
    know_entry = float(test_candidate[know_test].mean())
    enrichment = hall_entry / know_entry if know_entry > 0 else float("inf")
    return {
        "model": model,
        "train_2d": train_2d,
        "test_2d": test_2d,
        "test_labels": test_labels,
        "test_candidate": test_candidate,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_candidate": grid_candidate,
        "hall_entry": hall_entry,
        "know_entry": know_entry,
        "enrichment_ratio": enrichment,
        "know_support_threshold": know_support_threshold,
        "overall_density_threshold": overall_density_threshold,
        "train_candidate_rate": float(train_candidate.mean()),
        "feature_dim": int(train.shape[1]),
        "pca2_variance": float(pca.explained_variance_ratio_.sum()),
        "test_n": int(len(test_idx)),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.know_support_quantile < 1.0:
        raise ValueError("--know-support-quantile must be in (0, 1).")
    if not 0.0 < args.overall_density_quantile < 1.0:
        raise ValueError("--overall-density-quantile must be in (0, 1).")
    cases = [analyze_case(args.dataset, model, args) for model, _ in MODELS]

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8), squeeze=False)
    rows = []
    for ax, case, (_, model_name) in zip(axes.flat, cases, MODELS):
        if case["grid_candidate"].any():
            ax.contourf(
                case["grid_x"], case["grid_y"], case["grid_candidate"].astype(float),
                levels=[0.5, 1.5], colors=[VOID_COLOR], alpha=0.30,
            )
            ax.contour(
                case["grid_x"], case["grid_y"], case["grid_candidate"].astype(float),
                levels=[0.5], colors=[VOID_COLOR], linewidths=1.0,
            )
        points = case["test_2d"]
        labels = case["test_labels"]
        know = labels == 0
        hall = labels == 1
        ax.scatter(points[know, 0], points[know, 1], s=8, color=KNOW_COLOR, alpha=0.38, edgecolors="none")
        ax.scatter(points[hall, 0], points[hall, 1], s=8, color=HALL_COLOR, alpha=0.44, edgecolors="none")
        selected = case["test_candidate"]
        ax.scatter(
            points[selected, 0], points[selected, 1], s=24, facecolors="none",
            edgecolors=np.where(labels[selected] == 1, HALL_COLOR, KNOW_COLOR), linewidths=0.75,
        )
        er = case["enrichment_ratio"]
        er_text = r"$\infty$" if not np.isfinite(er) else f"{er:.2f}"
        ax.set_title(
            f"{model_name}\nHALL {100 * case['hall_entry']:.1f}% | "
            f"KNOW {100 * case['know_entry']:.1f}% | ER {er_text}\u00d7",
            fontsize=10,
        )
        ax.set_xlabel("PCA-2 dim 1")
        ax.set_ylabel("PCA-2 dim 2")
        ax.tick_params(labelsize=8)
        rows.append(
            {
                "dataset": args.dataset,
                "model": case["model"],
                "feature_dim": case["feature_dim"],
                "know_support_quantile": args.know_support_quantile,
                "overall_density_quantile": args.overall_density_quantile,
                "hall_entry_rate": case["hall_entry"],
                "know_entry_rate": case["know_entry"],
                "enrichment_ratio": case["enrichment_ratio"],
                "train_candidate_rate": case["train_candidate_rate"],
                "pca2_explained_variance": case["pca2_variance"],
                "test_n": case["test_n"],
            }
        )

    handles = [
        Line2D([0], [0], color=VOID_COLOR, linewidth=8, alpha=0.30, label="Train-defined candidate void"),
        Line2D([0], [0], marker="o", linestyle="none", color=KNOW_COLOR, label="Held-out KNOW"),
        Line2D([0], [0], marker="o", linestyle="none", color=HALL_COLOR, label="Held-out HALL"),
        Line2D([0], [0], marker="o", markerfacecolor="none", linestyle="none", color="#333333", label="Void-entry test point"),
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.suptitle("Label-free high-dimensional latent-void candidates", y=1.015, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_plot(fig, output_stem)
    pd.DataFrame(rows).to_csv(output_stem.with_suffix(".csv"), index=False)
    metadata = {
        "definition": "s_K below Q0.15 of train KNOW and p_all above Q0.05 of all train.",
        "region_construction_uses_hall_labels": False,
        "statistics_space": "full standardized hidden representation; no PCA",
        "visualization": "PCA-2 is train-fit only. Orange regions are nearest-train-point projections of high-dimensional candidate identities within the 95% train PCA occupancy radius.",
        "seed": args.seed,
        "k": args.k,
    }
    output_stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f">>> outputs: {output_stem}.png/.pdf/.csv/.json")


if __name__ == "__main__":
    main()
