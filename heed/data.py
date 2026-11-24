"""Feature-file access, splits and train-only standardisation."""
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .config import HBAR_FEATURE


def load_features(path, mmap: bool = True) -> dict:
    return torch.load(str(path), map_location="cpu", mmap=mmap)


def labels_of(data: dict) -> torch.Tensor:
    """0 = KNOW (correct recall), 1 = HALL (false recall).  Stored one-hot as [KNOW, HALL]."""
    return data["labels"].argmax(1).long()


def split_indices(labels: torch.Tensor, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 train/validation/test split (seed-dependent)."""
    idx = np.arange(len(labels))
    y = labels.numpy()
    tr, rest = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    return tr, va, te


def hidden_h(data: dict) -> torch.Tensor:
    """h: answer-token mean of the hidden states of decoder blocks -4, -9, -14, concatenated (3d)."""
    return data["probe_target_hidden_mean_answer"].float().reshape(len(data["labels"]), -1)


def scalar_column(data: dict, name: str, group: str = "scalar_features", names: str = "scalar_names") -> torch.Tensor:
    return data[group][:, list(data[names]).index(name)].float()


def hbar(data: dict) -> torch.Tensor:
    """H-bar: mean teacher-forced predictive entropy over the answer keywords (N x 1)."""
    return torch.nan_to_num(scalar_column(data, HBAR_FEATURE)).unsqueeze(1)


def train_stats(x: torch.Tensor, tr: np.ndarray):
    xt = x[torch.as_tensor(tr)]
    return xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp_min(1e-6)


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num((x.float() - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_train(x: torch.Tensor, tr: np.ndarray) -> torch.Tensor:
    mean, std = train_stats(x, tr)
    return standardize(x, mean, std)
