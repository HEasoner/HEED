"""Balance KNOW : HALL = 1 : 1 by down-sampling the majority class (seed 42).

    python scripts/balance_features.py --dataset popqa --model llama2_7b_chat     # -> features/<d>_<m>_balanced.pt
    python scripts/balance_features.py --all

Two equivalent-in-distribution orderings were used for the paper's files (both seed 42):
  "downsample"  random subset of each class, then shuffled                       (TriviaQA, MKQA, DeepSeek)
  "keep_know"   all KNOW rows in file order + random HALL subset, then shuffled  (PopQA / WebQuestions, other 5 LLMs)
LEGACY_KEEP_KNOW lists the settings of the second kind so that the released balanced files are reproduced
exactly (same rows, same order -> same 70/15/15 splits).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heed import config as C  # noqa: E402

LEGACY_KEEP_KNOW = {(d, m) for d in ("popqa", "webquestions") for m in C.MODELS if m != "deepseek_7b_chat"}


def balanced_indices(labels: torch.Tensor, seed: int = 42, keep_know: bool = False) -> torch.Tensor:
    know, hall = torch.where(labels == 0)[0], torch.where(labels == 1)[0]
    g = torch.Generator().manual_seed(seed)
    n = min(len(know), len(hall))
    if keep_know:
        assert len(know) <= len(hall), "keep_know assumes KNOW is the minority class"
        sel = torch.cat([know, hall[torch.randperm(len(hall), generator=g)[:n]]])
    else:
        sel = torch.cat([know[torch.randperm(len(know), generator=g)[:n]],
                         hall[torch.randperm(len(hall), generator=g)[:n]]])
    return sel[torch.randperm(len(sel), generator=g)]


def subset(value, idx: torch.Tensor, n: int):
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
        return value[idx]
    if isinstance(value, list) and len(value) == n:
        return [value[int(i)] for i in idx.tolist()]
    return value


def balance(d: str, m: str, seed: int = 42):
    data = torch.load(C.feature_path(d, m, balanced=False), map_location="cpu")
    labels = data["labels"].argmax(1).long()
    keep_know = (d, m) in LEGACY_KEEP_KNOW
    idx = balanced_indices(labels, seed, keep_know)
    out = {k: subset(v, idx, len(labels)) for k, v in data.items()}
    out["balance"] = dict(seed=seed, strategy="keep_know" if keep_know else "downsample",
                          source_counts=dict(know=int((labels == 0).sum()), hall=int((labels == 1).sum())),
                          selected_source_indices=idx.tolist())
    path = C.feature_path(d, m, balanced=True)
    torch.save(out, path)
    print(f">>> {path}: {len(idx)} rows (KNOW = HALL = {len(idx) // 2})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=C.DATASETS)
    ap.add_argument("--model", choices=list(C.MODELS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for d, m in (C.SETTINGS if a.all else [(a.dataset, a.model)]):
        balance(d, m)
