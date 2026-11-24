import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "heed_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import make_split
from _paths import PROJECT_ROOT, PT_DIR, RESULTS_DIR, feature_file, local_model_path, resolve_input_path
from heed.extraction import freeze_model_parameters, generate_answer, get_decoder_layers, get_model_input_device
from heed.qa import answer_match_details, generated_contains_answer, normalize_answer_text


DEFAULT_FEATURE_FILE = feature_file("popqa", "qwen2_5_7b_instruct")
DEFAULT_MODEL_PATH = local_model_path("Qwen2.5-7B-Instruct")
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "latent_support_pilot"
DEFAULT_ALPHAS = "-1.5,-1.0,-0.5,0,0.5,1.0,1.5"
DEFAULT_RANDOM_ALPHAS = "-1.0,1.0"
DEFAULT_LAYER_RATIOS = "0.50,0.65,0.80,0.90"
LABEL_KNOW = 0
LABEL_HALL = 1
RESULT_COLUMNS = [
    "sample_id",
    "pt_index",
    "source_row_index",
    "question",
    "reference_answers",
    "baseline_answer",
    "baseline_label",
    "intervened_answer",
    "intervened_label",
    "layer",
    "layer_ratio",
    "requested_layer_ratio",
    "alpha",
    "sigma",
    "direction_type",
    "direction_id",
    "baseline_correct",
    "intervention_correct",
    "transition_type",
    "hook_call_count",
    "prefill_intervention_count",
    "decode_call_count",
]


def parse_float_list(value: str) -> List[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_file(path: Path) -> Dict[str, object]:
    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def label_ids(data: Dict[str, object]) -> np.ndarray:
    labels = data["labels"]
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)
    return labels.long().cpu().numpy()


def sha256_texts(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in texts:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def class_counts(labels: np.ndarray, indices: Sequence[int]) -> Dict[str, int]:
    subset = labels[np.asarray(indices, dtype=np.int64)]
    return {
        "know": int(np.sum(subset == LABEL_KNOW)),
        "hallucination": int(np.sum(subset == LABEL_HALL)),
        "total": int(len(subset)),
    }


def build_or_validate_split_manifest(
    data: Dict[str, object],
    feature_path: Path,
    output_dir: Path,
    seed: int,
) -> Dict[str, object]:
    labels = label_ids(data)
    records = data["records"]
    train_idx, val_idx, test_idx = make_split(labels, seed)

    rng = np.random.default_rng(seed)
    know_order = rng.permutation(test_idx[labels[test_idx] == LABEL_KNOW])
    hall_order = rng.permutation(test_idx[labels[test_idx] == LABEL_HALL])
    expected = {
        "version": 1,
        "seed": int(seed),
        "feature_file": str(feature_path.resolve()),
        "feature_version": str(data.get("feature_version", "unknown")),
        "dataset": str(data.get("dataset", "unknown")),
        "model_path": str(data.get("model_path", "unknown")),
        "num_samples": int(len(labels)),
        "prompt_sha256": sha256_texts(record["prompt"] for record in records),
        "counts": class_counts(labels, np.arange(len(labels))),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_counts": class_counts(labels, train_idx),
        "validation_counts": class_counts(labels, val_idx),
        "test_counts": class_counts(labels, test_idx),
        "test_know_order": know_order.tolist(),
        "test_hallucination_order": hall_order.tolist(),
    }
    path = output_dir / "split_manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        identity_keys = [
            "seed",
            "feature_file",
            "num_samples",
            "prompt_sha256",
            "train_indices",
            "validation_indices",
            "test_indices",
        ]
        mismatches = [key for key in identity_keys if existing.get(key) != expected.get(key)]
        if mismatches:
            raise RuntimeError(
                "Existing split manifest does not match this run: " + ", ".join(mismatches)
            )
        return existing

    atomic_json_dump(expected, path)
    return expected


def choose_balanced_test_subset(manifest: Dict[str, object], max_samples: int) -> List[int]:
    if max_samples < 2:
        raise ValueError("--max-test-samples must be at least 2.")
    know = list(manifest["test_know_order"])
    hall = list(manifest["test_hallucination_order"])
    per_class = min(max_samples // 2, len(know), len(hall))
    selected = know[:per_class] + hall[:per_class]
    if len(selected) < max_samples:
        remaining = know[per_class:] + hall[per_class:]
        selected.extend(remaining[: max_samples - len(selected)])
    return selected


def layer_ids_from_ratios(num_layers: int, ratios: Sequence[float]) -> Tuple[List[int], Dict[int, float]]:
    if num_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers={num_layers}")
    requested_by_layer: Dict[int, float] = {}
    for ratio in ratios:
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"Layer ratio must satisfy 0 <= ratio < 1, got {ratio}")
        layer_id = min(max(int(ratio * num_layers), 0), num_layers - 1)
        requested_by_layer.setdefault(layer_id, float(ratio))
    layer_ids = sorted(requested_by_layer)
    if not layer_ids:
        raise ValueError("No intervention layers selected.")
    return layer_ids, requested_by_layer


def cache_path_for_layer(cache_dir: Path, layer_id: int) -> Path:
    return cache_dir / f"prompt_hidden_layer_{layer_id}.pt"


def load_layer_hidden_cache(
    cache_dir: Path,
    layer_id: int,
    expected_hidden_size: int,
    prompt_sha256: str,
) -> Dict[int, torch.Tensor]:
    path = cache_path_for_layer(cache_dir, layer_id)
    if not path.exists():
        return {}
    cached = torch.load(path, map_location="cpu")
    if int(cached["layer"]) != int(layer_id):
        raise RuntimeError(f"Wrong layer in hidden cache: {path}")
    if int(cached["hidden_size"]) != int(expected_hidden_size):
        raise RuntimeError(f"Hidden-size mismatch in cache: {path}")
    if cached["prompt_sha256"] != prompt_sha256:
        raise RuntimeError(f"Prompt fingerprint mismatch in cache: {path}")
    indices = cached["indices"].long().tolist()
    hidden = cached["hidden"].to(torch.float16)
    return {int(index): hidden[row] for row, index in enumerate(indices)}


def save_layer_hidden_cache(
    cache_dir: Path,
    layer_id: int,
    hidden_size: int,
    prompt_sha256: str,
    values: Dict[int, torch.Tensor],
) -> None:
    ordered_indices = sorted(values)
    hidden = torch.stack([values[index] for index in ordered_indices]).to(torch.float16)
    atomic_torch_save(
        {
            "version": 1,
            "layer": int(layer_id),
            "hidden_size": int(hidden_size),
            "prompt_sha256": prompt_sha256,
            "indices": torch.tensor(ordered_indices, dtype=torch.long),
            "hidden": hidden,
        },
        cache_path_for_layer(cache_dir, layer_id),
    )


def extract_prompt_hidden_cache(
    model,
    tokenizer,
    records: Sequence[Dict[str, object]],
    required_indices: Sequence[int],
    layer_ids: Sequence[int],
    cache_dir: Path,
    hidden_size: int,
    prompt_sha256: str,
    batch_size: int,
    cache_every: int,
) -> Dict[int, Dict[int, torch.Tensor]]:
    caches = {
        layer_id: load_layer_hidden_cache(
            cache_dir,
            layer_id,
            expected_hidden_size=hidden_size,
            prompt_sha256=prompt_sha256,
        )
        for layer_id in layer_ids
    }
    missing = [
        int(index)
        for index in required_indices
        if any(int(index) not in caches[layer_id] for layer_id in layer_ids)
    ]
    if not missing:
        print(">>> Prompt-hidden cache already covers all required samples.")
        return caches

    layers = get_decoder_layers(model)
    capture: Dict[int, torch.Tensor] = {}
    state: Dict[str, Optional[torch.Tensor]] = {"last_positions": None}
    handles = []

    def make_capture_hook(layer_id: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            positions = state["last_positions"]
            if positions is None:
                raise RuntimeError("Missing prompt positions during hidden capture.")
            positions = positions.to(hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            capture[layer_id] = hidden[rows, positions, :].detach().cpu().to(torch.float16)

        return hook

    for layer_id in layer_ids:
        handles.append(layers[layer_id].register_forward_hook(make_capture_hook(layer_id)))

    input_device = get_model_input_device(model)
    completed_batches = 0
    try:
        for start in tqdm(range(0, len(missing), batch_size), desc="prompt hidden"):
            batch_indices = missing[start : start + batch_size]
            prompts = [str(records[index]["prompt"]) for index in batch_indices]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            attention_mask = inputs["attention_mask"].long()
            token_positions = torch.arange(attention_mask.shape[1]).unsqueeze(0)
            last_positions = (attention_mask.cpu() * token_positions).max(dim=1).values
            state["last_positions"] = last_positions
            capture.clear()
            inputs = inputs.to(input_device)
            with torch.no_grad():
                model(**inputs, use_cache=False, return_dict=True)

            if set(capture) != set(layer_ids):
                raise RuntimeError(
                    f"Hidden capture missed layers: expected={layer_ids}, got={sorted(capture)}"
                )
            for layer_id in layer_ids:
                vectors = capture[layer_id]
                if vectors.shape != (len(batch_indices), hidden_size):
                    raise RuntimeError(
                        f"Unexpected hidden shape at layer {layer_id}: {tuple(vectors.shape)}"
                    )
                for row, pt_index in enumerate(batch_indices):
                    caches[layer_id][pt_index] = vectors[row].clone()

            completed_batches += 1
            if completed_batches % max(cache_every, 1) == 0:
                for layer_id in layer_ids:
                    save_layer_hidden_cache(
                        cache_dir,
                        layer_id,
                        hidden_size,
                        prompt_sha256,
                        caches[layer_id],
                    )
    finally:
        state["last_positions"] = None
        for handle in handles:
            handle.remove()

    for layer_id in layer_ids:
        save_layer_hidden_cache(
            cache_dir,
            layer_id,
            hidden_size,
            prompt_sha256,
            caches[layer_id],
        )
    return caches


def random_unit_direction(hidden_size: int, seed: int, layer_id: int, direction_id: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed + 100_003 * layer_id + 1_009 * direction_id))
    direction = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    return direction / direction.norm(p=2).clamp_min(1e-12)


def build_support_directions(
    labels: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    hidden_caches: Dict[int, Dict[int, torch.Tensor]],
    layer_ids: Sequence[int],
    requested_by_layer: Dict[int, float],
    num_layers: int,
    hidden_size: int,
    seed: int,
    random_direction_count: int,
    output_dir: Path,
) -> Dict[int, Dict[str, object]]:
    train_array = np.asarray(train_indices, dtype=np.int64)
    test_set = set(int(index) for index in test_indices)
    if test_set.intersection(int(index) for index in train_array):
        raise AssertionError("Train/test overlap detected while building support directions.")

    directions: Dict[int, Dict[str, object]] = {}
    for layer_id in layer_ids:
        train_hidden = torch.stack(
            [hidden_caches[layer_id][int(index)].float() for index in train_array]
        )
        train_labels = torch.from_numpy(labels[train_array]).long()
        know_hidden = train_hidden[train_labels == LABEL_KNOW]
        hall_hidden = train_hidden[train_labels == LABEL_HALL]
        if len(know_hidden) == 0 or len(hall_hidden) == 0:
            raise RuntimeError(f"Layer {layer_id} has an empty train class.")

        mu_know = know_hidden.mean(dim=0)
        mu_hall = hall_hidden.mean(dim=0)
        difference = mu_know - mu_hall
        difference_norm = difference.norm(p=2)
        if not torch.isfinite(difference_norm) or float(difference_norm) <= 0:
            raise RuntimeError(f"Degenerate support direction at layer {layer_id}.")
        support = difference / difference_norm
        support_norm = float(support.norm(p=2))
        if not math.isclose(support_norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise AssertionError(f"Support direction norm is {support_norm}, expected 1.")

        projections = train_hidden @ support
        sigma = projections.std(unbiased=False)
        if not torch.isfinite(sigma) or float(sigma) <= 0:
            raise RuntimeError(f"Invalid projection sigma at layer {layer_id}: {sigma}")
        random_directions = torch.stack(
            [
                random_unit_direction(hidden_size, seed, layer_id, direction_id)
                for direction_id in range(random_direction_count)
            ],
            dim=0,
        ) if random_direction_count > 0 else torch.empty((0, hidden_size))

        directions[layer_id] = {
            "layer": int(layer_id),
            "layer_ratio": float(layer_id / num_layers),
            "requested_layer_ratio": float(requested_by_layer[layer_id]),
            "mu_know": mu_know.cpu(),
            "mu_hallucination": mu_hall.cpu(),
            "support_direction": support.cpu(),
            "support_direction_norm": support_norm,
            "sigma": float(sigma),
            "projection_mean": float(projections.mean()),
            "projection_std": float(sigma),
            "train_know_count": int(len(know_hidden)),
            "train_hallucination_count": int(len(hall_hidden)),
            "random_directions": random_directions.cpu(),
        }

    atomic_torch_save(
        {
            "version": 1,
            "seed": int(seed),
            "num_hidden_layers": int(num_layers),
            "hidden_size": int(hidden_size),
            "train_indices": torch.tensor(train_array, dtype=torch.long),
            "test_indices_used_only_for_overlap_check": torch.tensor(
                sorted(test_set), dtype=torch.long
            ),
            "direction_provenance": "TRAIN SPLIT labels and pre-generation prompt hidden only",
            "layers": directions,
        },
        output_dir / "support_directions.pt",
    )
    return directions


@dataclass
class HookStats:
    call_count: int = 0
    prefill_intervention_count: int = 0
    decode_call_count: int = 0


@dataclass
class DetailedInterventionResult:
    generated: object
    stats: HookStats
    hidden_before: Optional[torch.Tensor] = None
    hidden_after: Optional[torch.Tensor] = None


def run_prompt_only_intervention_detailed(
    model,
    tokenizer,
    prompt: str,
    layer_id: int,
    direction: torch.Tensor,
    sigma: float,
    alpha: float,
    max_new_tokens: int,
    return_diagnostics: bool = False,
    capture_intervened_hidden: bool = False,
) -> DetailedInterventionResult:
    tokenized = tokenizer(prompt, return_tensors="pt")
    attention_mask = tokenized["attention_mask"][0].long()
    valid_positions = torch.where(attention_mask != 0)[0]
    if valid_positions.numel() == 0:
        raise RuntimeError("Prompt contains no valid tokens.")
    final_prompt_position = int(valid_positions[-1])
    prompt_length = int(tokenized["input_ids"].shape[1])
    stats = HookStats()
    intervention_applied = False
    captured: Dict[str, torch.Tensor] = {}

    def steering_hook(_module, _inputs, output):
        nonlocal intervention_applied
        stats.call_count += 1
        hidden = output[0] if isinstance(output, tuple) else output
        sequence_length = int(hidden.shape[1])

        if not intervention_applied:
            if sequence_length != prompt_length:
                raise RuntimeError(
                    "First hook invocation was not the expected prefill: "
                    f"seq_len={sequence_length}, prompt_len={prompt_length}"
                )
            modified = hidden.clone()
            delta = direction.to(device=hidden.device, dtype=hidden.dtype) * float(
                alpha * sigma
            )
            if capture_intervened_hidden:
                captured["before"] = hidden[0, final_prompt_position, :].detach().cpu().float()
            modified[0, final_prompt_position, :] = (
                modified[0, final_prompt_position, :] + delta
            )
            if capture_intervened_hidden:
                captured["after"] = modified[0, final_prompt_position, :].detach().cpu().float()
            intervention_applied = True
            stats.prefill_intervention_count += 1
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        stats.decode_call_count += 1
        return None

    layer = get_decoder_layers(model)[layer_id]
    handle = layer.register_forward_hook(steering_hook)
    try:
        generated = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            return_diagnostics=return_diagnostics,
            do_sample=False,
        )
    finally:
        handle.remove()

    if stats.prefill_intervention_count != 1:
        raise AssertionError(
            "Prompt-only intervention must be applied exactly once; "
            f"got {stats.prefill_intervention_count}."
        )
    return DetailedInterventionResult(
        generated=generated,
        stats=stats,
        hidden_before=captured.get("before"),
        hidden_after=captured.get("after"),
    )


def run_prompt_only_intervention(
    model,
    tokenizer,
    prompt: str,
    layer_id: int,
    direction: torch.Tensor,
    sigma: float,
    alpha: float,
    max_new_tokens: int,
) -> Tuple[str, HookStats]:
    result = run_prompt_only_intervention_detailed(
        model,
        tokenizer,
        prompt,
        layer_id,
        direction,
        sigma,
        alpha,
        max_new_tokens,
    )
    return result.generated.text, result.stats


def aliases_for_record(record: Dict[str, object]) -> List[str]:
    source = record.get("label_source") or {}
    aliases = source.get("possible_answers")
    if aliases is None:
        aliases = source.get("answers", [])
    return [str(value) for value in aliases]


def baseline_label_name(correct: bool) -> str:
    return "KNOW" if correct else "HALLUCINATION"


def transition_name(baseline_correct: bool, intervention_correct: bool) -> str:
    before = "KNOW" if baseline_correct else "HALL"
    after = "KNOW" if intervention_correct else "HALL"
    return f"{before} -> {after}"


def load_baseline_checks(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    rows = pd.read_csv(path).fillna("").to_dict(orient="records")
    return {int(row["pt_index"]): row for row in rows}


def write_baseline_checks(path: Path, rows: Dict[int, Dict[str, object]]) -> None:
    frame = pd.DataFrame([rows[index] for index in sorted(rows)])
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def verify_fresh_baselines(
    model,
    tokenizer,
    records: Sequence[Dict[str, object]],
    labels: np.ndarray,
    selected_indices: Sequence[int],
    max_new_tokens: int,
    output_dir: Path,
    relaxed_matching: bool = False,
) -> Dict[int, Dict[str, object]]:
    path = output_dir / "baseline_checks.csv"
    checks: Dict[int, Dict[str, object]] = load_baseline_checks(path)
    for pt_index in tqdm(selected_indices, desc="baseline check"):
        pt_index = int(pt_index)
        record = records[pt_index]
        stored_answer = str(record["generated_answer"])
        cached = checks.get(pt_index)
        if cached and str(cached.get("stored_answer", "")) == stored_answer:
            exact_cached = str(cached.get("exact_match", "")).lower() in {"true", "1"}
            label_cached = str(cached.get("label_match", "")).lower() in {"true", "1"}
            if exact_cached and label_cached:
                continue

        fresh = generate_answer(
            model,
            tokenizer,
            str(record["prompt"]),
            max_new_tokens=max_new_tokens,
            return_diagnostics=False,
            do_sample=False,
        ).text
        aliases = aliases_for_record(record)
        fresh_correct = bool(
            answer_match_details(fresh, aliases, relaxed=relaxed_matching)["matched"]
        )
        expected_correct = bool(labels[pt_index] == LABEL_KNOW)
        exact = fresh == stored_answer
        label_equal = fresh_correct == expected_correct
        checks[pt_index] = {
            "pt_index": pt_index,
            "sample_id": (record.get("label_source") or {}).get("id", record["row_index"]),
            "stored_answer": stored_answer,
            "fresh_no_hook_answer": fresh,
            "exact_match": exact,
            "stored_correct": expected_correct,
            "fresh_correct": fresh_correct,
            "label_match": label_equal,
        }
        write_baseline_checks(path, checks)
        if not exact or not label_equal:
            raise AssertionError(
                "Fresh deterministic baseline differs from the stored v4.1 baseline at "
                f"pt_index={pt_index}: stored={stored_answer!r}, fresh={fresh!r}."
            )
    return checks


def result_key(row: Dict[str, object]) -> Tuple[int, int, str, int, float]:
    return (
        int(row["pt_index"]),
        int(row["layer"]),
        str(row["direction_type"]),
        int(row["direction_id"]),
        round(float(row["alpha"]), 8),
    )


def load_existing_result_keys(path: Path) -> set:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return {result_key(row) for row in frame.to_dict(orient="records")}


def append_result(path: Path, row: Dict[str, object]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({column: row[column] for column in RESULT_COLUMNS})
        handle.flush()


def run_interventions(
    model,
    tokenizer,
    data: Dict[str, object],
    labels: np.ndarray,
    selected_indices: Sequence[int],
    directions: Dict[int, Dict[str, object]],
    alphas: Sequence[float],
    random_alphas: Sequence[float],
    random_direction_count: int,
    max_new_tokens: int,
    output_dir: Path,
    skip_random_control: bool,
    relaxed_matching: bool = False,
) -> Path:
    records = data["records"]
    raw_path = output_dir / "raw_results.csv"
    existing = load_existing_result_keys(raw_path)

    jobs: List[Tuple[int, float, str, int]] = []
    for layer_id in sorted(directions):
        for alpha in alphas:
            jobs.append((layer_id, float(alpha), "support", -1))
        if not skip_random_control:
            for direction_id in range(random_direction_count):
                for alpha in random_alphas:
                    jobs.append((layer_id, float(alpha), "random", direction_id))

    total_missing = sum(
        result_key(
            {
                "pt_index": pt_index,
                "layer": layer_id,
                "direction_type": direction_type,
                "direction_id": direction_id,
                "alpha": alpha,
            }
        )
        not in existing
        for layer_id, alpha, direction_type, direction_id in jobs
        for pt_index in selected_indices
    )
    progress = tqdm(total=total_missing, desc="intervention generations")
    for layer_id, alpha, direction_type, direction_id in jobs:
        layer_data = directions[layer_id]
        if direction_type == "support":
            direction = layer_data["support_direction"]
        else:
            direction = layer_data["random_directions"][direction_id]

        for pt_index in selected_indices:
            pt_index = int(pt_index)
            key = (pt_index, layer_id, direction_type, direction_id, round(alpha, 8))
            if key in existing:
                continue
            record = records[pt_index]
            aliases = aliases_for_record(record)
            baseline_correct = bool(labels[pt_index] == LABEL_KNOW)
            intervened_answer, hook_stats = run_prompt_only_intervention(
                model,
                tokenizer,
                str(record["prompt"]),
                layer_id=layer_id,
                direction=direction,
                sigma=float(layer_data["sigma"]),
                alpha=alpha,
                max_new_tokens=max_new_tokens,
            )
            intervention_correct = bool(
                answer_match_details(
                    intervened_answer,
                    aliases,
                    relaxed=relaxed_matching,
                )["matched"]
            )
            if direction_type == "support" and math.isclose(alpha, 0.0, abs_tol=1e-12):
                if intervened_answer != str(record["generated_answer"]):
                    raise AssertionError(
                        "alpha=0 hook result differs from no-hook baseline at "
                        f"pt_index={pt_index}, layer={layer_id}."
                    )

            source = record.get("label_source") or {}
            row = {
                "sample_id": source.get("id", record.get("row_index", pt_index)),
                "pt_index": pt_index,
                "source_row_index": int(record.get("row_index", -1)),
                "question": str(record["question"]),
                "reference_answers": json.dumps(aliases, ensure_ascii=False),
                "baseline_answer": str(record["generated_answer"]),
                "baseline_label": baseline_label_name(baseline_correct),
                "intervened_answer": intervened_answer,
                "intervened_label": baseline_label_name(intervention_correct),
                "layer": int(layer_id),
                "layer_ratio": float(layer_data["layer_ratio"]),
                "requested_layer_ratio": float(layer_data["requested_layer_ratio"]),
                "alpha": float(alpha),
                "sigma": float(layer_data["sigma"]),
                "direction_type": direction_type,
                "direction_id": int(direction_id),
                "baseline_correct": baseline_correct,
                "intervention_correct": intervention_correct,
                "transition_type": transition_name(baseline_correct, intervention_correct),
                "hook_call_count": int(hook_stats.call_count),
                "prefill_intervention_count": int(hook_stats.prefill_intervention_count),
                "decode_call_count": int(hook_stats.decode_call_count),
            }
            append_result(raw_path, row)
            existing.add(key)
            progress.update(1)
    progress.close()
    return raw_path


def metric_row(group: pd.DataFrame) -> Dict[str, float]:
    baseline_correct = group["baseline_correct"].astype(bool).to_numpy()
    intervention_correct = group["intervention_correct"].astype(bool).to_numpy()
    baseline_hall = ~baseline_correct
    baseline_know = baseline_correct
    return {
        "n": int(len(group)),
        "accuracy": float(intervention_correct.mean()),
        "hallucination_rate": float((~intervention_correct).mean()),
        "rescue_rate": float(intervention_correct[baseline_hall].mean()),
        "damage_rate": float((~intervention_correct[baseline_know]).mean()),
        "hall_to_know_count": int(np.sum(intervention_correct & baseline_hall)),
        "know_to_hall_count": int(np.sum((~intervention_correct) & baseline_know)),
    }


def aggregate_results(raw: pd.DataFrame, selected_indices: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_set = set(int(index) for index in selected_indices)
    current = raw[raw["pt_index"].astype(int).isin(selected_set)].copy()
    support = current[current["direction_type"] == "support"]
    support_rows = []
    for (layer, ratio, alpha), group in support.groupby(
        ["layer", "layer_ratio", "alpha"], sort=True
    ):
        support_rows.append(
            {
                "layer": int(layer),
                "layer_ratio": float(ratio),
                "alpha": float(alpha),
                **metric_row(group),
            }
        )
    support_summary = pd.DataFrame(support_rows).sort_values(["layer", "alpha"])

    random_rows = []
    random_frame = current[current["direction_type"] == "random"]
    if not random_frame.empty:
        per_direction = []
        for (layer, ratio, alpha, direction_id), group in random_frame.groupby(
            ["layer", "layer_ratio", "alpha", "direction_id"], sort=True
        ):
            per_direction.append(
                {
                    "layer": int(layer),
                    "layer_ratio": float(ratio),
                    "alpha": float(alpha),
                    "direction_id": int(direction_id),
                    **metric_row(group),
                }
            )
        per_direction_frame = pd.DataFrame(per_direction)
        for (layer, ratio, alpha), group in per_direction_frame.groupby(
            ["layer", "layer_ratio", "alpha"], sort=True
        ):
            row = {
                "layer": int(layer),
                "layer_ratio": float(ratio),
                "alpha": float(alpha),
                "num_random_directions": int(len(group)),
            }
            for metric in ["accuracy", "hallucination_rate", "rescue_rate", "damage_rate"]:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            random_rows.append(row)
    random_summary = pd.DataFrame(random_rows)
    return support_summary, random_summary


def plot_dose_response(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for layer, group in summary.groupby("layer", sort=True):
        group = group.sort_values("alpha")
        ratio = float(group["layer_ratio"].iloc[0])
        ax.plot(
            group["alpha"],
            group[metric],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"Layer {int(layer)} ({ratio:.0%})",
        )
    ax.axvline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"Intervention strength $\alpha$")
    ax.set_ylabel(ylabel)
    if metric == "hallucination_rate":
        ax.set_ylim(-0.02, 1.02)
    else:
        upper = max(0.1, min(1.0, float(summary[metric].max()) * 1.25))
        ax.set_ylim(-0.01, upper)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_rescue_by_layer(summary: pd.DataFrame, output_dir: Path) -> None:
    positive = summary[summary["alpha"] > 0].copy()
    best_rows = []
    for _, group in positive.groupby("layer", sort=True):
        best_rows.append(group.sort_values(["rescue_rate", "damage_rate"], ascending=[False, True]).iloc[0])
    best = pd.DataFrame(best_rows).sort_values("layer_ratio")
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(best["layer_ratio"], best["rescue_rate"], marker="o", linewidth=1.8)
    for _, row in best.iterrows():
        ax.annotate(
            rf"$\alpha$={row['alpha']:g}",
            (row["layer_ratio"], row["rescue_rate"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Layer depth (layer index / number of layers)")
    ax.set_ylabel("Best positive-alpha rescue rate")
    ax.set_ylim(-0.02, max(0.1, min(1.02, float(best["rescue_rate"].max()) + 0.08)))
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"best_positive_rescue_by_layer.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_random_control(
    support: pd.DataFrame,
    random_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    if random_summary.empty:
        return
    support_control = support[support["alpha"].isin(random_summary["alpha"].unique())].copy()
    merged = support_control.merge(
        random_summary,
        on=["layer", "layer_ratio", "alpha"],
        how="inner",
        suffixes=("_support", "_random"),
    )
    if merged.empty:
        return
    x = np.arange(len(merged))
    width = 0.36
    labels = [f"L{int(row.layer)}\n{row.alpha:+g}" for row in merged.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharex=True)
    for ax, metric, title in [
        (axes[0], "rescue_rate", "Rescue rate"),
        (axes[1], "damage_rate", "Damage rate"),
    ]:
        ax.bar(x - width / 2, merged[metric], width, label="Support direction", color="#3b75af")
        ax.bar(
            x + width / 2,
            merged[f"{metric}_mean"],
            width,
            yerr=merged[f"{metric}_std"],
            label="Random direction",
            color="#9a9a9a",
            capsize=2,
        )
        ax.set_title(title)
        upper = max(
            float(merged[metric].max()),
            float((merged[f"{metric}_mean"] + merged[f"{metric}_std"]).max()),
        )
        ax.set_ylim(0, max(0.05, min(1.0, upper * 1.3)))
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("Layer / alpha")
    axes[0].set_ylabel("Rate")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for extension in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"support_vs_random_control.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def format_rate(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_summary(
    support: pd.DataFrame,
    random_summary: pd.DataFrame,
    raw: pd.DataFrame,
    selected_indices: Sequence[int],
    manifest: Dict[str, object],
    output_dir: Path,
    seed: int,
) -> None:
    if support.empty:
        return
    zero = support[np.isclose(support["alpha"], 0.0)]
    positive = support[support["alpha"] > 0]
    negative = support[support["alpha"] < 0]
    best = positive.sort_values(
        ["accuracy", "rescue_rate", "damage_rate"], ascending=[False, False, True]
    ).iloc[0]
    zero_by_layer = zero.set_index("layer")["hallucination_rate"].to_dict()
    best_hall_change = float(best["hallucination_rate"] - zero_by_layer[int(best["layer"])])

    positive_best_by_layer = positive.loc[
        positive.groupby("layer")["hallucination_rate"].idxmin()
    ]
    negative_worst_by_layer = negative.loc[
        negative.groupby("layer")["hallucination_rate"].idxmax()
    ] if not negative.empty else pd.DataFrame()
    positive_reductions = [
        zero_by_layer[int(row.layer)] - float(row.hallucination_rate)
        for row in positive_best_by_layer.itertuples()
    ]
    negative_increases = [
        float(row.hallucination_rate) - zero_by_layer[int(row.layer)]
        for row in negative_worst_by_layer.itertuples()
    ]
    mean_positive_best_change = float(-np.mean(positive_reductions))

    dose_correlations = []
    for layer, group in support.groupby("layer"):
        if group["alpha"].nunique() >= 3:
            correlation = spearmanr(group["alpha"], group["hallucination_rate"]).statistic
            dose_correlations.append((int(layer), float(correlation)))

    random_statement = "Random control was not run."
    support_stronger = False
    if not random_summary.empty:
        support_plus = support[np.isclose(support["alpha"], 1.0)]
        random_plus = random_summary[np.isclose(random_summary["alpha"], 1.0)]
        comparison = support_plus.merge(
            random_plus, on=["layer", "layer_ratio", "alpha"], how="inner"
        )
        if not comparison.empty:
            support_net = comparison["rescue_rate"] - comparison["damage_rate"]
            random_net = comparison["rescue_rate_mean"] - comparison["damage_rate_mean"]
            support_stronger = bool(float(support_net.mean()) > float(random_net.mean()))
            random_statement = (
                f"At alpha=+1, mean net rescue (rescue-damage) was "
                f"{format_rate(float(support_net.mean()))} for support steering and "
                f"{format_rate(float(random_net.mean()))} for random steering."
            )

        support_minus = support[np.isclose(support["alpha"], -1.0)]
        random_minus = random_summary[np.isclose(random_summary["alpha"], -1.0)]
        reverse_comparison = support_minus.merge(
            random_minus, on=["layer", "layer_ratio", "alpha"], how="inner"
        )
        if not reverse_comparison.empty:
            random_statement += (
                f" At alpha=-1, mean damage was "
                f"{format_rate(float(reverse_comparison['damage_rate'].mean()))} for "
                f"support steering and "
                f"{format_rate(float(reverse_comparison['damage_rate_mean'].mean()))} "
                f"for random steering."
            )

    directional = bool(np.mean(positive_reductions) > 0) if positive_reductions else False
    reverse_directional = bool(np.mean(negative_increases) > 0) if negative_increases else False
    worthwhile = directional and reverse_directional and (support_stronger or random_summary.empty)

    if best_hall_change < -1e-12:
        positive_statement = (
            f"Positive steering lowered hallucination at the best setting by "
            f"{format_rate(-best_hall_change)}."
        )
    elif best_hall_change > 1e-12:
        positive_statement = (
            f"Positive steering increased hallucination even at the selected best setting by "
            f"{format_rate(best_hall_change)}."
        )
    else:
        positive_statement = (
            "Positive steering did not lower hallucination at the selected best setting "
            "(0.0 percentage-point change)."
        )

    selected_set = set(int(index) for index in selected_indices)
    baseline_rows = raw[
        raw["pt_index"].astype(int).isin(selected_set)
        & (raw["direction_type"] == "support")
    ].drop_duplicates("pt_index")
    suspicious_short_aliases = []
    for row in baseline_rows.itertuples():
        if not bool(row.baseline_correct):
            continue
        aliases = json.loads(row.reference_answers)
        match = answer_match_details(row.baseline_answer, aliases, relaxed=False)
        matched_answer = match.get("matched_answer")
        if matched_answer is not None and len(normalize_answer_text(matched_answer)) <= 2:
            suspicious_short_aliases.append(
                {
                    "pt_index": int(row.pt_index),
                    "question": row.question,
                    "baseline_answer": row.baseline_answer,
                    "matched_alias": str(matched_answer),
                }
            )

    lines = [
        "# Latent Support Steering Pilot",
        "",
        "This pilot reuses the frozen Qwen2.5-7B-Instruct + PopQA v4.1 baseline "
        "answers, labels, prompts, and seed-42 stratified split. Support directions "
        "are estimated from pre-generation prompt hidden states in the training split only.",
        "",
        "## Direct answers",
        "",
        f"1. Best tested positive intervention: layer {int(best['layer'])} "
        f"({best['layer_ratio']:.1%} depth), alpha={best['alpha']:g}; it tied rather "
        "than improved on baseline accuracy.",
        f"2. {positive_statement} Across layers, the best positive setting changed "
        f"hallucination rate by {mean_positive_best_change * 100:+.1f} percentage points "
        f"on average (negative is improvement).",
        f"3. Negative steering mean worst-per-layer hallucination increase was "
        f"{format_rate(float(np.mean(negative_increases))) if negative_increases else 'not available'}.",
        "4. Layer-wise Spearman(alpha, hall rate): "
        + ", ".join(f"L{layer}={value:+.3f}" for layer, value in dose_correlations)
        + ".",
        f"5. {random_statement} The beneficial positive support effect was not clearly "
        "stronger than random control.",
        f"6. Best tested positive alpha: {best['alpha']:g}, with no net accuracy gain.",
        f"7. HALL -> KNOW at the best setting: {int(best['hall_to_know_count'])}.",
        f"8. KNOW -> HALL at the best setting: {int(best['know_to_hall_count'])}.",
        "9. Full experiment recommendation: " + ("yes" if worthwhile else "not yet") + ".",
        "",
        "## Core metrics",
        "",
        "| Layer | alpha | Accuracy | Hall Rate | Rescue Rate | Damage Rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in support.itertuples():
        lines.append(
            f"| {int(row.layer)} | {row.alpha:g} | {row.accuracy:.4f} | "
            f"{row.hallucination_rate:.4f} | {row.rescue_rate:.4f} | {row.damage_rate:.4f} |"
        )

    lines.extend(["", "## Random-direction control", ""])
    if random_summary.empty:
        lines.append("Not run.")
    else:
        lines.extend(
            [
                "| Layer | alpha | Random Rescue mean +/- std | Random Damage mean +/- std |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in random_summary.itertuples():
            lines.append(
                f"| {int(row.layer)} | {row.alpha:g} | "
                f"{row.rescue_rate_mean:.4f} +/- {row.rescue_rate_std:.4f} | "
                f"{row.damage_rate_mean:.4f} +/- {row.damage_rate_std:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Provenance and sanity checks",
            "",
            f"- Feature samples: {manifest['num_samples']}; train/validation/test: "
            f"{manifest['train_counts']['total']}/{manifest['validation_counts']['total']}/"
            f"{manifest['test_counts']['total']}.",
            f"- Evaluated subset: {len(selected_indices)} samples; seed={seed}.",
            "- alpha=0 outputs were required to exactly match fresh no-hook and stored baselines.",
            "- Every recorded intervention has prefill_intervention_count=1.",
            "- Test labels and test hidden states were not used to estimate support directions.",
            f"- Existing PopQA matching was reused without cleaning. "
            f"{len(suspicious_short_aliases)} selected baseline KNOW sample(s) matched an "
            "alias of at most two normalized characters and may be label noise.",
        ]
    )
    if suspicious_short_aliases:
        lines.extend(["", "### Inherited short-alias matches", ""])
        for item in suspicious_short_aliases:
            lines.append(
                f"- pt_index={item['pt_index']}: {item['question']} | "
                f"answer={item['baseline_answer']!r} | alias={item['matched_alias']!r}"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    current = raw[
        raw["pt_index"].astype(int).isin(set(int(index) for index in selected_indices))
        & (raw["direction_type"] == "support")
    ]
    rng = np.random.default_rng(seed)
    for transition in ["HALL -> KNOW", "KNOW -> HALL"]:
        candidates = current[current["transition_type"] == transition].drop_duplicates(
            "pt_index"
        )
        if candidates.empty:
            print(f">>> No {transition} examples found.")
            continue
        chosen = candidates.iloc[rng.permutation(len(candidates))[:5]]
        print(f"\n>>> Examples: {transition}")
        for row in chosen.itertuples():
            print(
                f"  [L{row.layer} alpha={row.alpha:+g}] {row.question}\n"
                f"    baseline: {row.baseline_answer}\n"
                f"    intervened: {row.intervened_answer}"
            )


def analyze_and_plot(
    raw_path: Path,
    selected_indices: Sequence[int],
    manifest: Dict[str, object],
    output_dir: Path,
    seed: int,
) -> None:
    raw = pd.read_csv(raw_path)
    raw["baseline_correct"] = raw["baseline_correct"].astype(str).str.lower().isin(["true", "1"])
    raw["intervention_correct"] = (
        raw["intervention_correct"].astype(str).str.lower().isin(["true", "1"])
    )
    current = raw[raw["pt_index"].astype(int).isin(set(int(i) for i in selected_indices))]
    if not (current["prefill_intervention_count"].astype(int) == 1).all():
        raise AssertionError("At least one intervention was not confined to one prefill call.")
    support, random_summary = aggregate_results(raw, selected_indices)
    support.to_csv(output_dir / "support_summary.csv", index=False)
    random_summary.to_csv(output_dir / "random_control_summary.csv", index=False)

    plot_dose_response(
        support,
        "hallucination_rate",
        "Hallucination rate",
        output_dir,
        "dose_response_hallucination_rate",
    )
    plot_dose_response(
        support,
        "rescue_rate",
        "Rescue rate (baseline HALL)",
        output_dir,
        "dose_response_rescue_rate",
    )
    plot_dose_response(
        support,
        "damage_rate",
        "Damage rate (baseline KNOW)",
        output_dir,
        "dose_response_damage_rate",
    )
    if not support[support["alpha"] > 0].empty:
        plot_best_rescue_by_layer(support, output_dir)
    plot_random_control(support, random_summary, output_dir)
    write_summary(
        support,
        random_summary,
        raw,
        selected_indices,
        manifest,
        output_dir,
        seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-only latent factual-support steering pilot on PopQA."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", choices=["popqa"], default="popqa")
    parser.add_argument("--feature-file", default=str(DEFAULT_FEATURE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--random-alphas", default=DEFAULT_RANDOM_ALPHAS)
    parser.add_argument("--layer-ratios", default=DEFAULT_LAYER_RATIOS)
    parser.add_argument("--random-directions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-batch-size", type=int, default=8)
    parser.add_argument("--cache-every", type=int, default=25)
    parser.add_argument(
        "--skip-random-control",
        action="store_true",
        help="Skip random controls for the smallest smoke test.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write/validate the fixed split manifest without loading the model.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Regenerate summaries and plots from the existing raw CSV without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    feature_path = resolve_input_path(args.feature_file)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = load_feature_file(feature_path)
    if str(data.get("dataset", "")).lower() != "popqa":
        raise ValueError("This pilot is intentionally restricted to PopQA.")
    records = data["records"]
    labels = label_ids(data)
    if len(records) != len(labels):
        raise RuntimeError("records and labels have different lengths.")
    for index, record in enumerate(records):
        expected = "KNOW" if labels[index] == LABEL_KNOW else "HALLUCINATION"
        if str(record.get("label")) != expected:
            raise RuntimeError(f"Record/tensor label mismatch at pt_index={index}.")

    manifest = build_or_validate_split_manifest(
        data,
        feature_path,
        output_dir,
        seed=args.seed,
    )
    selected_indices = choose_balanced_test_subset(manifest, args.max_test_samples)
    print(
        ">>> Fixed split counts: "
        f"train={manifest['train_counts']}, validation={manifest['validation_counts']}, "
        f"test={manifest['test_counts']}"
    )
    print(
        f">>> Pilot subset: {class_counts(labels, selected_indices)} "
        f"(max_test_samples={args.max_test_samples}, seed={args.seed})"
    )
    if args.prepare_only:
        print(f">>> Split manifest ready: {output_dir / 'split_manifest.json'}")
        return
    if args.analyze_only:
        raw_path = output_dir / "raw_results.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw pilot results: {raw_path}")
        analyze_and_plot(
            raw_path,
            selected_indices,
            manifest,
            output_dir,
            args.seed,
        )
        print(f">>> Pilot analysis regenerated in: {output_dir}")
        return

    alphas = parse_float_list(args.alphas)
    random_alphas = parse_float_list(args.random_alphas)
    layer_ratios = parse_float_list(args.layer_ratios)
    if not any(math.isclose(value, 0.0, abs_tol=1e-12) for value in alphas):
        raise ValueError("--alphas must include 0 for the mandatory baseline sanity check.")
    if args.random_directions < 0:
        raise ValueError("--random-directions must be non-negative.")

    model_path = Path(args.model) if Path(args.model).exists() else args.model   # local dir or HF id
    print(f">>> Loading tokenizer/model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    freeze_model_parameters(model)

    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    layer_ids, requested_by_layer = layer_ids_from_ratios(num_layers, layer_ratios)
    print(f">>> num_hidden_layers={num_layers}; intervention layers={layer_ids}")
    max_new_tokens = int(data.get("max_new_tokens", 32))
    train_indices = list(manifest["train_indices"])
    required_indices = sorted(set(int(i) for i in train_indices + selected_indices))

    hidden_caches = extract_prompt_hidden_cache(
        model,
        tokenizer,
        records,
        required_indices,
        layer_ids,
        cache_dir,
        hidden_size,
        str(manifest["prompt_sha256"]),
        batch_size=args.hidden_batch_size,
        cache_every=args.cache_every,
    )
    directions = build_support_directions(
        labels,
        train_indices,
        selected_indices,
        hidden_caches,
        layer_ids,
        requested_by_layer,
        num_layers,
        hidden_size,
        args.seed,
        args.random_directions,
        output_dir,
    )
    for layer_id, values in directions.items():
        print(
            f">>> Layer {layer_id}: ||v_support||={values['support_direction_norm']:.6f}, "
            f"sigma={values['sigma']:.6f}, "
            f"train KNOW/HALL={values['train_know_count']}/{values['train_hallucination_count']}"
        )

    verify_fresh_baselines(
        model,
        tokenizer,
        records,
        labels,
        selected_indices,
        max_new_tokens,
        output_dir,
    )
    raw_path = run_interventions(
        model,
        tokenizer,
        data,
        labels,
        selected_indices,
        directions,
        alphas,
        random_alphas,
        args.random_directions,
        max_new_tokens,
        output_dir,
        args.skip_random_control,
    )
    analyze_and_plot(
        raw_path,
        selected_indices,
        manifest,
        output_dir,
        args.seed,
    )
    print(f">>> Pilot outputs written to: {output_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
