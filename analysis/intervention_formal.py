import argparse
import csv
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import mean_knn_distance
from _paths import PROJECT_ROOT, PT_DIR, RESULTS_DIR, feature_from_name, local_model_path, resolve_input_path
from heed.extraction import freeze_model_parameters
from heed.qa import answer_match_details
from intervention_steering import (
    LABEL_HALL,
    LABEL_KNOW,
    aliases_for_record,
    atomic_json_dump,
    atomic_torch_save,
    baseline_label_name,
    build_or_validate_split_manifest,
    build_support_directions,
    choose_balanced_test_subset,
    class_counts,
    extract_prompt_hidden_cache,
    label_ids,
    load_feature_file,
    random_unit_direction,
    run_interventions,
    run_prompt_only_intervention,
    run_prompt_only_intervention_detailed,
    set_seed,
    transition_name,
    verify_fresh_baselines,
)


FORMAL_ROOT = RESULTS_DIR / "latent_support_intervention"
PILOT_ROOT = RESULTS_DIR / "latent_support_pilot"
DEFAULT_ALPHAS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
DEFAULT_LAYER_RATIOS = [0.50, 0.65, 0.80, 0.90]
PRIMARY_RATIO = 0.80
PRIMARY_ALPHA = -1.0
CONTROL_COLUMNS = [
    "setting",
    "sample_id",
    "pt_index",
    "question",
    "reference_answers",
    "baseline_answer",
    "baseline_correct",
    "intervened_answer",
    "intervention_correct",
    "transition_type",
    "layer",
    "requested_layer_ratio",
    "alpha",
    "sigma",
    "direction_type",
    "direction_id",
    "direction_norm",
    "dot_support",
    "hook_call_count",
    "prefill_intervention_count",
    "decode_call_count",
]
QUALITY_COLUMNS = [
    "setting",
    "sample_id",
    "pt_index",
    "question",
    "reference_answers",
    "baseline_answer",
    "baseline_correct",
    "alpha",
    "intervened_answer",
    "intervention_correct",
    "transition_type",
    "token_length",
    "empty_response",
    "malformed_response",
    "repetition_indicator",
    "average_generated_token_log_probability",
    "hook_call_count",
    "prefill_intervention_count",
    "captured_hidden",
    "captured_delta_l2",
]


SETTINGS: Dict[str, Dict[str, str]] = {
    "qwen25_popqa": {
        "model": "Qwen2.5-7B-Instruct",
        "model_name": "Qwen2.5-7B-Instruct",
        "dataset": "popqa",
        "dataset_name": "PopQA",
        "feature": "popqa_qwen2_5_7b_instruct",
        "reuse_pilot": "true",
    },
    "qwen25_webq": {
        "model": "Qwen2.5-7B-Instruct",
        "model_name": "Qwen2.5-7B-Instruct",
        "dataset": "webquestions",
        "dataset_name": "WebQuestions",
        "feature": "webquestions_qwen2_5_7b_instruct",
    },
    "llama3_popqa": {
        "model": "Llama-3-8B-Instruct",
        "model_name": "Llama-3-8B-Instruct",
        "dataset": "popqa",
        "dataset_name": "PopQA",
        "feature": "popqa_llama3_8b_instruct",
    },
    "llama3_webq": {
        "model": "Llama-3-8B-Instruct",
        "model_name": "Llama-3-8B-Instruct",
        "dataset": "webquestions",
        "dataset_name": "WebQuestions",
        "feature": "webquestions_llama3_8b_instruct",
    },
    "gemma2_popqa": {
        "model": "gemma-2-9b-it",
        "model_name": "Gemma-2-9B-it",
        "dataset": "popqa",
        "dataset_name": "PopQA",
        "feature": "popqa_gemma_2_9b_it",
    },
    "gemma2_webq": {
        "model": "gemma-2-9b-it",
        "model_name": "Gemma-2-9B-it",
        "dataset": "webquestions",
        "dataset_name": "WebQuestions",
        "feature": "webquestions_gemma_2_9b_it",
    },
}

EXTRA_SETTINGS: Dict[str, Dict[str, str]] = {
    "llama2_popqa": {
        "model": "Llama-2-7b-chat-hf",
        "model_name": "Llama-2-7B-Chat",
        "dataset": "popqa",
        "dataset_name": "PopQA",
        "feature": "popqa_llama2_7b_chat",
    },
    "llama2_webq": {
        "model": "Llama-2-7b-chat-hf",
        "model_name": "Llama-2-7B-Chat",
        "dataset": "webquestions",
        "dataset_name": "WebQuestions",
        "feature": "webquestions_llama2_7b_chat",
    },
    "qwen2_popqa": {
        "model": "Qwen2-7B-Instruct",
        "model_name": "Qwen2-7B-Instruct",
        "dataset": "popqa",
        "dataset_name": "PopQA",
        "feature": "popqa_qwen2_7b_instruct",
    },
    "qwen2_webq": {
        "model": "Qwen2-7B-Instruct",
        "model_name": "Qwen2-7B-Instruct",
        "dataset": "webquestions",
        "dataset_name": "WebQuestions",
        "feature": "webquestions_qwen2_7b_instruct",
    },
    "deepseek_popqa": {
        "model": "deepseek-llm-7b-chat",
        "model_name": "DeepSeek-LLM-7B-Chat",
        "dataset": "popqa",
        "dataset_name": "PopQA",
        "feature": "popqa_deepseek_7b_chat",
    },
    "deepseek_webq": {
        "model": "deepseek-llm-7b-chat",
        "model_name": "DeepSeek-LLM-7B-Chat",
        "dataset": "webquestions",
        "dataset_name": "WebQuestions",
        "feature": "webquestions_deepseek_7b_chat",
    },
}
OTHER_MODEL_INFO = {
    "llama2": ("Llama-2-7b-chat-hf", "Llama-2-7B-Chat", "llama2_7b_chat"),
    "llama3": ("Llama-3-8B-Instruct", "Llama-3-8B-Instruct", "llama3_8b_instruct"),
    "qwen2": ("Qwen2-7B-Instruct", "Qwen2-7B-Instruct", "qwen2_7b_instruct"),
    "qwen25": ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "qwen2_5_7b_instruct"),
    "gemma2": ("gemma-2-9b-it", "Gemma-2-9B-it", "gemma_2_9b_it"),
    "deepseek": ("deepseek-llm-7b-chat", "DeepSeek-LLM-7B-Chat", "deepseek_7b_chat"),
}
OTHER_SETTINGS: Dict[str, Dict[str, str]] = {
    f"{key}_{dataset}": {
        "model": model,
        "model_name": model_name,
        "dataset": dataset,
        "dataset_name": "TriviaQA" if dataset == "triviaqa" else "MKQA",
        "feature": f"{dataset}_{slug}",
    }
    for key, (model, model_name, slug) in OTHER_MODEL_INFO.items()
    for dataset in ("triviaqa", "mkqa")
}
ALL_SETTINGS = {**SETTINGS, **EXTRA_SETTINGS, **OTHER_SETTINGS}


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def rounded_layer_ids(
    num_layers: int, ratios: Sequence[float]
) -> Tuple[List[int], Dict[int, float]]:
    requested_by_layer: Dict[int, float] = {}
    for ratio in ratios:
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"Layer ratio must satisfy 0 <= ratio < 1, got {ratio}")
        layer = min(max(int(math.floor(float(ratio) * num_layers + 0.5)), 0), num_layers - 1)
        requested_by_layer.setdefault(layer, float(ratio))
    return sorted(requested_by_layer), requested_by_layer


def append_csv(path: Path, row: Dict[str, object], columns: Sequence[str]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if new_file:
            writer.writeheader()
        writer.writerow({column: row[column] for column in columns})
        handle.flush()


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(["true", "1"])


def answer_is_correct(answer: str, aliases: Sequence[str], dataset: str) -> bool:
    return bool(
        answer_match_details(answer, list(aliases), relaxed=dataset != "popqa")["matched"]
    )


def control_key(row: Dict[str, object]) -> Tuple[int, str, int]:
    return int(row["pt_index"]), str(row["direction_type"]), int(row["direction_id"])


def make_control_directions(
    support: torch.Tensor,
    hidden_size: int,
    seed: int,
    layer_id: int,
    count: int,
) -> Dict[str, torch.Tensor]:
    random_directions = []
    orthogonal_directions = []
    for direction_id in range(count):
        random_direction = random_unit_direction(hidden_size, seed, layer_id, direction_id)
        random_directions.append(random_direction)

        source = random_unit_direction(hidden_size, seed + 7_919, layer_id, direction_id)
        orthogonal = source - torch.dot(source, support) * support
        orthogonal = orthogonal / orthogonal.norm(p=2).clamp_min(1e-12)
        if abs(float(torch.dot(orthogonal, support))) >= 1e-5:
            raise AssertionError("Orthogonal control is not orthogonal to support direction.")
        orthogonal_directions.append(orthogonal)

    return {
        "random": torch.stack(random_directions),
        "orthogonal": torch.stack(orthogonal_directions),
    }


def run_controls(
    model,
    tokenizer,
    setting: str,
    data: Dict[str, object],
    labels: np.ndarray,
    selected_indices: Sequence[int],
    primary_layer: int,
    layer_data: Dict[str, object],
    count: int,
    seed: int,
    max_new_tokens: int,
    output_dir: Path,
) -> Path:
    path = output_dir / "control_raw_results.csv"
    existing = set()
    if path.exists():
        existing = {
            control_key(row) for row in pd.read_csv(path).fillna("").to_dict(orient="records")
        }
    support = layer_data["support_direction"].float()
    controls = make_control_directions(support, len(support), seed, primary_layer, count)
    atomic_torch_save(
        {
            "version": 1,
            "setting": setting,
            "seed": seed,
            "layer": primary_layer,
            "sigma": float(layer_data["sigma"]),
            "support_direction": support,
            "controls": controls,
        },
        output_dir / "control_directions.pt",
    )

    records = data["records"]
    total = sum(
        (int(index), direction_type, direction_id) not in existing
        for direction_type in ["random", "orthogonal"]
        for direction_id in range(count)
        for index in selected_indices
    )
    progress = tqdm(total=total, desc=f"{setting} controls")
    for direction_type in ["random", "orthogonal"]:
        for direction_id, direction in enumerate(controls[direction_type]):
            dot_support = float(torch.dot(direction.float(), support))
            for pt_index in selected_indices:
                pt_index = int(pt_index)
                key = (pt_index, direction_type, direction_id)
                if key in existing:
                    continue
                record = records[pt_index]
                aliases = aliases_for_record(record)
                baseline_correct = bool(labels[pt_index] == LABEL_KNOW)
                answer, stats = run_prompt_only_intervention(
                    model,
                    tokenizer,
                    str(record["prompt"]),
                    primary_layer,
                    direction,
                    float(layer_data["sigma"]),
                    1.0,
                    max_new_tokens,
                )
                intervention_correct = answer_is_correct(
                    answer, aliases, str(data.get("dataset", ""))
                )
                source = record.get("label_source") or {}
                append_csv(
                    path,
                    {
                        "setting": setting,
                        "sample_id": source.get("id", record.get("row_index", pt_index)),
                        "pt_index": pt_index,
                        "question": str(record["question"]),
                        "reference_answers": json.dumps(aliases, ensure_ascii=False),
                        "baseline_answer": str(record["generated_answer"]),
                        "baseline_correct": baseline_correct,
                        "intervened_answer": answer,
                        "intervention_correct": intervention_correct,
                        "transition_type": transition_name(
                            baseline_correct, intervention_correct
                        ),
                        "layer": primary_layer,
                        "requested_layer_ratio": PRIMARY_RATIO,
                        "alpha": 1.0,
                        "sigma": float(layer_data["sigma"]),
                        "direction_type": direction_type,
                        "direction_id": direction_id,
                        "direction_norm": float(direction.float().norm()),
                        "dot_support": dot_support,
                        "hook_call_count": stats.call_count,
                        "prefill_intervention_count": stats.prefill_intervention_count,
                        "decode_call_count": stats.decode_call_count,
                    },
                    CONTROL_COLUMNS,
                )
                existing.add(key)
                progress.update(1)
    progress.close()
    frame = pd.read_csv(path)
    if not (frame["prefill_intervention_count"].astype(int) == 1).all():
        raise AssertionError("A control hook was not restricted to one prefill invocation.")
    orthogonal = frame[frame["direction_type"] == "orthogonal"]
    if not (orthogonal["dot_support"].abs() < 1e-5).all():
        raise AssertionError("Saved orthogonal control failed the dot-product tolerance.")
    return path


def fit_support_space(
    train_hidden: torch.Tensor,
    before_hidden: torch.Tensor,
    after_hidden: torch.Tensor,
    pca_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    mean = train_hidden.mean(dim=0, keepdim=True)
    std = train_hidden.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    def standardize(values: torch.Tensor) -> np.ndarray:
        values = torch.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        return values.numpy().astype(np.float32, copy=False)

    train_np = standardize(train_hidden)
    before_np = standardize(before_hidden)
    after_np = standardize(after_hidden)
    n_components = min(pca_dim, train_np.shape[0] - 1, train_np.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    train_embed = pca.fit_transform(train_np).astype(np.float32, copy=False)
    before_embed = pca.transform(before_np).astype(np.float32, copy=False)
    after_embed = pca.transform(after_np).astype(np.float32, copy=False)
    return train_embed, before_embed, after_embed, {
        "standardization": "per-dimension train mean/std",
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def capture_primary_hidden(
    model,
    tokenizer,
    setting: str,
    data: Dict[str, object],
    selected_indices: Sequence[int],
    primary_layer: int,
    layer_data: Dict[str, object],
    primary_raw_path: Path,
    max_new_tokens: int,
    output_dir: Path,
) -> Path:
    path = output_dir / "primary_intervened_hidden.pt"
    stored = torch.load(path, map_location="cpu") if path.exists() else {
        "version": 1,
        "setting": setting,
        "layer": primary_layer,
        "alpha": PRIMARY_ALPHA,
        "sigma": float(layer_data["sigma"]),
        "rows": {},
    }
    if int(stored["layer"]) != primary_layer or float(stored["alpha"]) != PRIMARY_ALPHA:
        raise RuntimeError("Existing primary hidden capture has a different configuration.")
    rows = stored["rows"]
    raw = pd.read_csv(primary_raw_path)
    primary = raw[
        (raw["direction_type"] == "support")
        & np.isclose(raw["requested_layer_ratio"].astype(float), PRIMARY_RATIO)
        & np.isclose(raw["alpha"].astype(float), PRIMARY_ALPHA)
    ].set_index("pt_index")
    if primary.index.duplicated().any():
        raise RuntimeError("Duplicate primary intervention outputs in raw sweep.")

    records = data["records"]
    pending = [int(index) for index in selected_indices if int(index) not in rows]
    progress = tqdm(pending, desc=f"{setting} actual hidden")
    for pt_index in progress:
        if pt_index not in primary.index:
            raise RuntimeError(f"Missing primary sweep row for pt_index={pt_index}.")
        result = run_prompt_only_intervention_detailed(
            model,
            tokenizer,
            str(records[pt_index]["prompt"]),
            primary_layer,
            layer_data["support_direction"],
            float(layer_data["sigma"]),
            PRIMARY_ALPHA,
            max_new_tokens,
            capture_intervened_hidden=True,
        )
        if result.hidden_before is None or result.hidden_after is None:
            raise AssertionError("Primary intervention hook did not capture before/after hidden.")
        rows[pt_index] = {
            "before": result.hidden_before.to(torch.float16),
            "after": result.hidden_after.to(torch.float16),
            "capture_answer": result.generated.text,
            "original_sweep_answer": str(primary.loc[pt_index, "intervened_answer"]),
            "answer_exact_match": result.generated.text
            == str(primary.loc[pt_index, "intervened_answer"]),
            "prefill_intervention_count": int(result.stats.prefill_intervention_count),
        }
        if result.stats.prefill_intervention_count != 1:
            raise AssertionError("Primary hidden capture did not apply exactly one intervention.")
        if len(rows) % 10 == 0:
            atomic_torch_save(stored, path)
    atomic_torch_save(stored, path)
    return path


def support_scores(
    train_embed: np.ndarray,
    train_labels: np.ndarray,
    query_embed: np.ndarray,
    k: int,
    eps: float,
) -> np.ndarray:
    d_know = mean_knn_distance(train_embed[train_labels == LABEL_KNOW], query_embed, k)
    d_hall = mean_knn_distance(train_embed[train_labels == LABEL_HALL], query_embed, k)
    return -np.log(d_know + eps) + np.log(d_hall + eps)


def run_support_analysis(
    setting: str,
    data: Dict[str, object],
    labels: np.ndarray,
    manifest: Dict[str, object],
    selected_indices: Sequence[int],
    hidden_cache: Dict[int, Dict[int, torch.Tensor]],
    primary_layer: int,
    layer_data: Dict[str, object],
    primary_raw_path: Path,
    captured_hidden_path: Path,
    output_dir: Path,
    seed: int,
    k: int,
    pca_dim: int,
    eps: float,
) -> Path:
    train_indices = np.asarray(manifest["train_indices"], dtype=np.int64)
    selected = np.asarray(selected_indices, dtype=np.int64)
    train_hidden = torch.stack(
        [hidden_cache[primary_layer][int(index)].float() for index in train_indices]
    )
    captured = torch.load(captured_hidden_path, map_location="cpu")
    captured_rows = captured["rows"]
    if set(map(int, selected)) - set(map(int, captured_rows)):
        raise RuntimeError("Primary intervention hidden capture is incomplete.")
    before = torch.stack([captured_rows[int(index)]["before"].float() for index in selected])
    after = torch.stack([captured_rows[int(index)]["after"].float() for index in selected])
    support = layer_data["support_direction"].float()
    sigma = float(layer_data["sigma"])
    train_embed, before_embed, after_embed, transform_info = fit_support_space(
        train_hidden, before, after, pca_dim, seed
    )
    train_labels = labels[train_indices]
    score_before = support_scores(train_embed, train_labels, before_embed, k, eps)
    score_after = support_scores(train_embed, train_labels, after_embed, k, eps)
    projection_before = (before @ support).numpy()
    projection_after = (after @ support).numpy()

    raw = pd.read_csv(primary_raw_path)
    raw["baseline_correct"] = bool_series(raw["baseline_correct"])
    raw["intervention_correct"] = bool_series(raw["intervention_correct"])
    primary = raw[
        (raw["direction_type"] == "support")
        & np.isclose(raw["requested_layer_ratio"].astype(float), PRIMARY_RATIO)
        & np.isclose(raw["alpha"].astype(float), PRIMARY_ALPHA)
        & raw["pt_index"].astype(int).isin(set(selected.tolist()))
    ].copy()
    if len(primary) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} primary rows for support analysis, got {len(primary)}."
        )
    primary = primary.set_index(primary["pt_index"].astype(int))
    records = data["records"]
    rows = []
    for row_id, pt_index in enumerate(selected):
        result = primary.loc[int(pt_index)]
        record = records[int(pt_index)]
        aliases = aliases_for_record(record)
        rows.append(
            {
                "setting": setting,
                "sample_id": (record.get("label_source") or {}).get(
                    "id", record.get("row_index", int(pt_index))
                ),
                "pt_index": int(pt_index),
                "question": str(record["question"]),
                "reference_answers": json.dumps(aliases, ensure_ascii=False),
                "baseline_answer": str(record["generated_answer"]),
                "intervened_answer": str(result["intervened_answer"]),
                "baseline_correct": bool(result["baseline_correct"]),
                "intervention_correct": bool(result["intervention_correct"]),
                "transition_type": str(result["transition_type"]),
                "layer": primary_layer,
                "requested_layer_ratio": PRIMARY_RATIO,
                "alpha": PRIMARY_ALPHA,
                "sigma": sigma,
                "knn_k": k,
                "support_score_before": float(score_before[row_id]),
                "support_score_after": float(score_after[row_id]),
                "delta_support_score": float(score_after[row_id] - score_before[row_id]),
                "support_projection_before": float(projection_before[row_id]),
                "support_projection_after": float(projection_after[row_id]),
                "delta_support_projection": float(
                    projection_after[row_id] - projection_before[row_id]
                ),
                "capture_answer_exact_match": bool(
                    captured_rows[int(pt_index)]["answer_exact_match"]
                ),
            }
        )
    path = output_dir / "support_change.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    expected_delta = PRIMARY_ALPHA * sigma
    observed = np.asarray([row["delta_support_projection"] for row in rows])
    projection_tolerance = max(0.05, abs(expected_delta) * 0.005)
    if not np.allclose(observed, expected_delta, rtol=0.0, atol=projection_tolerance):
        raise AssertionError("Support projection did not change by alpha * sigma.")
    atomic_json_dump(
        {
            "setting": setting,
            "definition": "log rho_KNOW - log rho_HALL; rho=1/(mean kNN L2 distance + eps)",
            "reference": "train split only",
            "k": k,
            "eps": eps,
            "after_hidden": "actual modified last prompt token captured in the prefill hook",
            "expected_delta_projection": expected_delta,
            "projection_check_tolerance": projection_tolerance,
            "max_projection_check_error": float(np.max(np.abs(observed - expected_delta))),
            "capture_answer_exact_match_fraction": float(
                np.mean([row["capture_answer_exact_match"] for row in rows])
            ),
            **transform_info,
        },
        output_dir / "support_analysis_metadata.json",
    )
    return path


def repetition_indicator(token_ids: torch.Tensor, ngram: int = 3) -> bool:
    values = [int(value) for value in token_ids.tolist()]
    if len(values) < 2 * ngram:
        return False
    seen = set()
    for start in range(len(values) - ngram + 1):
        item = tuple(values[start : start + ngram])
        if item in seen:
            return True
        seen.add(item)
    return False


def average_generated_log_probability(generated) -> float:
    token_ids = generated.raw_token_ids.long().tolist()
    scores = generated.generation_scores or ()
    usable = min(len(token_ids), len(scores))
    if usable == 0:
        return float("nan")
    values = []
    for index in range(usable):
        values.append(float(scores[index][0].float().log_softmax(dim=-1)[token_ids[index]]))
    return float(np.mean(values))


def malformed_response(text: str) -> bool:
    if not text.strip():
        return True
    return not any(character.isalnum() for character in text)


def run_generation_quality(
    model,
    tokenizer,
    setting: str,
    data: Dict[str, object],
    labels: np.ndarray,
    selected_indices: Sequence[int],
    primary_layer: int,
    layer_data: Dict[str, object],
    max_new_tokens: int,
    output_dir: Path,
) -> Path:
    path = output_dir / "generation_quality_raw.csv"
    existing = set()
    if path.exists():
        frame = pd.read_csv(path)
        existing = set(zip(frame["pt_index"].astype(int), frame["alpha"].astype(float)))
    records = data["records"]
    know_indices = [int(index) for index in selected_indices if labels[int(index)] == LABEL_KNOW]
    total = sum((index, alpha) not in existing for index in know_indices for alpha in [0.0, -1.0])
    progress = tqdm(total=total, desc=f"{setting} quality")
    support = layer_data["support_direction"].float()
    for alpha in [0.0, -1.0]:
        for pt_index in know_indices:
            if (pt_index, alpha) in existing:
                continue
            record = records[pt_index]
            result = run_prompt_only_intervention_detailed(
                model,
                tokenizer,
                str(record["prompt"]),
                primary_layer,
                support,
                float(layer_data["sigma"]),
                alpha,
                max_new_tokens,
                return_diagnostics=True,
                capture_intervened_hidden=True,
            )
            generated = result.generated
            if math.isclose(alpha, 0.0) and generated.text != str(record["generated_answer"]):
                raise AssertionError(
                    f"Quality alpha=0 differs from baseline at pt_index={pt_index}."
                )
            aliases = aliases_for_record(record)
            correct = answer_is_correct(
                generated.text, aliases, str(data.get("dataset", ""))
            )
            hidden_delta = float(
                (result.hidden_after - result.hidden_before).norm(p=2)
            ) if result.hidden_before is not None and result.hidden_after is not None else float("nan")
            append_csv(
                path,
                {
                    "setting": setting,
                    "sample_id": (record.get("label_source") or {}).get(
                        "id", record.get("row_index", pt_index)
                    ),
                    "pt_index": pt_index,
                    "question": str(record["question"]),
                    "reference_answers": json.dumps(aliases, ensure_ascii=False),
                    "baseline_answer": str(record["generated_answer"]),
                    "baseline_correct": True,
                    "alpha": alpha,
                    "intervened_answer": generated.text,
                    "intervention_correct": correct,
                    "transition_type": transition_name(True, correct),
                    "token_length": int(generated.raw_token_ids.numel()),
                    "empty_response": not bool(generated.text.strip()),
                    "malformed_response": malformed_response(generated.text),
                    "repetition_indicator": repetition_indicator(generated.raw_token_ids),
                    "average_generated_token_log_probability": average_generated_log_probability(
                        generated
                    ),
                    "hook_call_count": result.stats.call_count,
                    "prefill_intervention_count": result.stats.prefill_intervention_count,
                    "captured_hidden": result.hidden_before is not None,
                    "captured_delta_l2": hidden_delta,
                },
                QUALITY_COLUMNS,
            )
            existing.add((pt_index, alpha))
            progress.update(1)
    progress.close()
    frame = pd.read_csv(path)
    if not (frame["prefill_intervention_count"].astype(int) == 1).all():
        raise AssertionError("A quality hook was not restricted to one prefill invocation.")
    return path


def setting_metadata(
    setting: str,
    config: Dict[str, str],
    manifest: Dict[str, object],
    selected_indices: Sequence[int],
    labels: np.ndarray,
    layer_ids: Sequence[int],
    requested_by_layer: Dict[int, float],
    num_layers: int,
    output_dir: Path,
) -> None:
    atomic_json_dump(
        {
            "version": 1,
            "setting": setting,
            "model": config["model_name"],
            "dataset": config["dataset_name"],
            "seed": int(manifest["seed"]),
            "split": "stratified 70/15/15",
            "selected_indices": [int(index) for index in selected_indices],
            "selected_counts": class_counts(labels, selected_indices),
            "num_hidden_layers": num_layers,
            "layer_rounding": "floor(ratio * L + 0.5), clipped to [0,L-1]",
            "layer_ids": [int(layer) for layer in layer_ids],
            "requested_ratios": {
                str(layer): requested_by_layer[layer] for layer in layer_ids
            },
            "primary_layer_ratio": PRIMARY_RATIO,
            "primary_alpha": PRIMARY_ALPHA,
            "prompt_only": True,
        },
        output_dir / "setting_metadata.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal cross-model latent factual-support intervention experiment."
    )
    parser.add_argument("--setting", choices=sorted(ALL_SETTINGS), required=True)
    parser.add_argument("--output-root", default=str(FORMAL_ROOT))
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-ratios", default=",".join(map(str, DEFAULT_LAYER_RATIOS)))
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--control-directions", type=int, default=10)
    parser.add_argument("--hidden-batch-size", type=int, default=8)
    parser.add_argument("--cache-every", type=int, default=25)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Model placement, e.g. auto or cuda:5. This does not change the intervention.",
    )
    parser.add_argument(
        "--components",
        default="sweep,controls,capture,support,quality",
        help="Comma-separated: sweep,controls,capture,support,quality.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 4 KNOW + 4 HALL and only primary layer with alpha=-1,0,+1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = ALL_SETTINGS[args.setting]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_dir = output_root / ("smoke" if args.smoke else "") / args.setting
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = feature_from_name(config["feature"])
    data = load_feature_file(feature_path)
    labels = label_ids(data)
    records = data["records"]
    if len(records) != len(labels):
        raise RuntimeError("records and labels have different lengths.")
    if str(data.get("dataset", "")).lower() != config["dataset"]:
        raise RuntimeError("Feature dataset does not match the selected setting.")

    manifest_dir = PILOT_ROOT if config.get("reuse_pilot") == "true" and not args.smoke else output_dir
    manifest = build_or_validate_split_manifest(
        data, feature_path, manifest_dir, seed=args.seed
    )
    max_samples = 8 if args.smoke else args.max_test_samples
    selected_indices = choose_balanced_test_subset(manifest, max_samples)
    if config.get("reuse_pilot") == "true" and not args.smoke:
        pilot_selected = choose_balanced_test_subset(manifest, 100)
        if [int(value) for value in selected_indices] != [int(value) for value in pilot_selected]:
            raise AssertionError("Formal Qwen2.5+PopQA subset differs from the completed pilot.")
        atomic_json_dump(manifest, output_dir / "split_manifest.json")
    print(
        f">>> {args.setting}: train={manifest['train_counts']}, "
        f"validation={manifest['validation_counts']}, test={manifest['test_counts']}, "
        f"selected={class_counts(labels, selected_indices)}"
    )
    if args.prepare_only:
        return

    components = {item.strip() for item in args.components.split(",") if item.strip()}
    allowed = {"sweep", "controls", "capture", "support", "quality"}
    if not components <= allowed:
        raise ValueError(f"Unknown components: {sorted(components - allowed)}")
    alphas = [-1.0, 0.0, 1.0] if args.smoke else parse_float_list(args.alphas)
    layer_ratios = [PRIMARY_RATIO] if args.smoke else parse_float_list(args.layer_ratios)
    if 0.0 not in alphas:
        raise ValueError("The support sweep must include alpha=0.")

    model_path = local_model_path(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = "auto" if args.device == "auto" else {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map=device_map, torch_dtype=torch.float16
    )
    model.eval()
    freeze_model_parameters(model)
    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    layer_ids, requested_by_layer = rounded_layer_ids(num_layers, layer_ratios)
    primary_layers, _ = rounded_layer_ids(num_layers, [PRIMARY_RATIO])
    primary_layer = primary_layers[0]
    if primary_layer not in layer_ids:
        raise RuntimeError("Primary layer is missing from the requested layer sweep.")
    print(f">>> L={num_layers}, layers={layer_ids}, primary_layer={primary_layer}")

    train_indices = [int(index) for index in manifest["train_indices"]]
    required_indices = sorted(set(train_indices + [int(index) for index in selected_indices]))
    cache_dir = (
        PILOT_ROOT / "cache"
        if config.get("reuse_pilot") == "true" and not args.smoke
        else output_dir / "cache"
    )
    hidden_caches = extract_prompt_hidden_cache(
        model,
        tokenizer,
        records,
        required_indices,
        layer_ids,
        cache_dir,
        hidden_size,
        str(manifest["prompt_sha256"]),
        args.hidden_batch_size,
        args.cache_every,
    )
    if config.get("reuse_pilot") == "true" and not args.smoke:
        saved = torch.load(PILOT_ROOT / "support_directions.pt", map_location="cpu")
        directions = saved["layers"]
        missing = set(layer_ids) - set(directions)
        if missing:
            raise RuntimeError(f"Pilot direction file is missing layers: {sorted(missing)}")
        directions = {layer: directions[layer] for layer in layer_ids}
    else:
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
            0,
            output_dir,
        )
    for layer_id in layer_ids:
        if not math.isclose(
            float(directions[layer_id]["support_direction"].float().norm()),
            1.0,
            rel_tol=1e-4,
            abs_tol=1e-4,
        ):
            raise AssertionError(f"Support direction at layer {layer_id} is not unit norm.")

    setting_metadata(
        args.setting,
        config,
        manifest,
        selected_indices,
        labels,
        layer_ids,
        requested_by_layer,
        num_layers,
        output_dir,
    )
    max_new_tokens = int(data.get("max_new_tokens", 32))
    verify_fresh_baselines(
        model,
        tokenizer,
        records,
        labels,
        selected_indices,
        max_new_tokens,
        output_dir,
        relaxed_matching=config["dataset"] != "popqa",
    )

    reuse_pilot_sweep = config.get("reuse_pilot") == "true" and not args.smoke
    if "sweep" in components and not reuse_pilot_sweep:
        raw_path = run_interventions(
            model,
            tokenizer,
            data,
            labels,
            selected_indices,
            directions,
            alphas,
            [],
            0,
            max_new_tokens,
            output_dir,
            True,
            relaxed_matching=config["dataset"] != "popqa",
        )
    elif reuse_pilot_sweep:
        raw_path = PILOT_ROOT / "raw_results.csv"
    else:
        raw_path = output_dir / "raw_results.csv"

    if args.smoke:
        print(f">>> Smoke test passed: {args.setting}")
    else:
        primary_data = directions[primary_layer]
        if "controls" in components:
            run_controls(
                model,
                tokenizer,
                args.setting,
                data,
                labels,
                selected_indices,
                primary_layer,
                primary_data,
                args.control_directions,
                args.seed,
                max_new_tokens,
                output_dir,
            )
        captured_hidden_path = output_dir / "primary_intervened_hidden.pt"
        if "capture" in components:
            if not raw_path.exists():
                raise FileNotFoundError("Primary hidden capture requires the completed sweep.")
            captured_hidden_path = capture_primary_hidden(
                model,
                tokenizer,
                args.setting,
                data,
                selected_indices,
                primary_layer,
                primary_data,
                raw_path,
                max_new_tokens,
                output_dir,
            )
        if "support" in components:
            if not raw_path.exists():
                raise FileNotFoundError("Support analysis requires the completed support sweep.")
            if not captured_hidden_path.exists():
                raise FileNotFoundError("Support analysis requires actual post-hook hidden capture.")
            run_support_analysis(
                args.setting,
                data,
                labels,
                manifest,
                selected_indices,
                hidden_caches,
                primary_layer,
                primary_data,
                raw_path,
                captured_hidden_path,
                output_dir,
                args.seed,
                args.knn_k,
                args.pca_dim,
                args.eps,
            )
        if "quality" in components:
            run_generation_quality(
                model,
                tokenizer,
                args.setting,
                data,
                labels,
                selected_indices,
                primary_layer,
                primary_data,
                max_new_tokens,
                output_dir,
            )

    print(f">>> Setting outputs: {output_dir}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
