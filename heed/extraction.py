"""Model-side feature extraction (one greedy generation + one teacher-forced forward pass per question).

Functions below are copied unchanged from the code that produced the paper's feature files, except that the
gradient / self-evaluation features of earlier drafts are not computed any more.  The keyword-entropy
H-bar is computed by keyword_entropy() from the logits of the same teacher-forced pass that yields h.
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

from .qa import _PUNCT_TABLE, is_content_token, normalize_answer_text  # noqa: F401

DEFAULT_SAMPLE_TEMPERATURE = 1.0
DEFAULT_SAMPLE_TOP_P = 0.95
DEFAULT_LOGIT_TEMPERATURE = 0.7

EPS = 1e-8


LOGIT_BASELINE_NAMES = [
    "gen_mean_max_prob",
    "gen_min_max_prob",
    "gen_max_max_prob",
    "gen_mean_neg_log_max_prob",
    "gen_ppl_from_max_prob",
    "gen_mean_entropy_nats",
    "gen_max_entropy_nats",
    "gen_mean_entropy_bits",
    "gen_mean_temp_max_prob",
    "gen_mean_temp_entropy_nats",
    "gen_mean_energy",
    "gen_max_energy",
    "tf_mean_chosen_prob",
    "tf_min_chosen_prob",
    "tf_mean_chosen_logprob",
    "tf_min_chosen_logprob",
    "tf_mean_neg_logprob",
    "tf_ppl",
    "tf_mean_entropy_nats",
    "tf_max_entropy_nats",
    "tf_mean_margin",
    "tf_min_margin",
    "tf_frac_margin_lt_0_5",
    "tf_frac_margin_lt_1_0",
]


COE_FEATURE_NAMES = [
    "gen_coe_mag_mean",
    "gen_coe_mag_var",
    "gen_coe_ang_mean",
    "gen_coe_ang_var",
    "gen_coe_r",
    "gen_coe_c",
    "tf_coe_mag_mean",
    "tf_coe_mag_var",
    "tf_coe_ang_mean",
    "tf_coe_ang_var",
    "tf_coe_r",
    "tf_coe_c",
]


SAMPLED_BASELINE_NAMES = [
    "sample_count",
    "sample_unique_ratio",
    "sample_ln_entropy",
    "sample_mean_rouge_to_greedy",
    "sample_mean_pairwise_rouge",
    "sample_mean_length",
    "sample_length_var",
    "sample_eigenscore",
    "sample_embedding_trace",
]


@dataclass
class TokenGroup:
    text: str
    indices: List[int]


@dataclass
class GeneratedAnswer:
    text: str
    token_ids: torch.Tensor
    raw_text: str
    raw_token_ids: torch.Tensor
    generation_scores: Optional[Tuple[torch.Tensor, ...]] = None
    generation_hidden_states: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None


def get_model_input_device(model) -> torch.device:
    """Return the device expected by the first model shard."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def freeze_model_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def clean_generated_answer(text: str) -> str:
    """Remove obvious continuation artifacts while keeping multi-sentence answers."""
    text = text.strip()
    text = re.sub(r"^\s*(answer|a)\s*:\s*", "", text, flags=re.IGNORECASE)
    for marker in ["\nQuestion:", "\n\nQuestion:", "\nQ:", "\n\nQ:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    return text


def resolve_prompt_style(tokenizer, prompt_style: str) -> str:
    if prompt_style != "auto":
        return prompt_style
    if getattr(tokenizer, "chat_template", None):
        return "chat"
    return "plain"


def chat_template_kwargs(disable_thinking: bool) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    return kwargs


def apply_chat_template_safely(
    tokenizer,
    messages: List[Dict[str, str]],
    disable_thinking: bool = False,
) -> str:
    kwargs = chat_template_kwargs(disable_thinking)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)


def build_generation_prompt(
    tokenizer,
    question: str,
    prompt_style: str,
    disable_thinking: bool = False,
) -> str:
    if prompt_style == "plain":
        return f"Question: {question}\nAnswer:"

    if prompt_style == "llama2-chat":
        prompt_style = "chat"

    if prompt_style != "chat":
        raise ValueError("prompt_style must be one of: plain, chat, llama2-chat, auto")

    user_content = (
        "Answer the following question with a short factual answer only.\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": user_content}]
    if getattr(tokenizer, "chat_template", None):
        return apply_chat_template_safely(
            tokenizer,
            messages,
            disable_thinking=disable_thinking,
        )

    return f"<s>[INST] {user_content} [/INST]"


def safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def safe_max(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().max().item())


def safe_min(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().min().item())


def safe_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    if values.numel() == 1:
        return float(values.float().item())
    return float(torch.quantile(values.float(), q).item())


def trim_special_suffix(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    stop_ids = {
        token_id
        for token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id]
        if token_id is not None
    }
    if not stop_ids:
        return token_ids

    kept = []
    for token_id in token_ids.detach().cpu().tolist():
        if int(token_id) in stop_ids:
            break
        kept.append(int(token_id))
    return torch.tensor(kept, dtype=torch.long)


def encode_clean_answer(tokenizer, answer_text: str) -> torch.Tensor:
    return tokenizer(
        " " + answer_text.strip(),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].detach().cpu()


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    return_diagnostics: bool = False,
    do_sample: bool = False,
    temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
    top_p: float = DEFAULT_SAMPLE_TOP_P,
) -> GeneratedAnswer:
    device = get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": return_diagnostics,
        "output_scores": return_diagnostics,
        "output_hidden_states": return_diagnostics,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generation_output = model.generate(**generation_kwargs)

    output_ids = generation_output.sequences if return_diagnostics else generation_output
    raw_answer_ids = output_ids[0, inputs.input_ids.shape[1] :]
    raw_answer_ids = trim_special_suffix(raw_answer_ids, tokenizer)
    raw_answer = tokenizer.decode(raw_answer_ids, skip_special_tokens=True)
    clean_answer = clean_generated_answer(raw_answer)
    answer_ids = raw_answer_ids

    if clean_answer != raw_answer.strip():
        answer_ids = encode_clean_answer(tokenizer, clean_answer)

    generation_scores = None
    generation_hidden_states = None
    if return_diagnostics:
        generation_scores = tuple(score.detach().cpu() for score in generation_output.scores)
        generation_hidden_states = tuple(
            tuple(layer_hidden.detach().cpu() for layer_hidden in step_hidden)
            for step_hidden in generation_output.hidden_states
        )

    return GeneratedAnswer(
        text=clean_answer,
        token_ids=answer_ids,
        raw_text=raw_answer.strip(),
        raw_token_ids=raw_answer_ids,
        generation_scores=generation_scores,
        generation_hidden_states=generation_hidden_states,
    )


def prepare_answer_ids(
    tokenizer,
    answer_text: str,
    device: torch.device,
    answer_token_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if answer_token_ids is None:
        if not answer_text.strip():
            return None
        answer_ids = encode_clean_answer(tokenizer, answer_text)
    else:
        answer_ids = answer_token_ids.detach().cpu().long()

    if answer_ids.numel() == 0:
        return None
    if answer_ids.dim() == 1:
        answer_ids = answer_ids.unsqueeze(0)
    return answer_ids.to(device)


def get_best_competitors(
    log_probs: torch.Tensor, target_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the highest-probability token that is not the generated token."""
    top_k = min(8, log_probs.shape[-1])
    top_values, top_ids = torch.topk(log_probs, k=top_k, dim=-1)
    not_target = top_ids.ne(target_ids.unsqueeze(1))
    alt_rank = not_target.float().argmax(dim=1)
    competitor_log_probs = top_values.gather(1, alt_rank.unsqueeze(1)).squeeze(1)
    competitor_ids = top_ids.gather(1, alt_rank.unsqueeze(1)).squeeze(1)
    return competitor_log_probs, competitor_ids


def build_token_groups(tokenizer, target_ids: torch.Tensor) -> List[TokenGroup]:
    token_id_list = [int(token_id) for token_id in target_ids.detach().cpu()]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_id_list)
    groups: List[List[int]] = []

    for idx, raw_token in enumerate(raw_tokens):
        token_text = tokenizer.decode([token_id_list[idx]], skip_special_tokens=True)
        starts_word = (
            idx == 0
            or str(raw_token).startswith("▁")
            or str(raw_token).startswith("Ġ")
            or token_text.startswith(" ")
        )
        stripped = token_text.strip()
        is_punct = bool(stripped) and all(ch in _PUNCT_TABLE for ch in stripped)

        if (starts_word or is_punct) and groups:
            groups.append([idx])
        else:
            if not groups:
                groups.append([idx])
            else:
                groups[-1].append(idx)

    token_groups = []
    for indices in groups:
        text = tokenizer.decode(
            [token_id_list[idx] for idx in indices], skip_special_tokens=True
        ).strip()
        if text:
            token_groups.append(TokenGroup(text=text, indices=indices))
    return token_groups


def select_probe_word_groups(
    tokenizer,
    target_ids: torch.Tensor,
    margins: torch.Tensor,
    max_probe_tokens: int,
    prompt_text: Optional[str] = None,
) -> Tuple[List[TokenGroup], List[TokenGroup], torch.Tensor, torch.Tensor]:
    prompt_terms = set(normalize_answer_text(prompt_text or "").split())
    groups = build_token_groups(tokenizer, target_ids)
    content_groups = [
        group for group in groups if is_content_token(group.text, prompt_terms=prompt_terms)
    ]

    if not content_groups:
        content_groups = [
            group for group in groups if any(ch.isalnum() for ch in group.text)
        ] or groups

    group_margins = torch.stack(
        [
            margins[
                torch.tensor(group.indices, dtype=torch.long, device=margins.device)
            ].mean()
            for group in content_groups
        ]
    )
    probe_count = min(max_probe_tokens, len(content_groups))
    selected_order = torch.argsort(group_margins)[:probe_count]
    selected_groups = [content_groups[int(idx)] for idx in selected_order.detach().cpu()]
    selected_group_margins = group_margins[selected_order]

    return selected_groups, content_groups, selected_group_margins, group_margins


def tensor_or_zeros(values: List[float], expected_len: int) -> torch.Tensor:
    if len(values) != expected_len:
        values = (values + [0.0] * expected_len)[:expected_len]
    return torch.tensor(values, dtype=torch.float32)


def safe_rouge_l(
    scorer: rouge_scorer.RougeScorer,
    reference: str,
    candidate: str,
) -> float:
    if not reference.strip() or not candidate.strip():
        return 0.0
    return float(scorer.score(reference, candidate)["rougeL"].fmeasure)


def compute_generation_logit_features(
    generation_scores: Optional[Tuple[torch.Tensor, ...]],
    raw_token_ids: torch.Tensor,
    temperature: float = DEFAULT_LOGIT_TEMPERATURE,
) -> Dict[str, float]:
    if not generation_scores:
        return {name: 0.0 for name in LOGIT_BASELINE_NAMES[:12]}

    score_count = min(len(generation_scores), int(raw_token_ids.numel()))
    if score_count <= 0:
        return {name: 0.0 for name in LOGIT_BASELINE_NAMES[:12]}

    logits = torch.stack(
        [generation_scores[idx][0].float() for idx in range(score_count)],
        dim=0,
    )
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    max_probs = probs.max(dim=-1).values.clamp_min(EPS)
    entropies_nats = -(probs * log_probs).sum(dim=-1)
    entropies_bits = entropies_nats / math.log(2.0)

    temp_logits = logits / max(temperature, EPS)
    temp_log_probs = F.log_softmax(temp_logits, dim=-1)
    temp_probs = temp_log_probs.exp()
    temp_max_probs = temp_probs.max(dim=-1).values
    temp_entropies = -(temp_probs * temp_log_probs).sum(dim=-1)

    energy = -temperature * torch.logsumexp(temp_logits, dim=-1)
    mean_neg_log_max_prob = -safe_mean(max_probs.log())

    return {
        "gen_mean_max_prob": safe_mean(max_probs),
        "gen_min_max_prob": safe_min(max_probs),
        "gen_max_max_prob": safe_max(max_probs),
        "gen_mean_neg_log_max_prob": mean_neg_log_max_prob,
        "gen_ppl_from_max_prob": float(math.exp(min(mean_neg_log_max_prob, 20.0))),
        "gen_mean_entropy_nats": safe_mean(entropies_nats),
        "gen_max_entropy_nats": safe_max(entropies_nats),
        "gen_mean_entropy_bits": safe_mean(entropies_bits),
        "gen_mean_temp_max_prob": safe_mean(temp_max_probs),
        "gen_mean_temp_entropy_nats": safe_mean(temp_entropies),
        "gen_mean_energy": safe_mean(energy),
        "gen_max_energy": safe_max(energy),
    }


def compute_teacher_forced_outputs(
    model,
    tokenizer,
    prompt: str,
    answer_text: str,
    answer_token_ids: Optional[torch.Tensor],
):
    device = get_model_input_device(model)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    answer_ids = prepare_answer_ids(
        tokenizer,
        answer_text=answer_text,
        device=device,
        answer_token_ids=answer_token_ids,
    )
    if answer_ids is None or answer_ids.numel() == 0:
        return None

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]
    target_positions = torch.arange(
        prompt_len, prompt_len + answer_len, dtype=torch.long, device=device
    )
    logit_positions = target_positions - 1

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    return {
        "outputs": outputs,
        "full_ids": full_ids,
        "target_positions": target_positions,
        "logit_positions": logit_positions,
    }


def compute_teacher_forced_logit_features(tf_data) -> Dict[str, float]:
    if tf_data is None:
        return {name: 0.0 for name in LOGIT_BASELINE_NAMES[12:]}

    outputs = tf_data["outputs"]
    full_ids = tf_data["full_ids"]
    logits_device = outputs.logits.device
    target_positions = tf_data["target_positions"].to(full_ids.device)
    logit_positions = tf_data["logit_positions"].to(logits_device)
    logits = outputs.logits[0, logit_positions, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    target_ids = full_ids[0, target_positions].to(logits.device)
    chosen_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    chosen_probs = chosen_log_probs.exp()
    entropies = -(probs * log_probs).sum(dim=-1)
    competitor_log_probs, _ = get_best_competitors(log_probs, target_ids)
    margins = chosen_log_probs - competitor_log_probs
    mean_nll = -safe_mean(chosen_log_probs)

    return {
        "tf_mean_chosen_prob": safe_mean(chosen_probs),
        "tf_min_chosen_prob": safe_min(chosen_probs),
        "tf_mean_chosen_logprob": safe_mean(chosen_log_probs),
        "tf_min_chosen_logprob": safe_min(chosen_log_probs),
        "tf_mean_neg_logprob": mean_nll,
        "tf_ppl": float(math.exp(min(mean_nll, 20.0))),
        "tf_mean_entropy_nats": safe_mean(entropies),
        "tf_max_entropy_nats": safe_max(entropies),
        "tf_mean_margin": safe_mean(margins),
        "tf_min_margin": safe_min(margins),
        "tf_frac_margin_lt_0_5": safe_mean((margins < 0.5).float()),
        "tf_frac_margin_lt_1_0": safe_mean((margins < 1.0).float()),
    }


def cosine_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    similarity = F.cosine_similarity(
        a.float().unsqueeze(0),
        b.float().unsqueeze(0),
        dim=-1,
        eps=EPS,
    ).squeeze(0)
    return torch.acos(similarity.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def compute_coe_from_trajectory(hidden_by_layer: torch.Tensor):
    if hidden_by_layer is None or hidden_by_layer.dim() != 2 or hidden_by_layer.shape[0] < 2:
        zeros = torch.zeros(1, dtype=torch.float32)
        return {
            "step_magnitudes": zeros,
            "step_angles": zeros,
            "features": {name: 0.0 for name in COE_FEATURE_NAMES[:6]},
        }

    hidden = hidden_by_layer.float()
    layer_count = hidden.shape[0]
    mag_denominator = (hidden[-1] - hidden[0]).norm(p=2).clamp_min(EPS)
    step_vectors = hidden[1:] - hidden[:-1]
    step_magnitudes = step_vectors.norm(p=2, dim=-1) / mag_denominator

    angle_denominator = cosine_angle(hidden[-1], hidden[0]).clamp_min(EPS)
    step_angles = torch.stack(
        [cosine_angle(hidden[idx + 1], hidden[idx]) / angle_denominator for idx in range(layer_count - 1)]
    )

    mag_mean = safe_mean(step_magnitudes)
    mag_var = float(step_magnitudes.float().var(unbiased=False).item())
    ang_mean = safe_mean(step_angles)
    ang_var = float(step_angles.float().var(unbiased=False).item())
    coe_r = mag_mean - ang_mean
    x_values = step_magnitudes * torch.cos(step_angles)
    y_values = step_magnitudes * torch.sin(step_angles)
    coe_c = float(torch.sqrt(x_values.mean().pow(2) + y_values.mean().pow(2)).item())

    return {
        "step_magnitudes": step_magnitudes.detach().cpu().float(),
        "step_angles": step_angles.detach().cpu().float(),
        "features": {
            "mag_mean": mag_mean,
            "mag_var": mag_var,
            "ang_mean": ang_mean,
            "ang_var": ang_var,
            "coe_r": coe_r,
            "coe_c": coe_c,
        },
    }


def extract_generation_hidden_trajectory(
    generation_hidden_states: Optional[Tuple[Tuple[torch.Tensor, ...], ...]]
) -> Optional[torch.Tensor]:
    if not generation_hidden_states:
        return None

    output_len = len(generation_hidden_states)
    layer_count = len(generation_hidden_states[0])
    layer_vectors = []
    for layer_idx in range(layer_count):
        token_vectors = []
        for step_idx in range(output_len):
            hidden = generation_hidden_states[step_idx][layer_idx]
            # Match the official CoE implementation: each generated step
            # contributes one layer vector, then vectors are averaged by layer.
            token_vectors.append(hidden[0, 0, :].float())
        layer_vectors.append(torch.stack(token_vectors, dim=0).mean(dim=0))
    return torch.stack(layer_vectors, dim=0).detach().cpu().float()


def extract_teacher_hidden_trajectory(tf_data) -> Optional[torch.Tensor]:
    if tf_data is None:
        return None
    hidden_states = tf_data["outputs"].hidden_states
    layer_vectors = [
        layer_hidden[
            0,
            tf_data["target_positions"].to(layer_hidden.device),
            :,
        ].float().mean(dim=0).detach().cpu()
        for layer_hidden in hidden_states
    ]
    return torch.stack(layer_vectors, dim=0).float()


def hidden_state_index_for_decoder_layer(layer_idx: int, hidden_state_count: int) -> int:
    # hidden_states[0] is the embedding output; decoder layer 0 lives at index 1.
    hidden_idx = layer_idx + 1 if layer_idx >= 0 else layer_idx
    if not -hidden_state_count <= hidden_idx < hidden_state_count:
        raise IndexError(
            f"Layer index {layer_idx} maps to hidden_states[{hidden_idx}], "
            f"but there are {hidden_state_count} hidden-state tensors."
        )
    return hidden_idx


def extract_probe_activations(tf_data, target_layers: List[int]) -> Dict[str, torch.Tensor]:
    hidden_states = tf_data["outputs"].hidden_states
    target_positions = tf_data["target_positions"]
    final_target_position = target_positions[-1]

    last_hidden = hidden_states[-1]
    last_positions = target_positions.to(last_hidden.device)
    last_final = last_hidden[0, final_target_position.to(last_hidden.device), :].float()
    last_mean = last_hidden[0, last_positions, :].float().mean(dim=0)

    target_final_vectors = []
    target_mean_vectors = []
    for layer_idx in target_layers:
        hidden_idx = hidden_state_index_for_decoder_layer(layer_idx, len(hidden_states))
        layer_hidden = hidden_states[hidden_idx]
        layer_positions = target_positions.to(layer_hidden.device)
        layer_final_position = final_target_position.to(layer_hidden.device)
        target_final_vectors.append(layer_hidden[0, layer_final_position, :].float())
        target_mean_vectors.append(layer_hidden[0, layer_positions, :].float().mean(dim=0))

    return {
        "last_hidden_final_token": last_final.detach().cpu().to(torch.float16),
        "last_hidden_mean_answer": last_mean.detach().cpu().to(torch.float16),
        "target_hidden_final_token": torch.stack(target_final_vectors, dim=0)
        .detach()
        .cpu()
        .to(torch.float16),
        "target_hidden_mean_answer": torch.stack(target_mean_vectors, dim=0)
        .detach()
        .cpu()
        .to(torch.float16),
    }


def coe_feature_dict_with_prefix(prefix: str, coe_result: Dict[str, object]) -> Dict[str, float]:
    features = coe_result["features"]
    return {
        f"{prefix}_coe_mag_mean": float(features["mag_mean"]),
        f"{prefix}_coe_mag_var": float(features["mag_var"]),
        f"{prefix}_coe_ang_mean": float(features["ang_mean"]),
        f"{prefix}_coe_ang_var": float(features["ang_var"]),
        f"{prefix}_coe_r": float(features["coe_r"]),
        f"{prefix}_coe_c": float(features["coe_c"]),
    }


def compute_embedding_eigenscore(embeddings: List[torch.Tensor]) -> Tuple[float, float]:
    if len(embeddings) < 2:
        return 0.0, 0.0
    matrix = torch.stack([embedding.float().cpu() for embedding in embeddings], dim=0)
    matrix = F.normalize(matrix, p=2, dim=-1, eps=EPS)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    gram = centered @ centered.t() / max(1, matrix.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(1e-8)
    eigenscore = float(torch.log(eigenvalues).sum().item())
    trace = float(eigenvalues.sum().item())
    return eigenscore, trace


def compute_sampled_baseline_features(
    model,
    tokenizer,
    prompt: str,
    greedy_answer: str,
    sample_count: int,
    max_new_tokens: int,
    sample_temperature: float,
    sample_top_p: float,
    scorer: rouge_scorer.RougeScorer,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if sample_count <= 0:
        return torch.zeros(len(SAMPLED_BASELINE_NAMES), dtype=torch.float32), {
            "sampled_answers": [],
        }

    sampled_answers = []
    sample_embeddings = []
    sample_lengths = []

    for _ in range(sample_count):
        sampled = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            return_diagnostics=True,
            do_sample=True,
            temperature=sample_temperature,
            top_p=sample_top_p,
        )
        sampled_answers.append(sampled.text)
        sample_lengths.append(float(sampled.token_ids.numel()))
        trajectory = extract_generation_hidden_trajectory(sampled.generation_hidden_states)
        if trajectory is not None:
            sample_embeddings.append(trajectory[-1])

    normalized_answers = [normalize_answer_text(answer) for answer in sampled_answers]
    nonempty_answers = [answer for answer in normalized_answers if answer]
    unique_ratio = len(set(nonempty_answers)) / max(1, len(nonempty_answers))

    counts = {}
    for answer in nonempty_answers:
        counts[answer] = counts.get(answer, 0) + 1
    if counts:
        probs = torch.tensor(list(counts.values()), dtype=torch.float32) / len(nonempty_answers)
        ln_entropy = float((-(probs * probs.clamp_min(EPS).log()).sum() / math.log(max(2, sample_count))).item())
    else:
        ln_entropy = 0.0

    rouge_to_greedy = [
        safe_rouge_l(scorer, greedy_answer, answer)
        for answer in sampled_answers
        if answer.strip()
    ]
    pairwise_rouges = []
    for left_idx in range(len(sampled_answers)):
        for right_idx in range(left_idx + 1, len(sampled_answers)):
            pairwise_rouges.append(
                safe_rouge_l(scorer, sampled_answers[left_idx], sampled_answers[right_idx])
            )

    length_tensor = torch.tensor(sample_lengths, dtype=torch.float32) if sample_lengths else torch.zeros(1)
    eigenscore, embedding_trace = compute_embedding_eigenscore(sample_embeddings)
    values = [
        float(len(sampled_answers)),
        float(unique_ratio),
        float(ln_entropy),
        float(sum(rouge_to_greedy) / max(1, len(rouge_to_greedy))),
        float(sum(pairwise_rouges) / max(1, len(pairwise_rouges))),
        safe_mean(length_tensor),
        float(length_tensor.var(unbiased=False).item()) if length_tensor.numel() > 1 else 0.0,
        eigenscore,
        embedding_trace,
    ]
    return torch.tensor(values, dtype=torch.float32), {
        "sampled_answers": sampled_answers,
        "sample_lengths": sample_lengths,
        "rouge_to_greedy": rouge_to_greedy,
        "pairwise_rouge_mean": values[4],
    }


# ---------------------------------------------------------------------------------------------------------
KEYWORD_SCALAR_NAMES = ["answer_token_count", "keyword_count", "mean_entropy", "max_entropy"]


def keyword_entropy(tokenizer, tf_data, prompt: str, max_keywords: int = 12) -> Optional[Dict[str, object]]:
    """H-bar: teacher-forced predictive entropy averaged over answer keywords.

    Keywords are the answer's word groups that are content words (no stop words / punctuation) and do not
    occur in the prompt; the `max_keywords` keywords with the smallest teacher-forced margin
    log p(chosen) - log p(best competitor) are kept.  A keyword's entropy is the mean next-token entropy of
    its sub-word tokens; H-bar is the mean over keywords.
    """
    outputs, full_ids = tf_data["outputs"], tf_data["full_ids"]
    logits = outputs.logits[0, tf_data["logit_positions"].to(outputs.logits.device), :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
    target_ids = full_ids[0, tf_data["target_positions"].to(full_ids.device)].to(logits.device)
    chosen = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    competitor, _ = get_best_competitors(log_probs, target_ids)
    margins = chosen - competitor
    groups, _, _, _ = select_probe_word_groups(tokenizer, target_ids, margins, max_keywords, prompt_text=prompt)
    if not groups:
        return None
    idx = [torch.tensor(g.indices, dtype=torch.long, device=margins.device) for g in groups]
    ents = torch.stack([entropies[i].mean() for i in idx]).cpu().float()
    marg = torch.stack([margins[i].mean() for i in idx]).cpu().float()
    scalars = torch.tensor([float(target_ids.numel()), float(len(groups)), safe_mean(ents), safe_max(ents)],
                           dtype=torch.float32)
    return {"scalars": scalars, "keywords": [g.text for g in groups], "keyword_entropies": ents.tolist(),
            "keyword_margins": marg.tolist()}


def extract_example(model, tokenizer, prompt: str, generated: "GeneratedAnswer", cfg: dict,
                    scorer: rouge_scorer.RougeScorer) -> Optional[Dict[str, object]]:
    """All features of one example: h (and other hidden-state views for the probe baselines) and H-bar from
    ONE teacher-forced pass; logit / CoE baselines from the same generation and pass; optional sampled
    answers for the PSA / EigenScore baselines."""
    tf_data = compute_teacher_forced_outputs(model, tokenizer, prompt=prompt, answer_text=generated.text,
                                             answer_token_ids=generated.token_ids)
    if tf_data is None:
        return None
    kw = keyword_entropy(tokenizer, tf_data, prompt, cfg["max_keywords"])
    if kw is None:
        return None
    probe = extract_probe_activations(tf_data, list(cfg["target_layers"]))
    logit = {**compute_generation_logit_features(generated.generation_scores, generated.raw_token_ids,
                                                 temperature=cfg["logit_temperature"]),
             **compute_teacher_forced_logit_features(tf_data)}
    gen_coe = compute_coe_from_trajectory(extract_generation_hidden_trajectory(generated.generation_hidden_states))
    tf_coe = compute_coe_from_trajectory(extract_teacher_hidden_trajectory(tf_data))
    coe = {**coe_feature_dict_with_prefix("gen", gen_coe), **coe_feature_dict_with_prefix("tf", tf_coe)}
    del tf_data
    sampled, sampled_raw = compute_sampled_baseline_features(
        model, tokenizer, prompt=prompt, greedy_answer=generated.text, sample_count=cfg["baseline_samples"],
        max_new_tokens=cfg["max_new_tokens"], sample_temperature=cfg["sample_temperature"],
        sample_top_p=cfg["sample_top_p"], scorer=scorer)
    return {
        "scalar_features": kw["scalars"],
        "logit_baseline_features": tensor_or_zeros([float(logit.get(n, 0.0)) for n in LOGIT_BASELINE_NAMES],
                                                   len(LOGIT_BASELINE_NAMES)),
        "coe_features": tensor_or_zeros([float(coe.get(n, 0.0)) for n in COE_FEATURE_NAMES], len(COE_FEATURE_NAMES)),
        "sampled_baseline_features": sampled,
        "probe_last_hidden_final_token": probe["last_hidden_final_token"],
        "probe_last_hidden_mean_answer": probe["last_hidden_mean_answer"],
        "probe_target_hidden_final_token": probe["target_hidden_final_token"],
        "probe_target_hidden_mean_answer": probe["target_hidden_mean_answer"],
        "record": {"keywords": kw["keywords"], "keyword_entropies": kw["keyword_entropies"],
                   "keyword_margins": kw["keyword_margins"], "sampled_answers": sampled_raw.get("sampled_answers", [])},
    }


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_model = model.base_model.model
        if hasattr(base_model, "layers"):
            return base_model.layers
    raise ValueError("Cannot find decoder layers on this model.")
