"""Generate answers with a frozen LLM, label them KNOW / HALL and extract the detector inputs.

    python scripts/extract_features.py --dataset popqa --model llama2_7b_chat [--model-path /local/dir]
Writes features/<dataset>_<model>.pt (then run scripts/balance_features.py).

Per question: one greedy generation (max 32 new tokens) and one teacher-forced forward pass on [q; a]
(no backward pass).  Stored tensors (N rows):
  labels                            N x 2 one-hot [KNOW, HALL]
  probe_target_hidden_mean_answer   N x 3 x d  h: answer-token mean of decoder blocks -4, -9, -14 (fp16)
  scalar_features / scalar_names    keyword statistics; "mean_entropy" is H-bar
  -- used only by baselines --
  probe_last_hidden_final_token     LR probe / SAPLMA (last layer, last answer token)
  probe_target_hidden_final_token   SAPLMA layer variants (blocks -4, -9, -14, last answer token)
  probe_last_hidden_mean_answer
  logit_baseline_features / _names  MSP, perplexity, entropy, ...
  coe_features / coe_feature_names  CoE-R / CoE-C
  sampled_baseline_features / _names  PSA / EigenScore (5 extra sampled answers; --baseline-samples 0 disables)
  records                           question, prompt, answers, generated answer, label, match details, keywords
"""
import argparse
import gc
import sys
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heed import config as C  # noqa: E402
from heed.extraction import (COE_FEATURE_NAMES, KEYWORD_SCALAR_NAMES, LOGIT_BASELINE_NAMES,  # noqa: E402
                             SAMPLED_BASELINE_NAMES, build_generation_prompt, extract_example,
                             freeze_model_parameters, generate_answer, resolve_prompt_style)
from heed.qa import DATA_FILES, label_answer, load_questions, question_and_answers  # noqa: E402

TENSOR_KEYS = ["scalar_features", "logit_baseline_features", "coe_features", "sampled_baseline_features",
               "probe_last_hidden_final_token", "probe_last_hidden_mean_answer",
               "probe_target_hidden_final_token", "probe_target_hidden_mean_answer"]


def main():
    E = C.EXTRACTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--model-path", default=None, help="local checkpoint dir (default: the HF id in heed/config.py)")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--limit", type=int, default=E["row_limit"], help="-1 = all rows")
    ap.add_argument("--max-examples", type=int, default=None, help="stop after this many rows (debugging)")
    ap.add_argument("--baseline-samples", type=int, default=E["baseline_samples"])
    ap.add_argument("--seed", type=int, default=42, help="row sampling and answer sampling seed")
    a = ap.parse_args()
    cfg = dict(E, baseline_samples=a.baseline_samples)

    data_path = a.data_path or C.DATA_DIR / DATA_FILES[a.dataset]
    save_path = Path(a.save_path or C.feature_path(a.dataset, a.model, balanced=False))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = a.model_path or C.MODELS[a.model]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_style = resolve_prompt_style(tokenizer, cfg["prompt_style"])
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    freeze_model_parameters(model)

    df = load_questions(a.dataset, data_path, limit=None if a.limit < 0 else a.limit, seed=a.seed,
                        popqa_max_subject_pop=cfg["popqa_max_subject_pop"])
    if a.max_examples:
        df = df.iloc[:a.max_examples]
    torch.manual_seed(a.seed)  # sampled answers (PSA / EigenScore baselines only)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = {k: [] for k in TENSOR_KEYS}
    labels, records = [], []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        question, answers, source = question_and_answers(a.dataset, row)
        prompt = build_generation_prompt(tokenizer, question, prompt_style)
        try:
            gen = generate_answer(model, tokenizer, prompt, max_new_tokens=cfg["max_new_tokens"],
                                  return_diagnostics=True)
            if not gen.text:
                continue
            y, match = label_answer(a.dataset, gen.text, answers)
            ex = extract_example(model, tokenizer, prompt, gen, cfg, scorer)
            if ex is None or any(not torch.isfinite(ex[k].float()).all() for k in TENSOR_KEYS[:4]):
                continue
            for k in TENSOR_KEYS:
                rows[k].append(ex[k])
            labels.append(torch.tensor([1.0, 0.0]) if y == 0 else torch.tensor([0.0, 1.0]))
            records.append(dict(row_index=int(index), question=question, prompt=prompt, raw_generated_answer=gen.raw_text,
                                generated_answer=gen.text, label="KNOW" if y == 0 else "HALLUCINATION",
                                label_source=dict(source, match=match), **ex["record"]))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        finally:
            gc.collect()

    out = {k: torch.stack(v) for k, v in rows.items()}
    out.update(labels=torch.stack(labels), records=records, scalar_names=KEYWORD_SCALAR_NAMES,
               logit_baseline_names=LOGIT_BASELINE_NAMES, coe_feature_names=COE_FEATURE_NAMES,
               sampled_baseline_names=SAMPLED_BASELINE_NAMES, dataset=a.dataset, model=a.model,
               model_path=str(model_path), prompt_style=prompt_style, extraction_config=cfg, seed=a.seed)
    torch.save(out, save_path)
    n_know = int((out["labels"].argmax(1) == 0).sum())
    print(f">>> saved {len(labels)} examples (KNOW={n_know}, HALL={len(labels) - n_know}) to {save_path}")


if __name__ == "__main__":
    main()
