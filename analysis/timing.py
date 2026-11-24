"""Per-answer wall-clock cost on one GPU (Llama-2-7B-Chat, fp16, PopQA prompts from the stored records).

Untrained heads of the right shape are used: only the cost is measured.
Measured stages (torch.cuda.synchronize around each; first --warmup questions discarded):
  generate     greedy answer generation (max_new_tokens 32) - common to every detector
  heed         1 teacher-forced forward (output_hidden_states) + h (3-layer answer-token mean) + H-bar
               (content-word entropy) + e_h MLP (3d->128->32->2) + e_H linear head
  saplma       1 teacher-forced forward + single-layer last-token activation + 256-128-64 MLP
  sampling     5 sampled answers (T=1.0, top-p 0.95) + Rouge/EigenScore features
               (heed.extraction.compute_sampled_baseline_features), i.e. PSA / EigenScore
Writes outputs/analysis/timing.json.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from _paths import C, RESULTS_DIR, feature_file
import heed.extraction as INF


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=C.MODELS["llama2_7b_chat"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--skip-sampling", action="store_true")
    a = p.parse_args()
    data = torch.load(str(feature_file("popqa", "llama2_7b_chat")),
                      map_location="cpu", mmap=True)
    recs = data["records"][: a.n + a.warmup]
    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.float16).cuda().eval()
    layers = list(C.EXTRACTION["target_layers"])
    d = model.config.hidden_size
    dev = torch.device("cuda")
    e_h = nn.Sequential(nn.Linear(3 * d, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 32), nn.LayerNorm(32),
                        nn.GELU(), nn.Linear(32, 2), nn.Softplus()).to(dev).eval()
    e_H = nn.Sequential(nn.Linear(1, 2), nn.Softplus()).to(dev).eval()
    sap = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                        nn.Linear(64, 1), nn.Sigmoid()).to(dev).eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    T = {k: [] for k in ["generate", "heed", "saplma", "sampling"]}

    def heed(prompt, ans):
        tf = INF.compute_teacher_forced_outputs(model, tok, prompt, ans.text, ans.token_ids)
        hs, pos = tf["outputs"].hidden_states, tf["target_positions"]
        h = torch.cat([hs[INF.hidden_state_index_for_decoder_layer(l, len(hs))][0, pos].float().mean(0)
                       for l in layers])
        logits = tf["outputs"].logits[0, tf["logit_positions"]].float()
        lp = torch.log_softmax(logits, -1)
        ent = -(lp.exp() * lp).sum(-1)
        tgt = tf["full_ids"][0, pos]
        chosen = lp.gather(1, tgt[:, None]).squeeze(1)
        comp, _ = INF.get_best_competitors(lp, tgt)
        groups, *_ = INF.select_probe_word_groups(tok, tgt, chosen - comp, C.EXTRACTION["max_keywords"], prompt_text=prompt)
        hbar = torch.stack([ent[torch.tensor(g.indices, device=dev)].mean() for g in groups]).mean() if groups \
            else ent.mean()
        with torch.no_grad():
            alpha = e_h(h[None]) + e_H(hbar.view(1, 1)) + 1.0
        return float(alpha[0, 1] / alpha.sum())

    def saplma(prompt, ans):
        tf = INF.compute_teacher_forced_outputs(model, tok, prompt, ans.text, ans.token_ids)
        x = tf["outputs"].hidden_states[-9][0, tf["target_positions"][-1]].float()
        with torch.no_grad():
            return float(sap(x[None]))

    for i, r in enumerate(recs):
        prompt = r["prompt"]
        ans, t = timed(lambda: INF.generate_answer(model, tok, prompt, max_new_tokens=32))
        if not ans.text.strip():
            continue
        times = {"generate": t}
        _, times["heed"] = timed(lambda: heed(prompt, ans))
        _, times["saplma"] = timed(lambda: saplma(prompt, ans))
        if not a.skip_sampling:
            _, times["sampling"] = timed(lambda: INF.compute_sampled_baseline_features(
                model, tok, prompt, ans.text, 5, 32, 1.0, 0.95, scorer))
        if i >= a.warmup:
            for k, v in times.items():
                T[k].append(v)
    res = {k: dict(n=len(v), mean_ms=1000 * float(np.mean(v)), median_ms=1000 * float(np.median(v)),
                   std_ms=1000 * float(np.std(v))) for k, v in T.items() if v}
    res["setup"] = dict(model=a.model_path, gpu=torch.cuda.get_device_name(0), dtype="float16", n=a.n,
                        warmup=a.warmup, prompts="first records of the balanced PopQA/Llama-2 feature file",
                        max_new_tokens=32, samples=5)
    (RESULTS_DIR / "timing.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
