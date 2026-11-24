"""Baselines on the same seeds / splits as HEED.

    python scripts/run_baselines.py [--jobs popqa/llama2_7b_chat/42,...]

Training-free scores (direction s or -s chosen per (setting, seed) on the VALIDATION split):
  MSP            gen_mean_max_prob          mean max-token probability during generation
  Perplexity     tf_ppl                     teacher-forced perplexity of the answer
  Entropy        gen_mean_entropy_nats      mean next-token entropy during generation
  Keyword_Hbar   mean_entropy               H-bar (the e_H input of HEED) used alone
  CoE-R, CoE-C   gen_coe_r / gen_coe_c      Chain-of-Embedding scores (Wang et al., 2025)
  PSA            sample_mean_pairwise_rouge mean pairwise Rouge-L among 5 sampled answers
  EigenScore     sample_eigenscore          log-det of the sampled-answer embedding covariance (Chen et al., 2024)
Supervised (train split only):
  LR_probe       last layer, last answer token, train-standardised; LogisticRegression(C=1, balanced)
  SAPLMA         faithful re-implementation of Azaria & Mitchell (2023): 256-128-64 ReLU MLP, BCE, Adam 1e-3,
                 batch 32, 5 epochs, raw activation of the last answer token at ONE layer; the layer
                 (last, -4, -9, -14) is chosen per (setting, seed) on validation AUROC.
Outputs: outputs/scores/<method>/<dataset>_<model>_s<seed>.{npz,json}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heed import config as C  # noqa: E402
from heed.data import labels_of, load_features, split_indices  # noqa: E402

SCORES = C.OUTPUT_DIR / "scores"
FREE = [("MSP", "logit_baseline_features", "logit_baseline_names", "gen_mean_max_prob"),
        ("Perplexity", "logit_baseline_features", "logit_baseline_names", "tf_ppl"),
        ("Entropy", "logit_baseline_features", "logit_baseline_names", "gen_mean_entropy_nats"),
        ("Keyword_Hbar", "scalar_features", "scalar_names", "mean_entropy"),
        ("CoE-R", "coe_features", "coe_feature_names", "gen_coe_r"),
        ("CoE-C", "coe_features", "coe_feature_names", "gen_coe_c"),
        ("PSA", "sampled_baseline_features", "sampled_baseline_names", "sample_mean_pairwise_rouge"),
        ("EigenScore", "sampled_baseline_features", "sampled_baseline_names", "sample_eigenscore")]
SAPLMA_LAYERS = ["Llast", "L-4", "L-9", "L-14"]


def save(method, tag, te, y, score, info):
    out = SCORES / method / f"{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, test_idx=te, test_labels=y[te], score=np.asarray(score)[te].astype(np.float32))
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))


def column(data, group, names, field):
    return data[group][:, list(data[names]).index(field)].double().numpy()


def training_free(data, name, group, names, field, y, va):
    raw = column(data, group, names, field)
    bad = ~np.isfinite(raw)
    raw = np.where(bad, np.nanmedian(raw[~bad]) if (~bad).any() else 0.0, raw)
    if len(np.unique(raw)) <= 1:
        return None, dict(skipped="constant score")
    sign = 1.0 if roc_auc_score(y[va], raw[va]) >= 0.5 else -1.0
    return sign * raw, dict(field=field, sign=sign, n_nonfinite=int(bad.sum()))


def lr_probe(data, y, tr, seed):
    x = data["probe_last_hidden_final_token"].float()
    xt = x[torch.from_numpy(tr).long()]
    x = torch.nan_to_num((x - xt.mean(0)) / xt.std(0, unbiased=False).clamp_min(1e-6), nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000, class_weight="balanced",
                             random_state=seed)
    clf.fit(x[tr].numpy(), y[tr])
    return clf.predict_proba(x.numpy())[:, 1], dict(field="probe_last_hidden_final_token")


class SAPLMANet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        for mod in self.net:                     # Keras defaults: Glorot-uniform weights, zero bias
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def saplma_layer(data, layer):
    if layer == "Llast":
        x = data["probe_last_hidden_final_token"]
    else:
        x = data["probe_target_hidden_final_token"][:, {"L-4": 0, "L-9": 1, "L-14": 2}[layer]]
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def saplma_train_score(x, y, tr, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = SAPLMANet(x.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-7)
    lossf = nn.BCEWithLogitsLoss()
    xt, yt = x[tr].to(device), torch.from_numpy(y[tr]).float().to(device)
    g = torch.Generator().manual_seed(seed)
    for _ in range(5):
        net.train()
        for b in torch.randperm(len(tr), generator=g).split(32):
            opt.zero_grad(set_to_none=True)
            lossf(net(xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(net(c.to(device))).cpu() for c in x.split(4096)]).numpy()


def saplma(data, y, tr, va, seed, device):
    best = None
    vals = {}
    for layer in SAPLMA_LAYERS:
        s = saplma_train_score(saplma_layer(data, layer), y, tr, seed, device)
        vals[layer] = float(roc_auc_score(y[va], s[va]))
        if best is None or vals[layer] > best[0]:
            best = (vals[layer], layer, s)
    return best[2], dict(layer=best[1], val_auroc_by_layer=vals)


def run_job(d, m, seed, methods, device):
    tag = C.tag(d, m, seed)
    todo = [x for x in methods if not (SCORES / x / f"{tag}.json").exists()]
    if not todo:
        return
    data = load_features(C.feature_path(d, m))
    labels = labels_of(data)
    y = labels.numpy()
    tr, va, te = split_indices(labels, seed)
    for name in todo:
        if name == "LR_probe":
            s, info = lr_probe(data, y, tr, seed)
        elif name == "SAPLMA":
            s, info = saplma(data, y, tr, va, seed, device)
        else:
            s, info = training_free(data, *[f for f in FREE if f[0] == name][0], y, va)
        info.update(method=name, dataset=d, model=m, seed=seed)
        if s is None:
            (SCORES / name).mkdir(parents=True, exist_ok=True)
            (SCORES / name / f"{tag}.json").write_text(json.dumps(info, indent=1))
            continue
        info.update(val_auroc=float(roc_auc_score(y[va], s[va])))
        save(name, tag, te, y, s, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="")
    ap.add_argument("--methods", default=",".join([f[0] for f in FREE] + ["LR_probe", "SAPLMA"]))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    jobs = a.jobs.split(",") if a.jobs else [f"{d}/{m}/{s}" for d, m in C.SETTINGS for s in C.SEEDS]
    for job in jobs:
        d, m, s = job.split("/")
        run_job(d, m, int(s), a.methods.split(","), torch.device(a.device))
        print(f"done {job}", flush=True)


if __name__ == "__main__":
    main()
