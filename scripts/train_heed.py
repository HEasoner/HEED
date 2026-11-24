"""Train and evaluate HEED (and its two single-evidence ablations) on every (dataset, model, seed).

    python scripts/train_heed.py                                   # all 24 settings x 5 seeds
    python scripts/train_heed.py --jobs popqa/llama2_7b_chat/42    # one job
    python scripts/train_heed.py --variants heed,eh,eH

Variants (outputs/scores/<variant>/<dataset>_<model>_s<seed>.{npz,json}):
  heed  alpha = e_h + e_H + 1   (stage 1 e_h, frozen; stage 2 e_H)        -- the proposed detector
  eh    alpha = e_h + 1         (= stage 1 alone; written together with heed)
  eH    alpha = e_H + 1         (e_H trained alone with the same trainer)
The npz files store the test indices, test labels and test scores s_FRH; json files store validation AUROC,
best epochs and the learned (w, b) of e_H.  HEED checkpoints go to outputs/checkpoints/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heed import config as C  # noqa: E402
from heed.data import hbar, hidden_h, labels_of, load_features, split_indices, standardize, train_stats  # noqa: E402
from heed.model import evidence_of, fit_eh, fit_on_frozen, p_hall  # noqa: E402

SCORES = C.OUTPUT_DIR / "scores"
CKPT = C.OUTPUT_DIR / "checkpoints"


def save_scores(variant, tag, te, y, score, info):
    out = SCORES / variant / f"{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, test_idx=te, test_labels=y[te], score=score[te].astype(np.float32))
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))


def run_job(d, m, seed, variants, device):
    tag = C.tag(d, m, seed)
    todo = [v for v in variants if not (SCORES / v / f"{tag}.json").exists()]
    if not todo:
        return
    data = load_features(C.feature_path(d, m))
    labels = labels_of(data)
    y = labels.numpy()
    tr, va, te = split_indices(labels, seed)
    zH = None
    if {"heed", "eh", "eH"} & set(todo):
        hb = hbar(data)
        hb_mean, hb_std = train_stats(hb, tr)
        zH = standardize(hb, hb_mean, hb_std)
    base = dict(dataset=d, model=m, seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te))

    if {"heed", "eh"} & set(todo):
        t0 = time.time()
        fit1, mean, std = fit_eh(hidden_h(data), labels, tr, va, C.EH, seed, device)
        e_h = evidence_of(fit1.model, standardize(hidden_h(data), mean, std))
        s_eh = p_hall(e_h + 1.0)
        info_eh = dict(base, variant="eh", best_epoch=fit1.best_epoch, val_monitor=fit1.best_val,
                       val_auroc=float(roc_auc_score(y[va], s_eh[va])), seconds=time.time() - t0)
        save_scores("eh", tag, te, y, s_eh, info_eh)
        if "heed" in todo:
            fit2, alpha = fit_on_frozen(e_h, zH, labels, tr, va, C.EHBAR, seed, torch.device("cpu"))
            s = p_hall(alpha)
            lin = fit2.model.net[-1]
            w, b = lin.weight.detach().squeeze(1).tolist(), lin.bias.detach().tolist()
            info = dict(base, variant="heed", eh_best_epoch=fit1.best_epoch, eH_best_epoch=fit2.best_epoch,
                        val_auroc=float(roc_auc_score(y[va], s[va])), val_auroc_eh=info_eh["val_auroc"],
                        w_know=w[0], w_hall=w[1], b_know=b[0], b_hall=b[1], seconds=time.time() - t0)
            save_scores("heed", tag, te, y, s, info)
            CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(dict(eh=dict(state_dict=fit1.model.state_dict(), input_dim=int(mean.numel()),
                                    hidden_dim=C.EH["hidden_dim"], dropout=C.EH["dropout"], mean=mean, std=std),
                            eH=dict(state_dict=fit2.model.state_dict(), mean=hb_mean, std=hb_std),
                            config=dict(eh=C.EH, eH=C.EHBAR), info=info), CKPT / f"heed_{tag}.pt")

    if "eH" in todo:
        fit, alpha = fit_on_frozen(torch.zeros(len(y), 2), zH, labels, tr, va, C.EHBAR, seed, torch.device("cpu"))
        s = p_hall(alpha)
        lin = fit.model.net[-1]
        save_scores("eH", tag, te, y, s, dict(base, variant="eH", best_epoch=fit.best_epoch,
                                               val_auroc=float(roc_auc_score(y[va], s[va])),
                                               w=lin.weight.detach().squeeze(1).tolist(),
                                               b=lin.bias.detach().tolist()))


def all_jobs():
    return [f"{d}/{m}/{s}" for d, m in C.SETTINGS for s in C.SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="", help="comma list of dataset/model/seed (default: all 120)")
    ap.add_argument("--variants", default="heed,eh,eH")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device for stage 1 (e_h); stage 2 always runs on CPU")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    variants = a.variants.split(",")
    for job in (a.jobs.split(",") if a.jobs else all_jobs()):
        d, m, s = job.split("/")
        run_job(d, m, int(s), variants, torch.device(a.device))
        print(f"done {job}", flush=True)


if __name__ == "__main__":
    main()
