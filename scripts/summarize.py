"""Paper tables from outputs/scores -> outputs/tables/{main,ablation,baselines}.md

Aggregation: per setting, mean over the 5 seeds; overall = mean over the 24 settings; +/- = standard
deviation over seeds of the 24-setting average.  Wilcoxon signed-rank (two-sided) over the 24 per-setting
seed-mean differences.  HALL is the positive class; FPR@95 = FPR at 95% recall of HALL (lower is better).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heed import config as C  # noqa: E402
from heed.metrics import detection_metrics  # noqa: E402

SCORES = C.OUTPUT_DIR / "scores"
TABLES = C.OUTPUT_DIR / "tables"
K = ["auroc", "aupr", "fpr95"]
LABEL = {"heed": "HEED (e_h + e_H)", "eh": "e_h only", "eH": "e_H only", "SAPLMA": "SAPLMA", "LR_probe": "LR probe",
         "Keyword_Hbar": "Keyword entropy H-bar"}
SHORT_D = {"popqa": "PopQA", "webquestions": "WebQ", "triviaqa": "TriviaQA", "mkqa": "MKQA"}


def collect(method):
    """{(d, m): {metric: [per-seed values]}} or None if any run is missing / skipped."""
    out = {}
    for d, m in C.SETTINGS:
        for s in C.SEEDS:
            p = SCORES / method / f"{C.tag(d, m, s)}.npz"
            if not p.exists():
                return None
            z = np.load(p)
            for k, v in detection_metrics(z["test_labels"], z["score"]).items():
                out.setdefault((d, m), {}).setdefault(k, []).append(v)
    return out


def overall(v, k):
    arr = np.array([v[st][k] for st in C.SETTINGS])          # 24 x 5
    return f"{arr.mean(1).mean():.4f} ± {arr.mean(0).std(ddof=1):.4f}"


def seed_means(v, k="auroc"):
    return np.array([np.mean(v[st][k]) for st in C.SETTINGS])


def comparison_table(methods, ref, title):
    V = {x: collect(x) for x in methods}
    missing = [x for x, v in V.items() if v is None]
    V = {x: v for x, v in V.items() if v is not None}
    L = [f"# {title}\n", "| Method | AUROC | AUPR | FPR@95 ↓ | HEED better /24 | Wilcoxon p |", "|---|---|---|---|---|---|"]
    for x, v in V.items():
        cells = [overall(v, k) for k in K]
        if x == ref or ref not in V:
            cells += ["–", "–"]
        else:
            diff = seed_means(V[ref]) - seed_means(v)
            cells += [f"{int((diff > 0).sum())}", f"{stats.wilcoxon(diff).pvalue:.2g}" if np.any(diff) else "1"]
        L.append(f"| {LABEL.get(x, x)} | " + " | ".join(cells) + " |")
    if missing:
        L.append(f"\nIncomplete (not shown): {', '.join(missing)}")
    L += ["\n## Per-dataset mean (6 LLMs each): AUROC / AUPR / FPR@95\n",
          "| Method | " + " | ".join(SHORT_D[d] for d in C.DATASETS) + " |", "|" + "---|" * (len(C.DATASETS) + 1)]
    for x, v in V.items():
        L.append(f"| {LABEL.get(x, x)} | " + " | ".join(
            " / ".join(f"{np.mean([np.mean(v[(d, m)][k]) for m in C.MODELS]):.4f}" for k in K)
            for d in C.DATASETS) + " |")
    for k, kn in zip(K, ["AUROC", "AUPR", "FPR@95"]):
        L += [f"\n## Per-setting {kn} (mean ± std over seeds)\n",
              "| Dataset | Model | " + " | ".join(LABEL.get(x, x) for x in V) + " |", "|" + "---|" * (2 + len(V))]
        for d, m in C.SETTINGS:
            L.append(f"| {SHORT_D[d]} | {C.MODEL_NAMES[m]} | " + " | ".join(
                f"{np.mean(v[(d, m)][k]):.4f} ± {np.std(v[(d, m)][k], ddof=1):.4f}" for v in V.values()) + " |")
    return "\n".join(L) + "\n"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {
        "main": (["heed", "SAPLMA"], "HEED vs. SAPLMA (24 settings, seeds 42-46)"),
        "ablation": (["heed", "eh", "eH"], "Ablation of the two evidence sources"),
        "baselines": (["heed", "SAPLMA", "LR_probe", "MSP", "Perplexity", "Entropy", "Keyword_Hbar", "CoE-R", "CoE-C",
                       "PSA", "EigenScore"], "HEED vs. baselines"),
    }
    for name, (methods, title) in tables.items():
        text = comparison_table(methods, "heed", title)
        (TABLES / f"{name}.md").write_text(text)
        print(text.split("\n## ")[0])


if __name__ == "__main__":
    main()
