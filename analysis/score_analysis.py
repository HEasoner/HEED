"""Score analysis of HEED on the test splits (seeds 42-46).

1. s_FRH distribution on KNOW / HALL test answers: figures/sfrh_by_dataset.pdf (4 panels, pooled over models
   and seeds), figures/sfrh_small_multiples.pdf (24 settings), per-setting mean / median table.
2. Effect of e_H: per-setting AUROC(HEED) - AUROC(e_h only) (seed mean, error bar = seed std):
   figures/eH_delta_auroc.pdf.
3. Learned w of e_H = softplus(w * z + b) (z = train-standardised H-bar), read from the HEED run files.
4. Per-answer cost from outputs/analysis/timing.json (analysis/timing.py), if present.
Needs outputs/scores/{heed,eh} (scripts/train_heed.py).  Writes outputs/analysis/score_analysis.md.
"""
import json

import numpy as np

from _paths import FIG_DIR, RESULTS_DIR, C

SCORES = C.OUTPUT_DIR / "scores"
FIG = FIG_DIR
DATASETS, MODELS, SEEDS = C.DATASETS, list(C.MODELS), C.SEEDS
DN = {"popqa": "PopQA", "webquestions": "WebQuestions", "triviaqa": "TriviaQA", "mkqa": "MKQA"}
MN = {"llama2_7b_chat": "Llama-2-7B", "llama3_8b_instruct": "Llama-3-8B", "qwen2_7b_instruct": "Qwen2-7B",
      "qwen2_5_7b_instruct": "Qwen2.5-7B", "gemma_2_9b_it": "Gemma-2-9B", "deepseek_7b_chat": "DeepSeek-7B"}
KNOW_C, HALL_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    S = [(d, m) for d in DATASETS for m in MODELS]
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- scores
    full, eh = {}, {}
    for d, m in S:
        for s in SEEDS:
            tag = f"{d}_{m}_s{s}"
            zf, ze = np.load(SCORES / "heed" / f"{tag}.npz"), np.load(SCORES / "eh" / f"{tag}.npz")
            assert np.array_equal(zf["test_idx"], ze["test_idx"])
            full[(d, m, s)] = (zf["test_labels"], zf["score"])
            eh[(d, m, s)] = (ze["test_labels"], ze["score"])
    L = ["# Score analysis: HEED (e_h + e_H), test splits, seeds 42–46\n",
         "All numbers are on the test split of each seed (stratified 70/15/15, class-balanced). "
         "s_FRH = α_HALL / (α_HALL + α_KNOW) with α = e_h + e_H + 1. Positive class = HALL.\n"]

    # ---------- 1. s_FRH distributions
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=False)
    rows = []
    for ax, d in zip(axes, DATASETS):
        yk = np.concatenate([full[(d, m, s)][1][full[(d, m, s)][0] == 0] for m in MODELS for s in SEEDS])
        yh = np.concatenate([full[(d, m, s)][1][full[(d, m, s)][0] == 1] for m in MODELS for s in SEEDS])
        for v, c, lab in ((yk, KNOW_C, "KNOW"), (yh, HALL_C, "HALL")):
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=c)
            ax.hist(v, bins=bins, density=True, histtype="step", linewidth=1.5, color=c, label=lab)
        ax.set_title(DN[d], fontsize=9, color=INK)
        ax.set_xlabel(r"$s_{\mathrm{FRH}}$", color=INK)
        style(ax)
    axes[0].set_ylabel("density", color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "sfrh_by_dataset.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(4, 6, figsize=(12, 7.5), sharex=True)
    for i, d in enumerate(DATASETS):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            y = np.concatenate([full[(d, m, s)][0] for s in SEEDS])
            sc = np.concatenate([full[(d, m, s)][1] for s in SEEDS])
            for cls, c, lab in ((0, KNOW_C, "KNOW"), (1, HALL_C, "HALL")):
                ax.hist(sc[y == cls], bins=bins, density=True, histtype="step", linewidth=1.3, color=c, label=lab)
                rows.append(dict(dataset=d, model=m, cls=lab, n=int((y == cls).sum()),
                                 mean=float(sc[y == cls].mean()), median=float(np.median(sc[y == cls]))))
            if i == 0:
                ax.set_title(MN[m], fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(DN[d], fontsize=8, color=INK)
            style(ax)
            ax.tick_params(labelsize=6)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$s_{\mathrm{FRH}}$", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "sfrh_small_multiples.pdf")
    plt.close(fig)
    L += ["## 1. s_FRH on KNOW vs HALL test answers\n",
          "Figures: `figures/sfrh_by_dataset.pdf` (pooled over 6 models × 5 seeds per dataset), "
          "`figures/sfrh_small_multiples.pdf` (24 settings, pooled over seeds).\n",
          "| Dataset | KNOW mean | KNOW median | HALL mean | HALL median |", "|---|---|---|---|---|"]
    for d in DATASETS:
        yk = np.concatenate([full[(d, m, s)][1][full[(d, m, s)][0] == 0] for m in MODELS for s in SEEDS])
        yh = np.concatenate([full[(d, m, s)][1][full[(d, m, s)][0] == 1] for m in MODELS for s in SEEDS])
        L.append(f"| {DN[d]} | {yk.mean():.3f} | {np.median(yk):.3f} | {yh.mean():.3f} | {np.median(yh):.3f} |")
    L += ["\nPer setting (pooled over the 5 seeds' test sets):\n",
          "| Dataset | Model | n KNOW | KNOW mean | KNOW median | n HALL | HALL mean | HALL median |",
          "|---|---|---|---|---|---|---|---|"]
    for d, m in S:
        k = next(r for r in rows if r["dataset"] == d and r["model"] == m and r["cls"] == "KNOW")
        h = next(r for r in rows if r["dataset"] == d and r["model"] == m and r["cls"] == "HALL")
        L.append(f"| {DN[d]} | {MN[m]} | {k['n']} | {k['mean']:.3f} | {k['median']:.3f} | {h['n']} | "
                 f"{h['mean']:.3f} | {h['median']:.3f} |")

    # ---------- 2. ΔAUROC from e_H
    delta = {(d, m): np.array([roc_auc_score(*full[(d, m, s)]) - roc_auc_score(*eh[(d, m, s)]) for s in SEEDS])
             for d, m in S}
    fig, ax = plt.subplots(figsize=(10, 2.8))
    xs, xt, xl = [], [], []
    x = 0
    for d in DATASETS:
        start = x
        for m in MODELS:
            v = delta[(d, m)]
            ax.bar(x, v.mean(), width=0.72, color=KNOW_C, zorder=2)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color=MUTED, linewidth=1, capsize=2, zorder=3)
            xs.append(x); xt.append(x); xl.append(MN[m])
            x += 1
        ax.text((start + x - 1) / 2, 1.02, DN[d], transform=ax.get_xaxis_transform(), ha="center", fontsize=9,
                color=INK)
        x += 0.8
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"$\Delta$AUROC (Full $-$ $e_h$ only)", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "eH_delta_auroc.pdf")
    plt.close(fig)
    dm = np.array([delta[st].mean() for st in S])
    L += ["\n## 2. Effect of the keyword-entropy evidence e_H\n",
          "Figure: `figures/eH_delta_auroc.pdf` (bar = seed mean of AUROC(Full) − AUROC(e_h only); error bar = "
          "seed std).\n",
          f"ΔAUROC > 0 in {int((dm > 0).sum())}/24 settings; mean {dm.mean():+.4f}, min {dm.min():+.4f}, "
          f"max {dm.max():+.4f}. Per (setting, seed) run: > 0 in "
          f"{int(sum((delta[st] > 0).sum() for st in S))}/120.\n",
          "| Dataset | " + " | ".join(MN[m] for m in MODELS) + " | mean |", "|" + "---|" * 8]
    for d in DATASETS:
        L.append(f"| {DN[d]} | " + " | ".join(f"{delta[(d, m)].mean():+.4f} ± {delta[(d, m)].std(ddof=1):.4f}"
                                             for m in MODELS)
                 + f" | {np.mean([delta[(d, m)].mean() for m in MODELS]):+.4f} |")

    # ---------- 3. learned w
    W = [json.loads((SCORES / "heed" / f"{d}_{m}_s{s}.json").read_text()) for d, m in S for s in SEEDS]
    wk = np.array([r["w_know"] for r in W])
    wh = np.array([r["w_hall"] for r in W])
    L += ["\n## 3. Learned weights of e_H = softplus(w · z + b), z = train-standardised H̄\n",
          f"- w_HALL > 0 in {int((wh > 0).sum())}/120 runs; w_KNOW < 0 in {int((wk < 0).sum())}/120 runs; "
          f"both in {int(((wh > 0) & (wk < 0)).sum())}/120.",
          f"- Mean w_HALL = {wh.mean():+.3f} (min {wh.min():+.3f}); mean w_KNOW = {wk.mean():+.3f} "
          f"(max {wk.max():+.3f}).",
          "- Reading: w_HALL > 0 means higher keyword entropy adds HALL evidence; w_KNOW < 0 means it removes KNOW "
          "evidence.\n",
          "| Dataset | Model | w_KNOW mean ± std | w_HALL mean ± std | #seeds w_HALL>0 | #seeds w_KNOW<0 |",
          "|---|---|---|---|---|---|"]
    for d, m in S:
        r = [x for x in W if x["dataset"] == d and x["model"] == m]
        k = np.array([x["w_know"] for x in r])
        h = np.array([x["w_hall"] for x in r])
        L.append(f"| {DN[d]} | {MN[m]} | {k.mean():+.3f} ± {k.std(ddof=1):.3f} | {h.mean():+.3f} ± "
                 f"{h.std(ddof=1):.3f} | {int((h > 0).sum())}/5 | {int((k < 0).sum())}/5 |")

    # ---------- 4. timing
    tj = RESULTS_DIR / "timing.json"
    L.append("\n## 4. Per-answer cost\n")
    if tj.exists():
        t = json.loads(tj.read_text())
        su = t["setup"]
        g = t["generate"]["mean_ms"]
        L += [f"Setup: {su['model'].split('/')[-1]}, {su['gpu']}, {su['dtype']}, batch size 1, {t['generate']['n']} "
              f"PopQA questions (after {su['warmup']} warm-up), greedy answers ≤ {su['max_new_tokens']} new tokens. "
              "\n",
              "| Stage | mean ms / answer | median ms | relative to greedy generation |", "|---|---|---|---|",
              f"| Greedy answer generation (common to all detectors) | {g:.1f} | {t['generate']['median_ms']:.1f} | 1.00× |"]
        names = {"heed": "HEED: 1 teacher-forced forward + h, H̄ + e_h, e_H heads",
                 "saplma": "SAPLMA: 1 teacher-forced forward + 1-layer MLP",
                 "sampling": "PSA / EigenScore: 5 sampled answers + Rouge / eigen-score"}
        for k, n in names.items():
            if k in t:
                L.append(f"| {n} | {t[k]['mean_ms']:.1f} | {t[k]['median_ms']:.1f} | {t[k]['mean_ms'] / g:.2f}× |")
    else:
        L.append("timing.json not found (run analysis/timing.py).")
    (RESULTS_DIR / "score_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
