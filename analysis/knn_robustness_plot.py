"""Appendix figure: k-robustness of the relative factual-support gap over the five data splits (seeds 42-46).

Input: outputs/analysis/knn_robustness/s{42..46}.csv written by analysis/knn_robustness.py --seed S.
Output: figures/knn_robustness_relative_support_gap.pdf (2x3 models, one line per dataset = 5-seed mean,
band = ±1 seed std) and outputs/analysis/knn_robustness/summary.md.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import FIG_DIR, PROJECT_ROOT, RESULTS_DIR

R = RESULTS_DIR / "knn_robustness"
FIG = FIG_DIR
SEEDS = [42, 43, 44, 45, 46]
DATASETS = [("popqa", "PopQA"), ("webquestions", "WebQuestions"), ("triviaqa", "TriviaQA"), ("mkqa", "MKQA")]
MODELS = [("llama2_7b_chat", "Llama-2-7B-Chat"), ("llama3_8b_instruct", "Llama-3-8B-Instruct"),
          ("qwen2_7b_instruct", "Qwen2-7B-Instruct"), ("qwen2_5_7b_instruct", "Qwen2.5-7B-Instruct"),
          ("gemma_2_9b_it", "Gemma-2-9B-it"), ("deepseek_7b_chat", "DeepSeek-LLM-7B-Chat")]
COLORS = {"popqa": "#2a78d6", "webquestions": "#eb6834", "triviaqa": "#1baf7a", "mkqa": "#eda100"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def main():
    df = pd.concat([pd.read_csv(R / f"s{s}.csv").assign(seed=s) for s in SEEDS])
    ks = sorted(df.k.unique())
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey=True)
    for ax, (m, mn) in zip(axes.flat, MODELS):
        for d, dn in DATASETS:
            g = df[(df.model == m) & (df.dataset == d)].groupby("k").relative_factual_support_gap
            mu, sd = g.mean().reindex(ks).values, g.std(ddof=1).reindex(ks).values
            ax.fill_between(ks, mu - sd, mu + sd, color=COLORS[d], alpha=0.18, lw=0)
            ax.plot(ks, mu, marker="o", ms=4, lw=2, color=COLORS[d], label=dn)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_title(mn, fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("KNOW mean − HALL mean\nof relative support $s_{\\mathrm{sup}}$")
    for ax in axes[-1]:
        ax.set_xlabel("$k$ (nearest neighbours)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "knn_robustness_relative_support_gap.pdf")
    fig.savefig(FIG / "knn_robustness_relative_support_gap.png", dpi=150)

    # summary
    piv = df.groupby(["seed", "k"]).agg(pos=("relative_factual_support_gap", lambda x: int((x > 0).sum())),
                                        n=("relative_factual_support_gap", "size"),
                                        know_pos=("know_mean_relative_support", lambda x: int((x > 0).sum())),
                                        hall_neg=("hall_mean_relative_support", lambda x: int((x < 0).sum())),
                                        gap=("relative_factual_support_gap", "mean"),
                                        auroc=("auroc", "mean")).reset_index()
    L = ["# k-robustness of the relative factual support (seeds 42-46)\n",
         "`analysis/knn_robustness.py` (train-standardised h, PCA-128 fit on train, KNOW/HALL "
         "references from the train split, evaluated on the test split), run once per seed. Gap = mean s_sup on "
         "test KNOW − mean s_sup on test HALL.\n",
         "| seed | k | gap > 0 (settings) | KNOW mean > 0 | HALL mean < 0 | mean gap | mean AUROC of −s_sup |",
         "|---|---|---|---|---|---|---|"]
    for _, r in piv.iterrows():
        L.append(f"| {int(r.seed)} | {int(r.k)} | {int(r.pos)}/{int(r.n)} | {int(r.know_pos)}/{int(r.n)} | {int(r.hall_neg)}/{int(r.n)} | {r.gap:.3f} | "
                 f"{r.auroc:.3f} |")
    tot = len(df)
    L.append(f"\nOverall: gap > 0 in {int((df.relative_factual_support_gap > 0).sum())}/{tot} (setting, k, seed) "
             f"cases; KNOW mean > 0 in {int((df.know_mean_relative_support > 0).sum())}/{tot}; HALL mean < 0 in "
             f"{int((df.hall_mean_relative_support < 0).sum())}/{tot}.")
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
