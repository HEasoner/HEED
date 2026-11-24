"""Method overview figure (vector PDF): one teacher-forced forward pass -> h and H-bar -> e_h, e_H -> Dirichlet.

Writes figures/heed_pipeline.pdf (+ .png preview).  No method name is printed in the figure, so renaming the
method only needs the caption.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _paths import FIG_DIR

FIG = FIG_DIR
INK, MUTED = "#1f1f1e", "#6b6a66"
NEUTRAL_FC, NEUTRAL_EC = "#f3f2ef", "#8a8984"
H_EC, H_FC = "#6250d6", "#ecebfb"          # e_h branch (violet)
E_EC, E_FC = "#1baf7a", "#e3f5ee"          # e_H branch (aqua)
OUT_EC, OUT_FC = "#1f1f1e", "#ffffff"

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, fc, ec, fs=8.6, lw=1.1, bold_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, color=MUTED, lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(7.2, 2.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.004, 1.0)
    ax.set_ylim(-0.004, 0.355)
    ax.axis("off")

    yc = 0.1775
    # input + frozen LLM
    q = box(ax, 0.008, yc - 0.045, 0.104, 0.09, "question $q$\n$\\downarrow$\ngreedy\nanswer $\\hat a$", NEUTRAL_FC,
            NEUTRAL_EC, fs=8.2)
    f = box(ax, 0.130, yc - 0.075, 0.146, 0.15,
            "frozen LLM:\none teacher-forced\nforward pass\non $[q;\\hat a]$\n(no backward,\nno sampling)",
            NEUTRAL_FC, NEUTRAL_EC, fs=8.2)
    arrow(ax, (q[0] + q[2], yc), (f[0], yc))

    # two read-outs from the same pass
    yt, yb = 0.268, 0.087
    hh = box(ax, 0.298, yt - 0.052, 0.192, 0.104,
             "hidden states of layers\n$-4$, $-9$, $-14$, mean over\nanswer tokens: $\\mathbf{h}\\in\\mathbb{R}^{3d}$",
             H_FC, H_EC, fs=8.2)
    hb = box(ax, 0.298, yb - 0.052, 0.192, 0.104,
             "predictive entropy on answer\nkeywords (not in $q$), mean:\n$\\bar H\\in\\mathbb{R}$",
             E_FC, E_EC, fs=8.2)
    arrow(ax, (f[0] + f[2], yc + 0.03), (hh[0], yt), rad=0.0)
    arrow(ax, (f[0] + f[2], yc - 0.03), (hb[0], yb), rad=0.0)

    # evidence heads
    eh = box(ax, 0.522, yt - 0.045, 0.170, 0.09,
             "EDL probe\n$3d\\rightarrow128\\rightarrow32\\rightarrow2$\n$\\mathbf{e}_h=\\mathrm{softplus}(\\cdot)$",
             H_FC, H_EC)
    eH = box(ax, 0.522, yb - 0.045, 0.170, 0.09,
             "linear evidence\n$\\mathbf{e}_H=\\mathrm{softplus}(\\mathbf{w}\\bar H+\\mathbf{b})$\n(4 parameters)",
             E_FC, E_EC)
    arrow(ax, (hh[0] + hh[2], yt), (eh[0], yt), color=H_EC)
    arrow(ax, (hb[0] + hb[2], yb), (eH[0], yb), color=E_EC)
    ax.text(eh[0] + eh[2] / 2, eh[1] + eh[3] + 0.012, "stage 1: trained, then frozen", ha="center", fontsize=7.4,
            color=MUTED, style="italic")
    ax.text(eH[0] + eH[2] / 2, eH[1] - 0.022, "stage 2: trained with $\\mathbf{e}_h$ frozen", ha="center",
            fontsize=7.4, color=MUTED, style="italic")

    # sum -> Dirichlet -> score
    sx, sr = 0.733, 0.022
    ax.add_patch(plt.Circle((sx, yc), sr, fc=OUT_FC, ec=OUT_EC, lw=1.1))
    ax.text(sx, yc - 0.002, "+", ha="center", va="center", fontsize=13, color=INK)
    arrow(ax, (eh[0] + eh[2], yt), (sx - sr * 0.55, yc + sr * 0.85), color=H_EC)
    arrow(ax, (eH[0] + eH[2], yb), (sx - sr * 0.55, yc - sr * 0.85), color=E_EC)
    d = box(ax, 0.772, yc - 0.06, 0.212, 0.12,
            "$\\boldsymbol{\\alpha}=\\mathbf{e}_h+\\mathbf{e}_H+\\mathbf{1}$\n$\\mathrm{Dir}(\\mathbf{p}\\mid\\boldsymbol{\\alpha})$\n"
            "$s_{\\mathrm{FRH}}=\\alpha_{\\mathrm{HALL}}/(\\alpha_{\\mathrm{HALL}}+\\alpha_{\\mathrm{KNOW}})$",
            OUT_FC, OUT_EC, fs=7.8)
    arrow(ax, (sx + sr, yc), (d[0], yc), color=INK)

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "heed_pipeline"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=220)
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
