import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# INSET POSITION CONTROLS  ← adjust these two values to move the sub-figure
# Values are in figure-coordinate fractions (0 = left/bottom, 1 = right/top)
INSET_SHIFT_RIGHT = 0.05  # increase → move inset further right
INSET_SHIFT_UP = -0.06  # increase → move inset further up
# ══════════════════════════════════════════════════════════════════════════════


# ── Runtime equations ──────────────────────────────────────────────────────────
def af_calvados(x):
    return (
        1.99e-6 * x**2.75
        + 3.11e-6 * x**2.14
        + 3.47e-2 * x**1.06
        + 29.7 * x**0.02
        + 502.19
    )


def idpfold2(x):
    return 0.187 * x**1.595 + 2.916


def starling(x):
    return 1.42e-4 * x**1.73 + 23.82


STARLING_MAX_L = 380

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "af": "#E64B35",
    "idp": "#4DBBD5",
    "star": "#00A087",
    "bg": "#FFFFFF",
    "grid": "#E0E0E0",
    "zoom_bg": "#F0F4F8",
    "text": "#1A1A2E",
    "spine": "#888888",
}

# ── Global style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Palatino Linotype",
            "Palatino",
            "Book Antiqua",
            "DejaVu Serif",
            "Georgia",
        ],
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "legend.handlelength": 2.8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ── Data ──────────────────────────────────────────────────────────────────────
L_full = np.linspace(1, 1024, 2000)
L_zoom = np.linspace(1, 380, 800)
L_star_solid = np.linspace(1, STARLING_MAX_L, 600)
L_star_dot = np.linspace(STARLING_MAX_L, 1024, 600)


def plot_on(ax, lw=2.2, zoom=False):
    ax.plot(
        L_zoom if zoom else L_full,
        af_calvados(L_zoom if zoom else L_full) / 60,
        color=C["af"],
        lw=lw,
        ls="solid",
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        L_zoom if zoom else L_full,
        idpfold2(L_zoom if zoom else L_full) / 60,
        color=C["idp"],
        lw=lw,
        ls="solid",
        solid_capstyle="round",
        zorder=3,
    )
    if zoom:
        ax.plot(
            L_zoom,
            starling(L_zoom) / 60,
            color=C["star"],
            lw=lw,
            ls="solid",
            solid_capstyle="round",
            zorder=3,
        )
    else:
        ax.plot(
            L_star_solid,
            starling(L_star_solid) / 60,
            color=C["star"],
            lw=lw,
            ls="solid",
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            L_star_dot,
            starling(L_star_dot) / 60,
            color=C["star"],
            lw=lw,
            ls=(0, (3, 3)),
            dash_capstyle="round",
            zorder=3,
        )


# ── Figure & main axes ────────────────────────────────────────────────────────
# Main axes in figure coordinates: [left, bottom, width, height]
AX_L, AX_B, AX_W, AX_H = 0.09, 0.12, 0.86, 0.78
fig = plt.figure(figsize=(9, 5.5), facecolor=C["bg"])
ax = fig.add_axes([AX_L, AX_B, AX_W, AX_H], facecolor=C["bg"])

plot_on(ax)

ax.set_xlabel("Sequence length $L$ (residues)", labelpad=6)
ax.set_ylabel("Processing time (min)", labelpad=6)
ax.set_xlim(0, 1024)
ax.set_ylim(bottom=0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.grid(which="major", color=C["grid"], lw=0.6, zorder=0)
ax.grid(which="minor", color=C["grid"], lw=0.3, zorder=0, alpha=0.6)
for sp in ax.spines.values():
    sp.set_edgecolor(C["spine"])

# L_max marker
ax.axvline(STARLING_MAX_L, color=C["star"], lw=0.9, ls="--", alpha=0.45, zorder=2)
ax.text(
    STARLING_MAX_L + 10,
    ax.get_ylim()[1] * 0.97,
    r"$L_{\max}^{\mathrm{STARLING}}{=}380$",
    fontsize=7.5,
    color=C["star"],
    va="top",
    alpha=0.75,
)

ax.annotate(
    "Quadro RTX 6000 · 24 GB VRAM",
    xy=(0.99, 0.97),
    xycoords="axes fraction",
    fontsize=7.5,
    color="#555555",
    ha="right",
    va="top",
    style="italic",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", lw=0.7, alpha=0.9),
)

# ── Inset axes — placed with fig.add_axes for full control ────────────────────
# Base position (in figure coords): top-left corner of the main axes area,
# then shifted right/up by the two control variables above.
INS_W = AX_W * 0.44  # inset width  (fraction of figure)
INS_H = AX_H * 0.44  # inset height (fraction of figure)
INS_L = AX_L + INSET_SHIFT_RIGHT  # left edge
INS_B = AX_B + AX_H - INS_H + INSET_SHIFT_UP  # bottom edge (top-aligned with main)

axins = fig.add_axes([INS_L, INS_B, INS_W, INS_H], facecolor=C["zoom_bg"])

plot_on(axins, lw=1.9, zoom=True)

axins.set_xlim(0, 380)
axins.set_ylim(bottom=0)
axins.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
axins.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
axins.grid(which="major", color=C["grid"], lw=0.5, zorder=0)
axins.grid(which="minor", color=C["grid"], lw=0.25, zorder=0, alpha=0.6)
for sp in axins.spines.values():
    sp.set_edgecolor("#AAAAAA")
    sp.set_linewidth(0.8)
axins.spines["top"].set_visible(True)
axins.spines["right"].set_visible(True)
axins.tick_params(labelsize=7.5)
axins.set_xlabel("$L$ (residues)", fontsize=8.5, labelpad=3)
axins.set_ylabel("Time (min)", fontsize=8.5, labelpad=3)
axins.set_title(
    r"Zoom: $L \leq 380$", fontsize=8, pad=4, color=C["text"], style="italic"
)


# Connector lines from main axes to inset (manual, in figure coords)
# We draw lines from the zoom region corners on ax to the inset corners.
def ax_to_fig(ax_, x_data, y_data):
    """Convert data coords → figure coords."""
    disp = ax_.transData.transform((x_data, y_data))
    return fig.transFigure.inverted().transform(disp)


# Zoom region in main axes: x=[0,380], y=[0, top of inset y range]
axins_ylim = axins.get_ylim()
corners_main = [
    ax_to_fig(ax, 0, 0),
    ax_to_fig(ax, 380, 0),
]
corners_ins = [
    (INS_L, INS_B),  # bottom-left of inset
    (INS_L + INS_W, INS_B),  # bottom-right of inset
]
for (xm, ym), (xi, yi) in zip(corners_main, corners_ins):
    fig.add_artist(
        matplotlib.lines.Line2D(
            [xm, xi],
            [ym, yi],
            transform=fig.transFigure,
            color="#999999",
            lw=0.8,
            ls="--",
            zorder=10,
        )
    )

# ── Legend — anchored just to the right of the inset ─────────────────────────
# legend bbox_to_anchor in axes fraction: x = right edge of inset in ax coords
ins_right_fig = INS_L + INS_W
ins_top_fig = INS_B + INS_H
# convert figure coords → ax data-fraction
ins_right_ax = (ins_right_fig - AX_L) / AX_W
ins_top_ax = (ins_top_fig - AX_B) / AX_H

legend_elements = [
    Line2D(
        [0], [0], color=C["af"], lw=2.2, ls="solid", label="AF-CALVADOS  (1000 confs.)"
    ),
    Line2D(
        [0], [0], color=C["idp"], lw=2.2, ls="solid", label="IDPFold2  (100 confs.)"
    ),
    Line2D(
        [0],
        [0],
        color=C["star"],
        lw=2.2,
        ls="solid",
        label=r"STARLING  (400 confs., $L \leq 380$)",
    ),
    Line2D(
        [0],
        [0],
        color=C["star"],
        lw=2.2,
        ls=(0, (3, 3)),
        label=r"STARLING  (extrapolated, $L > 380$)",
    ),
]
leg = ax.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(ins_right_ax + 0.02, ins_top_ax),
    bbox_transform=ax.transAxes,
    frameon=True,
    fontsize=8.5,
    title="IDP ensemble models",
    title_fontsize=8.5,
    labelspacing=0.5,
    handlelength=2.8,
)
leg.get_title().set_style("italic")
leg.get_frame().set_linewidth(0.8)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Computational runtime vs. sequence length on a Quadro RTX 6000 GPU",
    fontsize=11,
    pad=10,
    color=C["text"],
)

# ── Caption ───────────────────────────────────────────────────────────────────
caption = (
    "Figure X | Predicted wall-clock runtime as a function of protein sequence "
    "length L for three IDP ensemble-generation models benchmarked on an NVIDIA "
    "Quadro RTX 6000 GPU (24 GB VRAM). Runtimes are expressed in minutes and were "
    "fitted from empirical observations on the target hardware. AF-CALVADOS was run "
    "at its default of 1,000 saved conformations; IDPFold2 at 100 conformations; and "
    "STARLING at 400 conformations. The solid STARLING curve (teal) covers its "
    "supported sequence-length range (L <= 380 residues); the dotted extension beyond "
    "L = 380 is an extrapolation outside the model's validated operating range and is "
    "shown for reference only. The inset provides a magnified view for short sequences "
    "(L <= 380), where IDPFold2 and STARLING runtimes diverge most clearly from "
    "AF-CALVADOS."
)
print("\n" + "=" * 80)
print("FIGURE CAPTION")
print("=" * 80)
print(caption)
print("=" * 80 + "\n")

# ── Export ────────────────────────────────────────────────────────────────────
out_png = "data/protein_timing_figure.png"
out_pdf = "data/protein_timing_figure.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=C["bg"])
fig.savefig(out_pdf, bbox_inches="tight", facecolor=C["bg"])
print("Saved:", out_png)
print("Saved:", out_pdf)
plt.close(fig)
