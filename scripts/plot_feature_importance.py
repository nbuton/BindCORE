#!/usr/bin/env python3
"""
scripts/plot_feature_importance.py
====================================
Publication-quality interpretability summary figure for BindCORE.

Reads DeepLIFT-SHAP feature_importance.csv files from:
    data/interpretability/BindCORE_{LIP|MoRF}_{AF_CALVADOS|IDPFold2|STARLING}/

Panels
------
a / b  Top-ranked features per group (global scalar, per-residue local,
       pairwise) for LIP (a) and MoRF (b). Horizontal bars = mean
       |DeepLIFT-SHAP| attribution ± s.d. across proteins, colour-coded
       by ensemble generator. Features ranked by mean importance across
       all three generators.

c      Spearman rank-correlation between ensemble generators for the
       complete feature importance ranking (all groups pooled), shown as
       annotated heatmaps for LIP (left) and MoRF (right).

Output
------
figures/figure_interpretability.pdf
figures/figure_interpretability.png  (300 dpi)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib / typography style  (Nature Methods: min 5 pt, sans-serif)
# ──────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":             7,
    "axes.labelsize":        7,
    "axes.titlesize":        7,
    "xtick.labelsize":       5.5,
    "ytick.labelsize":       6,
    "legend.fontsize":       6.5,
    "legend.title_fontsize": 7,
    "figure.dpi":            300,
    "pdf.fonttype":          42,   # embeds fonts as Type 1 (required by journals)
    "ps.fonttype":           42,
    "axes.linewidth":        0.7,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
    "xtick.major.size":      2.5,
    "ytick.major.size":      2.5,
    "xtick.minor.width":     0.5,
    "ytick.minor.width":     0.5,
})

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette  (NPG / Nature Publishing Group)
# ──────────────────────────────────────────────────────────────────────────────
ENSEMBLE_COLORS: dict[str, str] = {
    "AF-CALVADOS": "#3C5488",   # dark blue
    "IDPFold2":    "#00A087",   # teal-green
    "STARLING":    "#E64B35",   # red-orange
}

GROUP_BG: dict[str, str] = {
    "x_scalar":   "#F2F5FF",
    "x_local":    "#FFFCF0",
    "x_pairwise": "#F2FFF9",
}

GROUP_TITLE_COLOR: dict[str, str] = {
    "x_scalar":   "#3C5488",
    "x_local":    "#7A5900",
    "x_pairwise": "#006B52",
}

GROUP_LABELS: dict[str, str] = {
    "x_scalar":   "Global (scalar) features",
    "x_local":    "Per-residue (local) features",
    "x_pairwise": "Pairwise features",
}

# ──────────────────────────────────────────────────────────────────────────────
# Feature name formatting  (raw CSV name → display label)
# ──────────────────────────────────────────────────────────────────────────────
_FMT: dict[str, str] = {
    # --- global / scalar ---
    "avg_maximum_diameter":          "Max. end-to-end dist. (μ)",
    "std_maximum_diameter":          "Max. end-to-end dist. (σ)",
    "avg_squared_Ree":               "〈R²_ee〉",
    "std_squared_Ree":               "σ(R²_ee)",
    "scaling_exponent":              "Scaling exponent ν",
    "asphericity_mean":              "Asphericity (μ)",
    "asphericity_std":               "Asphericity (σ)",
    "prolateness_mean":              "Prolateness (μ)",
    "prolateness_std":               "Prolateness (σ)",
    "radius_of_gyration_mean":       "Radius of gyration (μ)",
    "radius_of_gyration_std":        "Radius of gyration (σ)",
    "normalized_acylindricity_mean": "Acylindricity (μ)",
    "normalized_acylindricity_std":  "Acylindricity (σ)",
    "rel_shape_anisotropy_mean":     "Shape anisotropy κ² (μ)",
    "rel_shape_anisotropy_std":      "Shape anisotropy κ² (σ)",
    "gyration_eigenvalues_l1_mean":  "Rg λ₁ (μ)",
    "gyration_eigenvalues_l1_std":   "Rg λ₁ (σ)",
    "gyration_eigenvalues_l2_mean":  "Rg λ₂ (μ)",
    "gyration_eigenvalues_l2_std":   "Rg λ₂ (σ)",
    "gyration_eigenvalues_l3_mean":  "Rg λ₃ (μ)",
    "gyration_eigenvalues_l3_std":   "Rg λ₃ (σ)",
    "gyration_l1_per_l2_mean":       "Rg λ₁/λ₂ (μ)",
    "gyration_l1_per_l2_std":        "Rg λ₁/λ₂ (σ)",
    "gyration_l1_per_l3_mean":       "Rg λ₁/λ₃ (μ)",
    "gyration_l1_per_l3_std":        "Rg λ₁/λ₃ (σ)",
    "gyration_l2_per_l3_mean":       "Rg λ₂/λ₃ (μ)",
    "gyration_l2_per_l3_std":        "Rg λ₂/λ₃ (σ)",
    # --- per-residue / local ---
    "phi_entropy":       "φ entropy",
    "phi_mean":          "φ (μ)",
    "phi_std":           "φ (σ)",
    "psi_entropy":       "ψ entropy",
    "psi_mean":          "ψ (μ)",
    "psi_std":           "ψ (σ)",
    "sasa_abs_mean":     "SASA abs. (μ)",
    "sasa_abs_std":      "SASA abs. (σ)",
    "sasa_rel_mean":     "SASA rel. (μ)",
    "sasa_rel_std":      "SASA rel. (σ)",
    "ss_propensity_C":   "SS propensity: coil",
    "ss_propensity_H":   "SS propensity: α-helix",
    "ss_propensity_E":   "SS propensity: β-strand",
    "ss_propensity_B":   "SS propensity: β-bridge",
    "ss_propensity_G":   "SS propensity: 3₁₀-helix",
    "ss_propensity_I":   "SS propensity: π-helix",
    "ss_propensity_S":   "SS propensity: bend",
    "ss_propensity_T":   "SS propensity: turn",
    # --- pairwise ---
    "dccm":                  "DCCM",
    "contact_map":           "Contact map",
    "distance_fluctuations": "Distance fluctuations",
}

def _fmt(name: str) -> str:
    return _FMT.get(name, name.replace("_", " "))


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR  = "data/interpretability"
TASKS     = ["LIP", "MoRF"]
ENSEMBLES = ["AF_CALVADOS", "IDPFold2", "STARLING"]
ENS_DISPLAY: dict[str, str] = {
    "AF_CALVADOS": "AF-CALVADOS",
    "IDPFold2":    "IDPFold2",
    "STARLING":    "STARLING",
}
GROUPS = ["x_scalar", "x_local", "x_pairwise"]
TOP_N: dict[str, int] = {"x_scalar": 10, "x_local": 8, "x_pairwise": 3}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & feature selection
# ──────────────────────────────────────────────────────────────────────────────
def load_all() -> dict[tuple[str, str], pd.DataFrame]:
    data: dict[tuple[str, str], pd.DataFrame] = {}
    for task in TASKS:
        for ens in ENSEMBLES:
            p = (
                Path(BASE_DIR)
                / f"BindCORE_{task}_{ens}"
                / "feature_importance.csv"
            )
            data[(task, ens)] = pd.read_csv(p)
    return data


def top_features(
    data: dict[tuple[str, str], pd.DataFrame],
    task: str,
) -> dict[str, list[str]]:
    """Top N features per group, ranked by mean importance across all ensembles."""
    all_df = pd.concat([data[(task, e)] for e in ENSEMBLES], ignore_index=True)
    result: dict[str, list[str]] = {}
    for grp in GROUPS:
        sub = all_df[all_df["feature_group"] == grp]
        mean_across = sub.groupby("feature_name")["mean_importance"].mean()
        n = min(TOP_N[grp], len(mean_across))
        result[grp] = mean_across.nlargest(n).index.tolist()
    return result


def spearman_matrix(
    data: dict[tuple[str, str], pd.DataFrame],
    task: str,
) -> tuple[np.ndarray, list[str]]:
    """
    3×3 Spearman rank-correlation matrix between ensemble generators
    for one task, pooling all feature groups.
    """
    vecs: dict[str, dict[str, float]] = {}
    for ens in ENSEMBLES:
        df = data[(task, ens)]
        vecs[ens] = dict(zip(df["feature_name"], df["mean_importance"]))

    common: list[str] = sorted(
        set.intersection(*[set(v.keys()) for v in vecs.values()])
    )

    mat = np.ones((3, 3))
    for i, e1 in enumerate(ENSEMBLES):
        for j, e2 in enumerate(ENSEMBLES):
            if i != j:
                v1 = [vecs[e1][f] for f in common]
                v2 = [vecs[e2][f] for f in common]
                res = stats.spearmanr(v1, v2)
                # scipy ≥ 1.9 returns SpearmanrResult; earlier returns tuple
                mat[i, j] = res.statistic if hasattr(res, "statistic") else res[0]

    return mat, [ENS_DISPLAY[e] for e in ENSEMBLES]


# ──────────────────────────────────────────────────────────────────────────────
# Panel drawing helpers
# ──────────────────────────────────────────────────────────────────────────────
def _draw_bar_panel(
    ax: mpl.axes.Axes,
    data: dict[tuple[str, str], pd.DataFrame],
    task: str,
    grp: str,
    feat_names: list[str],
    show_xlabel: bool = False,
) -> None:
    """Horizontal grouped bar chart for one (task, feature-group)."""
    n_feat = len(feat_names)
    n_ens  = len(ENSEMBLES)
    bw     = 0.22
    offsets = np.linspace(
        -(n_ens - 1) / 2 * bw,
         (n_ens - 1) / 2 * bw,
        n_ens,
    )
    yp = np.arange(n_feat)

    ax.set_facecolor(GROUP_BG[grp])

    for ei, ens in enumerate(ENSEMBLES):
        df        = data[(task, ens)]
        lbl       = ENS_DISPLAY[ens]
        means: list[float] = []
        stds:  list[float] = []
        for fn in feat_names:
            row = df[df["feature_name"] == fn]
            if len(row) == 0:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(float(row["mean_importance"].iloc[0]))
                stds.append(float(row["std_importance"].iloc[0]))

        ax.barh(
            yp + offsets[ei],
            means,
            bw,
            xerr=stds,
            color=ENSEMBLE_COLORS[lbl],
            alpha=0.88,
            label=lbl,
            linewidth=0,
            error_kw={
                "elinewidth": 0.45,
                "capsize":    1.2,
                "capthick":   0.45,
                "ecolor":     "dimgray",
                "alpha":      0.45,
            },
        )

    ax.set_yticks(yp)
    ax.set_yticklabels([_fmt(fn) for fn in feat_names], fontsize=6.0)
    ax.invert_yaxis()

    # Spine cleanup
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=5.5)

    # Group title
    ax.set_title(
        GROUP_LABELS[grp],
        fontsize=6.5,
        fontweight="bold",
        color=GROUP_TITLE_COLOR[grp],
        loc="left",
        pad=2,
    )

    # X axis: scientific notation, dotted grid
    ax.xaxis.set_major_formatter(
        ticker.ScalarFormatter(useMathText=True)
    )
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-4, 0))
    ax.xaxis.grid(True, linestyle=":", linewidth=0.35, color="gray", alpha=0.45)
    ax.set_axisbelow(True)

    if show_xlabel:
        ax.set_xlabel("Mean |SHAP attribution|", fontsize=6.5, labelpad=3)


def _draw_spearman(
    ax: mpl.axes.Axes,
    mat: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    """Annotated heatmap of Spearman rank-correlations."""
    im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="Blues", aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=0, pad=2)

    for i in range(3):
        for j in range(3):
            text_color = "white" if mat[i, j] > 0.85 else "#1a1a1a"
            fw = "bold" if i == j else "normal"
            ax.text(
                j, i,
                f"{mat[i, j]:.2f}",
                ha="center", va="center",
                fontsize=6.5, color=text_color, fontweight=fw,
            )

    ax.set_title(title, fontsize=7, fontweight="bold", pad=4, color="#333333")

    # Inset colourbar
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="7%", pad=0.05)
    cb  = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("ρ (Spearman)", fontsize=6, labelpad=2)


# ──────────────────────────────────────────────────────────────────────────────
# Main figure assembly
# ──────────────────────────────────────────────────────────────────────────────
def make_figure(
    data: dict[tuple[str, str], pd.DataFrame],
) -> mpl.figure.Figure:

    tops = {t: top_features(data, t) for t in TASKS}

    # Height ratios proportional to number of feature rows per group
    hr = [TOP_N["x_scalar"], TOP_N["x_local"], TOP_N["x_pairwise"]]

    # Figure: 183 mm wide (Nature Methods double column)
    fig = plt.figure(figsize=(183 / 25.4, 215 / 25.4))

    # ── Outer grid: [bar panels row] / [Spearman row] ────────────────────────
    outer = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[sum(hr) * 1.30, 3.6],
        hspace=0.60,
        left=0.03, right=0.97, top=0.88, bottom=0.05,
    )

    # ── Top row: LIP (a) and MoRF (b) bar panels ─────────────────────────────
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[0],
        wspace=0.50,
    )

    TASK_TITLE = {
        "LIP":  "LIP — linear interacting peptides",
        "MoRF": "MoRF — molecular recognition features",
    }
    PANEL_LETTER = {"LIP": "a", "MoRF": "b"}

    for ci, task in enumerate(TASKS):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1,
            subplot_spec=top_gs[ci],
            hspace=0.52,
            height_ratios=hr,
        )
        for ri, grp in enumerate(GROUPS):
            ax = fig.add_subplot(inner[ri])
            _draw_bar_panel(
                ax, data, task, grp,
                feat_names=tops[task][grp],
                show_xlabel=(ri == 2),
            )

        # Panel letter + title in figure coordinates
        bb = top_gs[ci].get_position(fig)
        fig.text(
            bb.x0 - 0.016, 0.93,
            PANEL_LETTER[task],
            fontsize=11, fontweight="bold",
            va="top", transform=fig.transFigure,
        )
        fig.text(
            bb.x0 + 0.012, 0.93,
            TASK_TITLE[task],
            fontsize=8, fontweight="bold",
            va="top", color="#222222",
            transform=fig.transFigure,
        )

    # ── Bottom row: Spearman rank-correlation heatmaps (c) ───────────────────
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 5,
        subplot_spec=outer[1],
        width_ratios=[0.05, 1, 0.45, 1, 0.05],
        wspace=0.0,
    )

    bb_bot = outer[1].get_position(fig)
    fig.text(
        bb_bot.x0 - 0.012,
        bb_bot.y1 + 0.04,
        "c",
        fontsize=11, fontweight="bold",
        va="bottom", transform=fig.transFigure,
    )
    fig.text(
        bb_bot.x0 + 0.018,
        bb_bot.y1 + 0.04,
        "Feature importance ranking consistency across ensemble generators",
        fontsize=8, fontweight="bold",
        va="bottom", color="#222222",
        transform=fig.transFigure,
    )

    for ci, task in enumerate(TASKS):
        ax_s = fig.add_subplot(bot_gs[ci * 2 + 1])
        mat, labels = spearman_matrix(data, task)
        _draw_spearman(ax_s, mat, labels, task)

    # ── Shared legend for ensemble colours ───────────────────────────────────
    handles = [
        Patch(facecolor=ENSEMBLE_COLORS[ENS_DISPLAY[e]], label=ENS_DISPLAY[e])
        for e in ENSEMBLES
    ]
    fig.legend(
        handles=handles,
        title="Ensemble generator",
        title_fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=6.5,
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        borderpad=0.65,
        handlelength=1.2,
        handleheight=0.85,
        ncol=3,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    print("Loading feature importance data …")
    data = load_all()

    print("Assembling figure …")
    fig = make_figure(data)

    for ext in ("pdf", "png"):
        out = out_dir / f"figure_interpretability.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved → {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()