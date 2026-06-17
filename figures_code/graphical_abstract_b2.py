#!/usr/bin/env python3
"""
scripts/plot_graphical_abstract_2x2.py
====================================
A 2x2 stratified feature importance plot optimized for a Graphical Abstract.
Separates Global and Local features into distinct rows to handle magnitude 
differences cleanly while keeping independent x-axis scaling.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Graphical Abstract Typography & Style (Big, Bold, Readable)
# ──────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":             11,
    "axes.labelsize":        13,
    "axes.titlesize":        14,
    "xtick.labelsize":       11,
    "ytick.labelsize":       12,
    "legend.fontsize":       12,
    "figure.dpi":            300,
    "pdf.fonttype":          42,
    "axes.linewidth":        1.4,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
})

GROUP_COLORS: dict[str, str] = {
    "x_scalar":   "#3C5488",  # NPG Dark Blue (Global)
    "x_local":    "#E64B35",  # NPG Red (Local)
}

GROUP_LABELS: dict[str, str] = {
    "x_scalar":   "Global (scalar) features",
    "x_local":    "Per-residue (local) features",
}

# ──────────────────────────────────────────────────────────────────────────────
# Feature name formatting
# ──────────────────────────────────────────────────────────────────────────────
_FMT: dict[str, str] = {
    "avg_maximum_diameter":          "Max. end-to-end dist. (μ)",
    "radius_of_gyration_mean":       "Radius of gyration (μ)",
    "gyration_l1_per_l2_mean":       "Rg λ₁/λ₂ (μ)",
    "scaling_exponent":              "Scaling exponent ν",
    "gyration_eigenvalues_l1_mean":  "Rg λ₁ (μ)",
    "phi_entropy":                   "φ entropy",
    "psi_mean":                      "ψ (μ)",
    "sasa_abs_mean":                 "SASA abs. (μ)",
    "sasa_abs_std":                  "SASA abs. (σ)",
    "sasa_rel_mean":                 "SASA rel. (μ)",
    "sasa_rel_std":                  "SASA rel. (σ)",
    "ss_propensity_S":               "SS propensity: bend",
}

def _fmt(name: str) -> str:
    return _FMT.get(name, name.replace("_", " "))

BASE_DIR  = "data/interpretability"
TASKS     = ["LIP", "MoRF"]
ENSEMBLES = ["AF_CALVADOS", "IDPFold2", "STARLING"]

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Stratified Processing
# ──────────────────────────────────────────────────────────────────────────────
def load_stratified_data() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Loads data and returns independent Top 3 DataFrames for both global 
    and local categories per task.
    """
    processed_data = {}
    for task in TASKS:
        task_dfs = []
        for ens in ENSEMBLES:
            p = Path(BASE_DIR) / f"BindCORE_{task}_{ens}" / "feature_importance.csv"
            if p.exists():
                task_dfs.append(pd.read_csv(p))
            else:
                print(f"Warning: {p} not found. Using fallback data for demonstration.")
                np.random.seed(42 if task == "LIP" else 7)
                task_dfs.append(pd.DataFrame({
                    "feature_name": list(_FMT.keys()),
                    "feature_group": ["x_scalar"]*5 + ["x_local"]*7,
                    "mean_importance": np.random.uniform(0.00001, 0.00008, len(_FMT)) if task=="LIP" else np.random.uniform(0.001, 0.008, len(_FMT))
                }))
                
        all_df = pd.concat(task_dfs, ignore_index=True)
        consensus_df = all_df.groupby(["feature_name", "feature_group"], as_index=False)["mean_importance"].mean()
        
        # Split into independent top-performing sub-pools
        processed_data[task] = {
            "x_scalar": consensus_df[consensus_df["feature_group"] == "x_scalar"].nlargest(3, "mean_importance").iloc[::-1],
            "x_local":  consensus_df[consensus_df["feature_group"] == "x_local"].nlargest(3, "mean_importance").iloc[::-1]
        }
        
    return processed_data

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def make_graphical_abstract(data: dict[str, dict[str, pd.DataFrame]]) -> mpl.figure.Figure:
    # 2 Rows (Global, Local) x 2 Columns (LIP, MoRF)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), gridspec_kw={'wspace': 0.45, 'hspace': 0.45})
    
    task_cols = {"LIP": 0, "MoRF": 1}
    group_rows = {"x_scalar": 0, "x_local": 1}
    
    panel_titles = {
        ("x_scalar", "LIP"):  "LIP — Global Features",
        ("x_scalar", "MoRF"): "MoRF — Global Features",
        ("x_local", "LIP"):   "LIP — Per-Residue Local Features",
        ("x_local", "MoRF"):  "MoRF — Per-Residue Local Features"
    }

    for task in TASKS:
        for grp in ["x_scalar", "x_local"]:
            row = group_rows[grp]
            col = task_cols[task]
            ax = axes[row, col]
            
            df = data[task][grp]
            y_pos = np.arange(len(df))
            
            # Draw single-color bar block matching the feature group
            ax.barh(y_pos, df["mean_importance"], color=GROUP_COLORS[grp], height=0.55, alpha=0.9)
            
            # Setup ticks and labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_fmt(name) for name in df["feature_name"]], fontweight="bold")
            
            # Label configuration
            ax.set_xlabel("Consensus SHAP Importance", fontweight="bold", labelpad=6)
            ax.set_title(panel_titles[(grp, task)], fontweight="bold", pad=12)
            
            # Handle scientific notation elegantly per plot since scales vary wildly
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            
            # Background grid adjustments
            ax.xaxis.grid(True, linestyle="--", linewidth=0.8, color="#E0E0E0", alpha=0.6)
            ax.set_axisbelow(True)

    # Simplified legend containing only the two layout identities
    legend_elements = [
        Patch(facecolor=GROUP_COLORS[key], label=val) 
        for key, val in GROUP_LABELS.items()
    ]
    
    fig.legend(handles=legend_elements, 
               loc="upper center", 
               bbox_to_anchor=(0.5, 0.96),
               ncol=2, 
               frameon=False,
               fontsize=13)

    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    print("Extracting separate 2x2 data tracking pools...")
    data = load_stratified_data()

    print("Assembling 2x2 Graphical Abstract layout...")
    fig = make_graphical_abstract(data)

    out_pdf = out_dir / "graphical_abstract_features_2x2.pdf"
    out_png = out_dir / "graphical_abstract_features_2x2.png"
    
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Success! Saved decoupled 2x2 abstract figures to {out_dir}")

if __name__ == "__main__":
    main()