import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
import string
from sklearn.metrics import precision_recall_curve, average_precision_score

from bindcore.data.io import parse_truth_file, parse_prediction_csv

# ---------------------------------------------------------------------------
# Configuration & Global Styling (Optimized for Graphical Abstract)
# ---------------------------------------------------------------------------

def set_abstract_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16,                  # Increased base font significantly
        "axes.titlesize": 20,             # Larger subplot titles
        "axes.labelsize": 18,             # Larger axis labels
        "xtick.labelsize": 14,            # Larger tick labels
        "ytick.labelsize": 14,
        "legend.fontsize": 16,            # Highly readable legend
        "axes.linewidth": 1.5,            # Thicker frame lines
        "lines.linewidth": 3.0,           # Thick, striking curves for abstract
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300, 
    })

# Fixed colors to ensure consistency across all subplots
MODEL_COLORS = {
    "CLIP": "#e63946",                 # Red
    "MoRFchibi": "#f4a261",            # Orange
    "BindCORE": "#2176ae",             # Unified Blue for best BindCORE
    "Random/Chance": "#888888"         # Gray
}

# Simplified abbreviations
MODEL_ABBREVIATIONS = {
    "CLIP": "CLIP",
    "MoRFchibi": "MoRFchibi 2.0",
    "BindCORE": "BindCORE",
    "Random/Chance": "Chance"
}

# ---------------------------------------------------------------------------
# Data Configuration (Filtered for Max 1024 Only)
# ---------------------------------------------------------------------------
EVAL_CONFIGS = [
    {
        "dataset_name": "LIP",
        "truth": "data/LIP_dataset/TE440_less_than_1024.txt",
        "preds": {
            "CLIP": "data/predictions/CLIP_TE440.csv",
            "BindCORE IDPFold2": "data/predictions/BindCORE_LIP_IDPFold2_TE440_less_than_1024.csv",
            "BindCORE AF-CALVADOS": "data/predictions/BindCORE_LIP_AF_CALVADOS_TE440_less_than_1024.csv",
        }
    },
    {
        "dataset_name": "MoRF",
        "truth": "data/MoRF_dataset/test.txt",
        "preds": {
            "MoRFchibi": "data/predictions/MoRFchibi_test.csv",
            "BindCORE IDPFold2": "data/predictions/BindCORE_MoRF_IDPFold2_test.csv",
            "BindCORE AF-CALVADOS": "data/predictions/BindCORE_MoRF_AF_CALVADOS_test.csv",
        }
    }
]

# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes) -> None:
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.tick_params(direction="out", length=6, width=1.5, colors="black")
    ax.tick_params(which="minor", direction="out", length=3.0, width=1.0)

def add_metric_text_smart(ax, metrics_dict, pct_imp=0.0):
    sorted_metrics = sorted(metrics_dict.items(), key=lambda item: item[1], reverse=True)
    abbreviated_metrics = [(MODEL_ABBREVIATIONS.get(name, name), score) for name, score in sorted_metrics]
    
    positions = [
        {"x": 0.96, "y": 0.04, "dx": -0.02, "dy": 0.12, "ha": 'right', "va": 'bottom'},
        {"x": 0.04, "y": 0.96, "dx": 0.02, "dy": -0.12, "ha": 'left', "va": 'top'},
        {"x": 0.96, "y": 0.96, "dx": -0.02, "dy": -0.12, "ha": 'right', "va": 'top'},
        {"x": 0.04, "y": 0.04, "dx": 0.02, "dy": 0.12, "ha": 'left', "va": 'bottom'}
    ]
    
    best_position = positions[2] # Default top right for PR curves usually
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    mid_x, mid_y = sum(xlim)/2, sum(ylim)/2
    
    point_counts = {"br": 0, "tl": 0, "tr": 0, "bl": 0}
    try:
        line_data_points = []
        for line in ax.get_lines():
            if line.get_linestyle() != '--':
                line_data_points.extend(zip(line.get_xdata(), line.get_ydata()))
        
        for x, y in line_data_points:
            if x > mid_x and y < mid_y: point_counts["br"] += 1
            elif x < mid_x and y > mid_y: point_counts["tl"] += 1
            elif x > mid_x and y > mid_y: point_counts["tr"] += 1
            elif x < mid_x and y < mid_y: point_counts["bl"] += 1
        
        min_quad = min(point_counts, key=point_counts.get)
        if min_quad == "tl": best_position = positions[1]
        elif min_quad == "tr": best_position = positions[2]
        elif min_quad == "bl": best_position = positions[3]
        elif min_quad == "br": best_position = positions[0]
    except Exception:
        pass
    
    x_base, y_base = best_position["x"], best_position["y"]
    dy = best_position["dy"]
    text_y = y_base
    
    for name, score in reversed(abbreviated_metrics):
        base_key = "BindCORE" if "BindCORE" in name else ("MoRFchibi" if "MoRFchibi" in name else name)
        color = MODEL_COLORS.get(base_key, "#000000")
        
        # Format label text to display percentage improvement for BindCORE
        if base_key == "BindCORE" and pct_imp > 0:
            label_text = f"{name}: {score:.2f} (+{pct_imp:.1f}%)"
        else:
            label_text = f"{name}: {score:.2f}"
            
        ax.annotate(
            label_text,
            xy=(x_base, text_y),
            xycoords='axes fraction',
            fontsize=14,                 # Increased text size inside panels
            fontweight='bold',
            color=color,
            ha=best_position["ha"],
            va=best_position["va"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85)
        )
        text_y += dy

def interpolate_curve(x, y, grid):
    sort_idx = np.argsort(x)
    return np.interp(grid, x[sort_idx], y[sort_idx])

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    set_abstract_style()
    
    # 1 row x 2 columns layout, scaled up for abstract clarity
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    global_handles = {} 

    for i, config in enumerate(EVAL_CONFIGS):
        print(f"Processing {config['dataset_name']}...")
        
        records = parse_truth_file(config['truth'])
        for model_name, pred_path in config['preds'].items():
            parse_prediction_csv(pred_path, records, model_name)
            
        y_true_global = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
        y_mask = y_true_global != -1
        y_true_global = y_true_global[y_mask]
        pos_rate = float(y_true_global.mean())

        ax_pr = axes[i]
        pr_curves = {}

        # 1. Compute PR metrics only
        for model_name in config['preds'].keys():
            y_score = np.concatenate([r.scores[model_name] for r in records.values()])[y_mask]
            
            # PR
            precision, recall, _ = precision_recall_curve(y_true_global, y_score)
            ap = average_precision_score(y_true_global, y_score)
            precision = np.maximum.accumulate(precision) 
            pr_curves[model_name] = (recall, precision, ap)

        # 2. Dynamic Best BindCORE Extraction & Percentage improvement calculation
        sota_name = "CLIP" if "CLIP" in config['preds'] else "MoRFchibi"
        best_bc_pr_key  = max([k for k in pr_curves if k.startswith("BindCORE")], key=lambda k: pr_curves[k][2])

        # ------------------- PR Curve & Fill -------------------
        rec_sota, prec_sota, ap_sota = pr_curves[sota_name]
        rec_bc, prec_bc, ap_bc = pr_curves[best_bc_pr_key]
        
        # Relative improvement calculation
        pr_pct_improvement = ((ap_bc - ap_sota) / ap_sota) * 100

        line_sota_pr, = ax_pr.plot(rec_sota, prec_sota, color=MODEL_COLORS[sota_name], alpha=0.8, zorder=10)
        line_bc_pr,   = ax_pr.plot(rec_bc, prec_bc, color=MODEL_COLORS["BindCORE"], alpha=0.9, zorder=11)
        
        rec_grid = np.linspace(0, 1, 1000)
        prec_sota_interp = interpolate_curve(rec_sota, prec_sota, rec_grid)
        prec_bc_interp = interpolate_curve(rec_bc, prec_bc, rec_grid)
        
        ax_pr.fill_between(rec_grid, prec_sota_interp, prec_bc_interp, 
                           where=(prec_bc_interp > prec_sota_interp), 
                           facecolor=MODEL_COLORS["BindCORE"], alpha=0.25, interpolate=True, zorder=5)

        metrics_pr = {sota_name: ap_sota, "BindCORE": ap_bc}

        # ------------------- Reference Baselines & Labels -------------------
        ax_pr.axhline(pos_rate, color=MODEL_COLORS["Random/Chance"], lw=2.0, ls="--", zorder=1)
        
        if sota_name not in global_handles:
            global_handles[sota_name] = line_sota_pr
        if "BindCORE" not in global_handles:
            global_handles["BindCORE"] = line_bc_pr
        if "Chance" not in global_handles:
            global_handles["Chance"] = Line2D([], [], color=MODEL_COLORS["Random/Chance"], lw=2.0, ls="--")

        ax_pr.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="Recall", ylabel="Precision")
        ax_pr.set_title(f"{config['dataset_name']}", fontweight="bold", pad=15)
        _style_ax(ax_pr)
        add_metric_text_smart(ax_pr, metrics_pr, pct_imp=pr_pct_improvement)

    # -----------------------------------------------------------------------
    # Graphical Abstract Assembly
    # -----------------------------------------------------------------------

    # Balanced spacing to accommodate larger fonts and legends
    fig.subplots_adjust(bottom=0.25, wspace=0.3)
    
    labels = [MODEL_ABBREVIATIONS.get(k, k) for k in global_handles.keys()]
    handles = list(global_handles.values())
    
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
               frameon=False, bbox_to_anchor=(0.5, 0.05))

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "graphical_abstract_pr_max1024.pdf"
    
    plt.savefig(save_path, transparent=True, bbox_inches="tight")
    print(f"\nSuccess! Graphical Abstract PR plots saved to → {save_path}")

if __name__ == "__main__":
    main()