import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
import string
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

from bindcore.data.io import parse_truth_file, parse_prediction_csv


# ---------------------------------------------------------------------------
# Configuration & Global Styling
# ---------------------------------------------------------------------------

def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300, 
    })

# Fixed colors to ensure consistency across all subplots
MODEL_COLORS = {
    "CLIP": "#e63946",                 # Red
    "MoRFchibi": "#f4a261",            # Orange
    "BindCORE StARLING": "#06d6a0",    # Green
    "BindCORE IDPFold2": "#2176ae",    # Blue
    "BindCORE AF-CALVADOS": "#9b5de5", # Purple
    "BindCORE pLM": "#f15bb5",         # Pink
    "Random/Chance": "#888888"         # Gray
}
# Model name abbreviations for cleaner display
MODEL_ABBREVIATIONS = {
    "CLIP": "CLIP",
    "MoRFchibi": "MoRFchibi",
    "BindCORE StARLING":r"$\mathbf{BC_{STA}}$",
    "BindCORE IDPFold2":  r"$\mathbf{BC_{IDPF2}}$", 
    "BindCORE AF-CALVADOS": r"$\mathbf{BC_{AFCAV}}$",
    "Random/Chance": "Chance"
}



# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------
# Define the 4 evaluation scenarios: [LIP Full, LIP <380, MoRF Full, MoRF <380]

EVAL_CONFIGS = [
    {
        "row_idx": 0,
        "col_offset": 0,
        "dataset_name": "LIP (Max 1024)",
        "truth": "data/LIP_dataset/TE440_less_than_1024.txt",
        "preds": {
            "CLIP": "data/predictions/CLIP_TE440.csv",
            "BindCORE IDPFold2": "data/predictions/BindCORE_LIP_IDPFold2_TE440_less_than_1024.csv",
            "BindCORE AF-CALVADOS": "data/predictions/BindCORE_LIP_AF_CALVADOS_TE440_less_than_1024.csv",
        }
    },
    {
        "row_idx": 0,
        "col_offset": 2,
        "dataset_name": "LIP (< 380)",
        "truth": "data/LIP_dataset/TE440_less_than_380.txt",
        "preds": {
            "CLIP": "data/predictions/CLIP_TE440.csv",
            "BindCORE StARLING": "data/predictions/BindCORE_LIP_STARLING_TE440_less_than_380.csv",
            "BindCORE IDPFold2": "data/predictions/BindCORE_LIP_IDPFold2_TE440_less_than_1024.csv",
            "BindCORE AF-CALVADOS": "data/predictions/BindCORE_LIP_AF_CALVADOS_TE440_less_than_1024.csv",
        }
    },
    {
        "row_idx": 1,
        "col_offset": 0,
        "dataset_name": "MoRF (Max 1024)",
        "truth": "data/MoRF_dataset/test.txt",
        "preds": {
            "MoRFchibi": "data/predictions/MoRFchibi_test.csv",
            "BindCORE IDPFold2": "data/predictions/BindCORE_MoRF_IDPFold2_test.csv",
            "BindCORE AF-CALVADOS": "data/predictions/BindCORE_MoRF_AF_CALVADOS_test.csv",
        }
    },
    {
        "row_idx": 1,
        "col_offset": 2,
        "dataset_name": "MoRF (< 380)",
        "truth": "data/MoRF_dataset/test_less_than_380.txt",
        "preds": {
            "MoRFchibi": "data/predictions/MoRFchibi_test.csv",
            "BindCORE StARLING": "data/predictions/BindCORE_MoRF_STARLING_test_less_than_380.csv",
            "BindCORE IDPFold2": "data/predictions/BindCORE_MoRF_IDPFold2_test.csv",
            "BindCORE AF-CALVADOS": "data/predictions/BindCORE_MoRF_AF_CALVADOS_test.csv",
        }
    }
]

# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes) -> None:
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.tick_params(direction="out", length=3, width=0.8, colors="black")
    ax.tick_params(which="minor", direction="out", length=1.5, width=0.6)

def add_metric_text_smart(ax, metrics_dict, is_pr=False):
    """
    Place abbreviated AUC/AP scores with smart positioning to minimize overlap.
    """
    sorted_metrics = sorted(metrics_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Abbreviate model names
    abbreviated_metrics = [(MODEL_ABBREVIATIONS.get(name, name), score) 
                          for name, score in sorted_metrics]
    
    # Try multiple positions to find the best fit
    positions = [
        {"x": 0.97, "y": 0.03, "dx": -0.02, "dy": 0.08, "ha": 'right', "va": 'bottom'},  # Bottom-right
        {"x": 0.03, "y": 0.97, "dx": 0.02, "dy": -0.08, "ha": 'left', "va": 'top'},     # Top-left
        {"x": 0.97, "y": 0.97, "dx": -0.02, "dy": -0.08, "ha": 'right', "va": 'top'},   # Top-right
        {"x": 0.03, "y": 0.03, "dx": 0.02, "dy": 0.08, "ha": 'left', "va": 'bottom'}    # Bottom-left
    ]
    
    best_position = positions[0]  # Default to bottom-right
    
    # Simple heuristic: prefer position with least data points nearby
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    mid_x = sum(xlim)/2
    mid_y = sum(ylim)/2
    
    # Count points in each quadrant
    point_counts = {"br": 0, "tl": 0, "tr": 0, "bl": 0}
    try:
        line_data_points = []
        for line in ax.get_lines():
            if line.get_linestyle() != '--':  # Skip baseline
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                line_data_points.extend(zip(xdata, ydata))
        
        for x, y in line_data_points:
            if x > mid_x and y < mid_y:
                point_counts["br"] += 1
            elif x < mid_x and y > mid_y:
                point_counts["tl"] += 1
            elif x > mid_x and y > mid_y:
                point_counts["tr"] += 1
            elif x < mid_x and y < mid_y:
                point_counts["bl"] += 1
        
        # Choose quadrant with minimum points
        min_quad = min(point_counts, key=point_counts.get)
        if min_quad == "tl":
            best_position = positions[1]
        elif min_quad == "tr":
            best_position = positions[2]
        elif min_quad == "bl":
            best_position = positions[3]
    except Exception:
        pass  # Fallback to default
    
    # Apply the chosen position
    x_base = best_position["x"]
    y_base = best_position["y"]
    dx = best_position["dx"]
    dy = best_position["dy"]
    ha = best_position["ha"]
    va = best_position["va"]
    
    text_y = y_base
    for name, score in reversed(abbreviated_metrics):
        color = MODEL_COLORS.get(next(k for k,v in MODEL_ABBREVIATIONS.items() if v==name), "#000000")
        metric_name = "AP" if is_pr else "AUC"
        txt = ax.annotate(
            f"{name}: {score:.2f}",
            xy=(x_base, text_y),
            xycoords='axes fraction',
            fontsize=10, 
            fontweight='bold',
            color=color,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7)
        )
        text_y += dy

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    set_nature_style()
    
    # 2 rows (LIP, MoRF) x 4 columns (ROC full, PR full, ROC <380, PR <380)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 7))
    
    global_handles = {} # To store unique legend handles

    for config in EVAL_CONFIGS:
        print(f"Processing {config['dataset_name']}...")
        
        # Load Data
        records = parse_truth_file(config['truth'])
        for model_name, pred_path in config['preds'].items():
            parse_prediction_csv(pred_path, records, model_name)
            
        # Aggregate true labels
        y_true_global = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
        y_mask = y_true_global != -1
        y_true_global = y_true_global[y_mask]
        pos_rate = float(y_true_global.mean())

        # Subplot axes
        ax_roc = axes[config['row_idx'], config['col_offset']]
        ax_pr  = axes[config['row_idx'], config['col_offset'] + 1]
        
        metrics_roc = {}
        metrics_pr = {}

        for model_name in config['preds'].keys():
            color = MODEL_COLORS.get(model_name, "#000000")
            y_score = np.concatenate([r.scores[model_name] for r in records.values()])[y_mask]
            
            # --- ROC ---
            fpr, tpr, _ = roc_curve(y_true_global, y_score)
            auc = roc_auc_score(y_true_global, y_score)
            metrics_roc[model_name] = auc
            line_roc, = ax_roc.plot(fpr, tpr, color=color, alpha=0.8, zorder=10)
            
            # --- PR ---
            precision, recall, _ = precision_recall_curve(y_true_global, y_score)
            ap = average_precision_score(y_true_global, y_score)
            metrics_pr[model_name] = ap
            precision = np.maximum.accumulate(precision) # interpolate
            line_pr, = ax_pr.plot(recall, precision, color=color, alpha=0.8, zorder=10)
            
            # Save for global legend
            if model_name not in global_handles:
                global_handles[model_name] = line_roc

        # Baselines
        ax_roc.plot([0, 1], [0, 1], color=MODEL_COLORS["Random/Chance"], lw=1.0, ls="--", zorder=1)
        ax_pr.axhline(pos_rate, color=MODEL_COLORS["Random/Chance"], lw=1.0, ls="--", zorder=1)
        
        if "Chance" not in global_handles:
            global_handles["Chance"] = Line2D([], [], color=MODEL_COLORS["Random/Chance"], lw=1.0, ls="--")

        # Styling ROC
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{config['dataset_name']} - ROC", fontweight="bold")
        _style_ax(ax_roc)
        add_metric_text_smart(ax_roc, metrics_roc, is_pr=False)
        
        # Styling PR
        ax_pr.set_xlim(-0.02, 1.02)
        ax_pr.set_ylim(-0.02, 1.02)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{config['dataset_name']} - PR", fontweight="bold")
        _style_ax(ax_pr)
        add_metric_text_smart(ax_pr, metrics_pr, is_pr=True)

    # -----------------------------------------------------------------------
    # Final Figure Assembly
    # -----------------------------------------------------------------------
    
    # Add a, b, c, d... panel labels
    for i, ax in enumerate(axes.flatten()):
        ax.text(-0.15, 1.05, string.ascii_lowercase[i], transform=ax.transAxes, 
                size=11, weight='bold')

    # Add the unified legend at the bottom
    fig.subplots_adjust(bottom=0.15, wspace=0.35, hspace=0.35)
    
    # Create the legend using the collected unique handles
    labels = list(global_handles.keys())
    handles = list(global_handles.values())
    
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
               frameon=False, bbox_to_anchor=(0.5, 0.02))

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "fig_residue_level_benchmark_performance_roc_pr_curves.pdf"
    
    plt.savefig(save_path, transparent=True, bbox_inches="tight")
    print(f"\nSuccess! Saved publication figure to → {save_path}")

if __name__ == "__main__":
    main()