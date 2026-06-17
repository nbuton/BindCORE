import os
import re
import warnings
import numpy as np
import pandas as pd
import string
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------
# COLOR PALETTE (Unified with Previous Script)
# ---------------------------------------------------------
MODEL_COLORS = {
    "BindCORE": "#2176ae",             # Unified Blue
    "CLIP": "#e63946",                 # Red
    "MoRFchibi": "#f4a261",            # Orange
    "Random/Chance": "#888888"         # Gray
}

# ---------------------------------------------------------
# GRAPHICS AND STYLE CONFIGURATION (Enlarged Abstract Style)
# ---------------------------------------------------------
def set_abstract_style():
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 16,                  # Significantly increased base font
        "axes.titlesize": 20,             # Larger subplot titles
        "axes.labelsize": 18,             # Larger axis labels
        "xtick.labelsize": 14,            # Larger tick labels
        "ytick.labelsize": 14,
        "legend.fontsize": 16,            # Highly readable legend
        "axes.linewidth": 1.5,            # Thicker frame lines
        "lines.linewidth": 3.0,           # Thick, striking curves
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300, 
    })

# ---------------------------------------------------------
# DATA LOADING UTILITIES
# ---------------------------------------------------------
def parse_annotation_file(path: str) -> dict:
    data = {}
    current_id, current_seq, current_lab = None, "", ""

    def _flush(pid, seq, lab):
        if pid is None or not lab:
            return
        raw = list(lab)
        mask = np.array([c != "-" for c in raw])
        labels = np.array([int(c) if c != "-" else 0 for c in raw])
        data[pid] = {"sequence": seq, "labels": labels, "mask": mask}

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush(current_id, current_seq, current_lab)
                current_id = line[1:]
                current_seq, current_lab = "", ""
            elif bool(re.fullmatch(r"[01\-]+", line)) and len(line) > 0:
                current_lab += line
            else:
                current_seq += line
    _flush(current_id, current_seq, current_lab)
    return data

def parse_prediction_file(path: str, max_len: int = None) -> dict:
    df = pd.read_csv(path)
    data = {}
    for _, row in df.iterrows():
        pid = str(row["protein_id"])
        scores = np.array([float(x) for x in str(row["predictions"]).split(",")])
        if max_len is not None and len(scores) > max_len:
            continue
        data[pid] = {"scores": scores}
    return data

def _make_rank_lut(arr):
    order = np.argsort(arr)
    sorted_vals = arr[order]
    pcts = np.arange(len(sorted_vals)) / (len(sorted_vals) - 1)
    return sorted_vals, pcts

def _global_rank_norm(scores, sorted_vals, pcts):
    return np.interp(scores, sorted_vals, pcts)

def prepare_dataset(ann, preds):
    models = list(preds.keys())
    rank_luts = {}
    for model in models:
        all_raw = []
        for pid, data in preds[model].items():
            if pid in ann:
                all_raw.extend(data["scores"])
        if len(all_raw) > 0:
            rank_luts[model] = _make_rank_lut(np.array(all_raw))

    out_labels = {}
    out_scores = {model: {} for model in models}
    lengths = {}

    for model in models:
        for pid, data in preds[model].items():
            if pid not in ann:
                continue

            mask = ann[pid]["mask"]
            lab = ann[pid]["labels"]
            sc_raw = data["scores"]

            if len(sc_raw) == 0:
                continue

            sorted_vals, pcts = rank_luts[model]
            sc_norm = _global_rank_norm(sc_raw, sorted_vals, pcts)

            n = min(len(lab), len(sc_norm))
            lab_n = lab[:n]
            mask_n = mask[:n]
            sc_norm_n = sc_norm[:n]

            lab_masked = lab_n[mask_n]
            sc_norm_masked = sc_norm_n[mask_n]

            if len(lab_masked) == 0:
                continue

            out_labels[pid] = lab_masked
            out_scores[model][pid] = sc_norm_masked
            lengths[pid] = len(lab_masked)

    return out_labels, out_scores, lengths

def _style_ax(ax: plt.Axes) -> None:
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(direction="out", length=6, width=1.5, colors="black")
    ax.tick_params(which="minor", direction="out", length=3.0, width=1.0)

def calculate_perf_curve(labels, scores, lengths, model):
    model_pids = list(scores[model].keys())
    model_lengths = [lengths[pid] for pid in model_pids]
    thresholds = np.unique(np.percentile(model_lengths, np.linspace(10, 100, 25)).astype(int))
    
    perf = []
    for t in thresholds:
        pids = [pid for pid in model_pids if lengths[pid] <= t]
        if len(pids) < 3:
            perf.append(np.nan)
            continue
        pool_labels = np.concatenate([labels[pid] for pid in pids])
        pool_scores = np.concatenate([scores[model][pid] for pid in pids])
        perf.append(average_precision_score(pool_labels, pool_scores) if len(np.unique(pool_labels)) > 1 else np.nan)
        
    return thresholds, np.array(perf)


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
print("Loading data...")
lip_ann = parse_annotation_file("data/LIP_dataset/TE440.txt")
morf_ann = parse_annotation_file("data/MoRF_dataset/test.txt")

lip_preds = {
    "IDPFold2": parse_prediction_file("data/predictions/BindCORE_LIP_IDPFold2_TE440_less_than_1024.csv"),
    "CLIP": parse_prediction_file("data/predictions/CLIP_TE440.csv", max_len=1024),
}

morf_preds = {
    "AF-CALVADOS": parse_prediction_file("data/predictions/BindCORE_MoRF_AF_CALVADOS_test.csv"),
    "MoRFchibi": parse_prediction_file("data/predictions/MoRFchibi_test.csv"),
}

lip_labels, lip_scores, lip_lengths = prepare_dataset(lip_ann, lip_preds)
morf_labels, morf_scores, morf_lengths = prepare_dataset(morf_ann, morf_preds)


print("Generating Unified Graphical Abstract...")
set_abstract_style()

# 1 row x 2 columns layout exactly matching the architecture of the previous script
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# --- Panel 1: LIP (BindCORE vs CLIP) ---
ax1 = axes[0]
t_bc_lip, p_bc_lip = calculate_perf_curve(lip_labels, lip_scores, lip_lengths, "IDPFold2")
t_sota_lip, p_sota_lip = calculate_perf_curve(lip_labels, lip_scores, lip_lengths, "CLIP")

x_grid_lip = np.linspace(min(t_bc_lip.min(), t_sota_lip.min()), max(t_bc_lip.max(), t_sota_lip.max()), 300)
p_bc_interp = np.interp(x_grid_lip, t_bc_lip, p_bc_lip)
p_sota_interp = np.interp(x_grid_lip, t_sota_lip, p_sota_lip)

p_bc_smooth = gaussian_filter1d(p_bc_interp, sigma=3)
p_sota_smooth = gaussian_filter1d(p_sota_interp, sigma=3)

# Shading with unified alpha profile
ax1.fill_between(x_grid_lip, p_sota_smooth, p_bc_smooth, where=(p_bc_smooth >= p_sota_smooth), 
                 color=MODEL_COLORS["BindCORE"], alpha=0.25, zorder=1)

ax1.plot(t_bc_lip, p_bc_lip, marker="o", lw=3.0, markersize=6, color=MODEL_COLORS["BindCORE"], zorder=3)
ax1.plot(t_sota_lip, p_sota_lip, marker="s", lw=2.5, markersize=6, color=MODEL_COLORS["CLIP"], zorder=2)

ax1.set_title("LIP", fontweight="bold", pad=15)
ax1.set_xlabel("Max Protein Length (k)")
ax1.set_ylabel("AUPRC (up to length k)")
_style_ax(ax1)


# --- Panel 2: MoRF (BindCORE vs MoRFchibi) ---
ax2 = axes[1]
t_bc_morf, p_bc_morf = calculate_perf_curve(morf_labels, morf_scores, morf_lengths, "AF-CALVADOS")
t_sota_morf, p_sota_morf = calculate_perf_curve(morf_labels, morf_scores, morf_lengths, "MoRFchibi")

x_grid_morf = np.linspace(min(t_bc_morf.min(), t_sota_morf.min()), max(t_bc_morf.max(), t_sota_morf.max()), 300)
p_bc_morf_interp = np.interp(x_grid_morf, t_bc_morf, p_bc_morf)
p_sota_morf_interp = np.interp(x_grid_morf, t_sota_morf, p_sota_morf)

p_bc_morf_smooth = gaussian_filter1d(p_bc_morf_interp, sigma=3)
p_sota_morf_smooth = gaussian_filter1d(p_sota_morf_interp, sigma=3)

ax2.fill_between(x_grid_morf, p_sota_morf_smooth, p_bc_morf_smooth, where=(p_bc_morf_smooth >= p_sota_morf_smooth), 
                 color=MODEL_COLORS["BindCORE"], alpha=0.25, zorder=1)

ax2.plot(t_bc_morf, p_bc_morf, marker="o", lw=3.0, markersize=6, color=MODEL_COLORS["BindCORE"], zorder=3)
ax2.plot(t_sota_morf, p_sota_morf, marker="^", lw=2.5, markersize=6, color=MODEL_COLORS["MoRFchibi"], zorder=2)

ax2.set_title("MoRF", fontweight="bold", pad=15)
ax2.set_xlabel("Max Protein Length (k)")
ax2.set_ylabel("AUPRC (up to length k)")
_style_ax(ax2)


# ---------------------------------------------------------
# ASSEMBLY & LABELS
# ---------------------------------------------------------

fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.25, wspace=0.3)

legend_elements = [
    Line2D([0], [0], color=MODEL_COLORS["BindCORE"], lw=3.0, marker='o', markersize=7, label="BindCORE"),
    Line2D([0], [0], color=MODEL_COLORS["CLIP"], lw=2.5, marker='s', markersize=7, label="CLIP (LIP SOTA)"),
    Line2D([0], [0], color=MODEL_COLORS["MoRFchibi"], lw=2.5, marker='^', markersize=7, label="MoRFchibi 2.0 (MoRF SOTA)"),
    Patch(facecolor=MODEL_COLORS["BindCORE"], alpha=0.25, edgecolor='none', label="Performance Gain")
]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.05),
    ncol=4, 
    frameon=False
)

out_path = "figures/graphical_abstract.pdf"
plt.savefig(out_path, transparent=True, bbox_inches="tight")
print(f"\nSuccess! Standardized graphical abstract plot saved to → {out_path}")