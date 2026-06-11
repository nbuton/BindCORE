"""
Compare Binding Predictions
===========================
Generates a multi-panel Nature methods-style figure comparing 
AF-CALVADOS, IDPFold2, and STARLING on LIP and MoRF datasets.

Subfigures:
  A) Performance (AUPRC) up to protein length k for LIP
  B) Performance (AUPRC) up to protein length k for MoRF
  C) Calibration (Reliability Diagram) for LIP
  D) Calibration (Reliability Diagram) for MoRF
  E) Agreement (Pearson correlation) between models for LIP
  F) Agreement (Pearson correlation) between models for MoRF
  G) Positive rate (binding residue fraction) vs protein length for LIP
  H) Positive rate (binding residue fraction) vs protein length for MoRF
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------
# DATA LOADING UTILITIES
# ---------------------------------------------------------

def parse_annotation_file(path: str) -> dict:
    data = {}
    current_id, current_seq, current_lab = None, "", ""

    def _flush(pid, seq, lab):
        if pid is None or not lab: return
        raw = list(lab)
        mask = np.array([c != "-" for c in raw])
        labels = np.array([int(c) if c != "-" else 0 for c in raw])
        data[pid] = {"sequence": seq, "labels": labels, "mask": mask}

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
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

# Rank normalisation for proper calibration / fair comparison
def _make_rank_lut(arr):
    order = np.argsort(arr)
    sorted_vals = arr[order]
    pcts = np.arange(len(sorted_vals)) / (len(sorted_vals) - 1)
    return sorted_vals, pcts

def _global_rank_norm(scores, sorted_vals, pcts):
    return np.interp(scores, sorted_vals, pcts)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print("Loading data...")

# Datasets
lip_ann = parse_annotation_file("data/LIP_dataset/TE440.txt")
morf_ann = parse_annotation_file("data/MoRF_dataset/test.txt")

# LIP Predictions
lip_preds = {
    "AF-CALVADOS": parse_prediction_file("data/predictions/BindCORE_LIP_AF_CALVADOS_TE440_less_than_1024.csv"),
    "IDPFold2": parse_prediction_file("data/predictions/BindCORE_LIP_IDPFold2_TE440_less_than_1024.csv"),
    "STARLING": parse_prediction_file("data/predictions/BindCORE_LIP_STARLING_TE440_less_than_380.csv"),
    "CLIP": parse_prediction_file("data/predictions/CLIP_TE440.csv", max_len=1024)
}

# MoRF Predictions
morf_preds = {
    "AF-CALVADOS": parse_prediction_file("data/predictions/BindCORE_MoRF_AF_CALVADOS_test.csv"),
    "IDPFold2": parse_prediction_file("data/predictions/BindCORE_MoRF_IDPFold2_test.csv"),
    "STARLING": parse_prediction_file("data/predictions/BindCORE_MoRF_STARLING_test_less_than_380.csv"),
    "MoRFchibi": parse_prediction_file("data/predictions/MoRFchibi_test.csv")
}

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
def prepare_dataset(ann, preds):
    """
    Returns:
    - all_labels: dict[pid, labels_array]
    - all_scores: dict[model, dict[pid, norm_scores_array]]
    - lengths: dict[pid, int]
    """
    models = list(preds.keys())
    
    # First, collect all scores for global rank normalization
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
            if pid not in ann: continue
            
            mask = ann[pid]["mask"]
            lab = ann[pid]["labels"]
            sc_raw = data["scores"]
            
            if len(sc_raw) == 0: continue
            
            # Normalise
            sorted_vals, pcts = rank_luts[model]
            sc_norm = _global_rank_norm(sc_raw, sorted_vals, pcts)
            
            # Match lengths
            n = min(len(lab), len(sc_norm))
            lab_n = lab[:n]
            mask_n = mask[:n]
            sc_norm_n = sc_norm[:n]
            
            # Apply mask
            lab_masked = lab_n[mask_n]
            sc_norm_masked = sc_norm_n[mask_n]
            
            if len(lab_masked) == 0: continue
            
            # We assume label is the same for a protein regardless of model, so just overwrite
            out_labels[pid] = lab_masked
            out_scores[model][pid] = sc_norm_masked
            lengths[pid] = len(lab_masked)
            
    return out_labels, out_scores, lengths

print("Preprocessing LIP data...")
lip_labels, lip_scores, lip_lengths = prepare_dataset(lip_ann, lip_preds)
print("Preprocessing MoRF data...")
morf_labels, morf_scores, morf_lengths = prepare_dataset(morf_ann, morf_preds)


# ---------------------------------------------------------
# FIGURE GENERATION
# ---------------------------------------------------------
print("Generating figures...")
fig = plt.figure(figsize=(24, 12))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.32)

COLORS = {
    "AF-CALVADOS": "#e41a1c", 
    "IDPFold2": "#377eb8", 
    "STARLING": "#4daf4a",
    "CLIP": "#984ea3",
    "MoRFchibi": "#ff7f00"
}

# --- Panel A & B: Performance up to length k ---
def plot_perf_length(ax, labels, scores, lengths, title):
    models = list(scores.keys())
    for model in models:
        model_pids = list(scores[model].keys())
        if not model_pids: continue
        
        # Get threshold lengths
        model_lengths = [lengths[pid] for pid in model_pids]
        thresholds = np.unique(np.percentile(model_lengths, np.linspace(10, 100, 20)).astype(int))
        
        perf = []
        for t in thresholds:
            pids = [pid for pid in model_pids if lengths[pid] <= t]
            if len(pids) < 3: 
                perf.append(np.nan)
                continue
            
            pool_labels = np.concatenate([labels[pid] for pid in pids])
            pool_scores = np.concatenate([scores[model][pid] for pid in pids])
            
            if len(np.unique(pool_labels)) > 1:
                perf.append(average_precision_score(pool_labels, pool_scores))
            else:
                perf.append(np.nan)
                
        ax.plot(thresholds, perf, marker='o', lw=2, color=COLORS[model], label=model)
        
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Max protein length (k)")
    ax.set_ylabel("AUPRC (up to length k)")
    
    # Unified legend for all models
    handles = [Line2D([0], [0], color=COLORS[m], lw=2, marker='o', label=m) for m in COLORS.keys()]
    ax.legend(handles=handles, loc="lower right")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, ls="--", alpha=0.5)

plot_perf_length(fig.add_subplot(gs[0, 0]), lip_labels, lip_scores, lip_lengths, "A. LIP Performance by Length")
plot_perf_length(fig.add_subplot(gs[1, 0]), morf_labels, morf_scores, morf_lengths, "B. MoRF Performance by Length")


# --- Panel C & D: Calibration ---
def plot_calibration(ax, labels, scores, title):
    models = list(scores.keys())
    bins = np.linspace(0, 1, 11)
    
    for model in models:
        model_pids = list(scores[model].keys())
        if not model_pids: continue
        
        all_l = np.concatenate([labels[pid] for pid in model_pids])
        all_s = np.concatenate([scores[model][pid] for pid in model_pids])
        
        bin_idx = np.digitize(all_s, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
        frac_pos = []
        for i in range(len(bins) - 1):
            mask = (bin_idx == i)
            if mask.sum() > 0:
                frac_pos.append(all_l[mask].mean())
            else:
                frac_pos.append(np.nan)
                
        mids = (bins[:-1] + bins[1:]) / 2
        ax.plot(mids, frac_pos, 's-', lw=2, color=COLORS[model], label=model)
        
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted score (Rank Normalized)")
    ax.set_ylabel("Fraction of positives")
    
    # Unified legend for all models + perfect calibration line
    handles = [Line2D([0], [0], color=COLORS[m], lw=2, marker='s', label=m) for m in COLORS.keys()]
    handles.append(Line2D([0], [0], color="k", linestyle="--", lw=1, label="Perfect calibration"))
    ax.legend(handles=handles, loc="upper left")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, ls="--", alpha=0.5)

plot_calibration(fig.add_subplot(gs[0, 1]), lip_labels, lip_scores, "C. LIP Calibration")
plot_calibration(fig.add_subplot(gs[1, 1]), morf_labels, morf_scores, "D. MoRF Calibration")


# --- Panel E & F: Agreement / Correlation Heatmap ---
def plot_agreement(ax, labels, scores, title):
    models = list(scores.keys())
    # find intersection of pids
    pids_sets = [set(scores[m].keys()) for m in models]
    common_pids = list(set.intersection(*pids_sets))
    
    if not common_pids:
        ax.text(0.5, 0.5, "No common proteins\nfor correlation", ha="center", va="center")
        ax.axis("off")
        return
        
    corr_matrix = np.zeros((len(models), len(models)))
    for i, m1 in enumerate(models):
        s1 = np.concatenate([scores[m1][pid] for pid in common_pids])
        for j, m2 in enumerate(models):
            s2 = np.concatenate([scores[m2][pid] for pid in common_pids])
            r, _ = pearsonr(s1, s2)
            corr_matrix[i, j] = r
            
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=models, yticklabels=models, vmin=0, vmax=1)
    ax.set_title(title, fontweight="bold")

plot_agreement(fig.add_subplot(gs[0, 2]), lip_labels, lip_scores, "E. LIP Model Agreement (Pearson r)")
plot_agreement(fig.add_subplot(gs[1, 2]), morf_labels, morf_scores, "F. MoRF Model Agreement (Pearson r)")


# --- Panel G & H: Positive rate vs protein length ---
def plot_positive_rate_by_length(ax, ann, title):
    """Scatter + running mean of binding residue fraction vs protein length."""
    pids = list(ann.keys())
    lengths_arr = []
    pos_rates = []
    for pid in pids:
        mask = ann[pid]["mask"]
        lab = ann[pid]["labels"]
        lab_masked = lab[mask]
        if len(lab_masked) == 0:
            continue
        lengths_arr.append(len(lab_masked))
        pos_rates.append(lab_masked.mean())

    lengths_arr = np.array(lengths_arr)
    pos_rates = np.array(pos_rates)

    # Scatter of individual proteins
    ax.scatter(lengths_arr, pos_rates, alpha=0.3, s=8, color="#888888", zorder=1)

    # Binned running mean ± std
    bin_edges = np.percentile(lengths_arr, np.linspace(0, 100, 21))
    bin_edges = np.unique(bin_edges.astype(int))
    bin_centers, bin_means, bin_stds = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        sel = (lengths_arr >= lo) & (lengths_arr < hi)
        if sel.sum() < 3:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_means.append(pos_rates[sel].mean())
        bin_stds.append(pos_rates[sel].std())

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax.plot(bin_centers, bin_means, color="#1a1a2e", lw=2.5, zorder=3, label="Binned mean")
    ax.fill_between(bin_centers,
                    np.clip(bin_means - bin_stds, 0, 1),
                    np.clip(bin_means + bin_stds, 0, 1),
                    alpha=0.25, color="#1a1a2e", zorder=2, label="±1 std")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Protein length (residues)")
    ax.set_ylabel("Positive rate (binding fraction)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, ls="--", alpha=0.5)

plot_positive_rate_by_length(fig.add_subplot(gs[0, 3]), lip_ann, "G. LIP Positive Rate by Length")
plot_positive_rate_by_length(fig.add_subplot(gs[1, 3]), morf_ann, "H. MoRF Positive Rate by Length")

plt.tight_layout()
out_path = "figures/nature_method_comparison_figure.pdf"
plt.savefig(out_path, dpi=300)
print(f"Saved figure to {out_path}")
