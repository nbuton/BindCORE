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
# COLOR PALETTE
# ---------------------------------------------------------
COLORS = {
    "AF-CALVADOS": "#D55E00",  
    "IDPFold2": "#0072B2",     
    "STARLING": "#009E73",     
    "CLIP": "#CC79A7",         
    "MoRFchibi": "#E69F00",    
}

# ---------------------------------------------------------
# GRAPHICS AND STYLE CONFIGURATION (Nature Methods Style)
# ---------------------------------------------------------
sns.set_style("ticks")

def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300, 
    })

set_nature_style()

LONG_LABELS = {
    "IDPFold2": "BindCORE IDPFold2",
    "STARLING": "BindCORE STARLING",
    "AF-CALVADOS": "BindCORE AF-CALVADOS",
    "CLIP": "CLIP",
    "MoRFchibi": "MoRFchibi 2.0"
}

SHORT_LABELS = {
    "IDPFold2": r"$\mathbf{BC_{IDPF2}}$", 
    "STARLING": r"$\mathbf{BC_{STA}}$",
    "AF-CALVADOS": r"$\mathbf{BC_{AFCAV}}$",
    "CLIP": r"$\mathbf{CLIP}$",
    "MoRFchibi": r"$\mathbf{MoRF2}$"
}

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

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print("Loading data...")
lip_ann = parse_annotation_file("data/LIP_dataset/TE440.txt")
morf_ann = parse_annotation_file("data/MoRF_dataset/test.txt")

lip_preds = {
    "AF-CALVADOS": parse_prediction_file("data/predictions/BindCORE_LIP_AF_CALVADOS_TE440_less_than_1024.csv"),
    "IDPFold2": parse_prediction_file("data/predictions/BindCORE_LIP_IDPFold2_TE440_less_than_1024.csv"),
    "STARLING": parse_prediction_file("data/predictions/BindCORE_LIP_STARLING_TE440_less_than_380.csv"),
    "CLIP": parse_prediction_file("data/predictions/CLIP_TE440.csv", max_len=1024),
}

morf_preds = {
    "AF-CALVADOS": parse_prediction_file("data/predictions/BindCORE_MoRF_AF_CALVADOS_test.csv"),
    "IDPFold2": parse_prediction_file("data/predictions/BindCORE_MoRF_IDPFold2_test.csv"),
    "STARLING": parse_prediction_file("data/predictions/BindCORE_MoRF_STARLING_test_less_than_380.csv"),
    "MoRFchibi": parse_prediction_file("data/predictions/MoRFchibi_test.csv"),
}

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
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

print("Preprocessing LIP data...")
lip_labels, lip_scores, lip_lengths = prepare_dataset(lip_ann, lip_preds)
print("Preprocessing MoRF data...")
morf_labels, morf_scores, morf_lengths = prepare_dataset(morf_ann, morf_preds)


# ---------------------------------------------------------
# NEW HELPER: ADD DETACHED PANEL LETTER
# ---------------------------------------------------------
def add_panel_letter(ax, letter):
    """Places a bold lowercase letter completely detached in the top-left margin."""
    ax.text(
        -0.22, 1.06, letter, 
        transform=ax.transAxes, 
        fontsize=14, 
        fontweight="bold", 
        va="bottom", 
        ha="left"
    )


# ---------------------------------------------------------
# FIGURE GENERATION
# ---------------------------------------------------------
print("Generating figures...")
fig = plt.figure(figsize=(14, 7))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.65, wspace=0.45) # Increased wspace slightly for letters
fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.20)

# --- Panel A & B: Performance up to length k ---
def plot_perf_length(ax, labels, scores, lengths, title, letter):
    models = list(scores.keys())
    for model in models:
        model_pids = list(scores[model].keys())
        if not model_pids:
            continue

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

        ax.plot(thresholds, perf, marker="o", lw=1.5, markersize=4, color=COLORS[model], label=LONG_LABELS.get(model, model))

    ax.set_title(title, fontweight="bold", pad=10)
    add_panel_letter(ax, letter)
    ax.set_xlabel("Max protein length (k)")
    ax.set_ylabel("AUPRC (up to length k)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(False)
    sns.despine(ax=ax)


# --- Panel C & D: Calibration ---
def plot_calibration(ax, labels, scores, title, letter):
    models = list(scores.keys())
    bins = np.linspace(0, 1, 11)

    for model in models:
        model_pids = list(scores[model].keys())
        if not model_pids:
            continue

        all_l = np.concatenate([labels[pid] for pid in model_pids])
        all_s = np.concatenate([scores[model][pid] for pid in model_pids])

        bin_idx = np.digitize(all_s, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
        frac_pos = []
        for i in range(len(bins) - 1):
            mask = bin_idx == i
            if mask.sum() > 0:
                frac_pos.append(all_l[mask].mean())
            else:
                frac_pos.append(np.nan)

        mids = (bins[:-1] + bins[1:]) / 2
        ax.plot(mids, frac_pos, "s-", lw=1.5, markersize=4, color=COLORS[model], label=LONG_LABELS.get(model, model))

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.set_title(title, fontweight="bold", pad=10)
    add_panel_letter(ax, letter)
    ax.set_xlabel("Predicted score (Rank Normalized)")
    ax.set_ylabel("Fraction of positives")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(False)
    sns.despine(ax=ax)


# --- Panel E & f: Agreement / Correlation Heatmap ---
def plot_agreement(ax, labels, scores, title, letter):
    models = list(scores.keys())
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

    short_names = [SHORT_LABELS.get(m, m) for m in models]
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        xticklabels=short_names,
        yticklabels=short_names,
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(title, fontweight="bold", pad=10)
    add_panel_letter(ax, letter)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


# --- Panel G & H: Positive rate vs protein length ---
def plot_positive_rate_by_length(ax, ann, title, letter):
    pids = list(ann.keys())
    lengths_arr = []
    pos_rates = []
    for pid in pids:
        mask = ann[pid]["mask"]
        lab = ann[pid]["labels"]
        lab_masked = lab[mask]
        if len(lab_masked) == 0 or len(lab_masked) > 1024:
            continue
        lengths_arr.append(len(lab_masked))
        pos_rates.append(lab_masked.mean())

    lengths_arr = np.array(lengths_arr)
    pos_rates = np.array(pos_rates)

    ax.scatter(lengths_arr, pos_rates, alpha=0.25, s=6, color="#888888", zorder=1)

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

    ax.plot(bin_centers, bin_means, color="#1a1a2e", lw=2, zorder=3, label="Binned mean")
    ax.fill_between(
        bin_centers,
        np.clip(bin_means - bin_stds, 0, 1),
        np.clip(bin_means + bin_stds, 0, 1),
        alpha=0.2,
        color="#1a1a2e",
        zorder=2,
        label="±1 std",
    )

    ax.set_title(title, fontweight="bold", pad=10)
    add_panel_letter(ax, letter)
    ax.set_xlabel("Protein length (residues)")
    ax.set_ylabel("Positive rate (binding fraction)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=False, fontsize="x-large")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(False)
    sns.despine(ax=ax)


# Render All Panels (Separating Title and Letter Strings)
plot_perf_length(fig.add_subplot(gs[0, 0]), lip_labels, lip_scores, lip_lengths, "LIP Performance by Length", "a")
plot_perf_length(fig.add_subplot(gs[1, 0]), morf_labels, morf_scores, morf_lengths, "MoRF Performance by Length", "b")
plot_calibration(fig.add_subplot(gs[0, 1]), lip_labels, lip_scores, "LIP Calibration", "c")
plot_calibration(fig.add_subplot(gs[1, 1]), morf_labels, morf_scores, "MoRF Calibration", "d")
plot_agreement(fig.add_subplot(gs[0, 2]), lip_labels, lip_scores, "LIP Model Agreement\n(Pearson r)", "e")
plot_agreement(fig.add_subplot(gs[1, 2]), morf_labels, morf_scores, "MoRF Model Agreement\n(Pearson r)", "f")
plot_positive_rate_by_length(fig.add_subplot(gs[0, 3]), lip_ann, "LIP Positive Rate by Length", "g")
plot_positive_rate_by_length(fig.add_subplot(gs[1, 3]), morf_ann, "MoRF Positive Rate by Length", "h")

# ---------------------------------------------------------
# GLOBAL UNIFIED LEGEND CREATION
# ---------------------------------------------------------
fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15)

shared_legend_elements = [
    Line2D([0], [0], color=COLORS["IDPFold2"], lw=2, marker='o', markersize=5, label=LONG_LABELS["IDPFold2"]),
    Line2D([0], [0], color=COLORS["STARLING"], lw=2, marker='o', markersize=5, label=LONG_LABELS["STARLING"]),
    Line2D([0], [0], color=COLORS["AF-CALVADOS"], lw=2, marker='o', markersize=5, label=LONG_LABELS["AF-CALVADOS"]),
    Line2D([0], [0], color=COLORS["CLIP"], lw=2, marker='o', markersize=5, label=LONG_LABELS["CLIP"]),
    Line2D([0], [0], color=COLORS["MoRFchibi"], lw=2, marker='o', markersize=5, label=LONG_LABELS["MoRFchibi"]),
    Line2D([0], [0], color='k', linestyle='--', lw=1.2, label='Perfect calibration')
]

fig.legend(
    handles=shared_legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.00),
    ncol=6, 
    frameon=False,
    fontsize=11
)

out_path = "figures/predictions_comparison.pdf"
plt.savefig(out_path, dpi=300)
print(f"Saved publication-ready figure to {out_path}")