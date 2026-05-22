"""
MoRF Prediction Model Comparison Analysis
==========================================
Compares two MoRF predictors (CORE-LIP and MoRFchibi) against ground truth labels.
Produces:
  - Per-protein and aggregate metrics (AUC, AUPRC, MCC, F1, precision, recall, etc.)
  - Correlation analysis between both models
  - Strength/weakness breakdown (agreement vs. disagreement regions)
  - Multiple publication-quality figures saved to ./figures/
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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────


def parse_annotation_file(path: str, debug: bool = True) -> dict:
    """
    Parse FASTA-like annotation file.
    Label characters: '1' = MoRF, '0' = non-MoRF, '-' = unknown/missing (masked out).
    Returns {protein_id: {"sequence": str, "labels": np.ndarray, "mask": np.ndarray}}
    where mask[i] = True means residue i has a known label.
    """
    data = {}
    current_id = None
    current_seq = ""
    current_lab = ""

    # Debug counters
    skipped_no_label = []
    skipped_label_line_rejected = []
    has_dash = []
    all_ids_seen = []

    def _is_label_line(line):
        """A label line contains only 0, 1, and/or - characters."""
        return bool(re.fullmatch(r"[01\-]+", line)) and len(line) > 0

    def _flush(pid, seq, lab):
        if pid is None:
            return
        all_ids_seen.append(pid)
        if not lab:
            skipped_no_label.append(pid)
            if debug:
                print(f"  [DEBUG] {pid}: NO label line found — skipped")
            return
        raw = list(lab)
        mask = np.array([c != "-" for c in raw])
        labels = np.array([int(c) if c != "-" else 0 for c in raw])
        if "-" in raw:
            has_dash.append(pid)
            if debug:
                n_dash = sum(1 for c in raw if c == "-")
                print(
                    f"  [DEBUG] {pid}: label line has {n_dash} '-' characters "
                    f"({100*n_dash/len(raw):.1f}% masked)"
                )
        data[pid] = {"sequence": seq, "labels": labels, "mask": mask}

    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush(current_id, current_seq, current_lab)
                current_id = line[1:]
                current_seq = ""
                current_lab = ""
            elif _is_label_line(line):
                current_lab += line
            else:
                # Could be sequence or a malformed label line — check for digits mixed with letters
                if re.search(r"[01\-]", line) and re.search(r"[A-Za-z]", line):
                    if debug:
                        print(
                            f"  [DEBUG] line {lineno} for {current_id!r} looks mixed "
                            f"(seq+label?): {line[:60]!r}"
                        )
                current_seq += line

    _flush(current_id, current_seq, current_lab)

    if debug:
        print(f"\n[parse_annotation_file] '{path}'")
        print(f"  Total >headers seen  : {len(all_ids_seen)}")
        print(f"  Successfully parsed  : {len(data)}")
        print(f"  Skipped (no label)   : {len(skipped_no_label)}")
        if skipped_no_label:
            print(f"    IDs: {skipped_no_label}")
        print(f"  Proteins with '-'    : {len(has_dash)}")
        if has_dash:
            print(f"    IDs: {has_dash}")

    return data


def parse_prediction_file(path: str) -> dict:
    """
    Parse CSV prediction file.
    Returns {protein_id: {"scores": np.ndarray, "binary": np.ndarray}}
    """
    df = pd.read_csv(path)
    data = {}
    for _, row in df.iterrows():
        pid = str(row["protein_id"])
        scores = np.array([float(x) for x in str(row["predictions"]).split(",")])
        binary = np.array([int(x) for x in str(row["binary_predictions"]).split(",")])
        data[pid] = {"scores": scores, "binary": binary}
    return data


print("Loading data …\n")
annotations = parse_annotation_file("data/MoRF_dataset/test.txt", debug=True)
bindcore = parse_prediction_file("data/predictions/bindcore_test.csv")
morfchibi = parse_prediction_file("data/predictions/MoRFchibi_test.csv")

print(f"\n[parse_prediction_file] bindcore_test.csv   : {len(bindcore)} proteins")
print(f"[parse_prediction_file] MoRFchibi_test.csv  : {len(morfchibi)} proteins")

# ─────────────────────────────────────────────
# 2.  INTERSECTION DEBUG + ALIGNED ARRAYS
# ─────────────────────────────────────────────

ann_ids = set(annotations)
core_ids = set(bindcore)
chibi_ids = set(morfchibi)

in_ann_not_core = ann_ids - core_ids
in_ann_not_chibi = ann_ids - chibi_ids
only_core = core_ids - ann_ids
only_chibi = chibi_ids - ann_ids
common_ids = sorted(ann_ids & core_ids & chibi_ids)

print(f"\n[Intersection debug]")
print(f"  Annotation IDs            : {len(ann_ids)}")
print(f"  CORE-LIP prediction IDs   : {len(core_ids)}")
print(f"  MoRFchibi prediction IDs  : {len(chibi_ids)}")
print(f"  Intersection (all 3)      : {len(common_ids)}")
print(f"  In annot. not CORE-LIP    : {len(in_ann_not_core)}")
if in_ann_not_core:
    print(f"    IDs: {sorted(in_ann_not_core)}")
print(f"  In annot. not MoRFchibi   : {len(in_ann_not_chibi)}")
if in_ann_not_chibi:
    print(f"    IDs: {sorted(in_ann_not_chibi)}")
print(f"  In CORE-LIP but not annot.: {len(only_core)}")
if only_core:
    print(f"    IDs: {sorted(only_core)}")
print(f"  In MoRFchibi but not annot: {len(only_chibi)}")
if only_chibi:
    print(f"    IDs: {sorted(only_chibi)}")

# Debug length mismatches
print(f"\n[Length mismatch debug]  (annotation vs predictions, for common IDs)")
n_mismatch = 0
for pid in common_ids:
    la = len(annotations[pid]["labels"])
    lc = len(bindcore[pid]["scores"])
    lm = len(morfchibi[pid]["scores"])
    if not (la == lc == lm):
        n_mismatch += 1
        print(f"  {pid}: annot={la}  bindcore={lc}  morfchibi={lm}")
if n_mismatch == 0:
    print("  All lengths match perfectly.")

print(f"\nProteins in all three files: {len(common_ids)}")

# -------------------------------------------------
# 2b. SCORE NORMALISATION
# -------------------------------------------------
# Raw scores are NOT probabilities (e.g. CORE-LIP is squashed into ~[0.4, 0.6]).
# We normalise globally using the empirical CDF (rank normalisation):
#   each score is mapped to its percentile rank across ALL residues.
# This is robust to outliers and handles any non-linear score scale.
# Min-max raw stats are also printed for reference.

from scipy.stats import rankdata as _rankdata


def _global_rank_norm(scores, sorted_vals, pcts):
    """Interpolate scores against the global sorted array to get percentile in [0,1]."""
    return np.interp(scores, sorted_vals, pcts)


def _make_rank_lut(arr):
    order = np.argsort(arr)
    sorted_vals = arr[order]
    pcts = np.arange(len(sorted_vals)) / (len(sorted_vals) - 1)
    return sorted_vals, pcts


# Collect ALL raw scores globally
_all_core_raw = np.concatenate([bindcore[pid]["scores"] for pid in common_ids])
_all_chibi_raw = np.concatenate([morfchibi[pid]["scores"] for pid in common_ids])

_core_sorted, _core_pct = _make_rank_lut(_all_core_raw)
_chibi_sorted, _chibi_pct = _make_rank_lut(_all_chibi_raw)

print("\n[Score normalisation -- raw score statistics]")
for name, raw in [("CORE-LIP", _all_core_raw), ("MoRFchibi", _all_chibi_raw)]:
    p1, p99 = np.percentile(raw, 1), np.percentile(raw, 99)
    print(
        f"  {name:10s}: min={raw.min():.4f}  max={raw.max():.4f}  "
        f"mean={raw.mean():.4f}  p1={p1:.4f}  p99={p99:.4f}"
    )
print("  -> rank-norm maps all scores to [0, 1] via global empirical CDF")

all_labels, all_core, all_chibi = [], [], []
all_core_raw_list, all_chibi_raw_list = [], []
all_core_bin, all_chibi_bin = [], []

per_protein = []  # per-protein metrics rows

for pid in common_ids:
    mask = annotations[pid]["mask"]
    lab = annotations[pid]["labels"]
    c_sc_raw = bindcore[pid]["scores"]
    m_sc_raw = morfchibi[pid]["scores"]
    c_bin = bindcore[pid]["binary"]
    m_bin = morfchibi[pid]["binary"]

    # Normalise BEFORE truncation (global rank-norm needs the full per-protein vector)
    c_sc = _global_rank_norm(c_sc_raw, _core_sorted, _core_pct)
    m_sc = _global_rank_norm(m_sc_raw, _chibi_sorted, _chibi_pct)

    # length alignment
    n = min(len(lab), len(c_sc), len(m_sc))
    lab, mask = lab[:n], mask[:n]
    c_sc_raw = c_sc_raw[:n]
    m_sc_raw = m_sc_raw[:n]
    c_sc = c_sc[:n]
    m_sc = m_sc[:n]
    c_bin = c_bin[:n]
    m_bin = m_bin[:n]

    # Apply mask: keep only residues with known labels
    lab = lab[mask]
    c_sc_raw = c_sc_raw[mask]
    m_sc_raw = m_sc_raw[mask]
    c_sc = c_sc[mask]
    m_sc = m_sc[mask]
    c_bin = c_bin[mask]
    m_bin = m_bin[mask]

    all_labels.append(lab)
    all_core.append(c_sc)
    all_chibi.append(m_sc)
    all_core_raw_list.append(c_sc_raw)
    all_chibi_raw_list.append(m_sc_raw)
    all_core_bin.append(c_bin)
    all_chibi_bin.append(m_bin)

    # per-protein metrics
    row = {"protein_id": pid, "length": len(lab), "morf_fraction": lab.mean()}
    for name, sc, bn in [("bindcore", c_sc, c_bin), ("MoRFchibi", m_sc, m_bin)]:
        if len(np.unique(lab)) > 1:
            row[f"{name}_AUC"] = roc_auc_score(lab, sc)
            row[f"{name}_AUPRC"] = average_precision_score(lab, sc)
        else:
            row[f"{name}_AUC"] = np.nan
            row[f"{name}_AUPRC"] = np.nan
        row[f"{name}_MCC"] = matthews_corrcoef(lab, bn)
        row[f"{name}_F1"] = f1_score(lab, bn, zero_division=0)
        row[f"{name}_Prec"] = precision_score(lab, bn, zero_division=0)
        row[f"{name}_Rec"] = recall_score(lab, bn, zero_division=0)
    row["score_pearson"] = pearsonr(c_sc, m_sc)[0]
    row["score_spearman"] = spearmanr(c_sc, m_sc)[0]
    per_protein.append(row)

pp_df = pd.DataFrame(per_protein)

# Flatten
y_true = np.concatenate(all_labels)
y_core = np.concatenate(all_core)
y_chibi = np.concatenate(all_chibi)
y_core_raw = np.concatenate(all_core_raw_list)
y_chibi_raw = np.concatenate(all_chibi_raw_list)
y_core_bin = np.concatenate(all_core_bin)
y_chibi_bin = np.concatenate(all_chibi_bin)


print(
    f"Total residues: {len(y_true):,}  |  MoRF residues: {y_true.sum():,} ({100*y_true.mean():.1f}%)"
)

# ─────────────────────────────────────────────
# 3.  AGGREGATE METRICS
# ─────────────────────────────────────────────


def aggregate_metrics(y_true, y_score, y_bin, name):
    metrics = {
        "Model": name,
        "AUC-ROC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score),
        "MCC": matthews_corrcoef(y_true, y_bin),
        "F1": f1_score(y_true, y_bin, zero_division=0),
        "Precision": precision_score(y_true, y_bin, zero_division=0),
        "Recall": recall_score(y_true, y_bin, zero_division=0),
        "Brier": brier_score_loss(y_true, y_score),
        "LogLoss": log_loss(y_true, y_score),
        "TP": int(((y_bin == 1) & (y_true == 1)).sum()),
        "TN": int(((y_bin == 0) & (y_true == 0)).sum()),
        "FP": int(((y_bin == 1) & (y_true == 0)).sum()),
        "FN": int(((y_bin == 0) & (y_true == 1)).sum()),
    }
    metrics["Specificity"] = metrics["TN"] / (metrics["TN"] + metrics["FP"] + 1e-9)
    metrics["Balanced_Acc"] = (metrics["Recall"] + metrics["Specificity"]) / 2
    return metrics


m1 = aggregate_metrics(y_true, y_core, y_core_bin, "CORE-LIP")
m2 = aggregate_metrics(y_true, y_chibi, y_chibi_bin, "MoRFchibi")
metrics_df = pd.DataFrame([m1, m2]).set_index("Model")

print("\n=== AGGREGATE METRICS ===")
print(metrics_df.to_string())

# ─────────────────────────────────────────────
# 4.  CORRELATION BETWEEN MODELS
# ─────────────────────────────────────────────

pear_r, pear_p = pearsonr(y_core, y_chibi)
spear_r, spear_p = spearmanr(y_core, y_chibi)
kend_r, kend_p = kendalltau(y_core, y_chibi)
binary_agree = (y_core_bin == y_chibi_bin).mean()

print(f"\n=== INTER-MODEL CORRELATION ===")
print(f"  Pearson  r = {pear_r:.4f}  (p={pear_p:.2e})")
print(f"  Spearman r = {spear_r:.4f}  (p={spear_p:.2e})")
print(f"  Kendall  τ = {kend_r:.4f}  (p={kend_p:.2e})")
print(f"  Binary agreement = {100*binary_agree:.1f}%")

# ─────────────────────────────────────────────
# 5.  DISAGREEMENT ANALYSIS
# ─────────────────────────────────────────────

# Quadrants: both correct, CORE-LIP only, MoRFchibi only, both wrong
both_correct = (y_core_bin == y_true) & (y_chibi_bin == y_true)
core_only = (y_core_bin == y_true) & (y_chibi_bin != y_true)
chibi_only = (y_core_bin != y_true) & (y_chibi_bin == y_true)
both_wrong = (y_core_bin != y_true) & (y_chibi_bin != y_true)

disagree_mask = y_core_bin != y_chibi_bin  # one says 1, other 0

print(f"\n=== AGREEMENT BREAKDOWN ===")
print(f"  Both correct  : {both_correct.sum():>7,}  ({100*both_correct.mean():.1f}%)")
print(f"  CORE-LIP only : {core_only.sum():>7,}  ({100*core_only.mean():.1f}%)")
print(f"  MoRFchibi only: {chibi_only.sum():>7,}  ({100*chibi_only.mean():.1f}%)")
print(f"  Both wrong    : {both_wrong.sum():>7,}  ({100*both_wrong.mean():.1f}%)")
print(f"  Disagreements : {disagree_mask.sum():>7,}  ({100*disagree_mask.mean():.1f}%)")

# Performance on disagreement subsets
print("\n  When models disagree:")
for mask_name, mask in [
    ("CORE-LIP=1, MoRFchibi=0", (y_core_bin == 1) & (y_chibi_bin == 0)),
    ("CORE-LIP=0, MoRFchibi=1", (y_core_bin == 0) & (y_chibi_bin == 1)),
]:
    if mask.sum() > 10:
        true_rate = y_true[mask].mean()
        print(
            f"    {mask_name}: {mask.sum():,} residues, true MoRF rate = {100*true_rate:.1f}%"
        )

# ─────────────────────────────────────────────
# 6.  FIGURES
# ─────────────────────────────────────────────

PALETTE = {"CORE-LIP": "#2C7BB6", "MoRFchibi": "#D7191C"}
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ── 6.1  ROC curves ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, y_score, y_bin) in zip(
    axes, [("CORE-LIP", y_core, y_core_bin), ("MoRFchibi", y_chibi, y_chibi_bin)]
):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax.plot(fpr, tpr, color=PALETTE[name], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} – ROC Curve")
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("figures/01_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.2  PR curves (both on same plot) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, y_score in [("CORE-LIP", y_core), ("MoRFchibi", y_chibi)]:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax.plot(rec, prec, color=PALETTE[name], lw=2, label=f"{name}  AP={ap:.3f}")
ax.axhline(y_true.mean(), color="grey", linestyle="--", lw=1, label="Random baseline")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend()
plt.tight_layout()
plt.savefig("figures/02_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.3  Score scatter + hexbin ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter (sample up to 10k for readability)
idx = np.random.choice(len(y_core), min(10_000, len(y_core)), replace=False)
sc = axes[0].scatter(
    y_core[idx],
    y_chibi[idx],
    c=y_true[idx],
    cmap="RdBu_r",
    alpha=0.4,
    s=5,
    vmin=0,
    vmax=1,
)
plt.colorbar(sc, ax=axes[0], label="True label")
axes[0].set_xlabel("CORE-LIP score")
axes[0].set_ylabel("MoRFchibi score")
axes[0].set_title(f"Score Scatter  (Pearson r={pear_r:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)

# Hexbin density
hb = axes[1].hexbin(y_core, y_chibi, gridsize=60, cmap="YlOrRd", mincnt=1)
plt.colorbar(hb, ax=axes[1], label="Count")
axes[1].set_xlabel("CORE-LIP score")
axes[1].set_ylabel("MoRFchibi score")
axes[1].set_title("Score Density (hexbin)")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)

plt.tight_layout()
plt.savefig("figures/03_score_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# -- 6.4a  Raw vs Normalised score distributions (before/after)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
pairs = [
    ("CORE-LIP", y_core_raw, y_core, PALETTE["CORE-LIP"]),
    ("MoRFchibi", y_chibi_raw, y_chibi, PALETTE["MoRFchibi"]),
]
for col_offset, (name, raw, norm, color) in enumerate(pairs):
    for row_i, (label_val, lbl) in enumerate([(1, "MoRF"), (0, "Non-MoRF")]):
        mask_lbl = y_true == label_val
        # Raw
        axes[row_i, col_offset * 2].hist(
            raw[mask_lbl],
            bins=60,
            color=color,
            alpha=0.7,
            edgecolor="none",
            density=True,
        )
        axes[row_i, col_offset * 2].set_title(f"{name} RAW -- {lbl}")
        axes[row_i, col_offset * 2].set_xlabel("Raw score")
        axes[row_i, col_offset * 2].set_ylabel("Density")
        # Normalised
        axes[row_i, col_offset * 2 + 1].hist(
            norm[mask_lbl],
            bins=60,
            color=color,
            alpha=0.7,
            edgecolor="none",
            density=True,
        )
        axes[row_i, col_offset * 2 + 1].set_title(f"{name} NORM -- {lbl}")
        axes[row_i, col_offset * 2 + 1].set_xlabel("Rank-normalised score")
        axes[row_i, col_offset * 2 + 1].set_ylabel("Density")
plt.suptitle(
    "Raw vs Rank-Normalised Score Distributions by True Label",
    y=1.01,
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    "figures/04_score_distributions_raw_vs_norm.png", dpi=150, bbox_inches="tight"
)
plt.close()

# -- 6.4b  Normalised score distributions by label (clean version)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row_i, label_val, lbl in [(0, 1, "MoRF (label=1)"), (1, 0, "Non-MoRF (label=0)")]:
    mask_lbl = y_true == label_val
    for col_i, (name, y_score) in enumerate(
        [("CORE-LIP", y_core), ("MoRFchibi", y_chibi)]
    ):
        axes[row_i, col_i].hist(
            y_score[mask_lbl],
            bins=60,
            color=PALETTE[name],
            alpha=0.7,
            edgecolor="none",
            density=True,
        )
        axes[row_i, col_i].set_title(f"{name} (normalised) -- {lbl}")
        axes[row_i, col_i].set_xlabel("Rank-normalised score")
        axes[row_i, col_i].set_ylabel("Density")
plt.suptitle(
    "Normalised Score Distributions by True Label",
    y=1.01,
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("figures/04_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.5  Per-protein AUC comparison ──────────────────────────────────────────
valid = pp_df.dropna(subset=["bindcore_AUC", "MoRFchibi_AUC"])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(
    valid["bindcore_AUC"], valid["MoRFchibi_AUC"], alpha=0.5, s=20, color="#555"
)
axes[0].plot([0, 1], [0, 1], "r--", lw=1)
axes[0].set_xlabel("CORE-LIP AUC")
axes[0].set_ylabel("MoRFchibi AUC")
axes[0].set_title("Per-protein AUC")

diff = valid["bindcore_AUC"] - valid["MoRFchibi_AUC"]
axes[1].hist(diff, bins=40, color="#555", edgecolor="none", alpha=0.8)
axes[1].axvline(0, color="red", lw=1.5)
axes[1].set_xlabel("CORE-LIP AUC  –  MoRFchibi AUC")
axes[1].set_ylabel("Proteins")
axes[1].set_title(f"AUC Difference  (mean={diff.mean():+.3f})")

plt.tight_layout()
plt.savefig("figures/05_per_protein_auc.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.6  Per-protein MCC comparison ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(
    pp_df["bindcore_MCC"], pp_df["MoRFchibi_MCC"], alpha=0.5, s=20, color="#555"
)
axes[0].plot([-1, 1], [-1, 1], "r--", lw=1)
axes[0].set_xlabel("CORE-LIP MCC")
axes[0].set_ylabel("MoRFchibi MCC")
axes[0].set_title("Per-protein MCC")

diff_mcc = pp_df["bindcore_MCC"] - pp_df["MoRFchibi_MCC"]
axes[1].hist(diff_mcc.dropna(), bins=40, color="#555", edgecolor="none", alpha=0.8)
axes[1].axvline(0, color="red", lw=1.5)
axes[1].set_xlabel("CORE-LIP MCC  –  MoRFchibi MCC")
axes[1].set_ylabel("Proteins")
axes[1].set_title(f"MCC Difference  (mean={diff_mcc.mean():+.3f})")

plt.tight_layout()
plt.savefig("figures/06_per_protein_mcc.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.7  Agreement breakdown bar ─────────────────────────────────────────────
cats = ["Both correct", "CORE-LIP only", "MoRFchibi only", "Both wrong"]
counts = [both_correct.sum(), core_only.sum(), chibi_only.sum(), both_wrong.sum()]
colors = ["#2ECC71", "#2C7BB6", "#D7191C", "#E74C3C"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(cats, counts, color=colors, edgecolor="white", width=0.6)
ax.set_ylabel("Residue count")
ax.set_title("Prediction Agreement Breakdown")
for bar, val in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + len(y_true) * 0.003,
        f"{val:,}\n({100*val/len(y_true):.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig("figures/07_agreement_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.8  Metric summary bar chart ────────────────────────────────────────────
metric_cols = ["AUC-ROC", "AUPRC", "MCC", "F1", "Precision", "Recall", "Balanced_Acc"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metric_cols))
w = 0.35
for i, (model, color) in enumerate(
    [("CORE-LIP", PALETTE["CORE-LIP"]), ("MoRFchibi", PALETTE["MoRFchibi"])]
):
    vals = [metrics_df.loc[model, m] for m in metric_cols]
    bars = ax.bar(x + i * w - w / 2, vals, w, label=model, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metric_cols, rotation=20, ha="right")
ax.set_ylim(0, 1.12)
ax.axhline(0.5, color="grey", linestyle="--", lw=0.8)
ax.set_title("Aggregate Metric Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("figures/08_metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.9  Confusion matrices ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, y_bin) in zip(
    axes, [("CORE-LIP", y_core_bin), ("MoRFchibi", y_chibi_bin)]
):
    cm = confusion_matrix(y_true, y_bin)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax.set_title(f"{name} Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/09_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.10  Per-protein score correlation distribution ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(
    pp_df["score_pearson"].dropna(),
    bins=40,
    color="#8E44AD",
    alpha=0.8,
    edgecolor="none",
    label="Pearson r",
)
ax.hist(
    pp_df["score_spearman"].dropna(),
    bins=40,
    color="#E67E22",
    alpha=0.6,
    edgecolor="none",
    label="Spearman ρ",
)
ax.axvline(pear_r, color="#8E44AD", linestyle="--", lw=2)
ax.axvline(spear_r, color="#E67E22", linestyle="--", lw=2)
ax.set_xlabel("Correlation coefficient")
ax.set_ylabel("Proteins")
ax.set_title("Per-protein inter-model score correlation")
ax.legend()
plt.tight_layout()
plt.savefig("figures/10_per_protein_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.11  MoRF fraction vs per-protein AUC ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, name, col in [
    (axes[0], "CORE-LIP", "bindcore_AUC"),
    (axes[1], "MoRFchibi", "MoRFchibi_AUC"),
]:
    sub = pp_df.dropna(subset=[col])
    ax.scatter(sub["morf_fraction"], sub[col], alpha=0.4, s=20, color=PALETTE[name])
    # trend line
    z = np.polyfit(sub["morf_fraction"], sub[col], 1)
    p = np.poly1d(z)
    xs = np.linspace(0, 1, 200)
    ax.plot(xs, p(xs), "k--", lw=1.5)
    ax.set_xlabel("MoRF fraction in protein")
    ax.set_ylabel("AUC")
    ax.set_title(f"{name}: MoRF fraction vs AUC")
plt.tight_layout()
plt.savefig("figures/11_morf_fraction_vs_auc.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6.12  Score calibration (reliability diagram) ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bins = np.linspace(0, 1, 11)
for name, y_score, color in [
    ("CORE-LIP", y_core, PALETTE["CORE-LIP"]),
    ("MoRFchibi", y_chibi, PALETTE["MoRFchibi"]),
]:
    bin_idx = np.digitize(y_score, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
    frac_pos = [
        y_true[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan
        for i in range(len(bins) - 1)
    ]
    mids = (bins[:-1] + bins[1:]) / 2
    ax.plot(mids, frac_pos, "o-", color=color, lw=2, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean predicted score")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve (Reliability Diagram)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/12_calibration.png", dpi=150, bbox_inches="tight")
plt.close()


# -------------------------------------------------
# 6.13  Performance vs Sequence Length
# -------------------------------------------------
# For each protein we have: length (post-mask) and per-protein AUPRC / AUC / MCC.
# We compute:
#   (a) Cumulative performance up to a given length threshold
#       (how well do models do on proteins <= L residues?)
#   (b) Rolling/binned performance as a function of length
#       (does performance degrade for longer / shorter proteins?)

# Build a length-annotated dataframe with raw residue arrays per protein
length_records = []
for pid in common_ids:
    row_pp = pp_df[pp_df["protein_id"] == pid]
    if row_pp.empty:
        continue
    length_records.append(
        {
            "protein_id": pid,
            "length": int(row_pp["length"].values[0]),
            "morf_frac": float(row_pp["morf_fraction"].values[0]),
            "bindcore_AUC": row_pp["bindcore_AUC"].values[0],
            "MoRFchibi_AUC": row_pp["MoRFchibi_AUC"].values[0],
            "bindcore_AUPRC": row_pp["bindcore_AUPRC"].values[0],
            "MoRFchibi_AUPRC": row_pp["MoRFchibi_AUPRC"].values[0],
            "bindcore_MCC": row_pp["bindcore_MCC"].values[0],
            "MoRFchibi_MCC": row_pp["MoRFchibi_MCC"].values[0],
        }
    )

len_df = pd.DataFrame(length_records).sort_values("length").reset_index(drop=True)

# ── (a) Cumulative performance: pool all residues for proteins <= L, compute AUPRC ──
thresholds = np.unique(
    np.percentile(len_df["length"], np.linspace(5, 100, 40)).astype(int)
)
thresholds = thresholds[thresholds >= 10]  # need enough residues

cum_results = []
# Build per-protein residue arrays indexed by protein_id
_residue_cache = {}
label_concat = np.concatenate(all_labels)
core_concat = np.concatenate(all_core)
chibi_concat = np.concatenate(all_chibi)

# Rebuild per-protein arrays from the already-processed lists
_pid_arrays = {}
for i, pid in enumerate(common_ids):
    _pid_arrays[pid] = {
        "labels": all_labels[i],
        "core": all_core[i],
        "chibi": all_chibi[i],
    }

for thresh in thresholds:
    subset = len_df[len_df["length"] <= thresh]
    if len(subset) < 3:
        continue
    labs_pool = np.concatenate([_pid_arrays[p]["labels"] for p in subset["protein_id"]])
    core_pool = np.concatenate([_pid_arrays[p]["core"] for p in subset["protein_id"]])
    chibi_pool = np.concatenate([_pid_arrays[p]["chibi"] for p in subset["protein_id"]])
    if len(np.unique(labs_pool)) < 2:
        continue
    cum_results.append(
        {
            "max_length": thresh,
            "n_proteins": len(subset),
            "n_residues": len(labs_pool),
            "bindcore_AUPRC": average_precision_score(labs_pool, core_pool),
            "MoRFchibi_AUPRC": average_precision_score(labs_pool, chibi_pool),
            "bindcore_AUC": roc_auc_score(labs_pool, core_pool),
            "MoRFchibi_AUC": roc_auc_score(labs_pool, chibi_pool),
        }
    )

cum_df = pd.DataFrame(cum_results)

# ── (b) Binned performance: split proteins into length bins, compute pooled AUPRC per bin ──
N_BINS = 8
len_df["length_bin"] = pd.qcut(len_df["length"], q=N_BINS, duplicates="drop")
bin_results = []
for bin_label, grp in len_df.groupby("length_bin", observed=True):
    pids = grp["protein_id"].tolist()
    labs_pool = np.concatenate([_pid_arrays[p]["labels"] for p in pids])
    core_pool = np.concatenate([_pid_arrays[p]["core"] for p in pids])
    chibi_pool = np.concatenate([_pid_arrays[p]["chibi"] for p in pids])
    bin_mid = (bin_label.left + bin_label.right) / 2
    row_b = {
        "bin": str(bin_label),
        "bin_mid": bin_mid,
        "n_proteins": len(grp),
        "n_residues": len(labs_pool),
        "length_mean": grp["length"].mean(),
        "length_median": grp["length"].median(),
    }
    if len(np.unique(labs_pool)) >= 2:
        row_b["bindcore_AUPRC"] = average_precision_score(labs_pool, core_pool)
        row_b["MoRFchibi_AUPRC"] = average_precision_score(labs_pool, chibi_pool)
        row_b["bindcore_AUC"] = roc_auc_score(labs_pool, core_pool)
        row_b["MoRFchibi_AUC"] = roc_auc_score(labs_pool, chibi_pool)
        row_b["bindcore_MCC_mean"] = grp["bindcore_MCC"].mean()
        row_b["MoRFchibi_MCC_mean"] = grp["MoRFchibi_MCC"].mean()
    else:
        for k in [
            "bindcore_AUPRC",
            "MoRFchibi_AUPRC",
            "bindcore_AUC",
            "MoRFchibi_AUC",
            "bindcore_MCC_mean",
            "MoRFchibi_MCC_mean",
        ]:
            row_b[k] = np.nan
    bin_results.append(row_b)
bin_df = pd.DataFrame(bin_results).dropna(subset=["bindcore_AUPRC"])

print(
    f"\n[Length-stratified analysis]  {len(cum_df)} cumulative thresholds,  {len(bin_df)} bins"
)
print(
    bin_df[
        [
            "bin",
            "n_proteins",
            "n_residues",
            "bindcore_AUPRC",
            "MoRFchibi_AUPRC",
            "bindcore_AUC",
            "MoRFchibi_AUC",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

# ── Figure 13: Cumulative AUPRC/AUC up to length L ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(
    cum_df["max_length"],
    cum_df["bindcore_AUPRC"],
    color=PALETTE["CORE-LIP"],
    lw=2,
    marker="o",
    ms=4,
    label="CORE-LIP",
)
ax.plot(
    cum_df["max_length"],
    cum_df["MoRFchibi_AUPRC"],
    color=PALETTE["MoRFchibi"],
    lw=2,
    marker="s",
    ms=4,
    label="MoRFchibi",
)
ax.set_xlabel("Max sequence length (proteins \u2264 L)")
ax.set_ylabel("AUPRC")
ax.set_title("Cumulative AUPRC vs Max Sequence Length")
ax.legend()
# Annotate protein counts at a few thresholds
for _, r in cum_df.iloc[::8].iterrows():
    ax.annotate(
        f"n={int(r['n_proteins'])}",
        (r["max_length"], r["bindcore_AUPRC"]),
        textcoords="offset points",
        xytext=(0, 8),
        fontsize=7,
        ha="center",
        color="grey",
    )

ax = axes[1]
ax.plot(
    cum_df["max_length"],
    cum_df["bindcore_AUC"],
    color=PALETTE["CORE-LIP"],
    lw=2,
    marker="o",
    ms=4,
    label="CORE-LIP",
)
ax.plot(
    cum_df["max_length"],
    cum_df["MoRFchibi_AUC"],
    color=PALETTE["MoRFchibi"],
    lw=2,
    marker="s",
    ms=4,
    label="MoRFchibi",
)
ax.set_xlabel("Max sequence length (proteins \u2264 L)")
ax.set_ylabel("AUC-ROC")
ax.set_title("Cumulative AUC-ROC vs Max Sequence Length")
ax.legend()

plt.suptitle(
    "Performance on Proteins UP TO a Given Length", fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    "figures/13_cumulative_performance_vs_length.png", dpi=150, bbox_inches="tight"
)
plt.close()

# ── Figure 14: Binned performance evolution ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
x = np.arange(len(bin_df))
w = 0.35
tick_labels = [
    f"{int(r['length_median'])} aa\n(n={int(r['n_proteins'])})"
    for _, r in bin_df.iterrows()
]

for ax, metric, ylabel, title in [
    (axes[0], "AUPRC", "AUPRC", "Pooled AUPRC by Length Bin"),
    (axes[1], "AUC", "AUC-ROC", "Pooled AUC-ROC by Length Bin"),
    (axes[2], "MCC_mean", "Mean MCC", "Mean per-protein MCC by Length Bin"),
]:
    c_col = f"bindcore_{metric}"
    m_col = f"MoRFchibi_{metric}"
    bars1 = ax.bar(
        x - w / 2,
        bin_df[c_col],
        w,
        label="CORE-LIP",
        color=PALETTE["CORE-LIP"],
        alpha=0.85,
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + w / 2,
        bin_df[m_col],
        w,
        label="MoRFchibi",
        color=PALETTE["MoRFchibi"],
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Length bin (median aa, proteins in bin)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.15, max(bin_df[c_col].max(), bin_df[m_col].max()) * 1.15))

plt.suptitle(
    "Model Performance Evolution Across Sequence Length Bins",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("figures/14_binned_performance_vs_length.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 15: Scatter — per-protein length vs AUPRC ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
valid_auprc = pp_df.dropna(subset=["bindcore_AUPRC", "MoRFchibi_AUPRC"])

for ax, name, col in [
    (axes[0], "CORE-LIP", "bindcore_AUPRC"),
    (axes[1], "MoRFchibi", "MoRFchibi_AUPRC"),
]:
    sc = ax.scatter(
        valid_auprc["length"],
        valid_auprc[col],
        c=valid_auprc["morf_fraction"],
        cmap="RdYlGn",
        alpha=0.7,
        s=40,
        edgecolors="none",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="MoRF fraction")
    # Trend line
    z = np.polyfit(valid_auprc["length"], valid_auprc[col].fillna(0), 1)
    xs = np.linspace(valid_auprc["length"].min(), valid_auprc["length"].max(), 200)
    ax.plot(xs, np.poly1d(z)(xs), "k--", lw=1.5, label=f"slope={z[0]*100:.3f}/100aa")
    ax.set_xlabel("Sequence length (post-mask residues)")
    ax.set_ylabel("AUPRC")
    ax.set_title(f"{name}: AUPRC vs Sequence Length")
    ax.legend(fontsize=9)

plt.suptitle(
    "Per-protein AUPRC vs Sequence Length (colour = MoRF fraction)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("figures/15_per_protein_auprc_vs_length.png", dpi=150, bbox_inches="tight")
plt.close()

# Save length-stratified tables
cum_df.round(4).to_csv("figures/cumulative_performance_vs_length.csv", index=False)
bin_df.round(4).to_csv("figures/binned_performance_vs_length.csv", index=False)

# ─────────────────────────────────────────────
# 7.  SAVE TABLES
# ─────────────────────────────────────────────

metrics_df.round(4).to_csv("figures/aggregate_metrics.csv")
pp_df.round(4).to_csv("figures/per_protein_metrics.csv", index=False)

# ─────────────────────────────────────────────
# 8.  PRINTED SUMMARY REPORT
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)

print("\n[1] Aggregate Performance")
print(
    metrics_df[["AUC-ROC", "AUPRC", "MCC", "F1", "Precision", "Recall", "Balanced_Acc"]]
    .round(4)
    .to_string()
)

print(f"\n[2] Inter-model Score Correlation")
print(f"  Pearson  r = {pear_r:.4f}")
print(f"  Spearman ρ = {spear_r:.4f}")
print(f"  Kendall  τ = {kend_r:.4f}")
print(f"  Binary agreement = {100*binary_agree:.1f}%")

print(f"\n[3] Strengths & Weaknesses  (all {len(pp_df)} proteins)")
# Use AUC where both models have it; fall back to MCC otherwise (single-class proteins after masking)
has_auc = pp_df["bindcore_AUC"].notna() & pp_df["MoRFchibi_AUC"].notna()
no_auc = ~has_auc
n_auc = has_auc.sum()
n_no_auc = no_auc.sum()

core_better_auc = (
    pp_df.loc[has_auc, "bindcore_AUC"] > pp_df.loc[has_auc, "MoRFchibi_AUC"]
).sum()
chibi_better_auc = (
    pp_df.loc[has_auc, "MoRFchibi_AUC"] > pp_df.loc[has_auc, "bindcore_AUC"]
).sum()
core_better_mcc = (
    pp_df.loc[no_auc, "bindcore_MCC"] > pp_df.loc[no_auc, "MoRFchibi_MCC"]
).sum()
chibi_better_mcc = (
    pp_df.loc[no_auc, "MoRFchibi_MCC"] > pp_df.loc[no_auc, "bindcore_MCC"]
).sum()

core_better_total = core_better_auc + core_better_mcc
chibi_better_total = chibi_better_auc + chibi_better_mcc
tie_total = len(pp_df) - core_better_total - chibi_better_total

print(
    f"  Metric used: AUC on {n_auc} proteins, MCC on {n_no_auc} proteins (single-class after masking)"
)
print(
    f"  CORE-LIP  better : {core_better_total:>3}/{len(pp_df)}  ({100*core_better_total/len(pp_df):.1f}%)  "
    f"[AUC: {core_better_auc}, MCC: {core_better_mcc}]"
)
print(
    f"  MoRFchibi better : {chibi_better_total:>3}/{len(pp_df)}  ({100*chibi_better_total/len(pp_df):.1f}%)  "
    f"[AUC: {chibi_better_auc}, MCC: {chibi_better_mcc}]"
)
print(
    f"  Tie              : {tie_total:>3}/{len(pp_df)}  ({100*tie_total/len(pp_df):.1f}%)"
)

# Which model is more conservative / aggressive?
print(f"\n[4] Threshold Behaviour (binary predictions)")
for name, y_bin in [("CORE-LIP", y_core_bin), ("MoRFchibi", y_chibi_bin)]:
    print(
        f"  {name}: predicted positives = {100*y_bin.mean():.1f}%  "
        f"(true positive rate = {y_true.mean()*100:.1f}%)"
    )

print(f"\n[5] Per-protein AUC stats")
for name, col in [("CORE-LIP", "bindcore_AUC"), ("MoRFchibi", "MoRFchibi_AUC")]:
    vals = pp_df[col].dropna()
    print(
        f"  {name}:  mean={vals.mean():.3f}  median={vals.median():.3f}  std={vals.std():.3f}  "
        f"min={vals.min():.3f}  max={vals.max():.3f}"
    )

print(f"\n[6] Calibration (Brier score – lower is better)")
for name in ["CORE-LIP", "MoRFchibi"]:
    print(
        f"  {name}: Brier = {metrics_df.loc[name,'Brier']:.4f}  "
        f"Log-loss = {metrics_df.loc[name,'LogLoss']:.4f}"
    )


print(f"\n[7] Performance vs Sequence Length (binned)")
print(
    bin_df[
        [
            "bin",
            "n_proteins",
            "bindcore_AUPRC",
            "MoRFchibi_AUPRC",
            "bindcore_AUC",
            "MoRFchibi_AUC",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

print("\n[8] Cumulative performance at key length thresholds")
key_pcts = [25, 50, 75, 100]
key_lengths = [int(np.percentile(len_df["length"], p)) for p in key_pcts]
for kl in key_lengths:
    row_c = (
        cum_df[cum_df["max_length"] >= kl].iloc[0]
        if (cum_df["max_length"] >= kl).any()
        else None
    )
    if row_c is not None:
        print(
            f"  <= {kl:4d} aa ({int(row_c['n_proteins']):3d} proteins): "
            f"CORE-LIP AUPRC={row_c['bindcore_AUPRC']:.3f}  MoRFchibi AUPRC={row_c['MoRFchibi_AUPRC']:.3f}"
        )

print("\nAll figures saved to ./figures/")
print(
    "Tables: aggregate_metrics.csv | per_protein_metrics.csv | "
    "cumulative_performance_vs_length.csv | binned_performance_vs_length.csv"
)
