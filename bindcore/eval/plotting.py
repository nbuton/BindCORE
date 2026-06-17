import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# ---------------------------------------------------------------------------
# Nature Paper Style Configuration
# ---------------------------------------------------------------------------

def set_nature_style():
    """Applies a clean, publication-ready style suitable for Nature."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300, 
    })

set_nature_style()

# ---------------------------------------------------------------------------
# Shared palette and utilities
# ---------------------------------------------------------------------------

_PALETTE = [
    "#e63946", "#2176ae", "#06d6a0", "#f4a261", "#9b5de5",
    "#f15bb5", "#118ab2", "#ffd166", "#06a77d", "#ef476f",
]

def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]

def _save(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches="tight")
        print(f"Saved → {save_path}")

def _style_ax(ax: plt.Axes) -> None:
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.tick_params(direction="out", length=4, width=1.0, colors="black")
    ax.tick_params(which="minor", direction="out", length=2.5, width=0.8)

# ---------------------------------------------------------------------------
# ROC Curves (Subfigure optimized)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    records: Dict[str, "ResidueExample"],
    model_names: List[str],
    title: str = "ROC",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    handles: List[Line2D] = []

    y_true_global = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
    y_mask = y_true_global != -1
    y_true_global = y_true_global[y_mask]

    for i, name in enumerate(model_names):
        color = _color(i)
        lw = 1.6 if i == 0 else 1.1 
        try:
            y_score = np.concatenate([r.scores[name] for r in records.values()])
            y_score = y_score[y_mask]
            fpr, tpr, _ = roc_curve(y_true_global, y_score)
            auc = roc_auc_score(y_true_global, y_score)
            
            ax.plot(fpr, tpr, color=color, lw=lw, alpha=0.75, zorder=10 - i)
            # Added '\n' to put the metric on the next line
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}\n(AUC={auc:.2f})")
            )
        except ValueError as exc:
            print(f"[plot_roc_curves] Skipping {name}: {exc}")

    # No-skill line
    ax.plot([0, 1], [0, 1], color="#888888", lw=1.0, ls="--", zorder=1)
    handles.append(Line2D([], [], color="#888888", lw=1.0, ls="--", label=f"Chance\n(AUC=0.50)"))
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    
    if title:
        ax.set_title(title, fontweight="bold", pad=8)
        
    # labelspacing=0.8 adds breathing room between the separate multi-line models
    ax.legend(
        handles=handles, 
        loc="best", 
        frameon=True, 
        facecolor="white", 
        edgecolor="none", 
        framealpha=0.9,
        labelspacing=0.8 
    )
    _style_ax(ax)
    
    fig.tight_layout()
    _save(fig, save_path)
    return fig

# ---------------------------------------------------------------------------
# Precision-Recall Curves (Subfigure optimized)
# ---------------------------------------------------------------------------

def plot_pr_curves(
    records: Dict[str, "ResidueExample"],
    model_names: List[str],
    title: str = "PR-AUC",
    save_path: Optional[str | Path] = None,
    interpolate: bool = True,
) -> plt.Figure:
    
    y_true_global = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
    y_mask = y_true_global != -1
    y_true_global = y_true_global[y_mask]
    pos_rate = float(y_true_global.mean())

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    handles: List[Line2D] = []

    for i, name in enumerate(model_names):
        color = _color(i)
        lw = 1.6 if i == 0 else 1.1
        try:
            y_score = np.concatenate([r.scores[name] for r in records.values()])
            y_score = y_score[y_mask]
            precision, recall, _ = precision_recall_curve(y_true_global, y_score)
            ap = average_precision_score(y_true_global, y_score)
            
            if interpolate:
                precision = np.maximum.accumulate(precision)
                
            ax.plot(recall, precision, color=color, lw=lw, alpha=0.75, zorder=10 - i)
            # Added '\n' to put the metric on the next line
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}\n(AP={ap:.2f})")
            )
        except ValueError as exc:
            print(f"[plot_pr_curves] Skipping {name}: {exc}")

    # No-skill baseline
    ax.axhline(pos_rate, color="#888888", lw=1.0, ls="--", zorder=1)
    handles.append(Line2D([], [], color="#888888", lw=1.0, ls="--", label=f"Random\n(AP={pos_rate:.2f})"))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    
    if title:
        ax.set_title(title, fontweight="bold", pad=8)
        
    ax.legend(
        handles=handles, 
        loc="best", 
        frameon=True, 
        facecolor="white", 
        edgecolor="none", 
        framealpha=0.9,
        labelspacing=0.8
    )
    _style_ax(ax)
    
    fig.tight_layout()
    _save(fig, save_path)
    return fig

# ---------------------------------------------------------------------------
# Metrics bar chart (Wide panel optimized)
# ---------------------------------------------------------------------------

_BAR_PANELS = [
    ("mcc", "MCC", False),
    ("f1", "F1 Score", False),
    ("avg_precision", "Avg. Precision", False),
    ("brier_score", "Brier Score (↓)", True),
]

def plot_metrics_bar(
    results: List[Dict],
    title: str = "Model Performance",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    
    model_names = [r["model"] for r in results]
    colors = [_color(i) for i in range(len(results))]
    x = np.arange(len(model_names))

    ncols = 4 
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 3.2)) 
    axes = np.array(axes).flatten()

    for ax, (key, ylabel, invert) in zip(axes, _BAR_PANELS):
        values = [r[key] for r in results]
        bars = ax.bar(
            x, values, width=0.6, color=colors, edgecolor="none", zorder=3
        )
        
        for bar, val in zip(bars, values):
            y_offset = 0.02 * max(values)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (y_offset if val >= 0 else -y_offset),
                f"{val:.2f}", 
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
                color="#333333",
            )
            
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontweight="bold")
        
        y_min = min(0.0, min(values)) - 0.05
        y_max = max(values) * 1.15 if max(values) > 0 else 0.1
        ax.set_ylim(y_min, y_max)
        
        ax.axhline(0, color="black", linewidth=1.0, zorder=2)
        _style_ax(ax)
        
        if invert:
            ax.invert_yaxis()

    for ax in axes[len(_BAR_PANELS):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontweight="bold", y=1.05)
        
    fig.subplots_adjust(wspace=0.4) 
    fig.tight_layout()
    _save(fig, save_path)
    return fig