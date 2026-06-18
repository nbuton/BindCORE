#!/usr/bin/env python3
"""Plot Ray Tune best-so-far validation performance from log files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


DEFAULT_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#4D4D4D",  # gray
)


def split_csv_arg(value: str, arg_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{arg_name} must contain at least one value.")
    return items


def parse_metric_value(line: str, metric: str) -> float | None:
    """Return the metric value from a Ray table/text line, if present."""
    metric_re = re.escape(metric)
    patterns = (
        rf"\b{metric_re}\b\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        rf"[│|]\s*{metric_re}\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[│|]",
        rf"['\"]{metric_re}['\"]\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return float(match.group(1))
    return None


def parse_trial_result_blocks(lines: Iterable[str], metric: str) -> list[float]:
    """Extract final trial metric values in chronological completion order."""
    values: list[float] = []
    pending_values: dict[str, float] = {}
    completed_trials: set[str] = set()
    current_trial: str | None = None
    in_result_block = False

    for line in lines:
        result_match = re.search(r"\bTrial\s+(\S+)\s+result\b", line)
        if result_match:
            current_trial = result_match.group(1)
            in_result_block = True
            continue

        completed_match = re.search(r"\bTrial\s+(\S+)\s+completed\b", line)
        if completed_match:
            trial_name = completed_match.group(1)
            if trial_name in pending_values and trial_name not in completed_trials:
                values.append(pending_values[trial_name])
                completed_trials.add(trial_name)
            in_result_block = False
            current_trial = None
            continue

        if in_result_block and current_trial is not None:
            value = parse_metric_value(line, metric)
            if value is not None:
                pending_values[current_trial] = value
                in_result_block = False
                current_trial = None
                continue

        if in_result_block and re.search(r"\bTrial\s+\S+\s+(errored|terminated)\b", line):
            in_result_block = False
            current_trial = None

    return values or list(pending_values.values())


def parse_final_table(lines: Iterable[str], metric: str) -> list[float]:
    """Fallback parser for logs that only contain the final Ray summary table."""
    last_table_values: list[float] = []
    current_table_values: list[float] = []
    metric_column: int | None = None
    row_prefix = re.compile(r"^\s*[│|]\s*trainable_\S+")

    for line in lines:
        if metric in line and "Trial name" in line:
            fields = [field.strip() for field in re.split(r"[│|]", line) if field.strip()]
            metric_column = fields.index(metric) if metric in fields else len(fields) - 1
            current_table_values = []
            continue

        if metric_column is not None and row_prefix.search(line):
            fields = [field.strip() for field in re.split(r"[│|]", line) if field.strip()]
            if not fields or len(fields) <= metric_column:
                continue
            raw_value = fields[metric_column]
            try:
                current_table_values.append(float(raw_value))
            except ValueError:
                continue

        if metric_column is not None and line.lstrip().startswith(("╰", "+")):
            if current_table_values:
                last_table_values = current_table_values
            metric_column = None

        if "BEST TRIAL" in line:
            break

    return last_table_values


def extract_trial_values(log_path: Path, metric: str) -> list[float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    trial_values = parse_trial_result_blocks(lines, metric)
    table_values = parse_final_table(lines, metric)

    if len(table_values) > len(trial_values):
        return table_values
    if trial_values:
        return trial_values
    if table_values:
        return table_values

    raise ValueError(f"Could not find any '{metric}' trial values in {log_path}.")


def best_so_far(values: list[float], max_runs: int | None = None) -> tuple[list[int], list[float]]:
    if max_runs is not None:
        values = values[:max_runs]

    best_values: list[float] = []
    current_best = float("-inf")
    for value in values:
        current_best = max(current_best, value)
        best_values.append(current_best)

    return list(range(1, len(best_values) + 1)), best_values


def improvement_points(runs: list[int], best_values: list[float]) -> tuple[list[int], list[float]]:
    """Return coordinates where the best-so-far curve increases."""
    improvement_runs: list[int] = []
    improvement_values: list[float] = []
    previous_best = float("-inf")

    for run, value in zip(runs, best_values, strict=True):
        if value > previous_best:
            improvement_runs.append(run)
            improvement_values.append(value)
            previous_best = value

    return improvement_runs, improvement_values


def apply_nature_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (3.7, 2.65),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "axes.labelcolor": "#222222",
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_curves(
    logs: list[Path],
    labels: list[str],
    metric: str,
    output: Path,
    max_runs: int | None,
    title: str | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires Matplotlib. Install it with "
            "`pip install matplotlib` or `conda install matplotlib`."
        ) from exc

    apply_nature_style()
    fig, ax = plt.subplots()

    all_best_values: list[float] = []
    all_run_counts: list[int] = []
    for index, (log_path, label) in enumerate(zip(logs, labels, strict=True)):
        values = extract_trial_values(log_path, metric)
        runs, best_values = best_so_far(values, max_runs=max_runs)
        if not runs:
            raise ValueError(f"No usable runs found in {log_path}.")

        all_best_values.extend(best_values)
        all_run_counts.append(runs[-1])
        color = DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        ax.step(
            runs,
            best_values,
            where="post",
            color=color,
            label=label,
            zorder=2,
        )
        ax.fill_between(
            runs,
            best_values,
            step="post",
            color=color,
            alpha=0.055 if len(labels) > 1 else 0.09,
            linewidth=0,
            zorder=1,
        )
        improved_runs, improved_values = improvement_points(runs, best_values)
        ax.scatter(
            improved_runs,
            improved_values,
            s=11,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            clip_on=False,
        )
        ax.scatter(
            runs[-1],
            best_values[-1],
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            clip_on=False,
        )

        print(
            f"{label}: parsed {len(values)} trial values, "
            f"plotted {len(runs)} runs, best {metric}={best_values[-1]:.4f}"
        )

    max_observed_run = max(all_run_counts)
    x_padding = max(1.0, max_observed_run * 0.025)
    ax.set_xlim(0, max_observed_run + x_padding)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlabel("Hyperparameter trials completed")
    ax.set_ylabel(f"Best validation {metric} so far")

    if all_best_values:
        y_min = min(all_best_values)
        y_max = max(all_best_values)
        padding = max((y_max - y_min) * 0.12, 0.01)
        ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))

    ax.grid(axis="y", color="#E1E1E1", linewidth=0.55, alpha=0.9)
    ax.grid(axis="x", color="#F0F0F0", linewidth=0.4, alpha=0.45)
    ax.tick_params(axis="both", direction="out")
    ax.margins(y=0.04)

    if title:
        ax.set_title(title, fontsize=8, pad=6, color="#222222")

    if len(labels) > 1:
        ax.legend(loc="lower right", handlelength=1.9, borderaxespad=0.4)

    fig.tight_layout(pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-style best-so-far curve from one or more "
            "Ray Tune logs."
        )
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Comma-separated Ray Tune log file paths.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated curve labels, one per log file.",
    )
    parser.add_argument(
        "--metric",
        default="PR-AUC",
        help="Metric name to extract from each trial result block. Default: PR-AUC.",
    )
    parser.add_argument(
        "--output",
        default="best_so_far_pr_auc.pdf",
        help="Output figure path. Use .pdf or .svg for manuscript figures.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Optional maximum number of completed trials to show. "
            "By default, all parsed runs are plotted."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional small panel title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = [Path(item).expanduser() for item in split_csv_arg(args.logs, "--logs")]
    labels = split_csv_arg(args.labels, "--labels")

    if len(logs) != len(labels):
        raise SystemExit(
            f"--logs has {len(logs)} value(s), but --labels has {len(labels)}."
        )

    missing_logs = [str(path) for path in logs if not path.is_file()]
    if missing_logs:
        raise SystemExit("Log file(s) not found: " + ", ".join(missing_logs))

    if args.max_runs is not None and args.max_runs <= 0:
        raise SystemExit("--max-runs must be a positive integer.")

    plot_curves(
        logs=logs,
        labels=labels,
        metric=args.metric,
        output=Path(args.output).expanduser(),
        max_runs=args.max_runs,
        title=args.title,
    )
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
