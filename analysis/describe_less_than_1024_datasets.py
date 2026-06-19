"""
Describe the less-than-1024 LIP and MoRF datasets.

The input files use a 3-line FASTA-like format:
    >protein_id
    SEQUENCE
    annotation_string

Annotations are interpreted as:
    1: positive residue
    0: negative residue
    -: unknown / masked residue
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

DEFAULT_DATASETS = {
    "MoRF": {
        "train": Path("data/MoRF_dataset/train.txt"),
        "test": Path("data/MoRF_dataset/test.txt"),
    },
    "LIP": {
        "train": Path("data/LIP_dataset/TR1000_less_than_1024.txt"),
        "test": Path("data/LIP_dataset/TE440_less_than_1024.txt"),
    },
}

LABELS = {
    "positive": "1",
    "negative": "0",
    "unknown": "-",
}


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    annotations: str


def read_records(file_path):
    """
    Parses a 3-line block file (Header, Sequence, Labels).
    Includes a debug print to track down exactly which file is being opened.
    """
    from pathlib import Path
    from types import SimpleNamespace

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Warning: File not found at {file_path}")
        return []

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    parsed_records = []

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break

        header = lines[i]
        sequence = lines[i + 1]
        labels = lines[i + 2]

        if not header.startswith(">") or len(sequence) != len(labels):
            continue

        record = SimpleNamespace(
            id=header[1:],
            protein_id=header[1:],
            sequence=sequence,
            annotations=labels,
            length=len(sequence),
        )
        parsed_records.append(record)

    # DEBUG LINE: This will reveal the exact path and count in your terminal!
    print(
        f"[DEBUG] read_records loaded {len(parsed_records)} proteins from: {file_path.resolve()}"
    )

    return parsed_records


def count_segments(annotation: str, label: str) -> tuple[int, list[int]]:
    segment_lengths: list[int] = []
    run_length = 0
    for char in annotation:
        if char == label:
            run_length += 1
        elif run_length:
            segment_lengths.append(run_length)
            run_length = 0
    if run_length:
        segment_lengths.append(run_length)
    return len(segment_lengths), segment_lengths


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def pct(part: int, total: int) -> float:
    return (100.0 * part / total) if total else 0.0


def summarize_split(
    dataset: str,
    split: str,
    path: Path,
    max_length: int | None,
) -> dict[str, str | int | float]:
    all_records = read_records(path)
    records = [
        record
        for record in all_records
        if max_length is None or len(record.sequence) < max_length
    ]

    lengths = [len(record.sequence) for record in records]
    length_stats = summarize_numbers(lengths)

    label_counts = {
        label_name: sum(record.annotations.count(label) for record in records)
        for label_name, label in LABELS.items()
    }
    total_residues = sum(lengths)
    labeled_residues = label_counts["positive"] + label_counts["negative"]

    proteins_with_positive = sum(
        1 for record in records if LABELS["positive"] in record.annotations
    )
    proteins_with_unknown = sum(
        1 for record in records if LABELS["unknown"] in record.annotations
    )
    positive_segment_lengths: list[int] = []
    positive_segments = 0
    for record in records:
        segment_count, segment_lengths = count_segments(
            record.annotations, LABELS["positive"]
        )
        positive_segments += segment_count
        positive_segment_lengths.extend(segment_lengths)
    segment_stats = summarize_numbers(positive_segment_lengths)

    unexpected_labels = sorted(
        {
            char
            for record in records
            for char in record.annotations
            if char not in LABELS.values()
        }
    )

    return {
        "dataset": dataset,
        "split": split,
        "path": str(path),
        "source_proteins": len(all_records),
        "proteins": len(records),
        "excluded_by_length": len(all_records) - len(records),
        "residues": total_residues,
        "positive_residues": label_counts["positive"],
        "positive_pct_all_residues": pct(label_counts["positive"], total_residues),
        "positive_pct_labeled_residues": pct(
            label_counts["positive"], labeled_residues
        ),
        "negative_residues": label_counts["negative"],
        "negative_pct_all_residues": pct(label_counts["negative"], total_residues),
        "unknown_residues": label_counts["unknown"],
        "unknown_pct_all_residues": pct(label_counts["unknown"], total_residues),
        "proteins_with_positive": proteins_with_positive,
        "proteins_with_positive_pct": pct(proteins_with_positive, len(records)),
        "proteins_with_unknown": proteins_with_unknown,
        "proteins_with_unknown_pct": pct(proteins_with_unknown, len(records)),
        "length_min": length_stats["min"],
        "length_median": length_stats["median"],
        "length_mean": length_stats["mean"],
        "length_max": length_stats["max"],
        "positive_segments": positive_segments,
        "positive_segment_length_min": segment_stats["min"],
        "positive_segment_length_median": segment_stats["median"],
        "positive_segment_length_mean": segment_stats["mean"],
        "positive_segment_length_max": segment_stats["max"],
        "unexpected_annotation_symbols": "".join(unexpected_labels),
    }


def add_overlap_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str]]:
    overlap_rows: list[dict[str, str]] = []
    for dataset, paths in DEFAULT_DATASETS.items():
        train_records = read_records(paths["train"])
        test_records = read_records(paths["test"])
        train_ids = {record.protein_id for record in train_records}
        test_ids = {record.protein_id for record in test_records}
        overlap = train_ids & test_ids
        overlap_rows.append(
            {
                "dataset": dataset,
                "train_test_id_overlap": str(len(overlap)),
                "example_overlapping_ids": ", ".join(sorted(overlap)[:5]),
            }
        )
    return overlap_rows


def format_value(value: str | int | float) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_markdown_table(
    rows: list[dict[str, str | int | float]], columns: list[str]
) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row[column]) for column in columns) + " |")


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe LIP and MoRF train/test datasets for paper reporting."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Analyze only proteins with sequence length strictly below this value. "
        "Use 0 to disable length filtering.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional path to write the full summary table as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_length = None if args.max_length == 0 else args.max_length

    rows: list[dict[str, str | int | float]] = []
    for dataset, split_paths in DEFAULT_DATASETS.items():
        for split, path in split_paths.items():
            rows.append(summarize_split(dataset, split, path, max_length))

    filter_label = f"length < {max_length}" if max_length is not None else "all lengths"
    print(f"\nDataset summary ({filter_label})\n")
    print_markdown_table(
        rows,
        [
            "dataset",
            "split",
            "source_proteins",
            "proteins",
            "excluded_by_length",
            "residues",
            "positive_residues",
            "positive_pct_all_residues",
            "negative_residues",
            "negative_pct_all_residues",
            "unknown_residues",
            "unknown_pct_all_residues",
        ],
    )

    print("\nAdditional paper-friendly descriptors\n")
    print_markdown_table(
        rows,
        [
            "dataset",
            "split",
            "proteins_with_positive",
            "proteins_with_positive_pct",
            "length_min",
            "length_median",
            "length_mean",
            "length_max",
            "positive_segments",
            "positive_segment_length_median",
            "positive_segment_length_mean",
        ],
    )

    print("\nTrain/test ID overlap in source files\n")
    print_markdown_table(
        add_overlap_rows(rows),
        ["dataset", "train_test_id_overlap", "example_overlapping_ids"],
    )

    if args.csv_output:
        write_csv(args.csv_output, rows)
        print(f"\nWrote full summary to {args.csv_output}")


if __name__ == "__main__":
    main()
