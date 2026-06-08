"""
bindcore/data/io.py
-------------------
Data preparation, parsing, feature extraction, and I/O utilities.
"""

from __future__ import annotations

import csv
import yaml
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Adjust this import based on your exact structure
from bindcore.data.datasets import AA_TO_INT, ProteinDataset
from bindcore.eval.structures import ResidueExample

# Allow reading large CSV fields for massive protein sequences
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# 1. Dataset I/O (FASTA-like & CLIP format)
# ===========================================================================


def read_protein_data(file_path: str | Path) -> pd.DataFrame:
    """Read a CLIP-format dataset file into a DataFrame."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            line1 = f.readline().strip()
            if not line1:
                break
            protein_id = line1.lstrip(">")
            sequence = f.readline().strip()
            annotations = f.readline().strip()
            data.append(
                {
                    "protein_id": protein_id,
                    "sequence": sequence,
                    "LIP_annotations": annotations,
                }
            )
    return pd.DataFrame(data)


def filter_protein_file(
    input_path: str | Path,
    protein_ids: list[str],
    output_path: str | Path,
) -> None:
    """Write a subset of a CLIP-format file, keeping only the given protein IDs."""
    target_ids = {f">{pid.strip()}" for pid in protein_ids}

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        while True:
            header = infile.readline()
            sequence = infile.readline()
            mask = infile.readline()
            if not header:
                break
            if header.strip() in target_ids:
                outfile.write(header)
                outfile.write(sequence)
                outfile.write(mask)

    print(f"Filtered file saved to: {output_path}")


# ===========================================================================
# 2. Evaluation Parsers (Truth & Predictions)
# ===========================================================================


def _read_blocks(path: str | Path) -> List[List[str]]:
    """Split a FASTA-like file into header+body blocks."""
    blocks: List[List[str]] = []
    current: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    blocks.append(current)
                current = [line]
            else:
                current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_binary_string(s: str) -> np.ndarray:
    return np.fromiter(
        (1 if c == "1" else -1 if c == "-" else 0 for c in s.strip()), dtype=np.int8
    )


def _parse_prob_string(s: str) -> np.ndarray:
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in s.split(",") if x], dtype=np.float64)


def _parse_binary_csv_string(s: str) -> np.ndarray:
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.int8)
    return np.array([int(x) for x in s.split(",") if x], dtype=np.int8)


def parse_truth_file(path: str | Path) -> Dict[str, ResidueExample]:
    """Parse a FASTA-like ground-truth file into ResidueExample objects."""
    records: Dict[str, ResidueExample] = {}
    for block in _read_blocks(path):
        if len(block) < 3:
            raise ValueError(f"Malformed truth block (expected ≥3 lines): {block}")
        protein_id = block[0][1:].strip()
        sequence = block[1].strip()
        y_true = _parse_binary_string("".join(block[2:]).strip())
        if len(sequence) != len(y_true):
            raise ValueError(
                f"Length mismatch for {protein_id}: seq={len(sequence)}, labels={len(y_true)}"
            )
        records[protein_id] = ResidueExample(protein_id, sequence, y_true)
    return records


def parse_prediction_csv(
    path: str | Path,
    records: Dict[str, ResidueExample],
    model_name: str,
) -> None:
    """Load per-residue predictions from a CSV into existing ResidueExample objects."""
    required = {"protein_id", "length", "predictions", "binary_predictions"}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            pid = row["protein_id"].strip()
            if pid not in records:
                continue
            expected_len = int(row["length"])
            scores = _parse_prob_string(row["predictions"])
            binary = _parse_binary_csv_string(row["binary_predictions"])

            if len(scores) != expected_len or len(binary) != expected_len:
                raise ValueError(
                    f"Length mismatch for {pid} in {path}: expected {expected_len}, "
                    f"got scores={len(scores)}, binary={len(binary)}"
                )
            records[pid].add_prediction(model_name, scores, binary)

    missing = [pid for pid in records if model_name not in records[pid].scores]
    if missing:
        raise ValueError(
            f"Model '{model_name}' missing predictions for {len(missing)} proteins. Missing: {missing}"
        )


# ===========================================================================
# 3. Feature Extraction & Statistics
# ===========================================================================


def prepare_data(
    df: pd.DataFrame,
    h5_data,
    scalar_features: list[str],
    local_features: list[str],
    pairwise_features: list[str],
    aa_to_int_dict: dict[str, int] = AA_TO_INT,
):
    """Extract and organise features from an HDF5 file for all proteins in df."""
    list_ids, X_scalar_list, X_local_list = [], [], []
    X_pairwise_list, seq_enc_list, y_list = [], [], []
    # Check for missing protein IDs before starting the loop
    if missing_pids := set(df["protein_id"]) - set(h5_data.keys()):
        print(
            f"Warning: The following {len(missing_pids)} protein IDs are missing from h5_data: {missing_pids}"
        )

    for _, row in df.iterrows():
        pid = row["protein_id"]
        seq_enc = np.array(
            [aa_to_int_dict.get(aa, 0) for aa in row["sequence"]], dtype=np.int64
        )
        scalar_feats = np.array(
            [h5_data[pid][f][()] for f in scalar_features], dtype=np.float32
        )
        local_feats = np.stack([h5_data[pid][f][()] for f in local_features], axis=0)

        if pairwise_features:
            pairwise_feats = np.stack(
                [h5_data[pid][f][()] for f in pairwise_features], axis=0
            )
        else:
            pairwise_feats = np.empty((0,), dtype=np.float32)

        # Convert your list, mapping '-' to -1
        mapping = {"0": 0, "1": 1, "-": -1}
        labels = np.array(
            [mapping[c] for c in row["LIP_annotations"]], dtype=np.float32
        )

        X_scalar_list.append(scalar_feats)
        X_local_list.append(local_feats)
        X_pairwise_list.append(pairwise_feats)
        seq_enc_list.append(seq_enc)
        y_list.append(labels)
        list_ids.append(pid)

    return X_scalar_list, X_local_list, X_pairwise_list, seq_enc_list, y_list, list_ids


def get_all_feature_stats(X_scalar_list, X_local_list, X_pairwise_list):
    """Computes means and stds for Scalar, Local, and Pairwise features."""
    stats = {}

    # Scalar Stats
    X_scalar_matrix = np.stack(X_scalar_list)
    s_scaler = StandardScaler().fit(X_scalar_matrix)
    stats["scalar"] = {
        "means": torch.from_numpy(s_scaler.mean_).float(),
        "stds": torch.from_numpy(s_scaler.scale_).float(),
    }

    # Local Stats
    X_local_flat = np.concatenate([arr.T for arr in X_local_list], axis=0)
    l_scaler = StandardScaler().fit(X_local_flat)
    stats["local"] = {
        "means": torch.from_numpy(l_scaler.mean_).float(),
        "stds": torch.from_numpy(l_scaler.scale_).float(),
    }

    # Pairwise Stats
    if X_pairwise_list and X_pairwise_list[0].size > 0:
        X_pair_flat = np.concatenate(
            [
                arr.transpose(1, 2, 0).reshape(-1, arr.shape[0])
                for arr in X_pairwise_list
            ],
            axis=0,
        )
        p_scaler = StandardScaler().fit(X_pair_flat)
        stats["pairwise"] = {
            "means": torch.from_numpy(p_scaler.mean_).float(),
            "stds": torch.from_numpy(p_scaler.scale_).float(),
        }
    else:
        stats["pairwise"] = {"means": torch.tensor([]), "stds": torch.tensor([])}

    return stats


# ===========================================================================
# 4. Clustering (External Tool Dispatch)
# ===========================================================================

def cluster_sequences_mmseqs2(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    id_col: str = "id",
    output_file: str = "data/mmseqs2_cluster.yaml",  # Changed extension to .yaml
    seq_identity: float = 0.3,
) -> dict:
    """Cluster sequences using MMseqs2 at a given sequence identity threshold."""
    # --- Cache Verification Logic ---
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            # safe_load automatically restores integer keys cleanly
            cached_dict = yaml.safe_load(f) or {}

        # Flatten all sequence IDs in the cached dictionary to check coverage
        cached_ids = set(
            seq_id for seq_list in cached_dict.values() for seq_id in seq_list
        )
        missing_ids = set(df[id_col]) - cached_ids

        if not missing_ids:
            print(f"[clustering] {output_file} exists and contains all IDs. Loading.")
            return cached_dict
        else:
            print(
                f"[clustering] {len(missing_ids)} IDs missing from cache. Re-clustering..."
            )
    # If the cache file for cluster does not exists
    accept_reclustering = input("Do you want to recluster?(y/n) This will change the validation set order")
    if accept_reclustering.lower()=='y' or accept_reclustering.lower()=='yes':
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "input.fasta")
            db_path = os.path.join(tmpdir, "seqdb")
            cluster_db = os.path.join(tmpdir, "clusterdb")
            tmp_path = os.path.join(tmpdir, "tmp")
            tsv_path = os.path.join(tmpdir, "clusters.tsv")

            with open(fasta_path, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    f.write(f">{row[id_col]}\n{row[sequence_col]}\n")

            subprocess.run(
                ["mmseqs", "createdb", fasta_path, db_path], check=True, capture_output=True
            )

            subprocess.run(
                [
                    "mmseqs",
                    "cluster",
                    db_path,
                    cluster_db,
                    tmp_path,
                    "--min-seq-id",
                    str(seq_identity),
                    "-c",
                    "0.8",
                    "--cov-mode",
                    "0",
                    "--cluster-mode",
                    "1",
                    "--threads",
                    "4",
                ],
                check=True,
                capture_output=True,
            )

            subprocess.run(
                ["mmseqs", "createtsv", db_path, db_path, cluster_db, tsv_path],
                check=True,
                capture_output=True,
            )

            cluster_df = pd.read_csv(
                tsv_path,
                sep="\t",
                header=None,
                names=["cluster_representative", id_col],
            )

        reps = cluster_df["cluster_representative"].unique()
        rep_to_idx = {rep: idx for idx, rep in enumerate(reps)}
        cluster_df["cluster"] = cluster_df["cluster_representative"].map(rep_to_idx)

        # Group by unique cluster integer mapping to a list of original sequence IDs
        cluster_dict = cluster_df.groupby("cluster")[id_col].apply(list).to_dict()

        # Save to clean human-readable YAML format
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(cluster_dict, f, default_flow_style=False)

        print(f"[clustering] Done. {len(cluster_dict)} clusters found.")
        print(f"[clustering] Saved to {output_file}")

        return cluster_dict
    else:
        raise RuntimeError("Cannot continue without re-clustering")


def ham_mask_val_labels(
    val_indices: list[int],
    train_indices: list[int],
    dataset: ProteinDataset,
    min_len: int = 10,
    min_identity: float = 0.8,
) -> None:
    """
    HAM-equivalent: for each val sequence, find aligned regions >= min_len residues
    with >= min_identity to any training sequence using MMseqs2 easy-search.
    Masks those residue positions in-place in dataset.labels (sets to -1).
    Only non-MoRF residues (label=0) are masked, MoRF residues (label=1) are left untouched,
    mirroring the HAM logic in the paper.
    """
    ids = dataset.ids
    seqs = dataset.seq_enc_list  # encoded, we need raw strings

    # Decode sequences back to amino acid strings
    IDX_TO_AA = {idx: aa for aa, idx in AA_TO_INT.items()}

    def decode_seq(enc: np.ndarray) -> str:
        return "".join(IDX_TO_AA.get(int(i), "X") for i in enc)

    val_ids = [ids[i] for i in val_indices]
    train_ids = [ids[i] for i in train_indices]
    val_seqs = [decode_seq(seqs[i]) for i in val_indices]
    train_seqs = [decode_seq(seqs[i]) for i in train_indices]

    with tempfile.TemporaryDirectory() as tmpdir:
        query_fasta = os.path.join(tmpdir, "val.fasta")
        target_fasta = os.path.join(tmpdir, "train.fasta")
        result_tsv = os.path.join(tmpdir, "hits.tsv")
        tmp_path = os.path.join(tmpdir, "tmp")

        with open(query_fasta, "w") as f:
            for pid, seq in zip(val_ids, val_seqs):
                f.write(f">{pid}\n{seq}\n")

        with open(target_fasta, "w") as f:
            for pid, seq in zip(train_ids, train_seqs):
                f.write(f">{pid}\n{seq}\n")

        # Columns: query, target, identity, alignment_length,
        #          mismatches, gap_opens, qstart, qend, tstart, tend, evalue, bitscore
        subprocess.run(
            [
                "mmseqs",
                "easy-search",
                query_fasta,
                target_fasta,
                result_tsv,
                tmp_path,
                "--min-seq-id",
                str(min_identity),
                "-c",
                "0.0",  # no coverage filter, we filter by length below
                "--cov-mode",
                "2",
                "--format-output",
                "query,target,fident,alnlen,qstart,qend",
                "--threads",
                "4",
                "-e",
                "10",  # loose e-value to catch short local hits
            ],
            check=True,
            capture_output=True,
        )

        hits = pd.read_csv(
            result_tsv,
            sep="\t",
            header=None,
            names=["query", "target", "fident", "alnlen", "qstart", "qend"],
        )

    # Filter to hits that are >= min_len residues and >= min_identity
    hits = hits[(hits["alnlen"] >= min_len) & (hits["fident"] >= min_identity)]

    if hits.empty:
        print("[HAM] No homologous regions found in val sequences.")
        return

    # Build a map: val_id -> list of (qstart, qend) 0-indexed intervals to mask
    id_to_val_idx = {ids[i]: i for i in val_indices}
    regions_to_mask: dict[str, list[tuple[int, int]]] = {}
    for _, row in hits.iterrows():
        qid = row["query"]
        # MMseqs2 uses 1-based indexing
        qstart = int(row["qstart"]) - 1
        qend = int(row["qend"])  # exclusive when converted from 1-based inclusive
        regions_to_mask.setdefault(qid, []).append((qstart, qend))

    # Apply masking in-place: only mask label=0 (non-MoRF) residues
    masked_residues = 0
    for qid, regions in regions_to_mask.items():
        if qid not in id_to_val_idx:
            continue
        idx = id_to_val_idx[qid]
        label_arr = dataset.labels[idx]  # np.ndarray, values in {-1, 0, 1}
        for qstart, qend in regions:
            for pos in range(qstart, min(qend, len(label_arr))):
                if label_arr[pos] == 0:
                    label_arr[pos] = -1
                    masked_residues += 1
        dataset.labels[idx] = label_arr

    print(
        f"[HAM] Masked {masked_residues} non-MoRF residues across "
        f"{len(regions_to_mask)} val sequences."
    )
