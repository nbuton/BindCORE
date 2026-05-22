"""
bindcore/data/conformation.py
----------------------------
Logic for processing molecular dynamics trajectories and extracting features.
"""

import os
import sys
import ctypes
import shutil
import logging
import warnings
from pathlib import Path

import h5py
import mdtraj as md
import numpy as np
from EnsembleMDP.analysis.orchestrator import ProteinAnalyzer

# Suppress noisy library warnings
warnings.filterwarnings("ignore", module="MDAnalysis")
warnings.filterwarnings("ignore", module="mdtraj")


class SuppressCStdout:
    """Context manager that redirects C-level stdout/stderr to /dev/null."""

    def __init__(self):
        try:
            self.libc = ctypes.CDLL(None)
        except Exception:
            self.libc = ctypes.cdll.msvcrt

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.old_stdout_fd = os.dup(sys.stdout.fileno())
        self.old_stderr_fd = os.dup(sys.stderr.fileno())
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull_fd, sys.stdout.fileno())
        os.dup2(self.devnull_fd, sys.stderr.fileno())

    def __exit__(self, *_):
        try:
            self.libc.fflush(None)
        except Exception:
            pass
        os.dup2(self.old_stdout_fd, sys.stdout.fileno())
        os.dup2(self.old_stderr_fd, sys.stderr.fileno())
        os.close(self.old_stdout_fd)
        os.close(self.old_stderr_fd)
        os.close(self.devnull_fd)


def convert_trajectory_format(folder_path: Path) -> None:
    """Rename topology and convert DCD to XTC within a protein folder."""
    old_pdb = folder_path / "aa_topology.pdb"
    new_pdb = folder_path / "top_AA.pdb"
    if old_pdb.exists():
        shutil.move(str(old_pdb), str(new_pdb))

    old_dcd = folder_path / "aa_traj.dcd"
    new_xtc = folder_path / "traj_AA.xtc"

    if old_dcd.exists() and new_pdb.exists():
        with SuppressCStdout():
            traj = md.load(str(old_dcd), top=str(new_pdb))
            traj.save_xtc(str(new_xtc))


def process_single_protein(
    protein_dir: Path, pdb_name: str, xtc_name: str, convert_dcd: bool = False
) -> tuple[str, dict]:
    """
    Worker function: Handles conversion and feature extraction for one protein.
    """
    # Use .name to get the actual folder ID (e.g., A0A0H3JS52)
    protein_id = protein_dir.name
    pdb_path = protein_dir / pdb_name
    xtc_path = protein_dir / xtc_name

    # 1. Handle DCD to XTC conversion if requested
    if not xtc_path.exists() and convert_dcd:
        dcd_path = xtc_path.with_suffix(".dcd")
        if dcd_path.exists():
            try:
                # Load DCD and save as XTC to local folder
                traj = md.load_dcd(str(dcd_path), top=str(pdb_path))
                traj.save_xtc(str(xtc_path))
            except Exception as e:
                return protein_id, {"error": f"DCD conversion failed: {e}"}
        else:
            return protein_id, {
                "error": f"No XTC found and no DCD available for conversion"
            }

    # 2. Final check before analysis
    if not xtc_path.exists():
        return protein_id, {"error": f"Missing trajectory: {xtc_path.name}"}
    if not pdb_path.exists():
        return protein_id, {"error": f"Missing topology: {pdb_path.name}"}

    # 3. Compute properties using your ProteinAnalyzer
    try:
        analyzer = ProteinAnalyzer(str(pdb_path), str(xtc_path))
        properties = analyzer.compute_all(
            sasa_n_sphere=1600,
            contact_cutoff=8.0,
            scaling_min_sep=5,
        )

        if properties is None:
            return protein_id, {"error": "Analyzer returned None"}

        return protein_id, properties

    except Exception as e:
        return protein_id, {"error": f"Extraction failed: {str(e)}"}


def save_properties_to_h5(dico_properties: dict, output_filepath: str | Path) -> None:
    """
    Saves or updates nested feature dictionary to an HDF5 file.
    If the file exists, it adds new keys or updates existing ones.
    """
    # Use "a" mode: Read/write if exists, create otherwise
    with h5py.File(output_filepath, "a") as h5f:
        for protein_id, props in dico_properties.items():

            # Get existing group or create a new one
            if protein_id in h5f:
                grp = h5f[protein_id]
            else:
                grp = h5f.create_group(protein_id)

            for name, value in props.items():
                val_arr = (
                    np.array(value) if not isinstance(value, np.ndarray) else value
                )

                # Check if the dataset already exists in the group
                if name in grp:
                    # HDF5 datasets cannot be resized/overwritten easily if shapes differ.
                    # Usually, it's safest to delete and recreate if you want to update.
                    del grp[name]

                # Create the dataset
                if val_arr.ndim > 0:
                    grp.create_dataset(
                        name, data=val_arr, compression="gzip", compression_opts=4
                    )
                else:
                    grp.create_dataset(name, data=val_arr)
