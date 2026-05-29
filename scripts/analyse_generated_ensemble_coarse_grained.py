#!/usr/bin/env python3
"""
Conformational Ensemble Quality Analysis – Source-Aware
=======================================================

This script applies quality checks that are appropriate for the internal
coordinate representation of each source, rather than blindly applying
Cα geometry checks to all trajectories.

Internal representations
------------------------
AF-CALVADOS  –  one COM-bead per residue (CALVADOS force field + AF2 Gō restraints)
                Folded residues  : bead at centre-of-mass of ALL heavy atoms
                Disordered residues: CALVADOS bead (~Cα equivalent, r₀ = 3.8 Å)
                → Cα bond / angle / clash checks are NOT applicable
                → frame_fully_ok  =  frame_shape_ok  (Rg / N^ν plausibility)
                Reference: von Bülow et al. (2025), bioRxiv 10.1101/2025.10.19.683306

IDPFold2     –  one Cα per residue (flow-matching + Mixture-of-Experts, AlphaFold3
                diffusion module backbone).  All Cα geometry checks are applied.
                → frame_fully_ok  =  bond_ok AND angle_ok AND clash_ok
                Reference: Zhu et al. (2026), bioRxiv 10.64898/2026.01.14.699584
Click and Read

STARLING     –  one Cα per residue (IDP-focused generative model).
                All Cα geometry checks are applied.
                → frame_fully_ok  =  bond_ok AND angle_ok AND clash_ok
                Reference: Novak et al. (2025), Nature / bioRxiv

Universal metrics  (valid for ALL sources, enable cross-method comparison)
    Rg, Ree, Rg / N^ν (Flory), mean pairwise bead distance

Geometry metrics  (Cα sources only: IDPFold2, STARLING)
    Cα–Cα bond lengths, Cα–Cα–Cα angles, pseudo-dihedrals, non-bonded clashes

Design notes
------------
- geometry_applicable column (0/1) marks which rows carry valid geometry data
- NaN geometry fields for COM sources; frame_bond/angle/clash_ok = True  (not
  failed, just not assessed)
- Shape (Rg / N^ν) is kept as a SOFT metric for all sources; it does NOT enter
  frame_fully_ok for Cα sources (consistent with original script convention)
- Cross-source comparison should use frame_shape_ok and shape descriptors

Usage
-----
python analyse_generated_ensemble_cg.py \\
    --input_dir  data/conformational_ensemble/AF-CALVADOS \\
    --source     AF-CALVADOS \\
    --output_dir data/conformations_analysis \\
    --limit_n_prot 2000 \\
    --workers    8
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("MDAnalysis not found.  pip install MDAnalysis")

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE REPRESENTATION REGISTRY
#  To add a new source: add an entry here and a case in _get_files().
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_REPR: dict[str, dict] = {
    "AF-CALVADOS": {
        "bead_type": "COM",
        "run_geometry": False,
        "frame_ok_basis": "shape",
        "description": (
            "Centre-of-mass (COM) bead simulation (CALVADOS 3 force field + "
            "AF2-derived pLDDT/PAE Gō restraints). Each residue is represented "
            "by ONE bead: the COM of all heavy atoms for folded residues (high "
            "pLDDT), or a CALVADOS bead (~Cα, r₀ = 3.8 Å) for disordered "
            "residues. Cα bond / angle / clash checks are NOT applicable – "
            "beads are not Cα atoms."
        ),
    },
    "IDPFold2": {
        "bead_type": "CA",
        "run_geometry": True,
        "frame_ok_basis": "geometry",
        "description": (
            "Cα-trace generative model (flow matching + Mixture-of-Experts "
            "architecture built on the AlphaFold3 diffusion module). One Cα "
            "per residue is generated; all-atom coordinates can be recovered "
            "via cg2all backmapping. All Cα geometry checks are applied."
        ),
    },
    "STARLING": {
        "bead_type": "CA",
        "run_geometry": True,
        "frame_ok_basis": "geometry",
        "description": (
            "Cα-trace IDP-focused generative model (STARLING). Optimised for "
            "intrinsically disordered regions; one Cα per residue. "
            "All Cα geometry checks are applied."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (Cα geometry thresholds – applicable to IDPFold2 / STARLING)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Sequential Cα–Cα virtual bond ─────────────────────────────────────────
CA_BOND_IDEAL = 3.81  # Å   standard peptide geometry
CA_BOND_MIN = 3.20  # Å   hard lower bound
CA_BOND_SOFT_LO = 3.50  # Å   soft lower bound
CA_BOND_SOFT_HI = 4.10  # Å   soft upper bound
CA_BOND_MAX = 4.50  # Å   hard upper bound (chain break above this)

MAX_BOND_BAD_RATIO = 0.01  # frame passes when ≤ 1 % of bonds violate hard limits

# ── 2. Cα–Cα–Cα pseudo-bond angles ───────────────────────────────────────────
CA_ANGLE_MIN = 70.0  # °
CA_ANGLE_MAX = 160.0  # °

MAX_ANGLE_BAD_RATIO = 0.02  # frame passes when ≤ 2 % of angles are outside limits

# ── 3. Pseudo-dihedrals ───────────────────────────────────────────────────────
CA_DIHEDRAL_EXTENDED_MIN = 150.0  # |φ| ≥ this → labelled "extended / β-like"

# ── 4. Non-bonded Cα clash ────────────────────────────────────────────────────
CA_CLASH_DIST = 3.50  # Å   hard-sphere floor for a Cα bead
CA_CLASH_SEQ_SKIP = 2  # skip pairs within ±2 positions in sequence

# ── 5–9. Shape descriptors (universal) ───────────────────────────────────────
FLORY_NU = 0.588  # Flory exponent (self-avoiding walk; IDPs)
RG_NORM_IDP_LO = 1.0  # Å   lower bound for Rg / N^ν
RG_NORM_IDP_HI = 6.0  # Å   upper bound

DEFAULT_CHUNK_SIZE = 500

# ── Per-frame CSV schema ──────────────────────────────────────────────────────
PER_FRAME_COLUMNS = [
    "protein",
    "frame",
    "n_residues",
    "n_segments",
    "geometry_applicable",  # 1 = Cα checks valid; 0 = COM-bead, checks skipped
    # Bond-length check (Cα sources only; NaN for COM sources)
    "n_bonds",
    "bonds_ok",
    "bonds_soft_ok",
    "bonds_bad",
    "bond_bad_ratio",
    "bond_mean",
    "bond_std",
    "bond_min",
    "bond_max",
    "frame_bond_ok",
    # Angle check (Cα sources only)
    "n_angles",
    "angles_bad",
    "angle_bad_ratio",
    "angle_mean",
    "angle_std",
    "frame_angle_ok",
    # Dihedral (informational; Cα sources only)
    "n_dihedrals",
    "dihedral_mean",
    "dihedral_std",
    "frac_extended",
    # Clash check (Cα sources only)
    "n_clashes",
    "frame_clash_ok",
    # Shape descriptors (UNIVERSAL – valid for all sources)
    "ree",
    "rg",
    "rg_norm",
    "mean_pairwise",
    "frame_shape_ok",
    # Overall viability (source-specific: see SOURCE_REPR[source]["frame_ok_basis"])
    "frame_fully_ok",
]


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Dihedral a–b–c–d in degrees (−180 to +180)."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / (np.linalg.norm(b2) + 1e-12)
    m1 = np.cross(n1, b2n)
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT-AWARE INDEX BUILDING
#  Restricts bond / angle / dihedral calculations to INTRA-SEGMENT consecutive
#  residues.  This prevents artificial bonds from last-residue-of-chain-A to
#  first-residue-of-chain-B in multi-chain or multi-domain topologies.
# ══════════════════════════════════════════════════════════════════════════════


def _sequential_pairs(ca_ag) -> np.ndarray:
    """(M, 2) int array of intra-segment consecutive Cα index pairs."""
    pairs: list[tuple[int, int]] = []
    for seg in ca_ag.segments:
        seg_ca = seg.atoms.select_atoms("name CA")
        if len(seg_ca) < 2:
            continue
        resids = seg_ca.resids
        indices = seg_ca.indices
        for k in range(len(resids) - 1):
            if resids[k + 1] == resids[k] + 1:
                pairs.append((indices[k], indices[k + 1]))
    return np.array(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)


def _intra_segment_triplets(ca_ag) -> list[tuple[int, int, int]]:
    """Intra-segment consecutive triplets for angle calculation."""
    triplets: list[tuple[int, int, int]] = []
    for seg in ca_ag.segments:
        seg_ca = seg.atoms.select_atoms("name CA")
        if len(seg_ca) < 3:
            continue
        resids = seg_ca.resids
        indices = seg_ca.indices
        for k in range(len(resids) - 2):
            if resids[k + 1] == resids[k] + 1 and resids[k + 2] == resids[k + 1] + 1:
                triplets.append((indices[k], indices[k + 1], indices[k + 2]))
    return triplets


def _intra_segment_quadruplets(ca_ag) -> list[tuple[int, int, int, int]]:
    """Intra-segment consecutive quadruplets for dihedral calculation."""
    quads: list[tuple[int, int, int, int]] = []
    for seg in ca_ag.segments:
        seg_ca = seg.atoms.select_atoms("name CA")
        if len(seg_ca) < 4:
            continue
        resids = seg_ca.resids
        indices = seg_ca.indices
        for k in range(len(resids) - 3):
            if (
                resids[k + 1] == resids[k] + 1
                and resids[k + 2] == resids[k + 1] + 1
                and resids[k + 3] == resids[k + 2] + 1
            ):
                quads.append(
                    (indices[k], indices[k + 1], indices[k + 2], indices[k + 3])
                )
    return quads


# ══════════════════════════════════════════════════════════════════════════════
#  PER-FRAME GEOMETRY CHECKS  (Cα sources – IDPFold2 / STARLING)
# ══════════════════════════════════════════════════════════════════════════════


def check_ca_bonds(pos: np.ndarray, pair_idx: np.ndarray) -> dict:
    """Sequential intra-segment Cα–Cα distances."""
    if len(pair_idx) == 0:
        return dict(
            n_bonds=0,
            bonds_ok=0,
            bonds_soft_ok=0,
            bonds_bad=0,
            bond_bad_ratio=float("nan"),
            bond_mean=float("nan"),
            bond_std=float("nan"),
            bond_min=float("nan"),
            bond_max=float("nan"),
            frame_bond_ok=True,
        )
    dists = np.linalg.norm(pos[pair_idx[:, 0]] - pos[pair_idx[:, 1]], axis=1)
    hard_ok = (dists >= CA_BOND_MIN) & (dists <= CA_BOND_MAX)
    soft_ok = (dists >= CA_BOND_SOFT_LO) & (dists <= CA_BOND_SOFT_HI)
    nb = len(dists)
    n_bad = int((~hard_ok).sum())
    bad_ratio = n_bad / nb
    return dict(
        n_bonds=nb,
        bonds_ok=int(hard_ok.sum()),
        bonds_soft_ok=int(soft_ok.sum()),
        bonds_bad=n_bad,
        bond_bad_ratio=round(bad_ratio, 6),
        bond_mean=round(float(dists.mean()), 4),
        bond_std=round(float(dists.std()), 4),
        bond_min=round(float(dists.min()), 4),
        bond_max=round(float(dists.max()), 4),
        frame_bond_ok=bad_ratio <= MAX_BOND_BAD_RATIO,
    )


def check_ca_angles(pos: np.ndarray, triplets: list[tuple[int, int, int]]) -> dict:
    """Cα–Cα–Cα pseudo-bond angles for intra-segment consecutive triplets."""
    if not triplets:
        return dict(
            n_angles=0,
            angles_bad=0,
            angle_bad_ratio=float("nan"),
            angle_mean=float("nan"),
            angle_std=float("nan"),
            frame_angle_ok=True,
        )
    tri = np.array(triplets, dtype=int)
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    u_ = a - b
    v_ = c - b
    denom = np.linalg.norm(u_, axis=1) * np.linalg.norm(v_, axis=1) + 1e-12
    cos_t = np.einsum("ij,ij->i", u_, v_) / denom
    angles = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
    bad = (angles < CA_ANGLE_MIN) | (angles > CA_ANGLE_MAX)
    na = len(angles)
    bad_ratio = float(bad.sum()) / na
    return dict(
        n_angles=na,
        angles_bad=int(bad.sum()),
        angle_bad_ratio=round(bad_ratio, 6),
        angle_mean=round(float(angles.mean()), 4),
        angle_std=round(float(angles.std()), 4),
        frame_angle_ok=bad_ratio <= MAX_ANGLE_BAD_RATIO,
    )


def check_ca_dihedrals(pos: np.ndarray, quads: list[tuple[int, int, int, int]]) -> dict:
    """Cα pseudo-dihedrals – informational only, no pass/fail."""
    if not quads:
        return dict(
            n_dihedrals=0,
            dihedral_mean=float("nan"),
            dihedral_std=float("nan"),
            frac_extended=float("nan"),
        )
    dihs = np.array([_dihedral(pos[i], pos[j], pos[k], pos[l]) for i, j, k, l in quads])
    extended = np.abs(dihs) >= CA_DIHEDRAL_EXTENDED_MIN
    return dict(
        n_dihedrals=len(dihs),
        dihedral_mean=round(float(dihs.mean()), 4),
        dihedral_std=round(float(dihs.std()), 4),
        frac_extended=round(float(extended.mean()), 6),
    )


def check_clashes(pos: np.ndarray) -> dict:
    """Vectorised non-bonded Cα clash detection."""
    n = len(pos)
    if n < CA_CLASH_SEQ_SKIP + 2:
        return dict(n_clashes=0, frame_clash_ok=True)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.triu(np.ones((n, n), dtype=bool), k=CA_CLASH_SEQ_SKIP + 1)
    n_clashes = int((dist[mask] < CA_CLASH_DIST).sum())
    return dict(n_clashes=n_clashes, frame_clash_ok=n_clashes == 0)


# ── Null geometry for COM-bead sources ────────────────────────────────────────
# "not applicable" is represented by NaN numeric fields and frame_*_ok = True
# (True = "no applicable check failed", not "check passed").


def _null_bonds() -> dict:
    return dict(
        n_bonds=0,
        bonds_ok=0,
        bonds_soft_ok=0,
        bonds_bad=0,
        bond_bad_ratio=float("nan"),
        bond_mean=float("nan"),
        bond_std=float("nan"),
        bond_min=float("nan"),
        bond_max=float("nan"),
        frame_bond_ok=True,
    )


def _null_angles() -> dict:
    return dict(
        n_angles=0,
        angles_bad=0,
        angle_bad_ratio=float("nan"),
        angle_mean=float("nan"),
        angle_std=float("nan"),
        frame_angle_ok=True,
    )


def _null_dihedrals() -> dict:
    return dict(
        n_dihedrals=0,
        dihedral_mean=float("nan"),
        dihedral_std=float("nan"),
        frac_extended=float("nan"),
    )


def _null_clashes() -> dict:
    return dict(n_clashes=0, frame_clash_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL SHAPE DESCRIPTORS  (valid for all bead representations)
# ══════════════════════════════════════════════════════════════════════════════


def shape_descriptors(pos: np.ndarray, ag) -> dict:
    """Ree, Rg, Rg / N^ν (Flory), mean pairwise distance."""
    n = len(pos)
    if n < 2:
        return dict(
            ree=float("nan"),
            rg=float("nan"),
            rg_norm=float("nan"),
            mean_pairwise=float("nan"),
            frame_shape_ok=True,
        )
    ree = float(np.linalg.norm(pos[-1] - pos[0]))
    try:
        rg = float(ag.radius_of_gyration())
    except Exception:
        centroid = pos.mean(axis=0)
        rg = float(np.sqrt(((pos - centroid) ** 2).sum(axis=1).mean()))
    rg_norm = rg / (n**FLORY_NU)

    if n <= 500:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(n, size=300, replace=False)
    sub = pos[idx]
    diff = sub[:, None, :] - sub[None, :, :]
    dists_pw = np.linalg.norm(diff, axis=-1)
    upper = dists_pw[np.triu_indices(len(sub), k=1)]
    mean_pw = float(upper.mean())

    shape_ok = RG_NORM_IDP_LO <= rg_norm <= RG_NORM_IDP_HI
    return dict(
        ree=round(ree, 3),
        rg=round(rg, 3),
        rg_norm=round(rg_norm, 4),
        mean_pairwise=round(mean_pw, 3),
        frame_shape_ok=bool(shape_ok),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════


def _get_files(protein_dir: Path, source: str) -> tuple[Path | None, Path | None]:
    pid = protein_dir.stem

    if source == "IDPFold2":
        for tname in ("topology.pdb", "top_CG.pdb", "top.pdb"):
            top = protein_dir / tname
            if top.exists():
                return top, protein_dir / "traj.dcd"
        return None, protein_dir / "traj.dcd"

    elif source == "AF-CALVADOS":
        return protein_dir / "top.pdb", protein_dir / f"{pid}.dcd"

    elif source == "STARLING":
        return protein_dir / "top.pdb", protein_dir / "traj.xtc"

    else:
        raise ValueError(
            f"Unknown source: {source!r}.  " f"Valid choices: {list(SOURCE_REPR)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PER-PROTEIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def analyse_protein(protein_dir: Path, source: str) -> list[dict]:
    cfg = SOURCE_REPR[source]
    run_geometry = cfg["run_geometry"]

    top_path, traj_path = _get_files(protein_dir, source)
    if (
        top_path is None
        or not top_path.exists()
        or traj_path is None
        or not traj_path.exists()
    ):
        return []

    try:
        u = mda.Universe(str(top_path), str(traj_path))
        ca = u.select_atoms("name CA")
    except Exception as e:
        print(f"  [ERROR] {protein_dir.name}: {e}", flush=True)
        return []

    if len(ca) == 0:
        print(f"  [SKIP] No CA atoms in {protein_dir.name}", flush=True)
        return []

    n_res = len(ca)

    # ── Build segment-aware index caches ONCE per protein ────────────────────
    pair_idx = _sequential_pairs(ca)
    triplets = _intra_segment_triplets(ca)
    quads = _intra_segment_quadruplets(ca)

    # Map Universe atom indices → local CA position-array indices
    idx_to_local: dict[int, int] = {int(i): k for k, i in enumerate(ca.indices)}

    pair_local = (
        np.array([[idx_to_local[a], idx_to_local[b]] for a, b in pair_idx], dtype=int)
        if len(pair_idx) > 0
        else np.empty((0, 2), dtype=int)
    )
    tri_local = [
        (idx_to_local[a], idx_to_local[b], idx_to_local[c]) for a, b, c in triplets
    ]
    quad_local = [
        (idx_to_local[a], idx_to_local[b], idx_to_local[c], idx_to_local[d])
        for a, b, c, d in quads
    ]

    rows: list[dict] = []

    for ts in u.trajectory:
        pos = ca.positions.copy()  # (n_res, 3) Å

        row: dict = {
            "protein": protein_dir.name,
            "frame": ts.frame,
            "n_residues": n_res,
            "n_segments": ca.n_segments,
            "geometry_applicable": int(run_geometry),
        }

        # ── Geometry checks (Cα sources) or null fill (COM sources) ──────────
        if run_geometry:
            row.update(check_ca_bonds(pos, pair_local))
            row.update(check_ca_angles(pos, tri_local))
            row.update(check_ca_dihedrals(pos, quad_local))
            row.update(check_clashes(pos))
        else:
            row.update(_null_bonds())
            row.update(_null_angles())
            row.update(_null_dihedrals())
            row.update(_null_clashes())

        # ── Shape descriptors (universal) ─────────────────────────────────────
        row.update(shape_descriptors(pos, ca))

        # ── frame_fully_ok: source-specific definition ────────────────────────
        #   COM sources (AF-CALVADOS):  only Rg plausibility is assessable
        #   Cα  sources (IDPFold2 / STARLING): bond + angle + clash
        #   Shape is kept SOFT for Cα sources (not included in fully_ok) to
        #   maintain consistency with the original script convention.
        if cfg["frame_ok_basis"] == "shape":
            row["frame_fully_ok"] = bool(row["frame_shape_ok"])
        else:
            row["frame_fully_ok"] = bool(
                row["frame_bond_ok"] and row["frame_angle_ok"] and row["frame_clash_ok"]
            )

        rows.append(row)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER  (multiprocessing-safe)
# ══════════════════════════════════════════════════════════════════════════════


def _worker(args: tuple[Path, str]) -> list[dict]:
    protein_dir, source = args
    try:
        return analyse_protein(protein_dir, source)
    except Exception as exc:
        print(f"  [WORKER ERROR] {protein_dir.name}: {exc}", flush=True)
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def build_summary(df: pd.DataFrame, source: str) -> pd.DataFrame:
    cfg = SOURCE_REPR[source]
    run_geo = cfg["run_geometry"]
    g = df.groupby("protein")

    summary = pd.DataFrame(
        {
            "n_residues": g["n_residues"].first(),
            "n_segments": g["n_segments"].first(),
            "n_frames": g["frame"].count(),
            "geometry_applicable": g["geometry_applicable"].first(),
        }
    )

    # Geometry columns (Cα sources only)
    if run_geo:
        summary["bond_mean_Å"] = g["bond_mean"].mean().round(3)
        summary["bond_std_Å"] = g["bond_std"].mean().round(3)
        summary["bond_bad_ratio"] = g["bond_bad_ratio"].mean().round(5)
        summary["frames_bond_ok_%"] = (g["frame_bond_ok"].mean() * 100).round(1)
        summary["angle_mean_deg"] = g["angle_mean"].mean().round(2)
        summary["angle_bad_ratio"] = g["angle_bad_ratio"].mean().round(5)
        summary["frames_angle_ok_%"] = (g["frame_angle_ok"].mean() * 100).round(1)
        summary["dihedral_mean_deg"] = g["dihedral_mean"].mean().round(2)
        summary["dihedral_std_deg"] = g["dihedral_std"].mean().round(2)
        summary["frac_extended"] = g["frac_extended"].mean().round(5)
        summary["mean_clashes/frame"] = g["n_clashes"].mean().round(2)
        summary["frames_clash_free_%"] = (g["frame_clash_ok"].mean() * 100).round(1)

    # Shape columns (universal)
    summary["mean_Ree_Å"] = g["ree"].mean().round(2)
    summary["mean_Rg_Å"] = g["rg"].mean().round(2)
    summary["mean_Rg_norm"] = g["rg_norm"].mean().round(4)
    summary["mean_pairwise_dist_Å"] = g["mean_pairwise"].mean().round(2)
    summary["frames_shape_ok_%"] = (g["frame_shape_ok"].mean() * 100).round(1)

    # Overall viability
    summary["frames_fully_ok_%"] = (g["frame_fully_ok"].mean() * 100).round(1)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════


def build_report(df: pd.DataFrame, summary: pd.DataFrame, source: str) -> str:
    cfg = SOURCE_REPR[source]
    N = len(df)
    n_prot = df["protein"].nunique()
    run_geo = cfg["run_geometry"]
    ok_basis = cfg["frame_ok_basis"]

    def pct(num, denom):
        return f"{100 * num / denom:.2f} %" if denom else "N/A"

    sep = "─" * 70

    lines = [
        "=" * 70,
        "  CONFORMATIONAL ENSEMBLE QUALITY REPORT",
        f"  Source         : {source}",
        f"  Representation : {cfg['bead_type']}  "
        f"({'Cα geometry checks valid' if run_geo else 'COM-bead – Cα geometry NOT applicable'})",
        "=" * 70,
        "",
        f"  {cfg['description']}",
        "",
        f"  Proteins analysed      : {n_prot}",
        f"  Total frames           : {N:,}",
        f"  Avg residues / protein : {df['n_residues'].mean():.0f}",
        f"  Avg segments / protein : {df['n_segments'].mean():.1f}  "
        f"(>1 = multi-chain; bonds/angles intra-segment only)",
        f"  frame_fully_ok basis   : {ok_basis.upper()}",
        "",
    ]

    # ── Geometry section (Cα sources only) ───────────────────────────────────
    if run_geo:
        tot_b = int(df["n_bonds"].sum())
        bad_b = int(df["bonds_bad"].sum())
        tot_a = int(df["n_angles"].sum())
        bad_a = int(df["angles_bad"].sum())

        lines += [
            "  Thresholds (Cα geometry)",
            f"    Bond  bad ratio (frame pass) : ≤ {MAX_BOND_BAD_RATIO*100:.1f} %",
            f"    Angle bad ratio (frame pass) : ≤ {MAX_ANGLE_BAD_RATIO*100:.1f} %",
            f"    Clash distance               : < {CA_CLASH_DIST} Å  "
            f"(skip ±{CA_CLASH_SEQ_SKIP} seq. neighbours)",
            f"    Rg normalisation             : Rg / N^{FLORY_NU}  "
            f"(Flory ν, expected {RG_NORM_IDP_LO}–{RG_NORM_IDP_HI} Å)",
            "",
            sep,
            f"  1. Cα–Cα VIRTUAL BOND LENGTHS   (ideal {CA_BOND_IDEAL} Å)",
            sep,
            f"  Total sequential bonds  : {tot_b:,}",
            f"  Within hard limits [{CA_BOND_MIN}–{CA_BOND_MAX} Å]  : "
            f"{pct(tot_b - bad_b, tot_b)}",
            f"  Within soft limits [{CA_BOND_SOFT_LO}–{CA_BOND_SOFT_HI} Å]  : "
            f"{pct(df['bonds_soft_ok'].sum(), tot_b)}",
            f"  Frames pass (≤{MAX_BOND_BAD_RATIO*100:.0f}% bad)  : "
            f"{df['frame_bond_ok'].sum():,} / {N}  "
            f"({pct(df['frame_bond_ok'].sum(), N)})",
            f"  Mean bond length        : {df['bond_mean'].mean():.3f} ± "
            f"{df['bond_std'].mean():.3f} Å",
            "",
            sep,
            f"  2. Cα–Cα–Cα PSEUDO-BOND ANGLES   ({CA_ANGLE_MIN}–{CA_ANGLE_MAX} °)",
            sep,
            f"  Total angles           : {tot_a:,}",
            f"  Within limits          : {pct(tot_a - bad_a, tot_a)}",
            f"  Frames pass (≤{MAX_ANGLE_BAD_RATIO*100:.0f}% bad)  : "
            f"{df['frame_angle_ok'].sum():,} / {N}  "
            f"({pct(df['frame_angle_ok'].sum(), N)})",
            f"  Mean angle             : {df['angle_mean'].mean():.2f} ± "
            f"{df['angle_std'].mean():.2f} °",
            "",
            sep,
            f"  3. Cα PSEUDO-DIHEDRALS   "
            f"(informational; |φ| ≥ {CA_DIHEDRAL_EXTENDED_MIN:.0f}° = extended/β-like)",
            sep,
            f"  Mean dihedral          : {df['dihedral_mean'].mean():.2f} °",
            f"  Std  dihedral          : {df['dihedral_std'].mean():.2f} °",
            f"  Mean frac extended     : {df['frac_extended'].mean():.4f}",
            "",
            sep,
            f"  4. NON-BONDED Cα CLASHES   (< {CA_CLASH_DIST} Å)",
            sep,
            f"  Clash-free frames      : {df['frame_clash_ok'].sum():,} / {N}  "
            f"({pct(df['frame_clash_ok'].sum(), N)})",
            f"  Mean clashes / frame   : {df['n_clashes'].mean():.2f}",
            f"  Max  clashes / frame   : {df['n_clashes'].max()}",
            "",
        ]
        shape_label = "5–9."
    else:
        lines += [
            sep,
            "  Cα GEOMETRY CHECKS  [NOT APPLICABLE – COM-bead representation]",
            sep,
            "  Bond / angle / clash checks are skipped because trajectories",
            "  store one COM-bead per residue, not Cα atoms. For folded",
            "  residues the stored bead is the centre-of-mass of all heavy",
            "  atoms; for disordered residues it is a CALVADOS bead (~Cα).",
            "  Shape descriptors below are the appropriate quality metric.",
            "",
        ]
        shape_label = "1–5."

    # ── Shape section (universal) ─────────────────────────────────────────────
    lines += [
        sep,
        f"  {shape_label} SHAPE DESCRIPTORS   "
        f"(universal – valid for ALL representations)",
        sep,
        f"  Rg normalisation       : Rg / N^{FLORY_NU}  "
        f"(Flory ν, expected {RG_NORM_IDP_LO}–{RG_NORM_IDP_HI} Å)",
        f"  Mean Ree               : {df['ree'].mean():.2f} Å",
        f"  Mean Rg                : {df['rg'].mean():.2f} Å",
        f"  Mean Rg / N^{FLORY_NU}      : {df['rg_norm'].mean():.3f} Å",
        f"  Mean pairwise dist     : {df['mean_pairwise'].mean():.2f} Å",
        f"  Frames plausible Rg    : {df['frame_shape_ok'].sum():,} / {N}  "
        f"({pct(df['frame_shape_ok'].sum(), N)})  "
        f"[{'PRIMARY metric for this source' if not run_geo else 'soft – not in fully_ok'}]",
        "",
    ]

    # ── Overall viability section ─────────────────────────────────────────────
    lines += [
        sep,
        f"  OVERALL VIABILITY   (frame_fully_ok = {ok_basis.upper()})",
        sep,
    ]
    if ok_basis == "shape":
        lines += [
            "  COM-bead source: frame_fully_ok = frame_shape_ok",
            f"  (Rg / N^ν ∈ [{RG_NORM_IDP_LO}, {RG_NORM_IDP_HI}] Å).  Geometry checks",
            "  are not applicable and do not contribute to this metric.",
            "",
        ]
    else:
        lines += [
            "  Cα source: frame_fully_ok = bond_ok AND angle_ok AND clash_ok.",
            "  Shape (Rg / N^ν) is kept SOFT – not included in frame_fully_ok.",
            "",
        ]

    lines += [
        f"  Fully viable frames    : {df['frame_fully_ok'].sum():,} / {N}  "
        f"({pct(df['frame_fully_ok'].sum(), N)})",
        "=" * 70,
        "",
    ]

    # ── Per-protein breakdown ─────────────────────────────────────────────────
    if run_geo:
        lines += [sep, "  WORST 10 PROTEINS  (by % fully-viable frames)", sep]
        for prot, row in summary.nsmallest(10, "frames_fully_ok_%").iterrows():
            lines.append(
                f"  {prot:<28s}  fully_ok={row['frames_fully_ok_%']:5.1f}%  "
                f"Rg={row['mean_Rg_Å']:.1f}Å  "
                f"clashes/fr={row['mean_clashes/frame']:.1f}  "
                f"bond_bad={row['bond_bad_ratio']:.4f}"
            )
        lines += ["", sep, "  BEST 10 PROTEINS  (by % fully-viable frames)", sep]
        for prot, row in summary.nlargest(10, "frames_fully_ok_%").iterrows():
            lines.append(
                f"  {prot:<28s}  fully_ok={row['frames_fully_ok_%']:5.1f}%  "
                f"Rg={row['mean_Rg_Å']:.1f}Å  "
                f"Rg/N^ν={row['mean_Rg_norm']:.3f}"
            )
    else:
        lines += [sep, "  BEST 10 PROTEINS   (by % frames with plausible Rg)", sep]
        for prot, row in summary.nlargest(10, "frames_shape_ok_%").iterrows():
            lines.append(
                f"  {prot:<28s}  shape_ok={row['frames_shape_ok_%']:5.1f}%  "
                f"Rg={row['mean_Rg_Å']:.1f}Å  "
                f"Rg/N^ν={row['mean_Rg_norm']:.3f}"
            )
        lines += ["", sep, "  BOTTOM 10 PROTEINS (by % frames with plausible Rg)", sep]
        for prot, row in summary.nsmallest(10, "frames_shape_ok_%").iterrows():
            lines.append(
                f"  {prot:<28s}  shape_ok={row['frames_shape_ok_%']:5.1f}%  "
                f"Rg={row['mean_Rg_Å']:.1f}Å  "
                f"Rg/N^ν={row['mean_Rg_norm']:.3f}"
            )

    # ── Cross-method comparison reminder ─────────────────────────────────────
    lines += [
        "",
        sep,
        "  CROSS-METHOD COMPARISON  (use shape metrics; geometry not comparable)",
        sep,
        "  The following metrics are valid for ALL three sources and enable",
        "  a fair comparison between AF-CALVADOS, IDPFold2, and STARLING:",
        "    •  mean_Rg_Å          •  mean_Ree_Å",
        "    •  mean_Rg_norm       •  mean_pairwise_dist_Å",
        "    •  frames_shape_ok_%  (Rg / N^ν plausibility)",
        "",
        "  Geometry metrics (bond / angle / clash) should only be compared",
        "  between IDPFold2 and STARLING (both Cα-trace models).",
        "",
        "  frame_fully_ok CANNOT be compared across COM and Cα sources",
        "  because it has different definitions (see frame_ok_basis above).",
        "",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════


def sanitize_and_filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Strip duplicate CSV headers that may appear in chunked output
    if "protein" in df.columns:
        df = df[df["protein"] != "protein"].copy()

    # 2. Boolean columns → integer 0 / 1
    bool_cols = [
        "geometry_applicable",
        "frame_bond_ok",
        "frame_angle_ok",
        "frame_clash_ok",
        "frame_shape_ok",
        "frame_fully_ok",
    ]

    # 3. Numeric columns
    numeric_cols = [
        "frame",
        "n_residues",
        "n_segments",
        "n_bonds",
        "bonds_ok",
        "bonds_soft_ok",
        "bonds_bad",
        "bond_mean",
        "bond_std",
        "bond_min",
        "bond_max",
        "bond_bad_ratio",
        "n_angles",
        "angles_bad",
        "angle_mean",
        "angle_std",
        "angle_bad_ratio",
        "n_dihedrals",
        "dihedral_mean",
        "dihedral_std",
        "frac_extended",
        "n_clashes",
        "ree",
        "rg",
        "rg_norm",
        "mean_pairwise",
    ]

    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].map(
                    {"True": 1, "False": 0, "1": 1, "0": 0, True: 1, False: 0}
                )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Drop corrupted / incomplete rows
    required = [c for c in ["rg", "n_residues"] if c in df.columns]
    if required:
        df = df.dropna(subset=required).copy()

    # 5. Remove stray header rows
    if "protein" in df.columns:
        df = df[df["protein"].notna()].copy()
        df = df[df["protein"] != ""].copy()

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CSV I/O INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════


def _existing_csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle), [])


def _validate_append_target(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    existing = _existing_csv_header(path)
    if existing != PER_FRAME_COLUMNS:
        raise SystemExit(
            f"Existing per-frame CSV has a different schema: {path}\n"
            f"Expected {len(PER_FRAME_COLUMNS)} columns, found {len(existing)}.\n"
            "Run without --append to overwrite, or choose a different --output_dir."
        )


def write_per_frame_chunk(
    rows: list[dict], per_frame_path: Path, write_header: bool
) -> bool:
    if not rows:
        return write_header
    chunk = pd.DataFrame(rows)
    extra = [c for c in chunk.columns if c not in PER_FRAME_COLUMNS]
    if extra:
        raise ValueError(f"Unexpected per-frame columns: {extra}")
    for col in PER_FRAME_COLUMNS:
        if col not in chunk.columns:
            chunk[col] = pd.NA
    chunk[PER_FRAME_COLUMNS].to_csv(
        per_frame_path, mode="a", header=write_header, index=False
    )
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Source-aware Cα / COM-bead ensemble quality analysis. "
            "Applies geometry checks only where appropriate for the "
            "internal representation of each source."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root directory with one sub-folder per protein.",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SOURCE_REPR),
        help="Trajectory source / naming convention.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/conformations_analysis",
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--limit_n_prot",
        type=int,
        default=2000,
        help="Maximum number of protein directories to process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows to buffer before writing to disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for protein sub-sampling.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append to an existing per-frame CSV if its header matches "
            "the current schema. By default existing outputs are overwritten."
        ),
    )

    args = parser.parse_args()

    DATA_ROOT = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_ROOT.exists():
        sys.exit(f"Input directory not found: {DATA_ROOT}")

    # Show source-specific configuration
    cfg = SOURCE_REPR[args.source]
    print(f"\n{'=' * 60}")
    print(f"  Source          : {args.source}")
    print(f"  Representation  : {cfg['bead_type']}")
    print(
        f"  Geometry checks : {'YES (Cα source)' if cfg['run_geometry'] else 'NO  (COM-bead source)'}"
    )
    print(f"  frame_fully_ok  : {cfg['frame_ok_basis'].upper()}")
    print(f"{'=' * 60}\n")

    random.seed(args.seed)
    all_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    n_sample = min(args.limit_n_prot, len(all_dirs))
    prot_dirs = random.sample(all_dirs, n_sample)
    print(
        f"Processing {n_sample} / {len(all_dirs)} protein directories "
        f"from {DATA_ROOT}  (workers={args.workers})\n"
    )

    per_frame_path = OUTPUT_DIR / f"per_frame_{args.source}.csv"
    if args.append:
        _validate_append_target(per_frame_path)
    elif per_frame_path.exists():
        per_frame_path.unlink()

    write_header = not (
        args.append and per_frame_path.exists() and per_frame_path.stat().st_size > 0
    )
    buf: list[dict] = []
    tasks = [(d, args.source) for d in prot_dirs]

    with mp.Pool(processes=args.workers) as pool:
        for rows in tqdm(
            pool.imap_unordered(_worker, tasks, chunksize=1),
            total=len(tasks),
            desc="proteins",
        ):
            buf.extend(rows)
            if len(buf) >= args.chunk_size:
                write_header = write_per_frame_chunk(buf, per_frame_path, write_header)
                buf.clear()

    if buf:
        write_per_frame_chunk(buf, per_frame_path, write_header)

    if not per_frame_path.exists() or per_frame_path.stat().st_size == 0:
        sys.exit("No data collected. Check input paths and file naming.")

    print(f"\nPer-frame results → {per_frame_path}")

    df = pd.read_csv(per_frame_path)
    df = sanitize_and_filter_dataframe(df)
    summary = build_summary(df, args.source)

    summary_path = OUTPUT_DIR / f"summary_per_protein_{args.source}.csv"
    summary.to_csv(summary_path)
    print(f"Per-protein summary → {summary_path}")

    report_text = build_report(df, summary, args.source)
    print("\n" + report_text)

    report_path = OUTPUT_DIR / f"report_{args.source}.txt"
    report_path.write_text(report_text + "\n")
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
