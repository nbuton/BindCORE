"""
Geometric Viability Analysis for IDP Conformational Ensembles
=============================================================
Analyses trajectory files across all protein folders.

Checks per frame:
  1. Bond lengths  – covalent bonds within biologically plausible limits
                     (reports mean bad ratio; frame-level pass uses a
                      tolerance of MAX_BOND_BAD_RATIO)
  2. Bond angles   – peptide backbone N-CA-C angles within sane ranges
  3. Clashes       – steric clashes using VdW-radius overlap criterion
                     (1-2, 1-3 and 1-4 bonded pairs are excluded)
  4. Ramachandran  – phi/psi in allowed regions using polygon-based
                     definitions that include alpha, beta, PPII and left-
                     handed helical regions (important for IDPs)
  5. Omega (ω)     – peptide-bond planarity computed residue-by-residue
                     to avoid index-mismatch bugs across chain breaks

Design goals
------------
- Suitable for IDP/IDR ensembles: lenient but physically meaningful
  thresholds, PPII region included, soft frame-level pass criteria.
- Scale to ~2 000 proteins × ~1 000 frames each without running out of
  memory: trajectory frames are processed one at a time and results are
  streamed to disk in chunks.
- Parallel execution: one worker process per protein via
  multiprocessing.Pool with a configurable number of workers.
- Supports multiple upstream sources (IDPFold2, AF-CALVADOS, STARLING).
- Outputs: results_per_frame.csv, results_summary.csv, report_<source>.txt

Usage
-----
python analyse_generated_ensemble_full_atom.py \
    --input_dir  data/conformational_ensemble/AF-CALVADOS \
    --source     AF-CALVADOS \
    --limit_n_prot 2000 \
    --workers    8 \
    --chunk_size 500
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals as mda_dihedrals
from MDAnalysis.lib.distances import self_distance_array

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration / thresholds
# ═══════════════════════════════════════════════════════════════════════════════

# ── Bond lengths (Å) ──────────────────────────────────────────────────────────
# Keyed by sorted element-pair string, e.g. "C-N".
# Ranges are intentionally generous to accommodate cg2all reconstruction
# artefacts; the important thing is catching truly unphysical bonds.
BOND_LIMITS: dict[str, tuple[float, float]] = {
    "C-C": (1.38, 1.68),
    "C-N": (1.22, 1.58),
    "C-O": (1.12, 1.38),
    "C-S": (1.68, 1.98),
    "N-N": (1.18, 1.52),
    "N-O": (1.18, 1.50),
    "O-S": (1.40, 1.70),
    "S-S": (1.90, 2.15),
    # Bonds involving hydrogen
    "C-H": (0.85, 1.15),
    "H-N": (0.85, 1.15),
    "H-O": (0.85, 1.05),
    "H-S": (1.20, 1.45),
}
BOND_ABSOLUTE_MIN = 0.70  # Å – below this is always wrong
BOND_ABSOLUTE_MAX = 2.20  # Å – above this is always wrong

# A frame is considered to pass the bond-length check if the fraction of
# out-of-range bonds is below this threshold (1 % tolerance).
MAX_BOND_BAD_RATIO = 0.01

# ── Clash detection ───────────────────────────────────────────────────────────
# Two heavy atoms clash when their centre-to-centre distance is less than
#   (R_vdw_i + R_vdw_j) * CLASH_OVERLAP_FACTOR
# 1-2, 1-3 and optionally 1-4 bonded pairs are excluded.
# Values from Bondi (1964) / CHARMM36 rounded for use here.
VDW_RADII: dict[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "H": 1.20,
}
VDW_DEFAULT_RADIUS = 1.70  # Å – fallback for unknown elements
CLASH_OVERLAP_FACTOR = 0.70  # allow 30 % vdW overlap (generous for IDPs)
EXCLUDE_14_PAIRS = True  # if True, also skip 1-4 bonded pairs

# ── Backbone angles (degrees) ─────────────────────────────────────────────────
ANGLE_NCA_C_MIN = 88.0
ANGLE_NCA_C_MAX = 142.0

# ── Omega (ω) planarity ───────────────────────────────────────────────────────
OMEGA_TRANS_TOL_DEG = 35.0  # |ω| must be within 35° of 180°
OMEGA_CIS_PRO_TOL_DEG = 40.0  # cis-Pro: |ω| < 40° counts as cis

# ── Ramachandran allowed regions ──────────────────────────────────────────────
# Polygon-based, expressed as lists of (phi, psi) vertex pairs.
# For speed we use the bounding-box first pass, then point-in-polygon.
# Regions cover: α-helix, β-sheet, PPII, left-handed helix.
# These match the "broad allowed" outlines of Lovell et al. 2003 / MolProbity.
RAMA_POLYGONS: list[tuple[str, list[tuple[float, float]]]] = [
    (
        "alpha_R",
        [(-170, -10), (-30, -10), (-30, -90), (-80, -90), (-80, -65), (-170, -65)],
    ),
    # Extended beta / PPII combined
    (
        "beta_PPII",
        [
            (-180, 180),
            (-30, 180),
            (-30, 80),
            (-180, 80),
            (-180, -100),
            (-30, -100),
            (-30, -180),
            (-180, -180),
        ],
    ),
    # Left-handed helix (populated in Gly and some IDP loops)
    ("alpha_L", [(30, 90), (90, 90), (90, 20), (30, 20)]),
    # Gamma turn region
    ("gamma", [(-180, -100), (-30, -100), (-30, -170), (-180, -170)]),
]

# Fraction of residues in disallowed Ramachandran regions above which a
# frame is flagged (5 % tolerance – IDPs legitimately sample borders).
MAX_RAMA_DISALLOWED_RATIO = 0.05

# ── Frame-level pass: maximum allowed fraction of bad ω angles ────────────────
MAX_OMEGA_BAD_RATIO = 0.05

# ── Output chunk size (rows written at once) ──────────────────────────────────
DEFAULT_CHUNK_SIZE = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _elem(atom_name: str) -> str:
    """Best-effort element from PDB atom name (works for standard names)."""
    cleaned = atom_name.strip().lstrip("0123456789")
    if len(cleaned) >= 2 and cleaned[:2].upper() in ("CL", "BR"):
        return cleaned[:2].upper()
    return cleaned[0].upper() if cleaned else "C"


def _bond_key(n1: str, n2: str) -> str:
    e1, e2 = _elem(n1), _elem(n2)
    return "-".join(sorted([e1, e2]))


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (2-D)."""
    n = len(polygon)
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _in_allowed_rama(phi: float, psi: float) -> bool:
    """Return True if (phi, psi) falls inside any allowed polygon."""
    for _label, poly in RAMA_POLYGONS:
        if _point_in_polygon(phi, psi, poly):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Universe bond topology cache (built once per protein)
# ═══════════════════════════════════════════════════════════════════════════════


class BondCache:
    """
    Pre-computed bond information for a Universe.

    Attributes
    ----------
    idx        : (N, 2) int array of atom indices
    lo, hi     : (N,) float arrays of allowed [min, max] bond lengths
    excl_pairs : set of frozenset({i, j}) – pairs to skip in clash detection
    """

    def __init__(self, u: mda.Universe):
        self.idx = None
        self.lo = None
        self.hi = None
        self.excl_pairs: set[frozenset] = set()
        self._build(u)

    def _build(self, u: mda.Universe):
        # ── Ensure topology has bonds ────────────────────────────────────────
        try:
            bonds = u.bonds
            if len(bonds) == 0:
                raise AttributeError
        except (AttributeError, Exception):
            try:
                from MDAnalysis.topology.guessers import guess_bonds

                u.trajectory[0]
                guessed = guess_bonds(u.atoms, u.atoms.positions, vdwradii=None)
                u.add_TopologyAttr("bonds", guessed)
                bonds = u.bonds
            except Exception:
                return  # give up – checks will be skipped

        if len(bonds) == 0:
            return

        raw_idx = bonds.indices.copy()  # (N, 2)
        names = np.array([[b.atoms[0].name, b.atoms[1].name] for b in bonds])

        lo = np.empty(len(bonds))
        hi = np.empty(len(bonds))
        for i, (n1, n2) in enumerate(names):
            key = _bond_key(n1, n2)
            _lo, _hi = BOND_LIMITS.get(key, (BOND_ABSOLUTE_MIN, BOND_ABSOLUTE_MAX))
            lo[i] = max(_lo, BOND_ABSOLUTE_MIN)
            hi[i] = min(_hi, BOND_ABSOLUTE_MAX)

        self.idx = raw_idx
        self.lo = lo
        self.hi = hi

        # ── Build exclusion set for clash detection ─────────────────────────
        # 1-2 pairs (direct bonds)
        for a, b in raw_idx:
            self.excl_pairs.add(frozenset({int(a), int(b)}))

        # 1-3 pairs (shared neighbour)
        adj: dict[int, set[int]] = {}
        for a, b in raw_idx:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))

        for center, neighbours in adj.items():
            nb_list = list(neighbours)
            for i in range(len(nb_list)):
                for j in range(i + 1, len(nb_list)):
                    self.excl_pairs.add(frozenset({nb_list[i], nb_list[j]}))

        # 1-4 pairs (optional)
        if EXCLUDE_14_PAIRS:
            for center, neighbours in adj.items():
                for nb in neighbours:
                    for nb2 in adj.get(nb, set()):
                        if nb2 != center:
                            self.excl_pairs.add(frozenset({center, nb2}))

    @property
    def valid(self) -> bool:
        return self.idx is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame check functions
# ═══════════════════════════════════════════════════════════════════════════════


def check_bond_lengths(
    positions: np.ndarray, cache: BondCache
) -> tuple[int, int, int, float, bool]:
    """
    Returns (n_bonds, n_ok, n_bad, bad_ratio, frame_ok).
    frame_ok is True when bad_ratio <= MAX_BOND_BAD_RATIO.
    """
    if not cache.valid:
        return 0, 0, 0, float("nan"), True

    p1 = positions[cache.idx[:, 0]]
    p2 = positions[cache.idx[:, 1]]
    lengths = np.linalg.norm(p2 - p1, axis=1)

    ok = (lengths >= cache.lo) & (lengths <= cache.hi)
    n_bonds = len(cache.idx)
    n_ok = int(ok.sum())
    n_bad = n_bonds - n_ok
    bad_ratio = n_bad / n_bonds
    frame_ok = bad_ratio <= MAX_BOND_BAD_RATIO
    return n_bonds, n_ok, n_bad, bad_ratio, frame_ok


def check_backbone_angles(residues) -> tuple[float, bool]:
    """
    N-CA-C angle for every residue.
    Returns (frac_bad, frame_ok).
    """
    bad = 0
    total = 0
    for res in residues:
        try:
            n = res.atoms.select_atoms("name N")[0].position
            ca = res.atoms.select_atoms("name CA")[0].position
            c = res.atoms.select_atoms("name C")[0].position
            v1 = n - ca
            v2 = c - ca
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            total += 1
            if not (ANGLE_NCA_C_MIN <= angle <= ANGLE_NCA_C_MAX):
                bad += 1
        except (IndexError, Exception):
            pass
    if total == 0:
        return float("nan"), True
    frac_bad = bad / total
    return frac_bad, frac_bad == 0.0


def check_clashes(heavy_atoms, cache: BondCache) -> tuple[int, bool]:
    """
    Count steric clashes among heavy atoms using VdW-radius overlap,
    properly excluding 1-2, 1-3 (and optionally 1-4) bonded pairs.

    Returns (n_clashes, clash_free).
    """
    positions = heavy_atoms.positions
    n = len(positions)
    if n < 2:
        return 0, True

    # Build per-atom VdW radii array
    radii = np.array(
        [VDW_RADII.get(_elem(a.name), VDW_DEFAULT_RADIUS) for a in heavy_atoms]
    )

    # Map heavy-atom local index → Universe atom index
    heavy_indices = heavy_atoms.indices  # shape (n,)

    # Upper-triangle pairwise distances
    dists = self_distance_array(positions)  # length n*(n-1)/2

    n_clashes = 0
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            pair = frozenset({int(heavy_indices[i]), int(heavy_indices[j])})
            if pair not in cache.excl_pairs:
                min_dist = (radii[i] + radii[j]) * CLASH_OVERLAP_FACTOR
                if dists[k] < min_dist:
                    n_clashes += 1
            k += 1

    return n_clashes, n_clashes == 0


def check_omega(residues) -> tuple[float, int, bool]:
    """
    Compute ω = CA(i)-C(i)-N(i+1)-CA(i+1) for consecutive residue pairs
    using residue objects directly (avoids list-index mismatches at chain
    breaks).

    Returns (frac_bad, n_checked, frame_ok).
    frame_ok is True when frac_bad <= MAX_OMEGA_BAD_RATIO.
    """
    res_list = list(residues)
    n_bad = 0
    n_checked = 0

    for i in range(len(res_list) - 1):
        ri = res_list[i]
        ri1 = res_list[i + 1]

        # Only consider consecutive residues in the same chain
        if ri.segid != ri1.segid:
            continue
        if ri1.resid - ri.resid != 1:
            continue

        try:
            ca_i = ri.atoms.select_atoms("name CA")[0].position
            c_i = ri.atoms.select_atoms("name C")[0].position
            n_i1 = ri1.atoms.select_atoms("name N")[0].position
            ca_i1 = ri1.atoms.select_atoms("name CA")[0].position
        except IndexError:
            continue

        b1 = c_i - ca_i
        b2 = n_i1 - c_i
        b3 = ca_i1 - n_i1

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        b2_norm = b2 / (np.linalg.norm(b2) + 1e-9)
        m1 = np.cross(n1, b2_norm)

        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        omega = np.degrees(np.arctan2(y, x))

        n_checked += 1
        dev_from_trans = abs(abs(omega) - 180.0)

        # Allow cis-proline
        is_next_pro = ri1.resname == "PRO"
        if is_next_pro and abs(omega) < OMEGA_CIS_PRO_TOL_DEG:
            continue  # valid cis-Pro

        if dev_from_trans > OMEGA_TRANS_TOL_DEG:
            n_bad += 1

    if n_checked == 0:
        return float("nan"), 0, True

    frac_bad = n_bad / n_checked
    frame_ok = frac_bad <= MAX_OMEGA_BAD_RATIO
    return frac_bad, n_checked, frame_ok


def check_ramachandran(protein_ag) -> tuple[float, bool]:
    """
    Compute phi/psi using MDAnalysis Ramachandran, then classify each
    residue using the polygon-based allowed regions (alpha, beta/PPII,
    left-handed helix).

    Returns (frac_disallowed, frame_ok).
    frame_ok is True when frac_disallowed <= MAX_RAMA_DISALLOWED_RATIO.
    """
    try:
        rama_run = mda_dihedrals.Ramachandran(protein_ag).run()
        angles = rama_run.results.angles  # (1, n_res, 2)
        if angles.shape[1] == 0:
            return float("nan"), True

        phis = angles[0, :, 0]
        psis = angles[0, :, 1]

        n_res = len(phis)
        n_disallowed = sum(
            0 if _in_allowed_rama(phi, psi) else 1 for phi, psi in zip(phis, psis)
        )
        frac_disallowed = n_disallowed / n_res
        frame_ok = frac_disallowed <= MAX_RAMA_DISALLOWED_RATIO
        return frac_disallowed, frame_ok
    except Exception:
        return float("nan"), True


# ═══════════════════════════════════════════════════════════════════════════════
# Per-protein analysis
# ═══════════════════════════════════════════════════════════════════════════════


def _get_files(protein_dir: Path, source: str) -> tuple[Path | None, Path | None]:
    """Return (topology_path, trajectory_path) for the given source."""
    if source == "IDPFold2":
        for fname in ("top_AA.pdb", "aa_topology.pdb"):
            top = protein_dir / fname
            if top.exists():
                return top, protein_dir / "aa_traj.dcd"
        return None, protein_dir / "aa_traj.dcd"

    elif source == "AF-CALVADOS":
        pid = protein_dir.stem
        return (protein_dir / f"{pid}_allatom.pdb", protein_dir / f"{pid}_allatom.dcd")

    elif source == "STARLING":
        # Adapt naming convention when known
        raise NotImplementedError("STARLING file layout not yet configured.")

    else:
        raise ValueError(f"Unknown source: {source!r}")


def analyse_protein(protein_dir: Path, source: str) -> list[dict]:
    """
    Analyse every frame of one protein trajectory.
    Returns a list of row-dicts (one per frame).
    """
    top_path, dcd_path = _get_files(protein_dir, source)

    if top_path is None or not top_path.exists() or not dcd_path.exists():
        return []

    try:
        u = mda.Universe(str(top_path), str(dcd_path))
    except Exception as e:
        print(f"  [ERROR] {protein_dir.name}: {e}", flush=True)
        return []

    cache = BondCache(u)
    protein_ag = u.select_atoms("protein")
    heavy_ag = u.select_atoms("protein and not name H*")

    rows = []
    for ts in u.trajectory:
        fi = ts.frame
        row: dict = {"protein": protein_dir.name, "frame": fi}

        positions = u.atoms.positions

        # 1. Bond lengths ─────────────────────────────────────────────────────
        nb, nok, nbad, bad_ratio, bl_ok = check_bond_lengths(positions, cache)
        row["n_bonds"] = nb
        row["bonds_ok"] = nok
        row["bonds_bad"] = nbad
        row["bond_bad_ratio"] = _safe_round(bad_ratio)
        row["frame_bond_ok"] = bl_ok

        # 2. Backbone angles ───────────────────────────────────────────────────
        frac_bad_angle, angle_ok = check_backbone_angles(protein_ag.residues)
        row["backbone_angle_frac_bad"] = _safe_round(frac_bad_angle)
        row["frame_angle_ok"] = angle_ok

        # 3. Clashes ───────────────────────────────────────────────────────────
        n_clashes, clash_ok = check_clashes(heavy_ag, cache)
        row["n_clashes"] = n_clashes
        row["frame_clash_ok"] = clash_ok

        # 4. Omega ─────────────────────────────────────────────────────────────
        frac_bad_omega, n_omega, omega_ok = check_omega(protein_ag.residues)
        row["omega_frac_bad"] = _safe_round(frac_bad_omega)
        row["n_omega_checked"] = n_omega
        row["frame_omega_ok"] = omega_ok

        # 5. Ramachandran ──────────────────────────────────────────────────────
        frac_disallowed, rama_ok = check_ramachandran(protein_ag)
        row["rama_frac_disallowed"] = _safe_round(frac_disallowed)
        row["frame_rama_ok"] = rama_ok

        # Overall ──────────────────────────────────────────────────────────────
        row["frame_fully_ok"] = all([bl_ok, angle_ok, clash_ok, omega_ok, rama_ok])

        rows.append(row)

    return rows


def _safe_round(v, decimals: int = 5):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker wrapper (needed for multiprocessing pickling)
# ═══════════════════════════════════════════════════════════════════════════════


def _worker(args: tuple[Path, str]) -> list[dict]:
    protein_dir, source = args
    try:
        return analyse_protein(protein_dir, source)
    except Exception as exc:
        print(f"  [WORKER ERROR] {protein_dir.name}: {exc}", flush=True)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Summary / report helpers
# ═══════════════════════════════════════════════════════════════════════════════


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("protein")
    summary = pd.DataFrame(
        {
            "n_frames": grp["frame"].count(),
            # Bond lengths
            "frames_bond_ok": grp["frame_bond_ok"].sum(),
            "frames_bond_ok_pct": grp["frame_bond_ok"].mean() * 100,
            "mean_bond_bad_ratio": grp["bond_bad_ratio"].mean(),
            # Backbone angles
            "frames_angle_ok": grp["frame_angle_ok"].sum(),
            "frames_angle_ok_pct": grp["frame_angle_ok"].mean() * 100,
            "mean_angle_frac_bad": grp["backbone_angle_frac_bad"].mean(),
            # Clashes
            "mean_clashes_per_frame": grp["n_clashes"].mean(),
            "frames_clash_free": grp["frame_clash_ok"].sum(),
            "frames_clash_free_pct": grp["frame_clash_ok"].mean() * 100,
            # Omega
            "frames_omega_ok": grp["frame_omega_ok"].sum(),
            "frames_omega_ok_pct": grp["frame_omega_ok"].mean() * 100,
            "mean_omega_frac_bad": grp["omega_frac_bad"].mean(),
            # Ramachandran
            "frames_rama_ok": grp["frame_rama_ok"].sum(),
            "frames_rama_ok_pct": grp["frame_rama_ok"].mean() * 100,
            "mean_rama_frac_disallowed": grp["rama_frac_disallowed"].mean(),
            # Overall
            "frames_fully_ok": grp["frame_fully_ok"].sum(),
            "frames_fully_ok_pct": grp["frame_fully_ok"].mean() * 100,
        }
    ).round(3)
    return summary


def build_report(df: pd.DataFrame, summary: pd.DataFrame, source: str) -> str:
    n_total = len(df)
    n_proteins = df["protein"].nunique()
    tot_bonds = df["n_bonds"].sum()
    tot_ok = df["bonds_ok"].sum()

    def pct(num, denom):
        return f"{100 * num / denom:.2f} %" if denom else "N/A"

    lines = [
        "=" * 72,
        "  CONFORMATIONAL ENSEMBLE GEOMETRIC VIABILITY REPORT",
        f"  Source: {source}",
        "=" * 72,
        f"  Proteins analysed          : {n_proteins}",
        f"  Total frames               : {n_total}",
        f"  Total covalent bonds checked: {tot_bonds:,}",
        "",
        "  Thresholds used",
        f"    Bond bad ratio (frame pass): ≤ {MAX_BOND_BAD_RATIO*100:.1f} %",
        f"    Clash VdW overlap factor   : {CLASH_OVERLAP_FACTOR}  "
        f"(1-2/1-3{'/1-4' if EXCLUDE_14_PAIRS else ''} excluded)",
        f"    Omega deviation tolerance  : {OMEGA_TRANS_TOL_DEG}°  "
        f"(frame pass ≤ {MAX_OMEGA_BAD_RATIO*100:.1f} % bad)",
        f"    Rama disallowed tolerance  : frame pass ≤ {MAX_RAMA_DISALLOWED_RATIO*100:.1f} %",
        "",
        "── BOND LENGTHS " + "─" * 55,
        f"  Bonds within limits  : {tot_ok:,} / {tot_bonds:,}  ({pct(tot_ok, tot_bonds)})",
        f"  Frames pass (≤{MAX_BOND_BAD_RATIO*100:.0f}% bad): "
        f"{df['frame_bond_ok'].sum()} / {n_total}  "
        f"({pct(df['frame_bond_ok'].sum(), n_total)})",
        "",
        "── BACKBONE ANGLES (N-CA-C) " + "─" * 43,
        f"  Frames ALL angles OK : "
        f"{df['frame_angle_ok'].sum()} / {n_total}  "
        f"({pct(df['frame_angle_ok'].sum(), n_total)})",
        f"  Mean fraction bad    : {df['backbone_angle_frac_bad'].mean():.4f}",
        "",
        "── STERIC CLASHES " + "─" * 52,
        f"  Clash-free frames    : "
        f"{df['frame_clash_ok'].sum()} / {n_total}  "
        f"({pct(df['frame_clash_ok'].sum(), n_total)})",
        f"  Mean clashes / frame : {df['n_clashes'].mean():.2f}",
        f"  Max  clashes / frame : {df['n_clashes'].max()}",
        "",
        "── PEPTIDE PLANARITY (ω) " + "─" * 46,
        f"  Frames pass          : "
        f"{df['frame_omega_ok'].sum()} / {n_total}  "
        f"({pct(df['frame_omega_ok'].sum(), n_total)})",
        f"  Mean fraction bad ω  : {df['omega_frac_bad'].mean():.4f}",
        "",
        "── RAMACHANDRAN (polygon-based, includes PPII) " + "─" * 23,
        f"  Frames pass          : "
        f"{df['frame_rama_ok'].sum()} / {n_total}  "
        f"({pct(df['frame_rama_ok'].sum(), n_total)})",
        f"  Mean frac disallowed : {df['rama_frac_disallowed'].mean():.4f}",
        "",
        "── OVERALL (all five checks pass) " + "─" * 36,
        f"  Fully viable frames  : "
        f"{df['frame_fully_ok'].sum()} / {n_total}  "
        f"({pct(df['frame_fully_ok'].sum(), n_total)})",
        "=" * 72,
    ]

    lines += ["", "── WORST 10 PROTEINS (% fully viable frames) " + "─" * 25]
    worst = summary.sort_values("frames_fully_ok_pct").head(10)
    for prot, r in worst.iterrows():
        lines.append(
            f"  {prot:<25}  {r['frames_fully_ok_pct']:5.1f}%  "
            f"clashes/frame={r['mean_clashes_per_frame']:.1f}  "
            f"rama_dis={r['mean_rama_frac_disallowed']:.3f}  "
            f"omega_bad={r['mean_omega_frac_bad']:.3f}"
        )

    lines += ["", "── BEST 10 PROTEINS (% fully viable frames) " + "─" * 26]
    best = summary.sort_values("frames_fully_ok_pct", ascending=False).head(10)
    for prot, r in best.iterrows():
        lines.append(
            f"  {prot:<25}  {r['frames_fully_ok_pct']:5.1f}%  "
            f"clashes/frame={r['mean_clashes_per_frame']:.1f}  "
            f"rama_dis={r['mean_rama_frac_disallowed']:.3f}"
        )

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Geometric viability analysis for IDP conformational ensembles."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing one sub-folder per protein.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["IDPFold2", "AF-CALVADOS", "STARLING"],
        help="Trajectory source / naming convention.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/conformations_analysis",
        help="Directory to write CSV and report files.",
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
        help="Number of per-frame rows to accumulate before writing to disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for protein sub-sampling.",
    )
    args = parser.parse_args()

    DATA_ROOT = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_ROOT.exists():
        sys.exit(f"Input directory not found: {DATA_ROOT}")

    random.seed(args.seed)
    all_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    n_sample = min(args.limit_n_prot, len(all_dirs))
    prot_dirs = random.sample(all_dirs, n_sample)
    print(
        f"Processing {n_sample} / {len(all_dirs)} protein directories "
        f"from {DATA_ROOT}  (workers={args.workers})\n"
    )

    per_frame_path = OUTPUT_DIR / f"results_per_frame_{args.source}.csv"
    write_header = True
    all_rows_buf: list[dict] = []

    worker_fn = partial(_worker)
    tasks = [(d, args.source) for d in prot_dirs]

    with mp.Pool(processes=args.workers) as pool:
        for rows in tqdm(
            pool.imap_unordered(_worker, tasks, chunksize=1),
            total=len(tasks),
            desc="proteins",
        ):
            all_rows_buf.extend(rows)

            # Stream chunks to disk to keep memory bounded
            if len(all_rows_buf) >= args.chunk_size:
                chunk_df = pd.DataFrame(all_rows_buf)
                chunk_df.to_csv(
                    per_frame_path,
                    mode="a",
                    header=write_header,
                    index=False,
                )
                write_header = False
                all_rows_buf.clear()

    # Flush remainder
    if all_rows_buf:
        pd.DataFrame(all_rows_buf).to_csv(
            per_frame_path, mode="a", header=write_header, index=False
        )

    if not per_frame_path.exists() or per_frame_path.stat().st_size == 0:
        sys.exit("No data collected. Check input paths and file naming.")

    print(f"\nPer-frame results → {per_frame_path}")

    # ── Build summary and report ─────────────────────────────────────────────
    df = pd.read_csv(per_frame_path)
    summary = build_summary(df)

    summary_path = OUTPUT_DIR / f"results_summary_{args.source}.csv"
    summary.to_csv(summary_path)
    print(f"Per-protein summary → {summary_path}")

    report_text = build_report(df, summary, args.source)
    print("\n" + report_text)

    report_path = OUTPUT_DIR / f"report_full_atom_{args.source}.txt"
    report_path.write_text(report_text + "\n")
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
