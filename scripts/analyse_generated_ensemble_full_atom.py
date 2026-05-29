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
    --frames_per_protein 100 \
    --workers    8 \
    --chunk_size 500
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals, self_distance_array

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
    "C-O": (1.12, 1.55),
    "C-S": (1.68, 1.98),
    "N-N": (1.18, 1.52),
    "N-O": (1.18, 1.50),
    "O-S": (1.40, 1.70),
    "S-S": (1.90, 2.15),
    # Bonds involving hydrogen
    "C-H": (0.85, 1.15),
    "H-N": (0.80, 1.15),
    "H-O": (0.80, 1.10),
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

# ── Frames to sample per protein trajectory ──────────────────────────────────
DEFAULT_FRAMES_PER_PROTEIN = 100


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


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _is_plausible_covalent_bond(atom_a, atom_b) -> bool:
    """Filter distance-guessed bonds that are likely non-covalent contacts."""
    same_residue = (
        atom_a.segid == atom_b.segid
        and atom_a.resid == atom_b.resid
        and atom_a.resname == atom_b.resname
    )
    if same_residue:
        return True

    if atom_a.segid != atom_b.segid:
        return False

    names = {atom_a.name.strip(), atom_b.name.strip()}
    resids = sorted((atom_a.resid, atom_b.resid))

    if names == {"C", "N"} and resids[1] - resids[0] == 1:
        return True

    if names == {"SG"} and atom_a.resname == "CYS" and atom_b.resname == "CYS":
        return True

    return False


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
    excl_pairs : set of sorted (i, j) tuples – pairs to skip in clash detection
    """

    def __init__(self, u: mda.Universe):
        self.idx = None
        self.lo = None
        self.hi = None
        self.excl_pairs: set[tuple[int, int]] = set()
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

        raw_idx_all = bonds.indices.copy()  # (N, 2)
        keep_bonds = [
            _is_plausible_covalent_bond(b.atoms[0], b.atoms[1]) for b in bonds
        ]
        raw_idx = raw_idx_all[np.asarray(keep_bonds, dtype=bool)]

        if len(raw_idx) == 0:
            return

        names = np.array([[u.atoms[i].name, u.atoms[j].name] for i, j in raw_idx])

        # IMPORTANT: allocate using the filtered bond count, not the original
        # topology bond count. Otherwise self.idx, self.lo, and self.hi can end
        # up with different lengths and broadcast errors will occur later.
        lo = np.empty(len(raw_idx), dtype=np.float64)
        hi = np.empty(len(raw_idx), dtype=np.float64)
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
            self.excl_pairs.add(_pair_key(int(a), int(b)))

        # 1-3 pairs (shared neighbour)
        adj: dict[int, set[int]] = {}
        for a, b in raw_idx:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))

        for center, neighbours in adj.items():
            nb_list = list(neighbours)
            for i in range(len(nb_list)):
                for j in range(i + 1, len(nb_list)):
                    self.excl_pairs.add(_pair_key(nb_list[i], nb_list[j]))

        # 1-4 pairs (optional)
        if EXCLUDE_14_PAIRS:
            for center, neighbours in adj.items():
                for nb in neighbours:
                    for nb2 in adj.get(nb, set()):
                        if nb2 != center:
                            self.excl_pairs.add(_pair_key(center, nb2))

    @property
    def valid(self) -> bool:
        return self.idx is not None


def _res_atom_index(residue, atom_name: str) -> int | None:
    atoms = residue.atoms.select_atoms(f"name {atom_name}")
    if len(atoms) == 0:
        return None
    return int(atoms[0].index)


def _select_heavy_protein_atoms(u: mda.Universe):
    protein_atoms = u.select_atoms("protein")
    heavy_indices = [atom.index for atom in protein_atoms if _elem(atom.name) != "H"]
    return u.atoms[heavy_indices]


class BackboneAngleCache:
    """Atom indices needed for per-frame N-CA-C angle checks."""

    def __init__(self, residues):
        n_idx = []
        ca_idx = []
        c_idx = []
        for res in residues:
            n = _res_atom_index(res, "N")
            ca = _res_atom_index(res, "CA")
            c = _res_atom_index(res, "C")
            if n is None or ca is None or c is None:
                continue
            n_idx.append(n)
            ca_idx.append(ca)
            c_idx.append(c)

        self.n_idx = np.asarray(n_idx, dtype=np.intp)
        self.ca_idx = np.asarray(ca_idx, dtype=np.intp)
        self.c_idx = np.asarray(c_idx, dtype=np.intp)

    @property
    def valid(self) -> bool:
        return len(self.n_idx) > 0


class ClashCache:
    """Pre-computed pair thresholds for vectorized clash checks."""

    def __init__(self, heavy_atoms, bond_cache: BondCache):
        self.thresholds = None
        self.distance_buffer = None
        self._build(heavy_atoms, bond_cache)

    def _build(self, heavy_atoms, bond_cache: BondCache):
        n_atoms = len(heavy_atoms)
        if n_atoms < 2:
            return

        n_pairs = n_atoms * (n_atoms - 1) // 2
        thresholds = np.empty(n_pairs, dtype=np.float32)
        radii = np.asarray(
            [
                VDW_RADII.get(_elem(atom.name), VDW_DEFAULT_RADIUS)
                for atom in heavy_atoms
            ],
            dtype=np.float32,
        )
        atom_indices = heavy_atoms.indices

        k = 0
        for i in range(n_atoms - 1):
            atom_i = int(atom_indices[i])
            radius_i = radii[i]
            for j in range(i + 1, n_atoms):
                atom_j = int(atom_indices[j])
                if _pair_key(atom_i, atom_j) in bond_cache.excl_pairs:
                    thresholds[k] = -np.inf
                else:
                    thresholds[k] = (radius_i + radii[j]) * CLASH_OVERLAP_FACTOR
                k += 1

        self.thresholds = thresholds
        self.distance_buffer = np.empty(n_pairs, dtype=np.float64)

    @property
    def valid(self) -> bool:
        return self.thresholds is not None and len(self.thresholds) > 0


class OmegaCache:
    """Atom indices needed for vectorized omega checks."""

    def __init__(self, residues):
        ca_i_idx = []
        c_i_idx = []
        n_next_idx = []
        ca_next_idx = []
        next_is_pro = []
        res_list = list(residues)

        for i in range(len(res_list) - 1):
            ri = res_list[i]
            ri1 = res_list[i + 1]
            if ri.segid != ri1.segid or ri1.resid - ri.resid != 1:
                continue

            ca_i = _res_atom_index(ri, "CA")
            c_i = _res_atom_index(ri, "C")
            n_i1 = _res_atom_index(ri1, "N")
            ca_i1 = _res_atom_index(ri1, "CA")
            if ca_i is None or c_i is None or n_i1 is None or ca_i1 is None:
                continue

            ca_i_idx.append(ca_i)
            c_i_idx.append(c_i)
            n_next_idx.append(n_i1)
            ca_next_idx.append(ca_i1)
            next_is_pro.append(ri1.resname == "PRO")

        self.ca_i_idx = np.asarray(ca_i_idx, dtype=np.intp)
        self.c_i_idx = np.asarray(c_i_idx, dtype=np.intp)
        self.n_next_idx = np.asarray(n_next_idx, dtype=np.intp)
        self.ca_next_idx = np.asarray(ca_next_idx, dtype=np.intp)
        self.next_is_pro = np.asarray(next_is_pro, dtype=bool)

    @property
    def valid(self) -> bool:
        return len(self.ca_i_idx) > 0


class RamachandranCache:
    """Atom indices needed for vectorized per-frame phi/psi checks."""

    def __init__(self, residues):
        c_prev_idx = []
        n_idx = []
        ca_idx = []
        c_idx = []
        n_next_idx = []
        res_list = list(residues)

        for i in range(1, len(res_list) - 1):
            prev_res = res_list[i - 1]
            res = res_list[i]
            next_res = res_list[i + 1]

            if prev_res.segid != res.segid or res.segid != next_res.segid:
                continue
            if res.resid - prev_res.resid != 1 or next_res.resid - res.resid != 1:
                continue

            c_prev = _res_atom_index(prev_res, "C")
            n = _res_atom_index(res, "N")
            ca = _res_atom_index(res, "CA")
            c = _res_atom_index(res, "C")
            n_next = _res_atom_index(next_res, "N")
            if None in (c_prev, n, ca, c, n_next):
                continue

            c_prev_idx.append(c_prev)
            n_idx.append(n)
            ca_idx.append(ca)
            c_idx.append(c)
            n_next_idx.append(n_next)

        self.c_prev_idx = np.asarray(c_prev_idx, dtype=np.intp)
        self.n_idx = np.asarray(n_idx, dtype=np.intp)
        self.ca_idx = np.asarray(ca_idx, dtype=np.intp)
        self.c_idx = np.asarray(c_idx, dtype=np.intp)
        self.n_next_idx = np.asarray(n_next_idx, dtype=np.intp)

    @property
    def valid(self) -> bool:
        return len(self.n_idx) > 0


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


def check_backbone_angles(
    positions: np.ndarray, cache: BackboneAngleCache
) -> tuple[float, bool]:
    """
    N-CA-C angle for every residue.
    Returns (frac_bad, frame_ok).
    """
    if not cache.valid:
        return float("nan"), True

    n = positions[cache.n_idx]
    ca = positions[cache.ca_idx]
    c = positions[cache.c_idx]
    v1 = n - ca
    v2 = c - ca
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
    cos_a = np.einsum("ij,ij->i", v1, v2) / denom
    angles = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    bad = (angles < ANGLE_NCA_C_MIN) | (angles > ANGLE_NCA_C_MAX)
    frac_bad = float(bad.mean())
    return frac_bad, frac_bad == 0.0


def check_clashes(heavy_atoms, cache: ClashCache) -> tuple[int, bool]:
    """
    Count steric clashes among heavy atoms using VdW-radius overlap,
    properly excluding 1-2, 1-3 (and optionally 1-4) bonded pairs.

    Returns (n_clashes, clash_free).
    """
    if not cache.valid:
        return 0, True

    dists = self_distance_array(heavy_atoms.positions, result=cache.distance_buffer)
    n_clashes = int(np.count_nonzero(dists < cache.thresholds))

    return n_clashes, n_clashes == 0


def check_omega(positions: np.ndarray, cache: OmegaCache) -> tuple[float, int, bool]:
    """
    Compute ω = CA(i)-C(i)-N(i+1)-CA(i+1) for consecutive residue pairs
    using pre-computed atom indices.

    Returns (frac_bad, n_checked, frame_ok).
    frame_ok is True when frac_bad <= MAX_OMEGA_BAD_RATIO.
    """
    if not cache.valid:
        return float("nan"), 0, True

    omega = np.degrees(
        calc_dihedrals(
            positions[cache.ca_i_idx],
            positions[cache.c_i_idx],
            positions[cache.n_next_idx],
            positions[cache.ca_next_idx],
        )
    )
    dev_from_trans = np.abs(np.abs(omega) - 180.0)
    valid_cis_pro = cache.next_is_pro & (np.abs(omega) < OMEGA_CIS_PRO_TOL_DEG)
    bad = (~valid_cis_pro) & (dev_from_trans > OMEGA_TRANS_TOL_DEG)
    n_bad = int(np.count_nonzero(bad))
    n_checked = len(omega)
    frac_bad = n_bad / n_checked
    frame_ok = frac_bad <= MAX_OMEGA_BAD_RATIO
    return frac_bad, n_checked, frame_ok


def check_ramachandran(
    positions: np.ndarray, cache: RamachandranCache
) -> tuple[float, bool]:
    """
    Compute phi/psi from cached atom indices, then classify each residue
    using the polygon-based allowed regions (alpha, beta/PPII, left-handed
    helix).

    Returns (frac_disallowed, frame_ok).
    frame_ok is True when frac_disallowed <= MAX_RAMA_DISALLOWED_RATIO.
    """
    if not cache.valid:
        return float("nan"), True

    phis = np.degrees(
        calc_dihedrals(
            positions[cache.c_prev_idx],
            positions[cache.n_idx],
            positions[cache.ca_idx],
            positions[cache.c_idx],
        )
    )
    psis = np.degrees(
        calc_dihedrals(
            positions[cache.n_idx],
            positions[cache.ca_idx],
            positions[cache.c_idx],
            positions[cache.n_next_idx],
        )
    )

    n_res = len(phis)
    n_disallowed = sum(
        0 if _in_allowed_rama(phi, psi) else 1 for phi, psi in zip(phis, psis)
    )
    frac_disallowed = n_disallowed / n_res
    frame_ok = frac_disallowed <= MAX_RAMA_DISALLOWED_RATIO
    return frac_disallowed, frame_ok


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
        return (protein_dir / f"top_AA.pdb", protein_dir / f"traj_AA.xtc")

    else:
        raise ValueError(f"Unknown source: {source!r}")


def _stable_seed(base_seed: int, protein_name: str) -> int:
    seed_material = f"{base_seed}:{protein_name}".encode("utf-8")
    digest = hashlib.blake2b(seed_material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _sample_frame_indices(
    n_frames: int, frames_per_protein: int, seed: int, protein_name: str
) -> list[int]:
    if frames_per_protein == -1 or n_frames <= frames_per_protein:
        return list(range(n_frames))

    rng = random.Random(_stable_seed(seed, protein_name))
    return sorted(rng.sample(range(n_frames), frames_per_protein))


def analyse_protein(
    protein_dir: Path, source: str, frames_per_protein: int, seed: int
) -> list[dict]:
    """
    Analyse a random subset of frames from one protein trajectory.
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
    heavy_ag = _select_heavy_protein_atoms(u)
    angle_cache = BackboneAngleCache(protein_ag.residues)
    clash_cache = ClashCache(heavy_ag, cache)
    omega_cache = OmegaCache(protein_ag.residues)
    rama_cache = RamachandranCache(protein_ag.residues)

    rows = []
    frame_indices = _sample_frame_indices(
        len(u.trajectory), frames_per_protein, seed, protein_dir.name
    )
    for frame_idx in frame_indices:
        ts = u.trajectory[frame_idx]
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
        frac_bad_angle, angle_ok = check_backbone_angles(positions, angle_cache)
        row["backbone_angle_frac_bad"] = _safe_round(frac_bad_angle)
        row["frame_angle_ok"] = angle_ok

        # 3. Clashes ───────────────────────────────────────────────────────────
        n_clashes, clash_ok = check_clashes(heavy_ag, clash_cache)
        row["n_clashes"] = n_clashes
        row["frame_clash_ok"] = clash_ok

        # 4. Omega ─────────────────────────────────────────────────────────────
        frac_bad_omega, n_omega, omega_ok = check_omega(positions, omega_cache)
        row["omega_frac_bad"] = _safe_round(frac_bad_omega)
        row["n_omega_checked"] = n_omega
        row["frame_omega_ok"] = omega_ok

        # 5. Ramachandran ──────────────────────────────────────────────────────
        frac_disallowed, rama_ok = check_ramachandran(positions, rama_cache)
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


def _worker(args: tuple[Path, str, int, int]) -> list[dict]:
    protein_dir, source, frames_per_protein, seed = args
    try:
        return analyse_protein(protein_dir, source, frames_per_protein, seed)
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


def sanitize_and_filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. REMOVE DUPLICATE HEADERS
    # If the script appended headers mid-file, the 'protein' column will contain the string 'protein'
    header_rows = df[df["protein"] == "protein"]
    if not header_rows.empty:
        print(f"🗑️  Removing {len(header_rows)} duplicate header rows found in CSV.")
        df = df[df["protein"] != "protein"].copy()

    # 2. DEFINE COLUMNS
    numeric_cols = [
        "bond_mean",
        "bond_std",
        "bond_bad_ratio",
        "bonds_ok",
        "bonds_bad",
        "angle_mean",
        "angle_bad_ratio",
        "backbone_angle_frac_bad",
        "dihedral_mean",
        "dihedral_std",
        "frac_extended",
        "n_bonds",
        "n_angles",
        "n_dihedrals",
        "n_clashes",
        "omega_frac_bad",
        "n_omega_checked",
        "rama_frac_disallowed",
        "ree",
        "rg",
        "rg_norm",
        "mean_pairwise",
    ]

    # These columns often contain 'True'/'False' strings which to_numeric fails on
    bool_cols = [
        "frame_bond_ok",
        "frame_angle_ok",
        "frame_clash_ok",
        "frame_omega_ok",
        "frame_rama_ok",
        "frame_shape_ok",
        "frame_fully_ok",
    ]

    # 3. IDENTIFY FUSED-STRING CORRUPTION (The smushed numbers)
    # Use the first available bond metric as the proxy. Anything still
    # unparseable after header removal is corruption.
    corruption_proxy = next(
        (
            col
            for col in ("n_bonds", "bond_bad_ratio", "bond_mean")
            if col in df.columns
        ),
        None,
    )
    if corruption_proxy is not None:
        is_corrupted = pd.to_numeric(df[corruption_proxy], errors="coerce").isna()

        if is_corrupted.any():
            corrupted_count = is_corrupted.sum()
            print(
                f"⚠️  Removing {corrupted_count} frames with fused-string corruption."
            )
            df = df[~is_corrupted].copy()

    # 4. CAST BOOLEANS FIRST
    # Converts string "True"/"False" to actual 1/0 or True/False objects
    for col in bool_cols:
        if col in df.columns:
            # Map strings to actual booleans if they are objects/strings
            if df[col].dtype == "object":
                df[col] = df[col].map(
                    {
                        "True": True,
                        "False": False,
                        "1": True,
                        "0": False,
                        1: True,
                        0: False,
                    }
                )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 5. CAST REMAINING NUMERICS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise")

    print("✅ Data cleaning complete. All columns cast to correct types.")
    return df


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
        "--append",
        action="store_true",
        help="Append to an existing per-frame CSV instead of starting a fresh run.",
    )
    parser.add_argument(
        "--frames_per_protein",
        type=int,
        default=DEFAULT_FRAMES_PER_PROTEIN,
        help=(
            "Number of random trajectory frames to analyse per protein. "
            "Use -1 for all frames."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for protein and frame sub-sampling.",
    )
    args = parser.parse_args()

    DATA_ROOT = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_ROOT.exists():
        sys.exit(f"Input directory not found: {DATA_ROOT}")

    if args.frames_per_protein == 0 or args.frames_per_protein < -1:
        sys.exit(
            "--frames_per_protein must be a positive integer, or -1 for all frames."
        )

    random.seed(args.seed)
    all_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    n_sample = min(args.limit_n_prot, len(all_dirs))
    prot_dirs = random.sample(all_dirs, n_sample)
    frame_msg = (
        "all frames"
        if args.frames_per_protein == -1
        else f"{args.frames_per_protein} random frame(s) per protein"
    )
    print(
        f"Processing {n_sample} / {len(all_dirs)} protein directories "
        f"from {DATA_ROOT}  (workers={args.workers}, frames={frame_msg})\n"
    )

    per_frame_path = OUTPUT_DIR / f"results_per_frame_{args.source}.csv"
    if per_frame_path.exists() and not args.append:
        per_frame_path.unlink()
    write_header = not per_frame_path.exists() or per_frame_path.stat().st_size == 0
    all_rows_buf: list[dict] = []

    tasks = [(d, args.source, args.frames_per_protein, args.seed) for d in prot_dirs]

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
    df = sanitize_and_filter_dataframe(df)
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
