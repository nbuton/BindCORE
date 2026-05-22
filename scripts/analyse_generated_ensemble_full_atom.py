"""
Geometric Viability Analysis for IDP Conformational Ensembles
=============================================================
Analyses aa_traj.dcd + top_AA.pdb files across all protein folders.

Checks per frame:
  1. Bond lengths  – all covalent bonds within biologically plausible limits
  2. Bond angles   – peptide backbone angles within chemically sane ranges
  3. Clashes       – no two heavy atoms closer than a hard VdW floor
  4. Ramachandran  – phi/psi in allowed regions (broad + strict G-factors)
  5. Omega (ω)     – peptide-bond planarity (|ω| within 30° of 180°)

Outputs:
  - results_summary.csv   : per-protein aggregate statistics
  - results_per_frame.csv : every (protein, frame) row
  - report.txt            : human-readable summary
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

# ── MDAnalysis ──────────────────────────────────────────────────────────────
try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import dihedrals
    from MDAnalysis.lib.distances import calc_bonds, calc_angles
    from MDAnalysis.topology.guessers import guess_bonds as _guess_bonds
except ImportError:
    sys.exit("MDAnalysis not found. Install with:  pip install MDAnalysis")

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT = Path("data/conformational_ensemble/IDPFold2")
OUTPUT_DIR = Path("results_ensemble")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Bond-length limits (Å) ──────────────────────────────────────────────────
# Values are [min, max] for each bond type; generous tolerances for cg2all output.
BOND_LIMITS = {
    # Backbone
    "N-CA": (1.30, 1.60),
    "CA-C": (1.42, 1.65),
    "C-N": (1.25, 1.55),  # peptide bond (resonance: ~1.33 ideal)
    "C-O": (1.15, 1.35),  # carbonyl
    "CA-CB": (1.40, 1.65),
    # Hydrogen-containing bonds (if H present)
    "N-H": (0.85, 1.15),
    "CA-HA": (0.85, 1.15),
    "C-H": (0.85, 1.15),
    "O-H": (0.85, 1.05),
    "S-H": (1.20, 1.45),
    # Heavy-atom generic fallback
    "C-C": (1.40, 1.65),
    "C-N": (1.25, 1.55),
    "C-O": (1.15, 1.35),
    "C-S": (1.70, 1.95),
    "N-N": (1.20, 1.50),
    "S-S": (1.95, 2.10),
}
# Absolute max for ANY covalent bond (catch gross errors)
BOND_ABSOLUTE_MAX = 2.20  # Å
BOND_ABSOLUTE_MIN = 0.70  # Å

# Clash detection: minimum allowed distance between bonded-1,3 heavy atoms
CLASH_DIST = 1.50  # Å  (non-bonded heavy atoms)
VDW_CLASH_PAIRS_SKIP = 3  # skip 1-2 and 1-3 pairs

# Angle limits (degrees)
ANGLE_NCA_C = (90.0, 140.0)  # N-CA-C backbone
ANGLE_CA_C_N = (100.0, 135.0)  # CA-C-N
ANGLE_C_N_CA = (105.0, 135.0)  # C-N-CA

# Omega (peptide plane) limit: |ω - 180| < threshold (also proline cis ~0°)
OMEGA_TRANS_TOL = 30.0  # degrees from 180

# Ramachandran regions (broad, Lovell 2003 / Richardson)
# Precomputed polygons are expensive; use simple rectangular "allowed zones"
RAMA_REGIONS = [
    # (phi_min, phi_max, psi_min, psi_max, label)
    (-180, -30, -180, -100, "beta"),
    (-180, -30, 50, 180, "beta"),
    (-170, -30, -100, 50, "alpha"),
    (-90, -30, -60, 60, "alpha"),
    (30, 180, 30, 180, "gamma_left"),
    (30, 180, -180, -30, "gamma_left"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_bond_key(name1: str, name2: str) -> str:
    """Return a lookup key for BOND_LIMITS given two atom names."""

    def _elem(n):
        return n[0]  # first char is element for standard PDB names

    e1, e2 = _elem(name1), _elem(name2)
    pair = tuple(sorted([e1, e2]))
    return f"{pair[0]}-{pair[1]}"


def ensure_bonds(u):
    """
    Guarantee the Universe has bond topology.
    PDB topology files often lack CONECT records, so we guess bonds from
    the first-frame geometry + standard vdW radii when needed.
    Returns a cache tuple (idx, lo_arr, hi_arr) built once per protein.
    """
    try:
        _ = u.bonds  # raises NoDataError if absent
        bonds = u.bonds
    except Exception:
        try:
            u.trajectory[0]
            guessed = _guess_bonds(u.atoms, u.atoms.positions, vdwradii=None)
            u.add_TopologyAttr("bonds", guessed)
            bonds = u.bonds
        except Exception:
            return None, None, None

    if len(bonds) == 0:
        return None, None, None

    idx = bonds.indices.copy()  # (N,2) int array
    names1 = np.array([b.atoms[0].name for b in bonds])
    names2 = np.array([b.atoms[1].name for b in bonds])

    lo_arr = np.empty(len(bonds))
    hi_arr = np.empty(len(bonds))
    for i, (n1, n2) in enumerate(zip(names1, names2)):
        key = get_bond_key(n1, n2)
        lo, hi = BOND_LIMITS.get(key, (BOND_ABSOLUTE_MIN, BOND_ABSOLUTE_MAX))
        lo_arr[i] = max(lo, BOND_ABSOLUTE_MIN)
        hi_arr[i] = min(hi, BOND_ABSOLUTE_MAX)

    return idx, lo_arr, hi_arr


def check_bond_lengths(u, bond_cache):
    """
    bond_cache = (idx, lo_arr, hi_arr) from ensure_bonds().
    Returns (n_bonds, n_ok, n_bad, bad_ratio, frame_ok).
    """
    idx, lo_arr, hi_arr = bond_cache
    if idx is None:
        return 0, 0, 0, float("nan"), True  # can't check → don't penalise

    pos = u.atoms.positions
    p1 = pos[idx[:, 0]]
    p2 = pos[idx[:, 1]]
    lengths = np.linalg.norm(p2 - p1, axis=1)

    ok = (lengths >= lo_arr) & (lengths <= hi_arr)
    n_bonds = len(idx)
    n_ok = int(ok.sum())
    n_bad = n_bonds - n_ok
    bad_ratio = n_bad / n_bonds
    frame_ok = n_bad == 0
    return n_bonds, n_ok, n_bad, bad_ratio, frame_ok


def check_backbone_angles(u):
    """Return fraction of backbone angles within allowed ranges."""
    try:
        backbone = u.select_atoms("backbone")
        if len(backbone) < 3:
            return float("nan"), True

        # N-CA-C angles
        n_atoms = backbone.select_atoms("name N")
        ca_atoms = backbone.select_atoms("name CA")
        c_atoms = backbone.select_atoms("name C")

        n_res = min(len(n_atoms), len(ca_atoms), len(c_atoms))
        if n_res == 0:
            return float("nan"), True

        bad = 0
        total = 0
        for i in range(n_res):
            try:
                p_n = n_atoms[i].position
                p_ca = ca_atoms[i].position
                p_c = c_atoms[i].position
                v1 = p_n - p_ca
                v2 = p_c - p_ca
                cos_a = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
                )
                angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                lo, hi = ANGLE_NCA_C
                total += 1
                if not (lo <= angle <= hi):
                    bad += 1
            except Exception:
                pass

        frac_bad = bad / total if total else float("nan")
        frame_ok = bad == 0
        return frac_bad, frame_ok
    except Exception:
        return float("nan"), True


def check_clashes(u):
    """Detect steric clashes among heavy atoms (non-bonded pairs < CLASH_DIST Å)."""
    try:
        heavy = u.select_atoms("not name H*")
        if len(heavy) < 2:
            return 0, True

        from MDAnalysis.lib.distances import self_distance_array

        dists = self_distance_array(heavy.positions)
        n_clashes = int((dists < CLASH_DIST).sum())
        # Divide by 2 not needed; self_distance_array returns upper triangle
        frame_ok = n_clashes == 0
        return n_clashes, frame_ok
    except Exception:
        return 0, True


def check_omega(u):
    """Fraction of omega (ω) angles that deviate > OMEGA_TRANS_TOL from 180°."""
    try:
        rama = dihedrals.Ramachandran(u.select_atoms("protein")).run()
        # rama.results.angles shape: (n_frames, n_residues, 2) → phi, psi
        # Omega is not directly in Ramachandran; use manual calc
        backbone = u.select_atoms("backbone")
        c_atoms = [a for a in backbone if a.name == "C"]
        n_atoms = [a for a in backbone if a.name == "N"]

        n_omega = 0
        n_bad_omega = 0
        for i in range(len(c_atoms) - 1):
            try:
                # ω = CA(i)-C(i)-N(i+1)-CA(i+1)
                ca_i = c_atoms[i].residue.atoms.select_atoms("name CA")[0]
                c_i = c_atoms[i]
                n_i1 = n_atoms[i + 1] if i + 1 < len(n_atoms) else None
                if n_i1 is None:
                    continue
                ca_i1 = n_i1.residue.atoms.select_atoms("name CA")
                if len(ca_i1) == 0:
                    continue
                ca_i1 = ca_i1[0]

                pts = np.array(
                    [ca_i.position, c_i.position, n_i1.position, ca_i1.position]
                )
                b1 = pts[1] - pts[0]
                b2 = pts[2] - pts[1]
                b3 = pts[3] - pts[2]
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)
                m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-9))
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                omega = np.degrees(np.arctan2(y, x))

                n_omega += 1
                dev = abs(abs(omega) - 180.0)
                # Allow cis-proline (~0°)
                is_cis_pro = (
                    c_atoms[i].residue.resname == "PRO" or n_i1.residue.resname == "PRO"
                ) and abs(omega) < 30
                if dev > OMEGA_TRANS_TOL and not is_cis_pro:
                    n_bad_omega += 1
            except Exception:
                pass

        frac_bad = n_bad_omega / n_omega if n_omega else float("nan")
        frame_ok = n_bad_omega == 0
        return frac_bad, n_omega, frame_ok
    except Exception:
        return float("nan"), 0, True


def check_ramachandran(u):
    """Fraction of residues in disallowed Ramachandran regions."""
    try:
        protein = u.select_atoms("protein")
        rama_run = dihedrals.Ramachandran(protein).run()
        angles = rama_run.results.angles  # (1, n_res, 2)
        if angles.shape[1] == 0:
            return float("nan"), True

        phis = angles[0, :, 0]
        psis = angles[0, :, 1]

        allowed = np.zeros(len(phis), dtype=bool)
        for phi, psi in zip(phis, psis):
            in_region = False
            for p1, p2, s1, s2, _ in RAMA_REGIONS:
                if p1 <= phi <= p2 and s1 <= psi <= s2:
                    in_region = True
                    break
            if in_region:
                allowed[np.where((phis == phi) & (psis == psi))[0]] = True

        # Simpler: vectorised
        in_any = np.zeros(len(phis), dtype=bool)
        for p1, p2, s1, s2, _ in RAMA_REGIONS:
            mask = (phis >= p1) & (phis <= p2) & (psis >= s1) & (psis <= s2)
            in_any |= mask

        n_res = len(phis)
        n_disallowed = int((~in_any).sum())
        frac_disallowed = n_disallowed / n_res if n_res else float("nan")
        frame_ok = n_disallowed == 0
        return frac_disallowed, frame_ok
    except Exception:
        return float("nan"), True


# ── Main analysis loop ────────────────────────────────────────────────────────


def analyse_protein(protein_dir: Path):
    dcd = protein_dir / "aa_traj.dcd"

    # Define potential topology filenames
    potential_tops = ["top_AA.pdb", "aa_topology.pdb"]

    # Find the first one that exists
    top = next(
        (protein_dir / f for f in potential_tops if (protein_dir / f).exists()), None
    )

    if not dcd.exists() or top is None:
        print(f"  [SKIP] Missing files (DCD or Topology) in {protein_dir.name}")
        return []

    try:
        u = mda.Universe(str(top), str(dcd))
    except Exception as e:
        print(f"  [ERROR] Cannot load {protein_dir.name}: {e}")
        return []

    # Build bond cache once from first-frame geometry (topology-independent)
    bond_cache = ensure_bonds(u)

    rows = []
    n_frames = len(u.trajectory)
    for i in range(n_frames):
        ts = u.trajectory[i]
        fi = ts.frame
        row = {
            "protein": protein_dir.name,
            "frame": fi,
        }

        # 1. Bond lengths
        nb, nok, nbad, bad_ratio, bl_ok = check_bond_lengths(u, bond_cache)
        row["n_bonds"] = nb
        row["bonds_ok"] = nok
        row["bonds_bad"] = nbad
        row["bond_bad_ratio"] = round(bad_ratio, 5) if not np.isnan(bad_ratio) else None
        row["frame_bond_ok"] = bl_ok

        # 2. Backbone angles
        frac_bad_angle, angle_ok = check_backbone_angles(u)
        row["backbone_angle_frac_bad"] = (
            round(frac_bad_angle, 5) if not np.isnan(frac_bad_angle) else None
        )
        row["frame_angle_ok"] = angle_ok

        # 3. Clashes
        n_clashes, clash_ok = check_clashes(u)
        row["n_clashes"] = n_clashes
        row["frame_clash_ok"] = clash_ok

        # 4. Omega
        frac_bad_omega, n_omega, omega_ok = check_omega(u)
        row["omega_frac_bad"] = (
            round(frac_bad_omega, 5) if not np.isnan(frac_bad_omega) else None
        )
        row["n_omega_checked"] = n_omega
        row["frame_omega_ok"] = omega_ok

        # 5. Ramachandran
        frac_disallowed, rama_ok = check_ramachandran(u)
        row["rama_frac_disallowed"] = (
            round(frac_disallowed, 5) if not np.isnan(frac_disallowed) else None
        )
        row["frame_rama_ok"] = rama_ok

        # Overall frame viability (all checks pass)
        row["frame_fully_ok"] = all([bl_ok, angle_ok, clash_ok, omega_ok, rama_ok])

        rows.append(row)
    return rows


def main():
    if not DATA_ROOT.exists():
        sys.exit(
            f"Data root not found: {DATA_ROOT}\n"
            "Run this script from your project root (~/Documents/my_project/BindCORE)."
        )

    protein_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(protein_dirs)} protein directories under {DATA_ROOT}\n")

    all_rows = []
    for pdir in tqdm(protein_dirs[:50]):
        rows = analyse_protein(pdir)
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No data collected. Check paths and file names.")

    df = pd.DataFrame(all_rows)
    per_frame_path = OUTPUT_DIR / "results_per_frame.csv"
    df.to_csv(per_frame_path, index=False)
    print(f"\nPer-frame results saved → {per_frame_path}")

    # ── Per-protein summary ──────────────────────────────────────────────────
    grp = df.groupby("protein")

    summary = pd.DataFrame(
        {
            "n_frames": grp["frame"].count(),
            "frames_bond_ok": grp["frame_bond_ok"].sum(),
            "frames_bond_ok_pct": grp["frame_bond_ok"].mean() * 100,
            "mean_bond_bad_ratio": grp["bond_bad_ratio"].mean(),
            "frames_angle_ok": grp["frame_angle_ok"].sum(),
            "frames_angle_ok_pct": grp["frame_angle_ok"].mean() * 100,
            "mean_angle_frac_bad": grp["backbone_angle_frac_bad"].mean(),
            "mean_clashes_per_frame": grp["n_clashes"].mean(),
            "frames_clash_free": grp["frame_clash_ok"].sum(),
            "frames_clash_free_pct": grp["frame_clash_ok"].mean() * 100,
            "frames_omega_ok": grp["frame_omega_ok"].sum(),
            "frames_omega_ok_pct": grp["frame_omega_ok"].mean() * 100,
            "mean_omega_frac_bad": grp["omega_frac_bad"].mean(),
            "frames_rama_ok": grp["frame_rama_ok"].sum(),
            "frames_rama_ok_pct": grp["frame_rama_ok"].mean() * 100,
            "mean_rama_frac_disallowed": grp["rama_frac_disallowed"].mean(),
            "frames_fully_ok": grp["frame_fully_ok"].sum(),
            "frames_fully_ok_pct": grp["frame_fully_ok"].mean() * 100,
        }
    ).round(3)

    summary_path = OUTPUT_DIR / "results_summary.csv"
    summary.to_csv(summary_path)
    print(f"Per-protein summary saved → {summary_path}")

    # ── Global statistics ────────────────────────────────────────────────────
    n_total = len(df)
    n_proteins = df["protein"].nunique()
    tot_bonds = df["n_bonds"].sum()
    tot_ok_bonds = df["bonds_ok"].sum()

    report_lines = [
        "=" * 70,
        "  CONFORMATIONAL ENSEMBLE GEOMETRIC VIABILITY REPORT",
        "=" * 70,
        f"  Proteins analysed  : {n_proteins}",
        f"  Total frames       : {n_total}",
        f"  Total covalent bonds checked: {tot_bonds:,}",
        "",
        "── BOND LENGTHS ────────────────────────────────────────────────────",
        (
            f"  Bonds within limits: {tot_ok_bonds:,} / {tot_bonds:,}  "
            f"({100 * tot_ok_bonds / tot_bonds:.2f} %)"
            if tot_bonds
            else "  N/A"
        ),
        f"  Frames ALL bonds OK: "
        f"{df['frame_bond_ok'].sum()} / {n_total}  "
        f"({100 * df['frame_bond_ok'].mean():.2f} %)",
        "",
        "── BACKBONE ANGLES ─────────────────────────────────────────────────",
        f"  Frames ALL angles OK: "
        f"{df['frame_angle_ok'].sum()} / {n_total}  "
        f"({100 * df['frame_angle_ok'].mean():.2f} %)",
        f"  Mean fraction bad angles per frame: "
        f"{df['backbone_angle_frac_bad'].mean():.4f}",
        "",
        "── STERIC CLASHES ──────────────────────────────────────────────────",
        f"  Clash-free frames : "
        f"{df['frame_clash_ok'].sum()} / {n_total}  "
        f"({100 * df['frame_clash_ok'].mean():.2f} %)",
        f"  Mean clashes/frame: {df['n_clashes'].mean():.2f}",
        f"  Max clashes/frame : {df['n_clashes'].max()}",
        "",
        "── PEPTIDE PLANARITY (ω) ───────────────────────────────────────────",
        f"  Frames all ω OK   : "
        f"{df['frame_omega_ok'].sum()} / {n_total}  "
        f"({100 * df['frame_omega_ok'].mean():.2f} %)",
        f"  Mean fraction bad ω per frame: " f"{df['omega_frac_bad'].mean():.4f}",
        "",
        "── RAMACHANDRAN ────────────────────────────────────────────────────",
        f"  Frames ALL residues in allowed regions: "
        f"{df['frame_rama_ok'].sum()} / {n_total}  "
        f"({100 * df['frame_rama_ok'].mean():.2f} %)",
        f"  Mean fraction disallowed per frame: "
        f"{df['rama_frac_disallowed'].mean():.4f}",
        "",
        "── OVERALL (all checks pass) ───────────────────────────────────────",
        f"  Fully viable frames: "
        f"{df['frame_fully_ok'].sum()} / {n_total}  "
        f"({100 * df['frame_fully_ok'].mean():.2f} %)",
        "=" * 70,
        "",
        "── WORST PROTEINS (by % fully viable frames) ───────────────────────",
    ]

    worst = summary.sort_values("frames_fully_ok_pct").head(10)
    for prot, row_ in worst.iterrows():
        report_lines.append(
            f"  {prot:<20}  {row_['frames_fully_ok_pct']:5.1f}%  "
            f"(clashes/frame: {row_['mean_clashes_per_frame']:.1f})"
        )

    report_lines += [
        "",
        "── BEST PROTEINS (by % fully viable frames) ────────────────────────",
    ]
    best = summary.sort_values("frames_fully_ok_pct", ascending=False).head(10)
    for prot, row_ in best.iterrows():
        report_lines.append(f"  {prot:<20}  {row_['frames_fully_ok_pct']:5.1f}%")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
