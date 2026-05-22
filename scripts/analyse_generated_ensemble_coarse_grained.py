"""
Coarse-Grained (Cα-only) Ensemble Geometric Viability Analysis
===============================================================
Input : topology.pdb  +  traj.dcd   (one bead = one residue = Cα)
Output: results_cg/
          report.txt
          summary_per_protein.csv
          per_frame.csv

Checks implemented
──────────────────
1. Cα–Cα virtual bond lengths        sequential Cα–Cα distance (3.5–4.1 Å ideal;
                                      flag >4.5 Å or <3.2 Å as broken/clashed)
2. Cα–Cα–Cα pseudo-bond angles       expect 80–150 °  (broad; IDPs are flexible)
3. Cα–Cα–Cα–Cα pseudo-dihedrals     distribution check; flag |ω| > 170 ° outliers
                                      (extended β) vs helix cluster  (–90° to –30°)
4. Cα clashes                         any two non-sequential Cα closer than 3.5 Å
   (non-bonded)                       (hard-sphere floor for a Cα bead)
5. End-to-end distance                Ree = distance between first and last Cα
6. Radius of gyration                 Rg  via MDAnalysis
7. Sequence-length normalised Rg      Rg / N^0.5  (expected ~2–4 Å for IDPs)
8. Local compactness                  mean of all pairwise Cα distances
9. Chain continuity                   no single sequential gap > MAX_BOND Å
   (frame-level pass/fail)

All thresholds are documented constants at the top of the file.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from tqdm import tqdm

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
except ImportError:
    sys.exit("MDAnalysis not found.  pip install MDAnalysis")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  edit these to tune thresholds
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT = Path("data/conformational_ensemble/IDPFold2")
OUTPUT_DIR = Path("results_cg")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Sequential Cα–Cα virtual bond ─────────────────────────────────────────
CA_BOND_IDEAL = 3.81  # Å  (peptide bond geometry → Cα–Cα)
CA_BOND_MIN = 3.20  # Å  absolute lower bound  (clash / bad backmapping)
CA_BOND_SOFT_LO = 3.50  # Å  soft lower bound  (warn, not fail)
CA_BOND_SOFT_HI = 4.10  # Å  soft upper bound
CA_BOND_MAX = 4.50  # Å  absolute upper bound  (chain break)

# ── 2. Cα–Cα–Cα pseudo-bond angle ────────────────────────────────────────────
CA_ANGLE_MIN = 70.0  # °
CA_ANGLE_MAX = 160.0  # °

# ── 3. Pseudo-dihedral (not used as pass/fail, reported as distribution) ──────
# (thresholds for flagging fully extended conformations)
CA_DIHEDRAL_EXTENDED_MIN = 150.0  # |φ| > this → extended β-like

# ── 4. Non-bonded Cα clash ────────────────────────────────────────────────────
CA_CLASH_DIST = 3.50  # Å  (two non-sequential Cα this close = steric clash)
CA_CLASH_SEQ_SKIP = 2  # ignore pairs within this many residues in sequence

# ── 5–9. Shape / structural descriptors ──────────────────────────────────────
RG_NORM_IDP_LO = 1.5  # Å  lower bound for Rg / sqrt(N)  (IDPs are expanded)
RG_NORM_IDP_HI = 6.0  # Å  upper bound

# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _vec_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a–b–c in degrees."""
    u = a - b
    v = c - b
    cos_t = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Dihedral a–b–c–d in degrees  (–180 to +180)."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    x = np.dot(n1, n2)
    y = np.dot(m, n2)
    return float(np.degrees(np.arctan2(y, x)))


# ══════════════════════════════════════════════════════════════════════════════
#  PER-FRAME CHECKS
# ══════════════════════════════════════════════════════════════════════════════


def check_ca_bonds(pos: np.ndarray):
    """
    Sequential Cα–Cα distances.
    Returns dict of counts + frame-level pass/fail (no bond outside [MIN, MAX]).
    """
    if len(pos) < 2:
        return dict(
            n_bonds=0,
            bonds_ok=0,
            bonds_soft_ok=0,
            bonds_bad=0,
            bond_bad_ratio=np.nan,
            bond_mean=np.nan,
            bond_std=np.nan,
            bond_min=np.nan,
            bond_max=np.nan,
            frame_bond_ok=True,
        )

    diffs = np.diff(pos, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    hard_ok = (dists >= CA_BOND_MIN) & (dists <= CA_BOND_MAX)
    soft_ok = (dists >= CA_BOND_SOFT_LO) & (dists <= CA_BOND_SOFT_HI)

    n = len(dists)
    return dict(
        n_bonds=n,
        bonds_ok=int(hard_ok.sum()),
        bonds_soft_ok=int(soft_ok.sum()),
        bonds_bad=int((~hard_ok).sum()),
        bond_bad_ratio=float((~hard_ok).sum() / n),
        bond_mean=float(dists.mean()),
        bond_std=float(dists.std()),
        bond_min=float(dists.min()),
        bond_max=float(dists.max()),
        frame_bond_ok=bool(hard_ok.all()),
    )


def check_ca_angles(pos: np.ndarray):
    """
    Cα–Cα–Cα pseudo-bond angles for triplets i, i+1, i+2.
    """
    if len(pos) < 3:
        return dict(
            n_angles=0,
            angles_bad=0,
            angle_bad_ratio=np.nan,
            angle_mean=np.nan,
            angle_std=np.nan,
            frame_angle_ok=True,
        )

    angles = np.array(
        [_vec_angle(pos[i], pos[i + 1], pos[i + 2]) for i in range(len(pos) - 2)]
    )

    bad = (angles < CA_ANGLE_MIN) | (angles > CA_ANGLE_MAX)
    n = len(angles)
    return dict(
        n_angles=n,
        angles_bad=int(bad.sum()),
        angle_bad_ratio=float(bad.sum() / n),
        angle_mean=float(angles.mean()),
        angle_std=float(angles.std()),
        frame_angle_ok=bool(not bad.any()),
    )


def check_ca_dihedrals(pos: np.ndarray):
    """
    Cα–Cα–Cα–Cα pseudo-dihedrals for quadruplets i..i+3.
    Returns distribution stats and fraction of 'extended' conformations.
    """
    if len(pos) < 4:
        return dict(
            n_dihedrals=0,
            dihedral_mean=np.nan,
            dihedral_std=np.nan,
            frac_extended=np.nan,
        )

    dihs = np.array(
        [
            _dihedral(pos[i], pos[i + 1], pos[i + 2], pos[i + 3])
            for i in range(len(pos) - 3)
        ]
    )

    extended = np.abs(dihs) >= CA_DIHEDRAL_EXTENDED_MIN
    return dict(
        n_dihedrals=len(dihs),
        dihedral_mean=float(dihs.mean()),
        dihedral_std=float(dihs.std()),
        frac_extended=float(extended.sum() / len(dihs)),
    )


def check_clashes(pos: np.ndarray):
    """
    Non-bonded Cα clashes: all pairs (i, j) with |i–j| > CA_CLASH_SEQ_SKIP
    that are closer than CA_CLASH_DIST Å.
    """
    n = len(pos)
    if n < CA_CLASH_SEQ_SKIP + 2:
        return dict(n_clashes=0, frame_clash_ok=True)

    n_clashes = 0
    for i in range(n):
        for j in range(i + CA_CLASH_SEQ_SKIP + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < CA_CLASH_DIST:
                n_clashes += 1

    return dict(
        n_clashes=n_clashes,
        frame_clash_ok=n_clashes == 0,
    )


def check_clashes_fast(pos: np.ndarray):
    """
    Vectorised version of check_clashes for large proteins (N > 200).
    Builds the full pairwise distance matrix once.
    """
    n = len(pos)
    if n < CA_CLASH_SEQ_SKIP + 2:
        return dict(n_clashes=0, frame_clash_ok=True)

    # Pairwise distance matrix via broadcasting
    diff = pos[:, None, :] - pos[None, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=-1)  # (N,N)

    # Mask: upper triangle, skip sequential neighbours
    mask = np.triu(np.ones((n, n), dtype=bool), k=CA_CLASH_SEQ_SKIP + 1)
    n_clashes = int((dist[mask] < CA_CLASH_DIST).sum())
    return dict(
        n_clashes=n_clashes,
        frame_clash_ok=n_clashes == 0,
    )


def shape_descriptors(pos: np.ndarray, ag):
    """
    Ree, Rg, normalised Rg, and mean pairwise distance.
    ag  = MDAnalysis AtomGroup (for the built-in Rg method).
    """
    n = len(pos)
    if n < 2:
        return dict(
            ree=np.nan,
            rg=np.nan,
            rg_norm=np.nan,
            mean_pairwise=np.nan,
            frame_shape_ok=True,
        )

    # End-to-end distance
    ree = float(np.linalg.norm(pos[-1] - pos[0]))

    # Radius of gyration (MDAnalysis, mass-weighted if masses available)
    try:
        rg = float(ag.radius_of_gyration())
    except Exception:
        centroid = pos.mean(axis=0)
        rg = float(np.sqrt(((pos - centroid) ** 2).sum(axis=1).mean()))

    rg_norm = rg / np.sqrt(n)

    # Mean pairwise Cα distance (upper triangle, all pairs)
    if n <= 500:
        diff = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        upper = dists[np.triu_indices(n, k=1)]
        mean_pw = float(upper.mean())
    else:
        # Subsample for very long chains
        idx = np.random.choice(n, size=min(n, 300), replace=False)
        sub = pos[idx]
        diff = sub[:, None, :] - sub[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        upper = dists[np.triu_indices(len(sub), k=1)]
        mean_pw = float(upper.mean())

    shape_ok = RG_NORM_IDP_LO <= rg_norm <= RG_NORM_IDP_HI
    return dict(
        ree=round(ree, 3),
        rg=round(rg, 3),
        rg_norm=round(rg_norm, 3),
        mean_pairwise=round(mean_pw, 3),
        frame_shape_ok=bool(shape_ok),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PER-PROTEIN LOOP
# ══════════════════════════════════════════════════════════════════════════════


def analyse_protein(protein_dir: Path):
    top = protein_dir / "topology.pdb"
    dcd = protein_dir / "traj.dcd"

    if not top.exists() or not dcd.exists():
        print(f"  [SKIP] Missing files in {protein_dir.name}")
        return []

    try:
        u = mda.Universe(str(top), str(dcd))
        ca = u.select_atoms("name CA")
    except Exception as e:
        print(f"  [ERROR] {protein_dir.name}: {e}")
        return []

    if len(ca) == 0:
        print(f"  [SKIP] No CA atoms in {protein_dir.name}")
        return []

    n_res = len(ca)
    n_frames = len(u.trajectory)
    print(
        f"  {protein_dir.name:25s}  {n_res:>4d} res  {n_frames:>4d} frames",
        end="",
        flush=True,
    )

    rows = []
    for ts in u.trajectory:
        pos = ca.positions.copy()  # (N,3) float32

        row = {"protein": protein_dir.name, "frame": ts.frame, "n_residues": n_res}

        # 1. Virtual bond lengths
        row.update(check_ca_bonds(pos))

        # 2. Pseudo-bond angles
        row.update(check_ca_angles(pos))

        # 3. Pseudo-dihedrals (distribution only)
        row.update(check_ca_dihedrals(pos))

        # 4. Non-bonded clashes
        row.update(check_clashes_fast(pos))

        # 5–9. Shape descriptors
        row.update(shape_descriptors(pos, ca))

        # Overall frame viability  (bond + angle + clash; shape is soft)
        row["frame_fully_ok"] = all(
            [
                row["frame_bond_ok"],
                row["frame_angle_ok"],
                row["frame_clash_ok"],
            ]
        )

        rows.append(row)

    # quick per-protein pass-rate for inline display
    ok_pct = 100 * sum(r["frame_fully_ok"] for r in rows) / len(rows)
    print(f"  →  {ok_pct:5.1f}% fully OK")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    if not DATA_ROOT.exists():
        sys.exit(f"Data root not found: {DATA_ROOT}\n" "Run from your project root.")

    protein_dirs = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    print(f"Found {len(protein_dirs)} directories under {DATA_ROOT}\n")

    all_rows = []
    for pdir in tqdm(protein_dirs[:100]):
        all_rows.extend(analyse_protein(pdir))

    if not all_rows:
        sys.exit("No data collected.")

    df = pd.DataFrame(all_rows)

    # ── Save per-frame CSV ────────────────────────────────────────────────────
    pf_path = OUTPUT_DIR / "per_frame.csv"
    df.to_csv(pf_path, index=False)

    # ── Per-protein summary ───────────────────────────────────────────────────
    g = df.groupby("protein")

    summary = pd.DataFrame(
        {
            "n_residues": g["n_residues"].first(),
            "n_frames": g["frame"].count(),
            # Bond lengths
            "bond_mean_Å": g["bond_mean"].mean().round(3),
            "bond_std_Å": g["bond_std"].mean().round(3),
            "bond_bad_ratio": g["bond_bad_ratio"].mean().round(4),
            "frames_bond_ok_%": (g["frame_bond_ok"].mean() * 100).round(1),
            # Angles
            "angle_mean_deg": g["angle_mean"].mean().round(2),
            "angle_bad_ratio": g["angle_bad_ratio"].mean().round(4),
            "frames_angle_ok_%": (g["frame_angle_ok"].mean() * 100).round(1),
            # Dihedrals
            "dihedral_mean_deg": g["dihedral_mean"].mean().round(2),
            "dihedral_std_deg": g["dihedral_std"].mean().round(2),
            "frac_extended": g["frac_extended"].mean().round(4),
            # Clashes
            "mean_clashes_per_frame": g["n_clashes"].mean().round(2),
            "frames_clash_free_%": (g["frame_clash_ok"].mean() * 100).round(1),
            # Shape
            "mean_Ree_Å": g["ree"].mean().round(2),
            "mean_Rg_Å": g["rg"].mean().round(2),
            "mean_Rg_norm": g["rg_norm"].mean().round(3),
            "mean_pairwise_dist_Å": g["mean_pairwise"].mean().round(2),
            "frames_shape_ok_%": (g["frame_shape_ok"].mean() * 100).round(1),
            # Overall
            "frames_fully_ok_%": (g["frame_fully_ok"].mean() * 100).round(1),
        }
    )

    sum_path = OUTPUT_DIR / "summary_per_protein.csv"
    summary.to_csv(sum_path)

    # ── Global statistics ─────────────────────────────────────────────────────
    N = len(df)
    n_prot = df["protein"].nunique()
    tot_bonds = df["n_bonds"].sum()
    bad_bonds = df["bonds_bad"].sum()
    tot_ang = df["n_angles"].sum()
    bad_ang = df["angles_bad"].sum()

    def pct(n, d):
        return f"{100*n/d:.2f}%" if d else "N/A"

    sep = "─" * 68
    lines = [
        "=" * 68,
        "  CG (Cα-only) ENSEMBLE GEOMETRIC VIABILITY REPORT",
        "=" * 68,
        f"  Proteins analysed   : {n_prot}",
        f"  Total frames        : {N:,}",
        f"  Avg residues/protein: {df['n_residues'].mean():.0f}",
        "",
        f"{sep}",
        f"  1. VIRTUAL Cα–Cα BOND LENGTHS   (ideal {CA_BOND_IDEAL} Å)",
        f"{sep}",
        f"  Total sequential bonds checked : {tot_bonds:,}",
        f"  Bonds within hard limits       : {pct(tot_bonds-bad_bonds, tot_bonds)}",
        f"  Bonds within soft limits       : {pct(df['bonds_soft_ok'].sum(), tot_bonds)}",
        f"  Frames ALL bonds hard-OK       : {df['frame_bond_ok'].sum():,} / {N}  "
        f"({pct(df['frame_bond_ok'].sum(), N)})",
        f"  Mean bond length (all frames)  : {df['bond_mean'].mean():.3f} Å  "
        f"± {df['bond_std'].mean():.3f}",
        "",
        f"{sep}",
        f"  2. Cα–Cα–Cα PSEUDO-BOND ANGLES   ({CA_ANGLE_MIN}–{CA_ANGLE_MAX} °)",
        f"{sep}",
        f"  Total angles checked           : {tot_ang:,}",
        f"  Angles within limits           : {pct(tot_ang-bad_ang, tot_ang)}",
        f"  Frames ALL angles OK           : {df['frame_angle_ok'].sum():,} / {N}  "
        f"({pct(df['frame_angle_ok'].sum(), N)})",
        f"  Mean angle (all frames)        : {df['angle_mean'].mean():.2f} °  "
        f"± {df['angle_std'].mean():.2f}",
        "",
        f"{sep}",
        f"  3. Cα PSEUDO-DIHEDRALS   (distribution; |φ| ≥ {CA_DIHEDRAL_EXTENDED_MIN}° = extended)",
        f"{sep}",
        f"  Mean dihedral                  : {df['dihedral_mean'].mean():.2f} °",
        f"  Std dihedral                   : {df['dihedral_std'].mean():.2f} °",
        f"  Mean fraction extended (|φ|≥{CA_DIHEDRAL_EXTENDED_MIN:.0f}°): "
        f"{df['frac_extended'].mean():.4f}",
        "",
        f"{sep}",
        f"  4. NON-BONDED Cα CLASHES   (< {CA_CLASH_DIST} Å, skip ±{CA_CLASH_SEQ_SKIP} neighbours)",
        f"{sep}",
        f"  Clash-free frames              : {df['frame_clash_ok'].sum():,} / {N}  "
        f"({pct(df['frame_clash_ok'].sum(), N)})",
        f"  Mean clashes per frame         : {df['n_clashes'].mean():.2f}",
        f"  Max clashes in one frame       : {df['n_clashes'].max()}",
        "",
        f"{sep}",
        f"  5–9. SHAPE DESCRIPTORS",
        f"{sep}",
        f"  Mean Ree                       : {df['ree'].mean():.2f} Å",
        f"  Mean Rg                        : {df['rg'].mean():.2f} Å",
        f"  Mean Rg / √N                   : {df['rg_norm'].mean():.3f} Å  "
        f"(IDP expected {RG_NORM_IDP_LO}–{RG_NORM_IDP_HI})",
        f"  Mean pairwise Cα distance      : {df['mean_pairwise'].mean():.2f} Å",
        f"  Frames with plausible Rg/√N    : {df['frame_shape_ok'].sum():,} / {N}  "
        f"({pct(df['frame_shape_ok'].sum(), N)})",
        "",
        f"{sep}",
        f"  OVERALL  (bond + angle + clash all pass)",
        f"{sep}",
        f"  Fully viable frames            : {df['frame_fully_ok'].sum():,} / {N}  "
        f"({pct(df['frame_fully_ok'].sum(), N)})",
        "",
    ]

    # Worst 10 proteins
    lines += [f"{sep}", "  WORST 10 PROTEINS  (by % fully-viable frames)", f"{sep}"]
    for prot, row in summary.nsmallest(10, "frames_fully_ok_%").iterrows():
        lines.append(
            f"  {prot:<28s}  {row['frames_fully_ok_%']:5.1f}%  "
            f"Rg={row['mean_Rg_Å']:.1f}Å  clashes/frame={row['mean_clashes_per_frame']:.1f}"
        )

    # Best 10
    lines += ["", f"{sep}", "  BEST 10 PROTEINS  (by % fully-viable frames)", f"{sep}"]
    for prot, row in summary.nlargest(10, "frames_fully_ok_%").iterrows():
        lines.append(
            f"  {prot:<28s}  {row['frames_fully_ok_%']:5.1f}%  "
            f"Rg={row['mean_Rg_Å']:.1f}Å"
        )

    lines += [
        "",
        "=" * 68,
        f"  Per-frame CSV  : {pf_path}",
        f"  Summary CSV    : {sum_path}",
        "=" * 68,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    rep_path = OUTPUT_DIR / "report.txt"
    with open(rep_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport → {rep_path}")


if __name__ == "__main__":
    main()
