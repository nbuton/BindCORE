import argparse
import logging
import concurrent.futures
from pathlib import Path
<<<<<<< HEAD
from bindcore.data.properties_extraction import save_properties_to_h5
import mdtraj as md
from tqdm import tqdm


def process_single_protein(
    protein_dir: Path, pdb_name: str, xtc_name: str, convert_dcd: bool = False
) -> tuple[str, dict]:
    """
    Processes a single protein directory, handles file resolution,
    optional DCD conversion, and property extraction.
    """
    protein_id = protein_dir.name
    pdb_path = protein_dir / pdb_name
    xtc_path = protein_dir / xtc_name

    # 1. Check for Topology
    if not pdb_path.exists():
        return protein_id, {"error": f"PDB not found: {pdb_path.name}"}

    # 2. Check for Trajectory / Handle DCD Conversion
    if not xtc_path.exists():
        if convert_dcd:
            dcd_path = xtc_path.with_suffix(".dcd")
            if dcd_path.exists():
                try:
                    # Convert DCD to XTC
                    traj = md.load(str(dcd_path), top=str(pdb_path))
                    traj.save_xtc(str(xtc_path))
                except Exception as e:
                    return protein_id, {"error": f"DCD conversion failed: {e}"}
            else:
                return protein_id, {
                    "error": f"Neither XTC nor DCD found for {protein_id}"
                }
        else:
            return protein_id, {"error": f"XTC not found: {xtc_path.name}"}

    # 3. Extraction Logic
    # Replace the placeholder below with your actual analysis functions
    try:
        # Example: props = calculate_metrics(pdb_path, xtc_path)
        props = {"status": "success"}  # Placeholder
        return protein_id, props
    except Exception as e:
        return protein_id, {"error": f"Extraction failed: {e}"}
=======
import h5py
from tqdm import tqdm

from bindcore.data.properties_extraction import (
    process_single_protein,
    save_properties_to_h5,
)
>>>>>>> 50f5de4aad0ab7fb839131a1ccf99a770fc62430


def main():
    parser = argparse.ArgumentParser(
        description="Compute conformational MD properties with dynamic file resolution."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing protein subfolders.",
    )
    parser.add_argument(
        "--workers", type=int, default=15, help="Number of parallel processes."
    )

    # Naming Arguments
    parser.add_argument(
        "--pdb_name",
        type=str,
        default="_allatom.pdb",
        help="Full filename or suffix (if --dynamic is used).",
    )
    parser.add_argument(
        "--xtc_name",
        type=str,
        default="_allatom.xtc",
        help="Full filename or suffix (if --dynamic is used).",
    )

    # Functional Flags
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Prepend folder name (Protein ID) to pdb/xtc arguments.",
    )
    parser.add_argument(
        "--convert_dcd",
        action="store_true",
        help="If XTC is missing, look for DCD and convert it.",
    )

    args = parser.parse_args()

<<<<<<< HEAD
    # Gather directories
    directories = [d for d in args.input_dir.iterdir() if d.is_dir()]
    if not directories:
        print(f"No subdirectories found in {args.input_dir}")
=======
    output_h5 = (
        Path("data/properties/") / f"{args.input_dir.stem}_derived_properties.h5"
    )

    directories = [d for d in args.input_dir.iterdir() if d.is_dir()]
    # 3. Filter out IDs already present in the HDF5 file
    if output_h5.exists():
        with h5py.File(output_h5, "r") as h5f:
            existing_ids = set(h5f.keys())

        # Filter directories: only keep those whose name (stem) isn't in the H5
        directories = [d for d in directories if d.stem not in existing_ids]

        print(f"Skipped {len(existing_ids)} proteins already present in the H5 file.")

    if not directories:
        print("All proteins are already processed. Exiting.")
>>>>>>> 50f5de4aad0ab7fb839131a1ccf99a770fc62430
        return

    print(
        f"Found {len(directories)} proteins. Processing with {args.workers} workers..."
    )

    results_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Construct names and submit jobs
        futures = {}
        for d in directories:
            pdb = f"{d.name}{args.pdb_name}" if args.dynamic else args.pdb_name
            xtc = f"{d.name}{args.xtc_name}" if args.dynamic else args.xtc_name

            job = executor.submit(
                process_single_protein,
                d,
                pdb_name=pdb,
                xtc_name=xtc,
                convert_dcd=args.convert_dcd,
            )
            futures[job] = d

        # Process results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            protein_dir = futures[future]
            try:
                pid, props = future.result()
                if "error" in props:
                    logging.error(f"Protein {pid}: {props['error']}")
                else:
                    results_dict[pid] = props
            except Exception as e:
                logging.error(f"Critical process failure for {protein_dir.name}: {e}")

<<<<<<< HEAD
    # Save Output
    if results_dict:
        output_h5 = (
            Path("data/properties/") / f"{args.input_dir.stem}_derived_properties.h5"
        )
        output_h5.parent.mkdir(parents=True, exist_ok=True)
=======
    output_h5.parent.mkdir(parents=True, exist_ok=True)
>>>>>>> 50f5de4aad0ab7fb839131a1ccf99a770fc62430

        save_properties_to_h5(results_dict, output_h5)
        print(
            f"Successfully processed {len(results_dict)} proteins. Saved to {output_h5}"
        )
    else:
        print("No successful results to save.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
