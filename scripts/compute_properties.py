import argparse
import logging
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

from bindcore.data.properties_extraction import process_single_protein


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

    # Gather directories
    directories = [d for d in args.input_dir.iterdir() if d.is_dir()]
    if not directories:
        print(f"No subdirectories found in {args.input_dir}")
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

    # Save Output
    if results_dict:
        output_h5 = (
            Path("data/properties/") / f"{args.input_dir.stem}_derived_properties.h5"
        )
        output_h5.parent.mkdir(parents=True, exist_ok=True)

        # save_properties_to_h5(results_dict, output_h5)
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
