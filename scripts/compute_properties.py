import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import logging
import concurrent.futures
from pathlib import Path
import h5py
from tqdm import tqdm
from filelock import FileLock 

from bindcore.data.properties_extraction import (
    process_single_protein,
    save_properties_to_h5,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute conformational MD properties with dynamic file resolution."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directory containing protein subfolders.",
    )
    parser.add_argument(
        "--workers", type=int, default=15,
        help="Number of parallel processes.",
    )
    parser.add_argument(
        "--pdb_name", type=str, default="_allatom.pdb",
        help="Full filename or suffix (if --dynamic is used).",
    )
    parser.add_argument(
        "--xtc_name", type=str, default="_allatom.xtc",
        help="Full filename or suffix (if --dynamic is used).",
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Prepend folder name (Protein ID) to pdb/xtc arguments.",
    )
    parser.add_argument(
        "--convert_dcd", action="store_true",
        help="If XTC is missing, look for DCD and convert it.",
    )
    parser.add_argument(
        "--n_subsample_trajectory", type=int, default=-1,
        help="Number of trajectory frames to use. -1 means all.",
    )
    parser.add_argument(
        "--job_index", type=int, default=0,
        help="0-based index of this job within the OAR array (default: 0).",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Total number of jobs in the OAR array. Directories are sharded "
             "across jobs via modulo so each job processes a non-overlapping subset.",
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    return parser.parse_args()


def resolve_filenames(d: Path, pdb_name: str, xtc_name: str, dynamic: bool):
    if dynamic:
        return f"{d.name}{pdb_name}", f"{d.name}{xtc_name}"
    return pdb_name, xtc_name


def submit_kwargs(d: Path, pdb: str, xtc: str, args) -> dict:
    return dict(
        pdb_name=pdb,
        xtc_name=xtc,
        convert_dcd=args.convert_dcd,
        n_subsample_trajectory=args.n_subsample_trajectory,
    )


def get_output_path(input_dir: Path) -> Path:
    return Path("data/properties/") / f"{input_dir.stem}_derived_properties.h5"


def get_lock_path(output_h5: Path) -> Path:
    """One lock file per HDF5 output, shared across all concurrent instances."""
    return output_h5.with_suffix(".lock")


def get_already_processed(output_h5: Path) -> set[str]:
    """Return the set of protein IDs already present in the HDF5 file.

    Must be called while holding the file lock when used for deduplication.
    """
    if not output_h5.exists():
        return set()
    with h5py.File(output_h5, "r") as h5f:
        return set(h5f.keys())


def save_incremental(pid: str, props: dict, output_h5: Path, lock: FileLock) -> None:
    """Append a single protein's result to the HDF5 file.

    The lock serialises all writers across concurrent jobs. The re-check inside
    the lock closes any startup race between jobs launched simultaneously.
    """
    with lock:
        if pid not in get_already_processed(output_h5):
            save_properties_to_h5({pid: props}, output_h5)


def handle_result(pid, props, output_h5: Path, lock: FileLock, results_count: list) -> None:
    """Validate, persist, and tally a single protein result."""
    if "error" in props:
        logging.error(f"Protein {pid}: {props['error']}")
    else:
        save_incremental(pid, props, output_h5, lock)
        results_count[0] += 1


def run_sequential(directories, args, output_h5: Path, lock: FileLock) -> int:
    results_count = [0]
    for d in tqdm(directories):
        pdb, xtc = resolve_filenames(d, args.pdb_name, args.xtc_name, args.dynamic)
        try:
            pid, props = process_single_protein(d, **submit_kwargs(d, pdb, xtc, args))
            handle_result(pid, props, output_h5, lock, results_count)
        except Exception as e:
            logging.error(f"Critical process failure for {d.name}: {e}")
    return results_count[0]


def run_parallel(directories, args, output_h5: Path, lock: FileLock) -> int:
    results_count = [0]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_protein, d,
                **submit_kwargs(d, *resolve_filenames(d, args.pdb_name, args.xtc_name, args.dynamic), args)
            ): d
            for d in directories
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            protein_dir = futures[future]
            try:
                pid, props = future.result()
                handle_result(pid, props, output_h5, lock, results_count)
            except Exception as e:
                logging.error(f"Critical process failure for {protein_dir.name}: {e}")
    return results_count[0]


def main():
    args = parse_args()

    # Sort for consistent ordering across all jobs — required for modulo sharding.
    directories = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    output_h5 = get_output_path(args.input_dir)
    
    # CRITICAL FIX 2: Ensure the parent directory exists BEFORE creating the FileLock
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the lock
    lock = FileLock(get_lock_path(output_h5))

    # CRITICAL FIX 3: Wrap the startup read inside the FileLock. 
    # If another OAR job is writing its first result, this job will patiently queue 
    # rather than crashing on a corrupted/half-written HDF5 state.
    with lock:
        already_done = get_already_processed(output_h5)
        
    directories = [d for d in directories if d.name not in already_done]

    # Shard directories across OAR array jobs using modulo.
    if args.n_jobs > 1:
        directories = [d for i, d in enumerate(directories) if i % args.n_jobs == args.job_index]

    print(f"[Job {args.job_index}/{args.n_jobs}] Skipping {len(already_done)} already-processed proteins.")
    print(f"[Job {args.job_index}/{args.n_jobs}] Assigned {len(directories)} proteins to this job.")

    if args.debug:
        directories = directories[:5]

    if not directories:
        print(f"[Job {args.job_index}/{args.n_jobs}] Nothing left to do.")
        return

    print(f"Processing with {args.workers} workers. Output: {output_h5}")

    runner = run_sequential if args.workers == 1 else run_parallel
    n_saved = runner(directories, args, output_h5, lock)

    print(f"[Job {args.job_index}/{args.n_jobs}] Done. Saved {n_saved} proteins.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()