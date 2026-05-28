"""
BindCore — Step 4: Make predictions
=====================================
Thin CLI wrapper around bindcore.predictor.

Usage
-----
    python scripts/predict.py \
        --model      data/models/bindcore.pt \
        --h5         data/protein_MD_properties.h5 \
        --datasets   data/CLIP_dataset/TE440_reduced.txt \
                     data/CLIP_dataset/TR1000_reduced.txt \
        --output_dir data/predictions/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch

from bindcore.engine.predictor import load_checkpoint, predict_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BindCore predictions.")
    parser.add_argument("--model", default="data/models/bindCORE.pt")
    parser.add_argument("--h5", default="data/protein_MD_properties.h5")
    parser.add_argument(
        "--datasets", nargs="+", default=["data/CLIP_dataset/TE440_reduced.txt"]
    )
    parser.add_argument("--output_dir", default="data/predictions/")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.model, device)
    print(f"Loaded checkpoint: {args.model}")

    model_stem = Path(args.model).stem
    h5_stem = Path(args.h5).stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as h5_features:
        for dataset_path in args.datasets:
            dataset_stem = Path(dataset_path).stem
            filename = f"{model_stem}_{h5_stem}_{dataset_stem}.csv"
            output_filepath = str(output_dir / filename)
            print(f"\nRunning inference on: {dataset_path}")
            predict_dataset(
                dataset_path=dataset_path,
                h5_features=h5_features,
                model=model,
                checkpoint=checkpoint,
                output_filepath=output_filepath,
                device=device,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
