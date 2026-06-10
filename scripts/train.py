import argparse
import torch
from bindcore.engine.trainer import bindcore_Trainer, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit-VRAM", action="store_true", help="Limit GPU memory to 49%%"
    )
    args = parser.parse_args()
 

    if args.limit_VRAM:
        torch.cuda.set_per_process_memory_fraction(0.49)

    cfg = get_config(args.config)

    trainer = bindcore_Trainer(cfg, args.config, device=args.device)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    best_auc = trainer.run()
    print("best_auc:", best_auc)
    trainer.plot()


if __name__ == "__main__":
    main()
