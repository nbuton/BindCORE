"""
bindcore/trainer.py
-------------------
Low-level training primitives and small helpers.

    - set_seed        : global reproducibility
    - get_config      : YAML → FullConfig
    - train_one_epoch : single epoch with gradient accumulation and clipping
"""

from __future__ import annotations

import math
import os
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random

import h5py
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import yaml
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# from popriskmin import PRM

from bindcore.config import FullConfig
from bindcore.data.datasets import ProteinDataset, collate_proteins
from bindcore.data.io import (
    cluster_sequences_mmseqs2,
    get_all_feature_stats,
    ham_mask_val_labels,
    prepare_data,
    read_protein_data,
)
from bindcore.eval.metrics import evaluate, select_threshold_cv
from bindcore.modeling.loss import AUCMarginLoss, FocalLoss, LDAMLoss
from bindcore.modeling.protein_multi_scale_transformer import (
    ProteinMultiScaleTransformer,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def get_config(yaml_path: str) -> FullConfig:
    """Parse a YAML config file and return a validated :class:`FullConfig`."""
    with open(yaml_path, "r") as fh:
        return FullConfig.model_validate(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class bindcore_Trainer:
    def __init__(self, cfg, config_path, threshold_selection=True, model_saving=True, device="cpu"):
        self.cfg = cfg
        self.train_cfg = cfg.training
        self.model_cfg = cfg.model
        self.device = torch.device(device)
        self.threshold_selection = threshold_selection
        self.model_saving=model_saving

        # Paths
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.model_save_path = os.path.join(self.config_dir, "bindCORE.pt")

        set_seed(self.train_cfg.seed)

        # Placeholders
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.stats = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_roc_auc": [],
        }
        self.ema_shadow = {}

    def prepare_loaders(self):
        """Handles data loading and OOD splitting logic."""
        with h5py.File(self.train_cfg.h5_properties, "r") as h5:
            df = read_protein_data(self.train_cfg.training_dataset)
            X_scalar, X_local, X_pairwise, seqs, y_list, ids = prepare_data(
                df,
                h5,
                self.train_cfg.SCALAR_FEATURES,
                self.train_cfg.LOCAL_FEATURES,
                self.train_cfg.PAIRWISE_FEATURES,
            )

        project_root = Path(__file__).resolve().parent.parent.parent
        plm_h5_path = os.path.join(project_root, "data/embeddings/esm3-large-2024-03_merged.h5")
        self.dataset = ProteinDataset(
            X_scalar,
            X_local,
            X_pairwise,
            seqs,
            y_list,
            ids=ids,
            plm_h5_path=(
                plm_h5_path
                if self.model_cfg.use_plm_embedding
                else None
            ),
        )
        self.stats = get_all_feature_stats(X_scalar, X_local, X_pairwise)
        self.y_list = y_list  # Keep for loss weight calculation

        # Handle split
        val_prop = self.train_cfg.val_prop

        if val_prop <= 0:
            print("[split] val_prop is 0. Using full dataset for training.")
            train_indices = list(range(len(ids)))
            val_indices = []
        else:
            seq_df = pd.DataFrame({"id": ids, "sequence": seqs})
            id_to_idx = {pid: i for i, pid in enumerate(ids)}

           # 1. Updated to expect a dictionary and write to a JSON file
            cluster_dict = cluster_sequences_mmseqs2(
                seq_df, output_file="data/mmseqs2_cluster.yaml", seq_identity=0.3
            )

            # 2. Get all cluster IDs from the dictionary keys
            all_clusters = list(cluster_dict.keys())
            rng = np.random.default_rng(self.train_cfg.seed)
            rng.shuffle(all_clusters)

            n_val_clusters = int(val_prop * len(all_clusters))
            
            val_clusters = all_clusters[:n_val_clusters]
            train_clusters = all_clusters[n_val_clusters:]

            # 3. Flatten the sequence ID lists for the chosen clusters
            val_ids = [pid for cid in val_clusters for pid in cluster_dict[cid]]
            val_indices = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]

            train_ids = [pid for cid in train_clusters for pid in cluster_dict[cid]]
            train_indices = [id_to_idx[pid] for pid in train_ids if pid in id_to_idx]

            print(
                f"[split] OOD split: {len(train_indices)} train, {len(val_indices)} val proteins."
            )
            # HAM-equivalent: mask val residues locally homologous to training sequences
            # ham_mask_val_labels(
            #     val_indices=val_indices,
            #     train_indices=train_indices,
            #     dataset=self.dataset,
            # )

        loader_kwargs = dict(
            batch_size=self.train_cfg.batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_proteins,
        )

        self.train_loader = DataLoader(
            Subset(self.dataset, train_indices), shuffle=True, **loader_kwargs
        )
        self.val_loader = (
            DataLoader(
                Subset(self.dataset, val_indices), shuffle=False, **loader_kwargs
            )
            if val_indices
            else None
        )

    def build_model(self):
        self.model_cfg.num_classes = 1
        self.model_cfg.nb_scalar = len(self.train_cfg.SCALAR_FEATURES)
        self.model_cfg.nb_local = len(self.train_cfg.LOCAL_FEATURES)
        self.model_cfg.nb_pairwise = len(self.train_cfg.PAIRWISE_FEATURES)

        self.model = ProteinMultiScaleTransformer(self.model_cfg, self.stats).to(
            self.device
        )
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f} M"
        )

    def build_criterion(self):
        total_pos = sum(y[y != -1].sum() for y in self.y_list)
        total_neg = sum((1 - np.array(y[y != -1])).sum() for y in self.y_list)

        loss_type = self.train_cfg.loss_type
        params = self.train_cfg.loss_params

        if loss_type == "focal":
            self.criterion = FocalLoss(reduction="none", **params)
        elif loss_type == "ldam":
            self.criterion = LDAMLoss(
                n_pos=total_pos, n_neg=total_neg, reduction="none", **params
            )
        elif loss_type == "auc_margin":
            self.criterion = AUCMarginLoss(
                n_pos=total_pos, n_neg=total_neg, reduction="none", **params
            )
        elif loss_type == "bce_with_logits_with_weight":
            pos_weight = torch.tensor(
                [total_neg / total_pos], device=self.device, dtype=torch.float32
            )
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction="none"
            )
        elif loss_type == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f"Unknown loss: {loss_type}")

        print(
            f"Initialized {loss_type} for imbalanced ranking (pos_ratio: {total_pos/(total_pos+total_neg):.2%})"
        )

    def save_checkpoint(self, auc=None, is_final=False):
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "cfg": self.model_cfg,
            "stats": self.stats,
            "scalar_features": self.train_cfg.SCALAR_FEATURES,
            "local_features": self.train_cfg.LOCAL_FEATURES,
            "pairwise_features": self.train_cfg.PAIRWISE_FEATURES,
            "best_val_auc": auc,
        }
        torch.save(save_dict, self.model_save_path)
        suffix = "(Final)" if is_final else f"(AUC: {auc:.4f})"
        print(f"  ✓ Checkpoint saved {suffix} → {self.model_save_path}")

    def run(self):
        eval_mode = "normal"
        self.prepare_loaders()
        self.build_model()
        self.build_criterion()

        if self.train_cfg.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_cfg.lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        # elif self.train_cfg.optimizer == "PRM":
        #     self.optimizer = PRM(
        #         self.model.parameters(),
        #         lr=self.train_cfg.lr,
        #         weight_decay=self.train_cfg.weight_decay,
        #         softness=1.0,
        #         warmup_steps=32,
        #         rho=0.99,
        #     )
        else:
            raise ValueError(f"Unknown {self.train_cfg.optimizer} optimizer type")

        if self.train_cfg.scheduler_type == "warmup_cosine":
            # Complexe scheduler
            total_steps = math.ceil(
                len(self.train_loader)
                / self.train_cfg.accumulation
                * self.train_cfg.epochs
            )
            # 1. Warmup for the first 10% of total steps
            warmup_steps = math.ceil(total_steps * 0.1)
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

            # 2. Cosine decay for the remaining 90%
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-6
            )

            # 3. Combine them
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        elif self.train_cfg.scheduler_type == "no_scheduler":
            # Dummy scheduler - Eq No
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: 1.0
            )
        else:
            raise ValueError(f"Unknown {self.train_cfg.scheduler_type} scheduler type")

        best_pr_auc = float("-inf")
        best_val_loss = float("inf")

        for epoch in range(1, self.train_cfg.epochs + 1):
            t_loss = self.train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.criterion,
                self.train_cfg.accumulation,
                self.device,
            )
            self.history["train_loss"].append(t_loss)

            log_str = f"Epoch {epoch:03d} | train_loss={t_loss:.4f}"

            if self.val_loader:
                use_ema_for_eval = self.train_cfg.use_ema and bool(self.ema_shadow)

                if use_ema_for_eval:
                    with torch.no_grad():
                        raw_state = {
                            n: p.clone()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad
                        }
                    self._apply_ema()

                val_loss, val_roc_auc, val_pr_auc = evaluate(
                    self.model, self.val_loader, self.criterion, self.device
                )
                self.history["val_loss"].append(val_loss)
                self.history["val_roc_auc"].append(val_roc_auc)
                self.history["val_pr_auc"].append(val_pr_auc)

                log_str += f" | val_loss={val_loss:.4f} | val_ROC-AUC={val_roc_auc:.4f} | val_PR-AUC={val_pr_auc:.4f}"
                print(log_str)

                if val_pr_auc > best_pr_auc:
                    best_pr_auc = val_pr_auc
                    best_val_loss = val_loss
                    if self.model_saving:
                        self.save_checkpoint(auc=best_pr_auc)
                else:
                    print(f"  - PR-AUC did not improve, checkpoint not updated.")

                if use_ema_for_eval:
                    with torch.no_grad():
                        for name, p in self.model.named_parameters():
                            if name in raw_state:
                                p.copy_(raw_state[name])

            else:
                print(log_str)
                if epoch == self.train_cfg.epochs:
                    if self.train_cfg.use_ema and bool(self.ema_shadow):
                        self._apply_ema()
                    if self.model_saving:
                        self.save_checkpoint(is_final=True)

        if self.threshold_selection:
            checkpoint = torch.load(
                self.model_save_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            best_thr = select_threshold_cv(
                self.model, self.train_loader, self.device, seed=self.train_cfg.seed
            )
            checkpoint["best_threshold"] = best_thr
            torch.save(checkpoint, self.model_save_path)
            print(f"Final threshold (CV-MCC): {best_thr:.6f}")

        # For hyperparameters tunning
        if len(self.history["val_pr_auc"]) != 0:
            peak_epoch = np.argmax(self.history["val_pr_auc"])
            peak_value = self.history["val_pr_auc"][peak_epoch]

            # Average of a window around the peak (+/-2 epochs)
            window = self.history["val_pr_auc"][max(0, peak_epoch - 2) : peak_epoch + 3]
            sustained_peak = np.mean(window)

            # Penalize peaks that happen in the first 20% of training
            total_epochs = len(self.history["val_pr_auc"])
            earliness_penalty = max(0, 0.2 - peak_epoch / total_epochs) * peak_value
            if eval_mode == "sustain":
                return sustained_peak - earliness_penalty
            else:
                return peak_value
        else:
            print("Empty validation so no value to optimize")
            return None

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history["train_loss"], label="Train")
        if self.history["val_loss"]:
            ax1.plot(self.history["val_loss"], label="Val")
        ax1.set_title("Loss")
        ax1.legend()

        if self.history["val_pr_auc"]:
            ax2.plot(self.history["val_pr_auc"])
            ax2.set_title("Validation PR-AUC")

        plt.tight_layout()
        plt.savefig("data/last_training_fig.png")
        plt.show()

    @torch.no_grad()
    def _ema_update(self):
        decay = self.train_cfg.ema_decay
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.ema_shadow:
                self.ema_shadow[name] = p.detach().clone()
            else:
                self.ema_shadow[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def _apply_ema(self):
        for name, p in self.model.named_parameters():
            if name in self.ema_shadow:
                p.copy_(self.ema_shadow[name])

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion,
        accumulation_steps: int,
        device: torch.device,
        grad_clip: float = 1.0,
    ) -> float:
        """
        Run one full training epoch with gradient accumulation.

        Returns
        -------
        float
            Mean training loss over all samples in *loader*.
        """
        model.train()
        total_loss, total = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_scalar, x_local, x_pairwise, seq, mask, y, plm_pad) in tqdm(
            enumerate(loader), total=len(loader)
        ):
            x_scalar = x_scalar.to(device)
            x_local = x_local.to(device)
            x_pairwise = x_pairwise.to(device)
            tokens = seq.long().to(device)
            mask = mask.to(device)
            y = y.to(device)
            if plm_pad is not None:
                plm_pad = plm_pad.to(device)

            logits = model(tokens, x_scalar, x_local, x_pairwise, mask, plm_pad)
            logits = logits.squeeze(-1)  # [batch, length]

            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite logits at batch {batch_idx}.")

            # 1. Identify valid labels (ignore -1)
            valid_label_mask = (y != -1).float()

            # 2. Combine the padding mask and label mask
            # If either is 0.0 (padded OR unknown), the combined mask becomes 0.0
            combined_mask = mask * valid_label_mask

            # 3. Clean y: Replace -1 with 0.0 so BCEWithLogitsLoss doesn't error out
            y_clean = y.clone().float()
            y_clean[y == -1] = 0.0

            loss_raw = criterion(logits, y_clean)
            loss = (
                (loss_raw * combined_mask).sum()
                / (combined_mask.sum() + 1e-8)
                / accumulation_steps
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {batch_idx}.")

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at batch {batch_idx}."
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if self.train_cfg.use_ema:
                    self._ema_update()
            # For debug
            # if isinstance(optimizer, PRM) and batch_idx % 10 == 0:
            #     print("Active fraction:", optimizer.get_mask_stats()["active_fraction"])

            # Undo the accumulation scaling to track the true loss magnitude
            total_loss += loss.item() * accumulation_steps * y.size(0)
            total += y.size(0)

        return total_loss / max(total, 1)
