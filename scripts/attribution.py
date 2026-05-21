"""
scripts/attribution.py
----------------------
Position-independent feature attribution for ProteinMultiScaleTransformer
using DeepLiftShap (Captum).

For each feature group (x_scalar, x_local, x_pairwise), produces one
importance score per feature channel — averaged over residue positions and
over all proteins in the test set — so the result is a global, position-
independent ranking of which input features drive LIP predictions.

Key design decisions
--------------------
* x_scalar  : global per-protein features, shape [F]         → no residue dim
* x_local   : per-residue features, stored as [F, L]         → transposed to [L, F] for attribution
* x_pairwise: pairwise features, stored as [C, L, L]         → importance averaged over (L, L)
* Baselines : three strategies — "sample" (random test seqs, recommended),
              "zeros", or "mean" (per-channel mean + small noise)
* Unknown labels (-1, -100, "-", NaN) are excluded from the residue-level
  pooling so they don't dilute the attribution signal.
* Fixed inputs inside _FeatureWrapper are expanded to match the dynamic
  batch size that DeepLiftShap uses internally (input + baselines stacked).

Usage
-----
    python scripts/attribution.py \
        --model   checkpoints/best_model.pt \
        --dataset data/test.txt \
        --h5      data/features.h5 \
        --output  results/feature_importance.csv \
        --n_baselines 20 \
        --strategy sample \
        --features x_scalar x_local
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import DeepLiftShap
from torch.utils.data import DataLoader

from core_lip.data.datasets import ProteinDataset, collate_proteins
from core_lip.data.io import prepare_data, read_protein_data
from core_lip.engine.predictor import load_checkpoint

# ============================================================================
# 1. Model wrapper
# ============================================================================


class _FeatureWrapper(nn.Module):
    """
    Wrapper so Captum sees f(x_flat) -> scalar [B], where x_flat is always 2-D.

    DeepLiftShap treats dim 0 as batch and dim 1 as features — it cannot
    handle 3-D inputs without misinterpreting the sequence dimension as batch.
    We therefore flatten any multi-dimensional feature tensor to [B, L*F] or
    [B, F] before attribution, and unflatten inside forward() before passing
    to the model.

    All inputs except the one being attributed are held fixed at the input
    protein's values and expanded to match B on the fly.

    Parameters
    ----------
    model        : ProteinMultiScaleTransformer (eval, frozen)
    fixed        : all model inputs except the attributed one, batch-dim = 1
    feature_key  : "x_scalar" | "x_local" | "x_pairwise"
    model_shape  : the shape the model expects for feature_key, WITHOUT batch
                   dim, e.g. (F,) or (F_l, L) or (C, L, L)
    mask         : [1, L] bool
    known_mask   : [1, L] bool
    """

    def __init__(
        self,
        model: nn.Module,
        fixed: dict,
        feature_key: str,
        model_shape: tuple,
        mask: torch.Tensor,
        known_mask: torch.Tensor,
    ):
        super().__init__()
        self.model = model
        self.fixed = fixed
        self.feature_key = feature_key
        self.model_shape = model_shape  # shape to restore before model call
        self.mask = mask  # [1, L]
        self.known_mask = known_mask  # [1, L]

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat : [B, D]  where D = prod(model_shape), B = Captum's internal batch
        returns : [B]     masked-mean logit
        """
        B = x_flat.shape[0]

        # Restore model-expected shape: [B, D] -> [B, *model_shape]
        x_attr = x_flat.view(B, *self.model_shape)

        # Expand every fixed input [1, ...] -> [B, ...] (zero-copy)
        kwargs: dict[str, torch.Tensor | None] = {}
        for k, v in self.fixed.items():
            kwargs[k] = v.expand((B,) + v.shape[1:]) if v is not None else None

        mask_b = self.mask.expand(B, -1)  # [B, L]
        known_mask_b = self.known_mask.expand(B, -1)  # [B, L]
        kwargs["mask"] = mask_b
        kwargs[self.feature_key] = x_attr

        # ── DEBUG ─────────────────────────────────────────────────────────
        print(f"\n[DBG forward] feature_key={self.feature_key}  B={B}")
        print(f"  x_flat.shape       = {x_flat.shape}")
        print(
            f"  x_attr.shape       = {x_attr.shape}  (after view to model_shape={self.model_shape})"
        )
        for k, v in kwargs.items():
            if v is not None:
                print(f"  kwargs[{k!r:<12}] = {tuple(v.shape)}")
            else:
                print(f"  kwargs[{k!r:<12}] = None")
        # ─────────────────────────────────────────────────────────────────

        logits = self.model(**kwargs)  # [B, L, 1] or [B, L]
        if logits.dim() == 3:
            logits = logits.squeeze(-1)  # [B, L]

        m = (mask_b & known_mask_b).float()
        valid_counts = m.sum(dim=1, keepdim=True).clamp(min=1)
        return (logits * m).sum(dim=1) / valid_counts.squeeze(1)  # [B]


# ============================================================================
# 2. Data loading
# ============================================================================


def _collect_samples(
    dataset_path: str,
    h5_features: h5py.File,
    checkpoint: dict,
    device: torch.device,
    n_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load the dataset and return a list of per-protein dicts.

    Each dict has keys:
        tokens, x_scalar, x_local, x_pairwise,
        mask, known_mask, plm_pad, protein_id

    All tensors keep their batch dimension (size 1) so they can be directly
    passed to the model or expanded inside _FeatureWrapper.

    known_mask is [1, L] bool: True where label not in {-1, -100, "-", NaN}.

    Parameters
    ----------
    n_samples : if set, randomly subsample this many proteins BEFORE building
                the dataset, so arrays and DataLoader stay in sync.
                Never subsample AFTER prepare_data — arrays would misalign.
    seed      : random seed for reproducible subsampling
    """
    df = read_protein_data(dataset_path)

    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
        print(f"Subsampled to {len(df)} proteins (seed={seed}).")

    X_scalar, X_local, X_pairwise, seqs, labels, ids = prepare_data(
        df,
        h5_features,
        checkpoint["scalar_features"],
        checkpoint["local_features"],
        checkpoint["pairwise_features"],
    )

    dataset = ProteinDataset(X_scalar, X_local, X_pairwise, seqs, ids=ids)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_proteins,
    )

    # protein_id -> raw label array
    label_map: dict[str, np.ndarray | None] = {
        pid: (np.asarray(lbl) if lbl is not None else None)
        for pid, lbl in zip(ids, labels)
    }

    samples = []
    for x_sc, x_lo, x_pw, seq, mask, prot_ids, plm_pad in loader:
        pid = prot_ids[0]
        raw_labels = label_map.get(pid)

        # Build known_mask: handles int (-1/-100), float (nan), str ("-")
        if raw_labels is None:
            L = int(mask.sum())
            known = torch.zeros(1, L, dtype=torch.bool)
        else:
            try:
                arr = raw_labels.astype(float)
                known_np = np.isfinite(arr) & (arr != -1.0) & (arr != -100.0)
            except (ValueError, TypeError):
                known_np = np.array(
                    [
                        str(v) not in ("-", "-1", "-100") and v not in (-1, -100)
                        for v in raw_labels
                    ],
                    dtype=bool,
                )
            known = torch.tensor(known_np, dtype=torch.bool).unsqueeze(0)  # [1, L]

        samples.append(
            {
                "tokens": seq.long(),  # [1, L]
                "x_scalar": x_sc,  # [1, F]
                "x_local": x_lo,  # [1, F_l, L]  channels-first
                "x_pairwise": x_pw,  # [1, C, L, L]
                "mask": mask,  # [1, L]  bool
                "known_mask": known,  # [1, L]  bool
                "plm_pad": plm_pad,  # [1, L, E] or None
                "protein_id": pid,
            }
        )

    print(f"Loaded {len(samples)} proteins.")
    for s in samples[:3]:
        n_known = int(s["known_mask"].sum())
        n_total = int(s["mask"].sum())
        print(f"  {s['protein_id']}: {n_known}/{n_total} residues with known labels")
    return samples


# ============================================================================
# 3. Baseline construction
# ============================================================================


def _build_baselines(
    samples: list[dict],
    feature_key: str,
    n_baselines: int,
    strategy: str,
) -> list[torch.Tensor]:
    """
    Return n_baselines tensors to use as DeepLiftShap baselines for feature_key.

    Each tensor has the native shape of that feature for one protein (no batch
    dim, variable L).  They are padded together with the input inside
    compute_attributions().

    Strategies
    ----------
    "sample" : pick n_baselines random proteins from the test set (recommended)
    "zeros"  : all-zero tensors matching the input shape
    "mean"   : per-channel mean across all residues + small Gaussian noise
    """

    def _unbatch(s: dict) -> torch.Tensor:
        return s[feature_key].squeeze(0)

    if strategy == "sample":
        idx = np.random.choice(
            len(samples),
            size=n_baselines,
            replace=(len(samples) < n_baselines),
        )
        return [_unbatch(samples[i]) for i in idx]

    ref = _unbatch(samples[0])

    if strategy == "zeros":
        return [torch.zeros_like(ref) for _ in range(n_baselines)]

    if strategy == "mean":
        if ref.dim() == 1:  # x_scalar [F]
            all_f = torch.stack([_unbatch(s) for s in samples])
            mean_feat = all_f.mean(0)
            return [
                mean_feat + torch.randn_like(mean_feat) * 0.01
                for _ in range(n_baselines)
            ]
        elif ref.dim() == 2:  # x_local [F, L]
            all_f = torch.cat([_unbatch(s) for s in samples], dim=1)
            mean_feat = all_f.mean(dim=1)
            L = ref.shape[1]
            base = mean_feat.unsqueeze(1).expand(-1, L)
            return [base + torch.randn_like(base) * 0.01 for _ in range(n_baselines)]
        else:
            raise ValueError(
                f"'mean' strategy not implemented for {feature_key} "
                f"with shape {ref.shape}. Use 'sample' or 'zeros'."
            )

    raise ValueError(f"Unknown strategy '{strategy}'. Choose: sample | zeros | mean.")


# ============================================================================
# 4. Padding utility
# ============================================================================


def _pad_along_seq(
    tensors: list[torch.Tensor],
    seq_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of variable-length tensors along their sequence dimension,
    then stack them into a single tensor.

    seq_dim
    -------
    -1  global [F]       : no padding, just stack
     0  [L, F]           : pad rows
     1  [F, L] or [C,L,L]: pad last dim(s)

    Returns
    -------
    stacked : [N, ...]
    lengths : [N]  original L per tensor
    """
    if seq_dim == -1:
        return torch.stack(tensors), torch.tensor([t.shape[0] for t in tensors])

    lengths = torch.tensor(
        [t.shape[seq_dim] if t.dim() > 1 else t.shape[0] for t in tensors]
    )
    L_max = int(lengths.max())

    padded = []
    for t in tensors:
        L = t.shape[seq_dim] if t.dim() > 1 else t.shape[0]
        pad_amt = L_max - L
        if pad_amt == 0:
            padded.append(t)
            continue
        if t.dim() == 1:  # [L] tokens
            p = torch.nn.functional.pad(t, (0, pad_amt))
        elif t.dim() == 2 and seq_dim == 0:  # [L, F]
            p = torch.nn.functional.pad(t, (0, 0, 0, pad_amt))
        elif t.dim() == 2 and seq_dim == 1:  # [F, L]
            p = torch.nn.functional.pad(t, (0, pad_amt))
        elif t.dim() == 3:  # [C, L, L]
            p = torch.nn.functional.pad(t, (0, pad_amt, 0, pad_amt))
        else:
            raise ValueError(f"Unexpected shape {t.shape} with seq_dim={seq_dim}")
        padded.append(p)

    return torch.stack(padded), lengths


# ============================================================================
# 5. Per-protein attribution
# ============================================================================


def compute_attributions(
    model: nn.Module,
    sample: dict,
    baseline_tensors: list[torch.Tensor],
    feature_key: str,
    device: torch.device,
) -> dict:
    """
    Run DeepLiftShap for one protein and one feature group.

    DESIGN: DeepLiftShap hard-codes dim 0 = batch, dim 1 = features.
    Passing a 3-D tensor [1, L, F] makes Captum treat L as the batch size
    inside forward(), which breaks all fixed-input expansions.

    Solution: flatten the feature tensor to [1, D] before .attribute(),
    then unflatten the returned [1, D] attributions back to the original
    shape for aggregation.  The wrapper receives [B, D] (B=1+n_baselines)
    and calls .view(B, *model_shape) to restore the model-expected layout.

    Returns
    -------
    {
      "feature_importance" : Tensor [F]     mean |attr| per channel
      "raw_attributions"   : Tensor [...]   native shape, padding removed
      "protein_id"         : str
      "n_known_residues"   : int
      "n_total_residues"   : int
    }
    """
    model.eval()

    raw_feat = sample[feature_key].squeeze(0)  # no batch dim
    feat_ndim = raw_feat.dim()

    is_global = feat_ndim == 1  # x_scalar  [F]
    is_pairwise = feat_ndim == 3  # x_pairwise [C, L, L]
    is_local = feat_ndim == 2  # x_local    [F, L]

    print(f"\n[DBG compute] pid={sample['protein_id']}  feature_key={feature_key}")
    print(
        f"  raw_feat.shape = {raw_feat.shape}  is_global={is_global}  is_local={is_local}  is_pairwise={is_pairwise}"
    )
    print(
        f"  n_baselines    = {len(baseline_tensors)}  baseline[0].shape = {baseline_tensors[0].shape}"
    )

    # ── Pad input + baselines to the same sequence length ─────────────────
    # global: no seq dim → seq_dim=-1 (just stack)
    # local:  [F, L]    → seq_dim=1
    # pairwise: [C,L,L] → seq_dim=1 (pads both spatial dims symmetrically)
    pad_seq_dim = -1 if is_global else 1
    all_feats, lengths = _pad_along_seq(
        [raw_feat] + baseline_tensors, seq_dim=pad_seq_dim
    )
    # all_feats: [1+n_bl, F] | [1+n_bl, F_l, L_max] | [1+n_bl, C, L_max, L_max]

    L_input = None if is_global else int(lengths[0])
    if is_global:
        L_max, pad_len = None, 0
    elif is_pairwise:
        L_max = all_feats.shape[2]  # [N, C, L_max, L_max]
        pad_len = L_max - L_input
    else:
        L_max = all_feats.shape[2]  # [N, F_l, L_max]
        pad_len = L_max - L_input

    # model_shape: what the model expects per-sample (no batch dim)
    model_shape = tuple(all_feats.shape[1:])
    D = int(torch.tensor(list(model_shape)).prod().item())

    print(f"  all_feats.shape = {all_feats.shape}")
    print(
        f"  model_shape={model_shape}  D={D}  L_input={L_input}  L_max={L_max}  pad_len={pad_len}"
    )

    # Flatten to [N, D] — Captum only sees 2-D input, so dim 0 is always batch
    input_flat = all_feats[[0]].reshape(1, D).to(device).requires_grad_(True)
    baseline_flat = all_feats[1:].reshape(-1, D).to(device)

    print(f"  input_flat.shape    = {input_flat.shape}")
    print(f"  baseline_flat.shape = {baseline_flat.shape}")

    # Also print fixed inputs that will go into the wrapper
    print(f"  fixed keys (before del): tokens, x_scalar, x_local, x_pairwise, plm_pad")

    # ── Pad masks ─────────────────────────────────────────────────────────
    orig_mask = sample["mask"]  # [1, L_true]
    orig_known = sample["known_mask"]  # [1, L_true]

    if is_global:
        mask_padded = orig_mask.to(device)
        known_padded = orig_known.to(device)
    else:
        mask_padded = torch.nn.functional.pad(orig_mask, (0, pad_len), value=False).to(
            device
        )
        known_padded = torch.nn.functional.pad(
            orig_known, (0, pad_len), value=False
        ).to(device)

    # ── Fixed inputs: pad sequence-dimension to L_max, keep batch dim = 1 ──
    # Each key has a known layout — we must pad the correct dimension and
    # NEVER pad global features (x_scalar [F]) regardless of pad_len.
    def _pad_fixed(key: str) -> torch.Tensor | None:
        v = sample.get(key)
        if v is None:
            return None
        t = v.squeeze(0)  # remove batch dim → native shape

        if key == "x_scalar":
            # Global feature [F] — no sequence dim, never pad
            p = t

        elif key == "tokens":
            # [L] integer sequence — pad along dim 0
            p = torch.nn.functional.pad(t, (0, pad_len))

        elif key == "x_local":
            # [F_l, L] channels-first — pad along last dim (L)
            p = torch.nn.functional.pad(t, (0, pad_len))

        elif key == "x_pairwise":
            # [C, L, L] — pad both spatial dims symmetrically
            p = torch.nn.functional.pad(t, (0, pad_len, 0, pad_len))

        elif key == "plm_pad":
            # [L, E] if present — pad along dim 0 (L)
            p = torch.nn.functional.pad(t, (0, 0, 0, pad_len))

        else:
            raise ValueError(f"Unknown key '{key}' in _pad_fixed")

        return p.unsqueeze(0).to(device)  # restore batch dim → [1, ...]

    fixed = {
        "tokens": _pad_fixed("tokens"),
        "x_scalar": _pad_fixed("x_scalar"),
        "x_local": _pad_fixed("x_local"),
        "x_pairwise": _pad_fixed("x_pairwise"),
        "plm_pad": _pad_fixed("plm_pad"),
    }
    del fixed[feature_key]  # injected as x_flat via wrapper

    print(f"  fixed shapes after del '{feature_key}':")
    for k, v in fixed.items():
        print(f"    fixed[{k!r:<12}] = {tuple(v.shape) if v is not None else None}")
    print(f"  mask_padded.shape  = {mask_padded.shape}")
    print(f"  known_padded.shape = {known_padded.shape}")

    wrapper = _FeatureWrapper(
        model=model,
        fixed=fixed,
        feature_key=feature_key,
        model_shape=model_shape,  # wrapper calls .view(B, *model_shape)
        mask=mask_padded,
        known_mask=known_padded,
    )

    dl_shap = DeepLiftShap(wrapper)
    attrs_flat = dl_shap.attribute(
        inputs=input_flat,  # [1, D]
        baselines=baseline_flat,  # [n_bl, D]
    )
    # attrs_flat: [1, D]

    # ── Unflatten and aggregate ───────────────────────────────────────────
    attrs = attrs_flat.reshape(model_shape).detach().cpu()
    # attrs: (F,) | (F_l, L_max) | (C, L_max, L_max)

    if is_global:
        feature_importance = attrs.abs()  # [F]
        raw_attributions = attrs
        n_known = int(orig_known.sum())
        n_total = int(orig_mask.sum())

    elif is_local:
        # attrs: [F_l, L_max] — trim padding, average |attr| over known residues
        attrs_valid = attrs[:, :L_input]  # [F_l, L_input]
        known_1d = orig_known.squeeze(0)  # [L_input] bool
        cols = attrs_valid[:, known_1d] if known_1d.any() else attrs_valid
        feature_importance = cols.abs().mean(dim=1)  # [F_l]
        raw_attributions = attrs_valid
        n_known = int(known_1d.sum())
        n_total = L_input

    else:  # pairwise
        attrs_valid = attrs[:, :L_input, :L_input]  # [C, L, L]
        feature_importance = attrs_valid.abs().mean(dim=(1, 2))  # [C]
        raw_attributions = attrs_valid
        n_known = int(orig_known.sum())
        n_total = L_input

    return {
        "feature_importance": feature_importance,  # [F] guaranteed 1-D
        "raw_attributions": raw_attributions,
        "protein_id": sample["protein_id"],
        "n_known_residues": n_known,
        "n_total_residues": n_total,
    }


# ============================================================================
# 6. Dataset-level aggregation
# ============================================================================


def dataset_attribution(
    model: nn.Module,
    samples: list[dict],
    feature_keys: list[str],
    n_baselines: int,
    strategy: str,
    checkpoint: dict,
    device: torch.device,
    min_known_residues: int = 5,
) -> pd.DataFrame:
    """
    Run attribution for every protein x every feature group.

    Proteins with fewer than `min_known_residues` known labels are skipped
    for per-residue features (local / pairwise).

    Returns a DataFrame sorted by mean_importance with columns:
        feature_group | feature_index | feature_name |
        mean_importance | std_importance | n_proteins
    """
    records = []

    for fkey in feature_keys:
        print(f"\n── Attributing '{fkey}' ──")

        baselines = _build_baselines(samples, fkey, n_baselines, strategy)
        importances: list[torch.Tensor] = []
        skipped = 0

        for i, sample in enumerate(samples):
            pid = sample["protein_id"]
            n_known = int(sample["known_mask"].sum())
            n_total = int(sample["mask"].sum())

            is_global = sample[fkey].squeeze(0).dim() == 1
            if not is_global and n_known < min_known_residues:
                print(
                    f"  [{i+1:>3}/{len(samples)}] {pid}: skipped "
                    f"({n_known} known residues < {min_known_residues})"
                )
                skipped += 1
                continue

            print(
                f"  [{i+1:>3}/{len(samples)}] {pid} " f"(L={n_total}, known={n_known})",
                end=" ... ",
                flush=True,
            )

            try:
                result = compute_attributions(model, sample, baselines, fkey, device)
                imp = result["feature_importance"]

                if imp.dim() != 1:
                    raise ValueError(
                        f"Expected 1-D feature_importance, got shape {imp.shape}"
                    )

                importances.append(imp)
                print(f"ok  (shape={imp.shape})")

            except Exception as exc:
                print(f"WARN — skipped: {exc}")
                traceback.print_exc()
                skipped += 1

        print(f"  -> {len(importances)} proteins attributed, {skipped} skipped.")

        if not importances:
            print(f"  [ERROR] No attributions computed for '{fkey}'. Skipping.")
            continue

        shapes = {imp.shape for imp in importances}
        if len(shapes) > 1:
            print(f"  [ERROR] Inconsistent shapes {shapes} for '{fkey}'. Skipping.")
            continue

        stacked = torch.stack(importances)  # [N, F]
        mean_imp = stacked.mean(dim=0).numpy()
        std_imp = stacked.std(dim=0).numpy()

        fname_key = {
            "x_scalar": "scalar_features",
            "x_local": "local_features",
            "x_pairwise": "pairwise_features",
        }.get(fkey)
        feat_names = (
            checkpoint.get(fname_key)
            if fname_key and checkpoint.get(fname_key)
            else [f"{fkey}_{j}" for j in range(len(mean_imp))]
        )
        if len(feat_names) != len(mean_imp):
            print(
                f"  [WARN] feat_names length ({len(feat_names)}) != "
                f"n_features ({len(mean_imp)}) for '{fkey}'. Using generic names."
            )
            feat_names = [f"{fkey}_{j}" for j in range(len(mean_imp))]

        for j, (name, mu, sigma) in enumerate(zip(feat_names, mean_imp, std_imp)):
            records.append(
                {
                    "feature_group": fkey,
                    "feature_index": j,
                    "feature_name": name,
                    "mean_importance": float(mu),
                    "std_importance": float(sigma),
                    "n_proteins": len(importances),
                }
            )

    if not records:
        print("\n[ERROR] No attributions were computed for any feature group.")
        return pd.DataFrame(
            columns=[
                "feature_group",
                "feature_index",
                "feature_name",
                "mean_importance",
                "std_importance",
                "n_proteins",
            ]
        )

    return pd.DataFrame(records).sort_values("mean_importance", ascending=False)


# ============================================================================
# 7. CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepLiftShap feature attribution for CORE-LIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--dataset", required=True, help="Path to test .txt dataset file"
    )
    parser.add_argument("--h5", required=True, help="Path to .h5 MD features file")
    parser.add_argument(
        "--output", default="results/feature_importance.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--n_baselines",
        type=int,
        default=20,
        help="Number of baseline sequences for DeepLiftShap",
    )
    parser.add_argument(
        "--strategy",
        default="sample",
        choices=["sample", "zeros", "mean"],
        help=(
            "'sample' = random test sequences (recommended), "
            "'zeros'  = all-zero baseline, "
            "'mean'   = per-channel mean + noise"
        ),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["x_scalar", "x_local"],
        choices=["x_scalar", "x_local", "x_pairwise"],
        help="Feature groups to attribute (pairwise is memory-intensive)",
    )
    parser.add_argument(
        "--min_known",
        type=int,
        default=5,
        help="Min known-label residues to include a protein",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device, e.g. 'cpu' or 'cuda:0'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for baseline sampling reproducibility",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help=(
            "If set, randomly subsample this many proteins from the "
            "dataset before attribution. Subsampling happens before "
            "prepare_data so all arrays stay aligned."
        ),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device : {device}")

    model, checkpoint = load_checkpoint(args.model, device)
    model.eval()
    print(f"Model  : {args.model}")

    with h5py.File(args.h5, "r") as h5:
        samples = _collect_samples(
            args.dataset,
            h5,
            checkpoint,
            device,
            n_samples=args.n_samples,
            seed=args.seed,
        )

    if not samples:
        raise RuntimeError("No samples loaded — check --dataset and --h5 paths.")

    df = dataset_attribution(
        model=model,
        samples=samples,
        feature_keys=args.features,
        n_baselines=args.n_baselines,
        strategy=args.strategy,
        checkpoint=checkpoint,
        device=device,
        min_known_residues=args.min_known,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}")

    for fkey in args.features:
        sub = df[df["feature_group"] == fkey]
        if sub.empty:
            continue
        print(
            f"\nTop features — {fkey} ({len(sub)} total, "
            f"{sub['n_proteins'].iloc[0]} proteins):"
        )
        print(
            sub.head(10).to_string(
                index=False,
                columns=["feature_name", "mean_importance", "std_importance"],
            )
        )


if __name__ == "__main__":
    main()
