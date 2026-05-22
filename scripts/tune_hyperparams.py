"""
Meta-learning with hyperparameter optimization using Ray Tune.

A single search-space YAML controls everything: fixed parameters (value:)
and tuned parameters (type: + bounds) alike. No separate base config is needed
at runtime — the YAML IS the full parameter description.

Usage:
    python tune_train.py --search-space search_space.yaml --device cpu --num-samples 20 --max-epochs 10
"""

import argparse
import copy
import os
import tempfile
from pathlib import Path
import numpy as np
from typing import Any

import yaml

# -- Ray Tune -----------------------------------------------------------------
import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.search.optuna import OptunaSearch

# -- Your project -------------------------------------------------------------
from core_lip.engine.trainer import CORE_LIP_Trainer, get_config


# -----------------------------------------------------------------------------
# Preset look-up tables
# -----------------------------------------------------------------------------

FEATURE_SETS: dict[str, list[str]] = {
    "scalar_only": ["scalar_features"],
    "scalar_local": ["scalar_features", "local_features"],
    "scalar_local_pairwise": ["scalar_features", "local_features", "pairwise_features"],
    "all": ["token_embedding","positional_embeddings", "scalar_features", "local_features", "pairwise_features"],
}

DILATION_PRESETS: dict[str, list[int]] = {
    "none": [1],
    "narrow": [1, 2, 4],
    "standard": [1, 3, 7],
    "wide": [1, 3, 7, 15],
    "deep": [1, 2, 4, 8, 16],
}

# Maps flat param name -> (config section, config key).
# Use a list of keys to fan one param out to multiple config fields.
# Maps flat param name -> (config section, config key).
# Use a list of keys to fan one param out to multiple config fields.
PARAM_TO_CONFIG: dict[str, tuple[str, str | list[str]]] = {
    # training
    "optimizer": ("training", "optimizer"),
    "lr": ("training", "lr"),
    "weight_decay": ("training", "weight_decay"),
    "batch_size": ("training", "batch_size"),
    "accumulation": ("training", "accumulation"),
    "scheduler_type": ("training", "scheduler_type"),
    "loss_type": ("training", "loss_type"),
    "loss_params": ("training", "loss_params"),
    
    # model
    "embed_dim": ("model", "embed_dim"),
    "num_blocks": ("model", "num_blocks"),
    "num_heads": ("model", "num_heads"),
    "ffn_expansion": ("model", "ffn_expansion"),
    "dropout": ("model", "dropout"),
    "share_block_weights": ("model", "share_block_weights"),
    "activate_classical_attention": ("model", "activate_classical_attention"),
    
    # MLP hidden mapping
    "mlp_hidden": ("model", ["local_mlp_hidden", "scalar_mlp_hidden"]),
    "local_mlp_hidden": ("model", "local_mlp_hidden"),
    "scalar_mlp_hidden": ("model", "scalar_mlp_hidden"),
    
    "inputs_features": ("model", "inputs_features"),  # expanded via FEATURE_SETS
    "activate_pairwise_bias": ("model", "activate_pairwise_bias"),
    "pairwise_cnn_channels": ("model", "pairwise_cnn_channels"),
    "pairwise_cnn_kernel": ("model", "pairwise_cnn_kernel"),
    "dilatations_cnn": ("model", "dilatations_cnn"),  # expanded via DILATION_PRESETS
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def build_config_from_trial(trial_params: dict, static_cfg: dict) -> dict:
    """
    Build a full nested trainer config by layering trial_params on top of
    static_cfg. The mapping is driven entirely by PARAM_TO_CONFIG, so adding
    a new tunable param only requires a YAML entry + one line in that table.
    """
    cfg = copy.deepcopy(static_cfg)

    for param, value in trial_params.items():
        if param not in PARAM_TO_CONFIG:
            continue  # unknown param — ignore

        section, key = PARAM_TO_CONFIG[param]

        # Expand preset keys
        if param == "inputs_features":
            value = FEATURE_SETS[value]
        elif param == "dilatations_cnn":
            value = DILATION_PRESETS[value]

        cfg.setdefault(section, {})

        if isinstance(key, list):  # fan-out: one param -> many config keys
            for k in key:
                cfg[section][k] = value
        else:
            cfg[section][key] = value

    return cfg


# -----------------------------------------------------------------------------
# Search-space YAML loading
# -----------------------------------------------------------------------------


def _parse_tune_distribution(param_name: str, spec: dict) -> Any:
    """
    Convert one parameter spec dict into a Ray Tune distribution.

    Spec formats
    ------------
    Fixed (never sampled):
        value: <anything>

    Categorical:
        type: choice
        values: [a, b, c]

    Log-uniform float:
        type: loguniform
        low: 1e-6
        high: 1e-3

    Uniform float:
        type: uniform
        low: 0.1
        high: 0.9

    Integer (upper-exclusive):
        type: randint
        low: 1
        high: 5

    Dependent expression:
        type: sample_from
        expr: |
            {"focal": {"gamma": 2.0}}.get(spec.config.loss_type, {})
    """
    if "value" in spec:
        return tune.choice([spec["value"]])  # single-element → always fixed

    dist_type = spec.get("type", "").lower()

    if dist_type == "choice":
        return tune.choice(spec["values"])
    if dist_type == "loguniform":
        return tune.loguniform(float(spec["low"]), float(spec["high"]))
    if dist_type == "uniform":
        return tune.uniform(float(spec["low"]), float(spec["high"]))
    if dist_type == "randint":
        return tune.randint(int(spec["low"]), int(spec["high"]))
    if dist_type == "sample_from":
        fn = eval(f"lambda spec: {spec['expr']}")  # noqa: S307
        return tune.sample_from(fn)

    raise ValueError(
        f"[{param_name}] Unknown type '{dist_type}'. "
        "Use: choice | loguniform | uniform | randint | sample_from | value."
    )


def load_search_space_yaml(path: str) -> tuple[dict, dict]:
    """
    Parse the single search-space YAML and return:

    tune_space   : {param_name: Ray Tune distribution}
                   Passed directly to tune.Tuner as param_space.

    static_cfg   : nested dict
                   Everything under the special ``_static_config`` key — fields
                   the trainer needs that are NOT hyperparameters (dataset paths,
                   vocab_size, max_seq_len, seed, …).

    ── YAML top-level structure ──────────────────────────────────────────────

    _static_config:             # verbatim config fields, passed through as-is
      training:
        epochs: 10
        seed: 42
        val_prop: 0.0
      data:
        h5_properties: data/...
        training_dataset: data/...
      model:
        vocab_size: 25
        max_seq_len: 1024

    # Every other key = a hyperparameter (fixed or searched)
    lr:
      type: loguniform
      low: 1e-5
      high: 1e-2

    batch_size:
      value: 2              # fixed

    inputs_features:
      type: choice
      values: [scalar_only, scalar_local, scalar_local_pairwise, all]
    """
    with open(path) as f:
        raw: dict = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Search-space YAML must be a top-level mapping, got {type(raw)}"
        )

    static_cfg: dict = raw.pop("_static_config", {})

    tune_space: dict[str, Any] = {}
    errors: list[str] = []

    for param_name, spec in raw.items():
        if not isinstance(spec, dict):
            errors.append(
                f"  '{param_name}': expected a mapping, got {type(spec).__name__}"
            )
            continue
        try:
            tune_space[param_name] = _parse_tune_distribution(param_name, spec)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"  '{param_name}': {exc}")

    if errors:
        raise ValueError(
            "Errors while parsing search-space YAML:\n" + "\n".join(errors)
        )

    return tune_space, static_cfg


def _is_fixed(dist: Any) -> bool:
    return (
        isinstance(dist, tune.search.sample.Categorical) and len(dist.categories) == 1
    )


# -----------------------------------------------------------------------------
# Trainable (one trial)
# -----------------------------------------------------------------------------

def trainable(
    trial_params: dict[str, Any],
    *,
    static_cfg: dict,
    search_space_path: str,
    device: str,
    max_epochs: int,
    num_seeds: int = 1,  
) -> None:
    """Called by Ray Tune for every trial."""
    cfg = build_config_from_trial(trial_params, static_cfg)
    cfg.setdefault("training", {})["epochs"] = max_epochs

    scores = []
    base_seed = cfg.get("training", {}).get("seed", 42)

    for i in range(num_seeds):                          
        cfg["training"]["seed"] = base_seed + i       

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            loaded_cfg = get_config(tmp_path)
            trainer = CORE_LIP_Trainer(
                loaded_cfg,
                search_space_path,
                threshold_selection=False,
                device=device,
            )
            auc = trainer.run()
            if auc is not None:
                scores.append(auc)
        finally:
            os.unlink(tmp_path)

    best_auc = float(np.mean(scores)) if scores else 0.0  # ← average
    tune.report({"PR-AUC": best_auc})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimisation for CORE-LIP with Ray Tune"
    )
    parser.add_argument(
        "--search-space",
        required=True,
        help=(
            "Single YAML that fully describes the experiment. "
            "Use 'value:' to fix a parameter, 'type:' to search it. "
            "Put non-hyperparameter fields (paths, seed …) under '_static_config:'."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=250)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--cpus-per-trial", type=float, default=4.0)
    parser.add_argument("--gpus-per-trial", type=float, default=1.0)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--output-dir", default="./ray_results")
    parser.add_argument("--exp-name", default="core_MoRF_hpo_two_seeds_V3")
    args = parser.parse_args()

    # -- Load the single YAML -------------------------------------------------
    search_space_path = Path(args.search_space).resolve()
    tune_space, static_cfg = load_search_space_yaml(str(search_space_path))

    # Resolve relative paths inside _static_config to absolute
    _PATH_KEYS = {
        "h5_properties",
        "training_dataset",
        "validation_dataset",
        "test_dataset",
    }
    for section in static_cfg.values():
        if not isinstance(section, dict):
            continue
        for key, val in section.items():
            if key in _PATH_KEYS and isinstance(val, str):
                p = Path(val)
                if not p.is_absolute():
                    section[key] = str((Path.cwd() / p).resolve())

    # -- Summary --------------------------------------------------------------
    fixed = [k for k, v in tune_space.items() if _is_fixed(v)]
    tuned = [k for k in tune_space if k not in fixed]

    print(f"Search space  : {search_space_path}")
    print(f"Device        : {args.device}")
    print(f"Num samples   : {args.num_samples}")
    print(f"Max epochs    : {args.max_epochs}")
    print(f"Output dir    : {args.output_dir}")
    print()
    print(f"Fixed params  ({len(fixed):2d}): {', '.join(fixed) or 'none'}")
    print(f"Tuned params  ({len(tuned):2d}): {', '.join(tuned) or 'none'}")
    print()

    # -- Ray ------------------------------------------------------------------
    ray.init(ignore_reinit_error=True)
    searcher = OptunaSearch(metric="PR-AUC", mode="max")

    trainable_with_args = tune.with_parameters(
        trainable,
        static_cfg=static_cfg,
        search_space_path=str(search_space_path),
        device=args.device,
        max_epochs=args.max_epochs,
        num_seeds=2,            
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable_with_args,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        param_space=tune_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            search_alg=searcher,
        ),
        run_config=RunConfig(
            name=args.exp_name,
            storage_path=str(Path(args.output_dir).resolve()),
            verbose=2,
        ),
    )

    results = tuner.fit()

    # -- Best result ----------------------------------------------------------
    best = results.get_best_result(metric="PR-AUC", mode="max")
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  PR-AUC : {best.metrics['PR-AUC']:.4f}")
    print("  Params:")
    for k, v in best.config.items():
        print(f"    {k:30s}: {v}")

    best_cfg = build_config_from_trial(best.config, static_cfg)
    best_cfg_path = Path(args.output_dir) / args.exp_name / "best_config.yaml"
    best_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_cfg_path, "w") as f:
        yaml.safe_dump(best_cfg, f)
    print(f"\nBest config saved to: {best_cfg_path}")

    # -- Final retrain --------------------------------------------------------
    print("\nRetraining final model with best hyperparameters ...")
    loaded_best_cfg = get_config(str(best_cfg_path))
    final_trainer = CORE_LIP_Trainer(
        loaded_best_cfg, str(best_cfg_path), device=args.device
    )
    final_auc = final_trainer.run()
    print(f"Final model AUC: {final_auc:.4f}")
    ray.shutdown()


if __name__ == "__main__":
    main()
