from pydantic import BaseModel, model_validator
from pathlib import Path
from typing import List, Tuple
from typing import Dict, Any


class ProteinModelConfig(BaseModel):
    """Central configuration for all model hyperparameters."""

    # ── Vocabulary / sequence ──────────────────────────────────────────────
    vocab_size: int = 25
    pad_token_id: int = 0

    # ── Feature dimensions ────────────────────────────────────────────────
    nb_scalar: int = 16
    nb_local: int = 32
    nb_pairwise: int = 8
    use_scalar_features: bool = True
    use_local_features: bool = True
    use_pairwise_features: bool = True
    use_token_embedding: bool = False
    use_positional_embeddings: bool = False
    use_plm_embedding: bool = False

    # ── Embedding / model width ───────────────────────────────────────────
    embed_dim: int = 128
    max_seq_len: int = 1024
    activate_pairwise_bias: bool = True
    activate_classical_attention: bool = True

    # ── Transformer blocks ────────────────────────────────────────────────
    num_blocks: int = 4
    share_block_weights: bool = False
    num_heads: int = 8
    ffn_expansion: int = 2
    dropout: float = 0.1

    # ── Pairwise CNN (inside each block) ──────────────────────────────────
    pairwise_cnn_channels: int = 32
    pairwise_cnn_kernel: int = 3
    dilatations_cnn: Tuple[int, ...] = (1, 2, 3)

    # ── Classification head ───────────────────────────────────────────────
    num_classes: int = 1

    # ── MLP hidden sizes (defaults derived from embed_dim) ────────────────
    local_mlp_hidden: int = -1
    scalar_mlp_hidden: int = -1

    # Add pre trainned embeddings
    plm_dim: int = 6144

    # ── Post-Initialization & Validation ──────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_inputs_features(cls, data):
        if not isinstance(data, dict) or "inputs_features" not in data:
            return data

        migrated = data.copy()
        legacy_value = migrated.pop("inputs_features") or []
        if isinstance(legacy_value, str):
            legacy_value = {
                "plm_only": ["plm_embedding"],
                "scalar_only": ["scalar_features"],
                "scalar_local": ["scalar_features", "local_features"],
                "scalar_local_pairwise": [
                    "scalar_features",
                    "local_features",
                    "pairwise_features",
                ],
                "scalar_local_pairwise_res": [
                    "token_embedding",
                    "scalar_features",
                    "local_features",
                    "pairwise_features",
                ],
                "scalar_local_pairwise_res_pos": [
                    "token_embedding",
                    "positional_embeddings",
                    "scalar_features",
                    "local_features",
                    "pairwise_features",
                ],
                "all_structural": [
                    "token_embedding",
                    "scalar_features",
                    "local_features",
                    "pairwise_features",
                ],
            }.get(legacy_value, [legacy_value])

        inputs_features = set(legacy_value)
        if "seq_emb" in inputs_features:
            inputs_features.add("token_embedding")

        feature_flags = {
            "scalar_features": "use_scalar_features",
            "local_features": "use_local_features",
            "pairwise_features": "use_pairwise_features",
            "token_embedding": "use_token_embedding",
            "positional_embeddings": "use_positional_embeddings",
            "plm_embedding": "use_plm_embedding",
        }
        for feature_name, flag_name in feature_flags.items():
            migrated.setdefault(flag_name, feature_name in inputs_features)

        return migrated

    @model_validator(mode="after")
    def validate_and_set_defaults(self) -> "ProteinModelConfig":
        # 1. Set dynamic defaults
        if self.local_mlp_hidden < 0:
            self.local_mlp_hidden = self.embed_dim
        if self.scalar_mlp_hidden < 0:
            self.scalar_mlp_hidden = self.embed_dim

        # 2. Replicate your assertions as proper Pydantic validations
        if self.embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even (pairwise windowing)")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        return self


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    accumulation: int
    scheduler_type: str
    optimizer: str
    loss_type: str
    loss_params: Dict[str, Any] = {}
    val_prop: float
    lr: float
    weight_decay: float
    seed: int
    use_ema: bool = False
    ema_decay: float = 0.999
    h5_properties: Path
    training_dataset: Path
    SCALAR_FEATURES: List
    LOCAL_FEATURES: List
    PAIRWISE_FEATURES: List


# You can now hook this up to your main config just like before:
class FullConfig(BaseModel):
    training: TrainingConfig  # (From the previous example)
    model: ProteinModelConfig
