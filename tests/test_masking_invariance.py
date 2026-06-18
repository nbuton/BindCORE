import unittest

import torch

from bindcore.config import ProteinModelConfig
from bindcore.modeling.protein_multi_scale_transformer import (
    BiasedMultiHeadAttention,
    PairwiseCNN,
    PairwiseUpdateBlock,
    ProteinMultiScaleTransformer,
)


def _assert_close(testcase, actual, expected, atol=1e-5):
    max_diff = (actual - expected).abs().max().item()
    testcase.assertTrue(
        torch.allclose(actual, expected, atol=atol, rtol=0.0),
        f"max_abs_diff={max_diff}",
    )


class MaskingInvarianceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def _cfg(self):
        return ProteinModelConfig(
            vocab_size=25,
            nb_scalar=3,
            nb_local=4,
            nb_pairwise=3,
            embed_dim=24,
            num_blocks=2,
            num_heads=4,
            max_seq_len=64,
            dropout=0.0,
            pair_dropout_rate=0.0,
            pairwise_cnn_channels=6,
            dilatations_cnn=(1, 2),
            use_token_embedding=True,
            use_positional_embeddings=True,
            use_scalar_features=True,
            use_local_features=True,
            use_pairwise_features=True,
            use_plm_embedding=True,
            plm_dim=5,
        )

    def _stats(self, cfg):
        return {
            "scalar": {
                "means": torch.tensor([0.5, -0.25, 1.25]),
                "stds": torch.tensor([1.5, 0.75, 2.0]),
            },
            "local": {
                "means": torch.linspace(-0.4, 0.3, cfg.nb_local),
                "stds": torch.linspace(0.8, 1.3, cfg.nb_local),
            },
            "pairwise": {
                "means": torch.tensor([0.2, -0.7, 1.1]),
                "stds": torch.tensor([1.2, 0.6, 2.1]),
            },
        }

    def _sample(self, cfg, length):
        pairwise = torch.randn(cfg.nb_pairwise, length, length)
        pairwise = (pairwise + pairwise.transpose(-1, -2)) / 2
        return {
            "tokens": torch.randint(1, cfg.vocab_size, (length,)),
            "x_scalar": torch.randn(cfg.nb_scalar),
            "x_local": torch.randn(cfg.nb_local, length),
            "x_pairwise": pairwise,
            "plm": torch.randn(length, cfg.plm_dim),
            "length": length,
        }

    def _batch(self, cfg, samples, pad_len):
        batch_size = len(samples)
        tokens = torch.randint(1, cfg.vocab_size, (batch_size, pad_len))
        x_scalar = torch.stack([sample["x_scalar"] for sample in samples])
        x_local = torch.randn(batch_size, cfg.nb_local, pad_len)
        x_pairwise = torch.randn(batch_size, cfg.nb_pairwise, pad_len, pad_len)
        plm = torch.randn(batch_size, pad_len, cfg.plm_dim)
        mask = torch.zeros(batch_size, pad_len, dtype=torch.bool)

        for idx, sample in enumerate(samples):
            length = sample["length"]
            tokens[idx, :length] = sample["tokens"]
            x_local[idx, :, :length] = sample["x_local"]
            x_pairwise[idx, :, :length, :length] = sample["x_pairwise"]
            plm[idx, :length] = sample["plm"]
            mask[idx, :length] = True

        return tokens, x_scalar, x_local, x_pairwise, mask, plm

    def test_attention_masks_keys_queries_and_padding_size(self):
        attention = BiasedMultiHeadAttention(
            embed_dim=16,
            num_heads=4,
            dropout=0.0,
            activate_bias=True,
            activate_classical_attention=True,
        ).eval()

        length = 7
        x = torch.randn(1, length, 16)
        bias = torch.randn(1, 4, length, length)
        mask = torch.ones(1, length, dtype=torch.bool)

        with torch.no_grad():
            expected = attention(x, bias, mask)

            padded_len = 13
            x_padded = torch.randn(1, padded_len, 16)
            bias_padded = torch.randn(1, 4, padded_len, padded_len)
            mask_padded = torch.zeros(1, padded_len, dtype=torch.bool)
            x_padded[:, :length] = x
            bias_padded[:, :, :length, :length] = bias
            mask_padded[:, :length] = True
            actual = attention(x_padded, bias_padded, mask_padded)

        _assert_close(self, actual[:, :length], expected)
        self.assertEqual(actual[:, length:].abs().max().item(), 0.0)

    def test_pairwise_cnn_bias_is_padding_size_invariant(self):
        cfg = self._cfg()
        cnn = PairwiseCNN(
            nb_pairwise=cfg.nb_pairwise,
            cnn_channels=cfg.pairwise_cnn_channels,
            num_heads=cfg.num_heads,
            kernel_size=cfg.pairwise_cnn_kernel,
            dilations=cfg.dilatations_cnn,
            dropout=0.0,
        ).eval()

        length = 8
        x_pairwise = torch.randn(1, cfg.nb_pairwise, length, length)
        mask = torch.ones(1, length, dtype=torch.bool)

        with torch.no_grad():
            expected = cnn(x_pairwise, mask)

            padded_len = 15
            x_padded = torch.randn(1, cfg.nb_pairwise, padded_len, padded_len)
            mask_padded = torch.zeros(1, padded_len, dtype=torch.bool)
            x_padded[:, :, :length, :length] = x_pairwise
            mask_padded[:, :length] = True
            actual = cnn(x_padded, mask_padded)

        _assert_close(self, actual[:, :, :length, :length], expected)
        self.assertEqual(actual[:, :, length:, :].abs().max().item(), 0.0)
        self.assertEqual(actual[:, :, :, length:].abs().max().item(), 0.0)

    def test_outer_product_pairwise_update_is_padding_size_invariant(self):
        cfg = self._cfg()
        block = PairwiseUpdateBlock(
            embed_dim=cfg.embed_dim,
            nb_pairwise=cfg.nb_pairwise,
            dropout=0.0,
        ).eval()

        length = 9
        x_pairwise = torch.randn(1, cfg.nb_pairwise, length, length)
        x = torch.randn(1, length, cfg.embed_dim)
        mask = torch.ones(1, length, dtype=torch.bool)

        with torch.no_grad():
            expected = block(x_pairwise, x, mask)

            padded_len = 16
            x_pairwise_padded = torch.randn(1, cfg.nb_pairwise, padded_len, padded_len)
            x_padded = torch.randn(1, padded_len, cfg.embed_dim)
            mask_padded = torch.zeros(1, padded_len, dtype=torch.bool)
            x_pairwise_padded[:, :, :length, :length] = x_pairwise
            x_padded[:, :length] = x
            mask_padded[:, :length] = True
            actual = block(x_pairwise_padded, x_padded, mask_padded)

        _assert_close(self, actual[:, :, :length, :length], expected)
        self.assertEqual(actual[:, :, length:, :].abs().max().item(), 0.0)
        self.assertEqual(actual[:, :, :, length:].abs().max().item(), 0.0)

    def test_full_model_outputs_do_not_depend_on_batch_or_padding_size(self):
        cfg = self._cfg()
        model = ProteinMultiScaleTransformer(
            cfg,
            self._stats(cfg),
            pairwise_features=["dccm", "contact_map", "other"],
        ).eval()

        sample_a = self._sample(cfg, 7)
        sample_b = self._sample(cfg, 11)
        sample_c = self._sample(cfg, 17)

        with torch.no_grad():
            out_single = model(*self._batch(cfg, [sample_a], sample_a["length"]))
            out_padded = model(*self._batch(cfg, [sample_a], 19))
            out_batch_two = model(*self._batch(cfg, [sample_a, sample_b], 11))
            out_batch_three = model(*self._batch(cfg, [sample_a, sample_b, sample_c], 17))

        length = sample_a["length"]
        expected = out_single[:, :length]
        _assert_close(self, out_padded[:, :length], expected)
        _assert_close(self, out_batch_two[:1, :length], expected)
        _assert_close(self, out_batch_three[:1, :length], expected)
        self.assertEqual(out_padded[:, length:].abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
