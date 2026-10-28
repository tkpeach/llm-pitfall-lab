#!/usr/bin/env python3
"""Validate declared conversion rules against real pinned upstream transformations on tiny tensors."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parents[2] / "tools/model_artifacts/qwen35"))
from qwen35_conversion import load_rules, converter_identity, audit_sanitization, audit_quantization, sanitization_plan

import mlx.core as mx
from mlx_lm.models.qwen3_5 import Model, TextModel


def upstream_sanitize(weights):
    # Exercise actual upstream sanitizer methods without constructing 4B random weights.
    text = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=True))
    model = SimpleNamespace(language_model=SimpleNamespace(sanitize=lambda w: TextModel.sanitize(text, w)))
    return Model.sanitize(model, weights)


def tiny_source():
    return {
        "model.language_model.embed_tokens.weight": mx.arange(256, dtype=mx.float32).reshape(4, 64).astype(mx.bfloat16),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.arange(16, dtype=mx.float32).reshape(4, 1, 4).astype(mx.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": mx.array([0., 1., -1., 0.5], mx.bfloat16),
        "model.language_model.layers.0.linear_attn.norm.weight": mx.array([1., 2., 3., 4.], mx.float32),
        "model.language_model.norm.weight": mx.zeros((4,), mx.bfloat16),
        "model.visual.blocks.0.weight": mx.ones((2, 2)),
        "mtp.fc.weight": mx.ones((2, 2)),
    }


class ConversionRulesTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules()
        self.raw = tiny_source()
        self.sanitized = upstream_sanitize(self.raw)

    def test_real_upstream_matches_declared_rules(self):
        audit = audit_sanitization(self.raw, self.sanitized, self.rules)
        self.assertEqual(audit["retained"], 5)
        self.assertEqual(audit["dropped"], {"vision": 1, "mtp": 1})
        self.assertEqual(audit["operations"], {"rename_prefix": 5, "moveaxis_2_to_1": 1, "norm_weight_plus_one": 2})
        self.assertEqual(self.sanitized["language_model.model.layers.0.linear_attn.conv1d.weight"].shape, (4, 4, 1))
        self.assertEqual(self.sanitized["language_model.model.layers.0.input_layernorm.weight"].tolist(), [1., 2., 0., 1.5])
        self.assertEqual(self.raw["model.language_model.layers.0.input_layernorm.weight"].tolist(), [0., 1., -1., 0.5])

    def test_missing_norm_offset_is_rejected(self):
        self.sanitized["language_model.model.layers.0.input_layernorm.weight"] = self.raw["model.language_model.layers.0.input_layernorm.weight"]
        with self.assertRaisesRegex(ValueError, "values differ"):
            audit_sanitization(self.raw, self.sanitized, self.rules)

    def test_wrong_conv_axis_order_is_rejected(self):
        self.sanitized["language_model.model.layers.0.linear_attn.conv1d.weight"] = self.raw["model.language_model.layers.0.linear_attn.conv1d.weight"]
        with self.assertRaisesRegex(ValueError, "layout differs"):
            audit_sanitization(self.raw, self.sanitized, self.rules)

    def test_unlisted_matrix_change_is_rejected(self):
        self.sanitized["language_model.model.embed_tokens.weight"] = self.sanitized["language_model.model.embed_tokens.weight"] + 1
        with self.assertRaisesRegex(ValueError, "unlisted modification"):
            audit_sanitization(self.raw, self.sanitized, self.rules)

    def test_missing_output_and_unknown_input_are_rejected(self):
        del self.sanitized["language_model.model.norm.weight"]
        with self.assertRaisesRegex(ValueError, "output tensor names"):
            audit_sanitization(self.raw, self.sanitized, self.rules)
        self.raw["unknown.weight"] = mx.ones((1,))
        with self.assertRaisesRegex(ValueError, "outside the fixed"):
            sanitization_plan(self.raw, self.rules)

    def test_quantized_tensor_contract_and_missing_bias(self):
        audit = audit_sanitization(self.raw, self.sanitized, self.rules)
        result = dict(self.sanitized)
        name = "language_model.model.embed_tokens"
        packed, scale, bias = mx.quantize(result[name + ".weight"], group_size=64, bits=4, mode="affine")
        result.update({name + ".weight": packed, name + ".scales": scale, name + ".biases": bias})
        output = audit_quantization(audit, result, self.rules)
        self.assertEqual(output["quantized_matrices"], 1)
        self.assertEqual(output["output_tensors"], 7)
        del result[name + ".biases"]
        with self.assertRaisesRegex(ValueError, "shape/name differs"):
            audit_quantization(audit, result, self.rules)

    def test_pinned_converter_identity_and_mismatch(self):
        identity = converter_identity(self.rules)
        self.assertEqual(identity["packages"], self.rules["packages"])
        wrong = copy.deepcopy(self.rules)
        wrong["critical_files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "converter file differs"):
            converter_identity(wrong)


if __name__ == "__main__":
    unittest.main()
