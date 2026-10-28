"""Functional coverage for checkpoint integrity, INT8 reload, masking and preprocessing."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from ngt_model.datasets.collide2v_data import build_split_indices, preprocess_dataset
from quantized_model.core import (FORMAT, build_model, choose_engine, infer, load_bundle,
                                  load_fp32, prepare_inputs, quantize)


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.config = dict(n_constituents=4, n_features=5, latent_dim=8, d_model=16,
                           ff_dim=32, num_heads=2, num_attention_blocks=2,
                           pooling_mode="attention", classifier_head="linear",
                           class_names=["minbias", "qcd", "dy", "top"], pid_encoding="raw",
                           norm=dict(pt_min=0., pt_max=10., eta_min=-5., eta_max=5.,
                                     phi_min=-np.pi, phi_max=np.pi, dxy_min=-10., dxy_max=10.))

    def test_real_int8_state_roundtrip_and_zero_mask(self):
        engine = choose_engine()
        model, selected = quantize(build_model(self.config))
        self.assertEqual(model.input_proj.weight.dtype, torch.float32)
        self.assertEqual(model.attention_blocks[0].q_proj.weight().dtype, torch.qint8)
        x = torch.randn(3, 4, 5)
        x[0] = 0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bundle.pt"
            torch.save(dict(format=FORMAT, config=self.config, engine=engine,
                            excluded_layers=["input_proj"], quantized_layers=selected,
                            model_state_dict=model.state_dict()), path)
            restored, _ = load_bundle(path)
            with torch.inference_mode():
                expected, actual = model(x), restored(x)
            for name in ("latent", "logits"):
                torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual[name]).all())

    def test_frozen_raw_preprocessing_and_batch_inference(self):
        raw = np.zeros((5, 4, 5), dtype=np.float32)
        raw[:, :2] = np.array([30, 1, .3, 2, 211], dtype=np.float32)
        expected, _ = preprocess_dataset(raw, norm=self.config["norm"], pid_mapping={}, add_pid_ohe=False)
        np.testing.assert_array_equal(prepare_inputs(raw, self.config, raw=True), expected)
        result = infer(build_model(self.config), raw, self.config, batch_size=2, raw=True)
        self.assertEqual(result["latent"].shape, (5, 8))
        self.assertEqual(result["logits"].shape, (5, 4))
        with self.assertRaises(ValueError):
            prepare_inputs(raw[..., :4], self.config)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_inputs(raw, self.config)

    def test_strict_checkpoint_no_random_fallback(self):
        model = build_model(self.config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "epoch.pt"
            torch.save(dict(epoch=4, model_state_dict=model.state_dict()), path)
            restored, epoch = load_fp32(path, self.config)
            self.assertEqual(epoch, 4)
            torch.testing.assert_close(restored.input_proj.weight, model.input_proj.weight)
            state = model.state_dict()
            del state["input_proj.weight"]
            torch.save(dict(model_state_dict=state), path)
            with self.assertRaises(RuntimeError):
                load_fp32(path, self.config)

    def test_holdout_permutation_matches_original_split(self):
        names = self.config["class_names"]
        labels = np.repeat(np.arange(4), 80)
        mapping = {name: i for i, name in enumerate(names)}
        train, val, evaluation = build_split_indices(labels, names, mapping, 42, 10, 10)
        rng = np.random.default_rng(42)
        for i in range(4):
            permutation = rng.permutation(80) + i * 80
            self.assertTrue(set(permutation[:10]).issubset(set(evaluation)))
            self.assertTrue(set(permutation[10:20]).issubset(set(val)))
            self.assertTrue(set(permutation[:20]).isdisjoint(set(train)))

    def test_invalid_exclusion_fails(self):
        with self.assertRaises(ValueError):
            quantize(build_model(self.config), ["does_not_exist"])


if __name__ == "__main__":
    unittest.main()
