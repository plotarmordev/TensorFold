"""Boundary and refusal checks for the H3 image comparison gate."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

from h3_image_compare import compare, latent


class ImageGateTest(unittest.TestCase):
    def test_pixel_boundary_and_latent_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, candidate = root / "a.f32", root / "b.f32"
            a, b = root / "a.png", root / "b.png"
            values = np.ones(24 * 2 * 2, dtype="<f4")
            values.tofile(reference)
            values.tofile(candidate)
            Image.new("RGB", (32, 32), (128, 128, 128)).save(a)
            Image.new("RGB", (32, 32), (129, 128, 128)).save(b)
            result = compare(reference, candidate, a, b, 32)
            self.assertTrue(result["latent_pass"])
            self.assertTrue(result["image_pass"])
            self.assertEqual(result["pixel_max_abs"], 1)
            Image.new("RGB", (32, 32), (130, 128, 128)).save(b)
            self.assertFalse(compare(reference, candidate, a, b, 32)["image_pass"])
            (-values).tofile(candidate)
            self.assertFalse(compare(reference, candidate, a, b, 32)["latent_pass"])

    def test_malformed_and_nonfinite_latents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.f32"
            np.zeros(95, dtype="<f4").tofile(path)
            with self.assertRaises(ValueError):
                latent(path, 32)
            values = np.ones(96, dtype="<f4")
            values[0] = np.nan
            values.tofile(path)
            with self.assertRaises(ValueError):
                latent(path, 32)
            path = path.with_suffix(".latent")
            save_file({"latent_tensor": np.ones((2, 12, 2, 2), dtype="<f4")}, path)
            with self.assertRaises(ValueError):
                latent(path, 32)

    def test_zero_norm_and_wrong_image_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, image = root / "zero.f32", root / "image.png"
            np.zeros(96, dtype="<f4").tofile(path)
            with self.assertRaises(ValueError):
                compare(path, path, image, image, 32)
            np.ones(96, dtype="<f4").tofile(path)
            Image.new("RGB", (16, 32)).save(image)
            with self.assertRaises(ValueError):
                compare(path, path, image, image, 32)


if __name__ == "__main__":
    unittest.main()
