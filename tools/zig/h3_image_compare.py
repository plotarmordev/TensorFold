"""Compare normalized H3 F32 latents and decoded RGB with fixed pre-run gates."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file


def latent(path, size):
    if path.suffix == ".latent":
        values = load_file(path)["latent_tensor"]
        if values.shape not in ((1, 24, size // 16, size // 16), (1, 24, 1, size // 16, size // 16)):
            raise ValueError(f"unexpected latent shape: {values.shape}")
    else:
        if path.stat().st_size != 24 * (size // 16) ** 2 * 4:
            raise ValueError("wrong raw latent byte size")
        values = np.fromfile(path, dtype="<f4")
    if values.size != 24 * (size // 16) ** 2 or not np.isfinite(values).all():
        raise ValueError("wrong latent size or non-finite values")
    return values.astype(np.float64).ravel()


def compare(reference, candidate, reference_png, candidate_png, size):
    a, b = latent(reference, size), latent(candidate, size)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        raise ValueError("zero-norm latent")
    cosine = float(np.dot(a, b) / denominator)
    images = []
    for path in (reference_png, candidate_png):
        with Image.open(path) as image:
            if image.size != (size, size):
                raise ValueError(f"unexpected image size: {image.size}")
            images.append(np.asarray(image.convert("RGB"), dtype=np.int16))
    difference = np.abs(images[0] - images[1])
    return {
        "latent_cosine": cosine,
        "latent_relative_error": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "pixel_max_abs": int(difference.max()),
        "pixel_mae": float(difference.mean()),
        "latent_pass": cosine >= 0.99,
        "image_pass": int(difference.max()) <= 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_latent", type=Path)
    parser.add_argument("candidate_latent", type=Path)
    parser.add_argument("reference_png", type=Path)
    parser.add_argument("candidate_png", type=Path)
    parser.add_argument("--size", type=int, choices=(2048, 4096), required=True)
    args = parser.parse_args()
    result = compare(args.reference_latent, args.candidate_latent, args.reference_png, args.candidate_png, args.size)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["latent_pass"] and result["image_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
