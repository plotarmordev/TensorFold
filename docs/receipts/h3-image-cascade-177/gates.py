"""Idea 1's picture gates, cascaded PNG against the uncascaded one: `gates.py BASE.png CASCADED.png`.

Prints the cascaded picture's top-octave radial spectrum energy over the base's (gate: at least 0.90) and
LPIPS (AlexNet, on the CPU; gate: at most 0.08).
"""
import sys

import lpips
import numpy as np
import torch
from PIL import Image


def top_octave(path):
    """Luma power at radial frequencies between half Nyquist and Nyquist."""
    y = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    power = np.abs(np.fft.fft2(y - y.mean())) ** 2
    fy = np.fft.fftfreq(y.shape[0])[:, None]
    fx = np.fft.fftfreq(y.shape[1])[None, :]
    radius = np.hypot(fy, fx)
    return power[(radius >= 0.25) & (radius <= 0.5)].sum()


def tensor(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(rgb).permute(2, 0, 1)[None]


base, cascaded = sys.argv[1:3]
energy = top_octave(cascaded) / top_octave(base)
with torch.no_grad():
    distance = lpips.LPIPS(net="alex", verbose=False)(tensor(base), tensor(cascaded)).item()
verdict = "pass" if energy >= 0.90 and distance <= 0.08 else "fail"
print(f"{cascaded}: top-octave energy {energy:.3f}, LPIPS {distance:.3f}, {verdict}")
