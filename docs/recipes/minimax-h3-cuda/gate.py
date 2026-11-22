"""Episode 1's picture gates on TOP, which has no numpy or PIL: latent cosine, PSNR and mean difference of two runs.

usage: python3 gate.py REFERENCE_PREFIX CANDIDATE_PREFIX  (each PREFIX.latent.f32 and PREFIX.png, tf-h3-image's files)
Our PNGs are one IDAT of 8-bit RGB rows with filter 0 (moss_image/png.zig), so a row is its bytes after the filter byte.
"""

import math
import sys
import zlib
from array import array


def latent(path):
    values = array("f")
    with open(path, "rb") as f:
        values.frombytes(f.read())
    return values


def pixels(path):
    with open(path, "rb") as f:
        data = f.read()
    at, idat, width, height = 8, b"", 0, 0
    while at < len(data):
        n = int.from_bytes(data[at : at + 4], "big")
        kind, body = data[at + 4 : at + 8], data[at + 8 : at + 8 + n]
        if kind == b"IHDR":
            width, height = int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big")
            if body[8:10] != b"\x08\x02":
                raise ValueError(f"{path}: not 8-bit RGB")
        elif kind == b"IDAT":
            idat += body
        at += 12 + n
    rows = zlib.decompress(idat)
    stride = 3 * width + 1
    if len(rows) != stride * height or any(rows[r * stride] for r in range(height)):
        raise ValueError(f"{path}: not filter-0 rows")
    return b"".join(rows[r * stride + 1 : (r + 1) * stride] for r in range(height))


def main():
    reference, candidate = sys.argv[1:3]
    a, b = latent(reference + ".latent.f32"), latent(candidate + ".latent.f32")
    if len(a) != len(b):
        raise ValueError("latent sizes differ")
    cosine = math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(math.fsum(x * x for x in a) * math.fsum(y * y for y in b))
    p, q = pixels(reference + ".png"), pixels(candidate + ".png")
    if len(p) != len(q):
        raise ValueError("picture sizes differ")
    absolute = squared = 0
    for x, y in zip(p, q):
        d = x - y
        absolute += abs(d)
        squared += d * d
    mean, mse = absolute / len(p), squared / len(p)
    psnr = math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)
    passed = cosine >= 0.99 and psnr >= 28.2 and mean <= 4.1
    print(f"{candidate}: latent cosine {cosine:.4f}, PSNR {psnr:.2f} dB, mean diff {mean:.2f}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
