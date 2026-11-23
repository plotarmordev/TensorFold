# bandwidth.py: the GPU's streaming bandwidth over 4 GiB arrays, medians of six timed passes. Run it with an MLX Python
# (oMLX's venv has one) under `lockf -k /tmp/tensorfold-model.lock`.
import statistics
import time

import mlx.core as mx

n = 1 << 30
a = mx.random.uniform(shape=(n,))
b = mx.random.uniform(shape=(n,))
mx.eval(a, b)


def rate(fn, bytes_moved, reps=7):
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t)
    return bytes_moved / statistics.median(ts[1:]) / 1e9


print(f"read (sum)   {rate(lambda: a.sum(), 4 * n):.0f} GB/s")
print(f"add (2R+1W)  {rate(lambda: a + b, 12 * n):.0f} GB/s")
print(f"copy (1R+1W) {rate(lambda: a * 1.0, 8 * n):.0f} GB/s")
