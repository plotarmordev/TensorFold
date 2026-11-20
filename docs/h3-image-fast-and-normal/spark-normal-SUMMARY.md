# MiniMax-H3-Image CUDA benchmark

Every selftest passed.

- commit: fa508cd6d291c203d4674c4838a09523ec981819
- GPU, driver: NVIDIA GB10, 580.173.02
- prompt: /assets/drama3-pilot/h3-image-pilot/prompt.txt, sha256 cc969bcab79793c5035d2f0186e09a75efefe9feea34963052940925cd1af499
- seed: 42
- sizes: 2048
- transformer linear layers: int8
- draft steps: 0
- runs per size: 4 (1 cold, 3 warm)

Selftests:

- transformer PASS
- decoder PASS
- prompt PASS

Run 1 is cold (start to first picture); 2-4 are warm repeats. Energy is average power times wall.

| run | wall s | prompt s | load s | case s | transformer s | decoder load s | decoder s | steps | peak MiB | avg W | kJ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048² cold | 151.2 | 1.2 | 54.0 | 0.0 | 42.9 | 3.2 | 1.9 | 6 | 86666.0 | 20.0 | 3.0 |
| 2048² warm 1 | 14.1 | 0.0 | — | 0.0 | 12.9 | — | 1.2 | 6 | 78562.0 | 83.8 | 1.2 |
| 2048² warm 2 | 14.2 | 0.0 | — | 0.0 | 12.9 | — | 1.3 | 6 | 78562.0 | 82.8 | 1.2 |
| 2048² warm 3 | 14.2 | 0.0 | — | 0.0 | 12.9 | — | 1.3 | 6 | 78562.0 | 83.6 | 1.2 |
| 2048² warm mean | 14.2 | 0.0 | — | 0.0 | 12.9 | — | 1.3 | 6.0 | 78562.0 | 83.4 | 1.2 |
