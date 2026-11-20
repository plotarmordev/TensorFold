# MiniMax-H3-Image CUDA benchmark

Every selftest passed.

- commit: fa508cd6d291c203d4674c4838a09523ec981819
- GPU, driver: NVIDIA GB10, 580.173.02
- prompt: /assets/drama3-pilot/h3-image-pilot/prompt.txt, sha256 cc969bcab79793c5035d2f0186e09a75efefe9feea34963052940925cd1af499
- seed: 42
- sizes: 2048
- transformer linear layers: int8
- draft steps: 3
- runs per size: 4 (1 cold, 3 warm)

Selftests:

- transformer PASS
- decoder PASS
- prompt PASS

Run 1 is cold (start to first picture); 2-4 are warm repeats. Energy is average power times wall.

| run | wall s | prompt s | load s | case s | transformer s | decoder load s | decoder s | steps | peak MiB | avg W | kJ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048² cold | 152.7 | 1.2 | 53.3 | 0.0 | 37.2 | 3.1 | 1.8 | 6 | 86666.0 | 18.8 | 2.9 |
| 2048² warm 1 | 9.5 | 0.0 | — | 0.0 | 8.2 | — | 1.2 | 6 | 78562.0 | 81.0 | 0.8 |
| 2048² warm 2 | 9.9 | 0.0 | — | 0.0 | 8.3 | — | 1.6 | 6 | 78562.0 | 78.9 | 0.8 |
| 2048² warm 3 | 10.4 | 0.0 | — | 0.0 | 9.1 | — | 1.2 | 6 | 78562.0 | 75.6 | 0.8 |
| 2048² warm mean | 9.9 | 0.0 | — | 0.0 | 8.6 | — | 1.4 | 6.0 | 78562.0 | 78.5 | 0.8 |
