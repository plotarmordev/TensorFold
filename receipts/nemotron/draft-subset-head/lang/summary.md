| run | commit | server sha256 | costs measured fresh | ready s | result |
|---|---|---|---|---|---|
| lang-base-1 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | True | 64.0 | completed |
| lang-pr-1 | f58dc04b4a98ed2e5b89c17bb5966ef58a1df4c7 | 3eefef75610ce5b42262f9c4ad5a347c8d36f04ba62022811edf3afe782a2596 | True | 60.0 | completed |
| lang-pr-2 | f58dc04b4a98ed2e5b89c17bb5966ef58a1df4c7 | 3eefef75610ce5b42262f9c4ad5a347c8d36f04ba62022811edf3afe782a2596 | False | 18.0 | completed |
| lang-base-2 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | False | 16.0 | completed |
| lang-pr-3 | f58dc04b4a98ed2e5b89c17bb5966ef58a1df4c7 | 3eefef75610ce5b42262f9c4ad5a347c8d36f04ba62022811edf3afe782a2596 | False | 18.0 | completed |
| lang-base-3 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | False | 16.0 | completed |

| run | admission |
|---|---|
| lang-base-1 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.3 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| lang-pr-1 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.22 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.1 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| lang-pr-2 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.22 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.9 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| lang-base-2 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.2 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| lang-pr-3 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.22 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.9 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| lang-base-3 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.3 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |

| run | head | level ms | window ms at 1 / 2 / 4 / 8 / 16 / 17 / 24 / 32 rows |
|---|---|---|---|
| lang-base-1 | full | 3.492 | 47.25 / 55.34 / 67.57 / 87.02 / 118.83 / 123.16 / 144.62 / 165.52 |
| lang-pr-1 | full | 3.517 | 47.60 / 55.69 / 67.90 / 87.28 / 119.13 / 123.47 / 144.79 / 165.61 |
| lang-pr-1 | subset | 2.053 | 47.60 / 55.69 / 67.90 / 87.28 / 119.13 / 123.47 / 144.79 / 165.61 |
| lang-pr-2 | full | 3.517 | 47.60 / 55.69 / 67.90 / 87.28 / 119.13 / 123.47 / 144.79 / 165.61 |
| lang-pr-2 | subset | 2.053 | 47.60 / 55.69 / 67.90 / 87.28 / 119.13 / 123.47 / 144.79 / 165.61 |
| lang-base-2 | full | 3.492 | 47.25 / 55.34 / 67.57 / 87.02 / 118.83 / 123.16 / 144.62 / 165.52 |
| lang-pr-3 | full | 3.517 | 47.60 / 55.69 / 67.90 / 87.28 / 119.13 / 123.47 / 144.79 / 165.61 |
| lang-pr-3 | subset | 2.053 | 47.60 / 55.69 / 67.90 / 87.28 / 119.13 / 123.47 / 144.79 / 165.61 |
| lang-base-3 | full | 3.492 | 47.25 / 55.34 / 67.57 / 87.02 / 118.83 / 123.16 / 144.62 / 165.52 |

| run | prompt | drafted tok/s, median of 3 (all) | tokens a round | acceptance | draft:false tok/s | token_sha256 | drafted == draft:false, stable |
|---|---|---|---|---|---|---|---|
| lang-base-1 | lang_en | 32.49 (32.49, 32.49, 32.47) | 2.46 | 0.633 | 20.51 | 44c9ab4037035f03 | True |
| lang-base-1 | lang_es | 34.10 (34.10, 34.08, 34.10) | 2.56 | 0.660 | 20.50 | bc0d0e8304606314 | True |
| lang-base-1 | lang_fr | 32.02 (32.02, 32.02, 32.03) | 2.37 | 0.622 | 20.49 | da60dd14386b3366 | True |
| lang-base-1 | lang_de | 32.17 (32.17, 32.17, 32.18) | 2.44 | 0.606 | 20.49 | 8c93dbb17a55769e | True |
| lang-base-1 | lang_it | 35.55 (35.55, 35.56, 35.51) | 2.64 | 0.726 | 20.49 | 0ca37a8469fc06fa | True |
| lang-base-1 | lang_ja | 42.34 (42.35, 42.34, 42.33) | 3.41 | 0.786 | 20.49 | 07f8403ff9649a39 | True |
| lang-base-1 | multilingual | 41.13 (41.12, 41.13, 41.13) | 3.46 | 0.726 | 20.50 | 757c6ebb3cd489cb | True |
| lang-pr-1 | lang_en | 34.03 (34.05, 34.02, 34.03) | 2.46 | 0.623 | 20.48 | 44c9ab4037035f03 | True |
| lang-pr-1 | lang_es | 34.16 (34.16, 34.16, 34.17) | 2.56 | 0.668 | 20.48 | bc0d0e8304606314 | True |
| lang-pr-1 | lang_fr | 31.99 (31.99, 31.99, 31.99) | 2.37 | 0.622 | 20.48 | da60dd14386b3366 | True |
| lang-pr-1 | lang_de | 32.16 (32.15, 32.16, 32.16) | 2.44 | 0.606 | 20.47 | 8c93dbb17a55769e | True |
| lang-pr-1 | lang_it | 35.49 (35.49, 35.49, 35.51) | 2.64 | 0.726 | 20.48 | 0ca37a8469fc06fa | True |
| lang-pr-1 | lang_ja | 42.37 (42.37, 42.39, 42.37) | 3.41 | 0.789 | 20.47 | 07f8403ff9649a39 | True |
| lang-pr-1 | multilingual | 41.18 (41.18, 41.16, 41.18) | 3.46 | 0.712 | 20.48 | 757c6ebb3cd489cb | True |
| lang-pr-2 | lang_en | 34.10 (34.09, 34.10, 34.10) | 2.46 | 0.623 | 20.58 | 44c9ab4037035f03 | True |
| lang-pr-2 | lang_es | 34.23 (34.23, 34.22, 34.24) | 2.56 | 0.668 | 20.58 | bc0d0e8304606314 | True |
| lang-pr-2 | lang_fr | 32.05 (32.05, 32.05, 32.05) | 2.37 | 0.622 | 20.57 | da60dd14386b3366 | True |
| lang-pr-2 | lang_de | 32.20 (32.21, 32.20, 32.19) | 2.44 | 0.606 | 20.56 | 8c93dbb17a55769e | True |
| lang-pr-2 | lang_it | 35.54 (35.54, 35.56, 35.54) | 2.64 | 0.726 | 20.56 | 0ca37a8469fc06fa | True |
| lang-pr-2 | lang_ja | 42.43 (42.44, 42.43, 42.43) | 3.41 | 0.789 | 20.56 | 07f8403ff9649a39 | True |
| lang-pr-2 | multilingual | 41.25 (41.22, 41.25, 41.25) | 3.46 | 0.712 | 20.57 | 757c6ebb3cd489cb | True |
| lang-base-2 | lang_en | 32.44 (32.45, 32.44, 32.40) | 2.46 | 0.633 | 20.45 | 44c9ab4037035f03 | True |
| lang-base-2 | lang_es | 34.05 (34.06, 34.05, 34.05) | 2.56 | 0.660 | 20.45 | bc0d0e8304606314 | True |
| lang-base-2 | lang_fr | 31.98 (31.98, 31.99, 31.98) | 2.37 | 0.622 | 20.45 | da60dd14386b3366 | True |
| lang-base-2 | lang_de | 32.14 (32.15, 32.14, 32.14) | 2.44 | 0.606 | 20.45 | 8c93dbb17a55769e | True |
| lang-base-2 | lang_it | 35.53 (35.53, 35.54, 35.53) | 2.64 | 0.726 | 20.45 | 0ca37a8469fc06fa | True |
| lang-base-2 | lang_ja | 42.31 (42.30, 42.33, 42.31) | 3.41 | 0.786 | 20.44 | 07f8403ff9649a39 | True |
| lang-base-2 | multilingual | 41.10 (41.10, 41.10, 41.10) | 3.46 | 0.726 | 20.47 | 757c6ebb3cd489cb | True |
| lang-pr-3 | lang_en | 34.22 (34.24, 34.22, 34.22) | 2.46 | 0.623 | 20.54 | 44c9ab4037035f03 | True |
| lang-pr-3 | lang_es | 34.32 (34.31, 34.32, 34.32) | 2.56 | 0.668 | 20.54 | bc0d0e8304606314 | True |
| lang-pr-3 | lang_fr | 32.12 (32.13, 32.12, 32.12) | 2.37 | 0.622 | 20.53 | da60dd14386b3366 | True |
| lang-pr-3 | lang_de | 32.29 (32.29, 32.28, 32.29) | 2.44 | 0.606 | 20.53 | 8c93dbb17a55769e | True |
| lang-pr-3 | lang_it | 35.66 (35.66, 35.67, 35.66) | 2.64 | 0.726 | 20.54 | 0ca37a8469fc06fa | True |
| lang-pr-3 | lang_ja | 42.55 (42.55, 42.54, 42.55) | 3.41 | 0.789 | 20.53 | 07f8403ff9649a39 | True |
| lang-pr-3 | multilingual | 41.38 (41.38, 41.36, 41.38) | 3.46 | 0.712 | 20.55 | 757c6ebb3cd489cb | True |
| lang-base-3 | lang_en | 32.46 (32.46, 32.46, 32.46) | 2.46 | 0.633 | 20.51 | 44c9ab4037035f03 | True |
| lang-base-3 | lang_es | 34.05 (34.05, 34.06, 34.05) | 2.56 | 0.660 | 20.51 | bc0d0e8304606314 | True |
| lang-base-3 | lang_fr | 31.98 (31.98, 31.98, 31.98) | 2.37 | 0.622 | 20.50 | da60dd14386b3366 | True |
| lang-base-3 | lang_de | 32.13 (32.13, 32.13, 32.13) | 2.44 | 0.606 | 20.44 | 8c93dbb17a55769e | True |
| lang-base-3 | lang_it | 35.15 (35.15, 35.23, 34.92) | 2.64 | 0.726 | 20.14 | 0ca37a8469fc06fa | True |
| lang-base-3 | lang_ja | 42.29 (42.29, 42.29, 42.26) | 3.41 | 0.786 | 20.50 | 07f8403ff9649a39 | True |
| lang-base-3 | multilingual | 41.07 (41.07, 41.05, 41.07) | 3.46 | 0.726 | 20.52 | 757c6ebb3cd489cb | True |

lang-base-1 concurrent run 1: multilingual == solo, lang_ja == solo, lang_de == solo, lang_en == solo
lang-base-1 concurrent run 2: multilingual == solo, lang_ja == solo, lang_de == solo, lang_en == solo
lang-pr-1 concurrent run 1: multilingual == solo, lang_ja == solo, lang_de == solo, lang_en == solo
lang-pr-1 concurrent run 2: lang_ja == solo, multilingual == solo, lang_en == solo, lang_de == solo
lang-pr-2 concurrent run 1: multilingual == solo, lang_ja == solo, lang_de == solo, lang_en == solo
lang-pr-2 concurrent run 2: multilingual == solo, lang_ja == solo, lang_en == solo, lang_de == solo
lang-base-2 concurrent run 1: multilingual == solo, lang_ja == solo, lang_de == solo, lang_en == solo
lang-base-2 concurrent run 2: multilingual == solo, lang_ja == solo, lang_en == solo, lang_de == solo
lang-pr-3 concurrent run 1: multilingual == solo, lang_ja == solo, lang_de == solo, lang_en == solo
lang-pr-3 concurrent run 2: lang_ja == solo, multilingual == solo, lang_de == solo, lang_en == solo
lang-base-3 concurrent run 1: multilingual == solo, lang_ja == solo, lang_de == solo, lang_en == solo
lang-base-3 concurrent run 2: multilingual == solo, lang_ja == solo, lang_en == solo, lang_de == solo

| prompt | base median tok/s (n) | PR median tok/s (n) | change | base tokens a round | PR tokens a round | hashes equal across builds |
|---|---|---|---|---|---|---|
| lang_en | 32.46 (9) | 34.10 (9) | +5.1% | 2.46 | 2.46 | True |
| lang_es | 34.06 (9) | 34.23 (9) | +0.5% | 2.56 | 2.56 | True |
| lang_fr | 31.98 (9) | 32.05 (9) | +0.2% | 2.37 | 2.37 | True |
| lang_de | 32.14 (9) | 32.20 (9) | +0.2% | 2.44 | 2.44 | True |
| lang_it | 35.53 (9) | 35.54 (9) | +0.0% | 2.64 | 2.64 | True |
| lang_ja | 42.31 (9) | 42.43 (9) | +0.3% | 3.41 | 3.41 | True |
| multilingual | 41.10 (9) | 41.25 (9) | +0.4% | 3.46 | 3.46 | True |
