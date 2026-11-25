# PR 3733 asks 3 to 5

## 3. A planted control for every new kernel

Each of the 19 decoder and prompt kernels has a "first block only" control (`zig/tests/cuda/contract_h3_register.zig`).
TOP job `top-a4a49aacbcd1` ran `tf-cuda-test kernel-contract -tf_md_conv3x3` at `93f592f6b` on the GB10, and every control
went red at `shapes`, the gate it is planted for. Each control's gate line and verdict are in
`controls-top-a4a49aacbcd1.txt`, 38 lines for 19 controls. For example:

```
FAIL shapes cuda h3 image decoder h3d_attention control: first block only: at 1x261x64 output 64 is -nan where float64 gives 9.543158e-1: nan of its bound
CONTROL cuda h3 image decoder h3d_attention control: first block only: red at shapes, as planted
FAIL shapes cuda h3 image prompt h3p_heads rotary control: first block only: at 8x128x128 output 1024 is 3.984375e-1 where float64 gives 4.902344e-1: 11.95 of its bound
CONTROL cuda h3 image prompt h3p_heads rotary control: first block only: red at shapes, as planted
```

In the same log (`contract-h3-top-a4a49aacbcd1.txt`), every H3 gate line without a control is `PASS` (86) or `N/A` (10).

## 4. An H3 clip checkpoint on main against this head

This comparison can't be run on the Spark. Ash's main at `a6cde902d` has two tools that print H3 velocities
from a checkpoint. `tf-h3-dit` handles FastH3 clip cases and `tf-h3-image` handles picture cases (`zig/build/h3_host.zig:104-105`).
Both run only on Metal. CUDA's H3 on main is `libtf_h3.so` (`zig build tf-h3`). That library runs the transformer blocks
on rows the caller prepares. It has no case reader, no input and output projections, and no velocity output. The
clip case exporter, `tools/zig/h3_case.py`, lives on the `drowzeys/TensorFold` fork (`docs/recipes/minimax-h3-metal.md:47-49`).
It is in neither tree. On this Mac, `find /Users/agents /Volumes -maxdepth 6` matched 122 `*.h3case.safetensors` files under
`h3-image-runs/`. All of them are MiniMax-H3-Image picture cases (`text`, `video`, `video_velocity`, no `audio`). The only
models are `MiniMax-H3-Image` and `ComfyUI-MiniMax-H3-Image`; there is no FastH3 clip checkpoint. On TOP, job
`top-a4a49aacbcd1`'s `find / -xdev` for `*fasth3*`, `*h3*case*`, `case.safetensors` and `*minimax*h3*` found 0 paths.

The transformer diff against Ash's main is empty for `h3.zig`, `geometry.zig`, `block.zig` and the CUDA block sources.
The shared Metal loader does change: `tensors.zig` now uses the span-based `read_ahead.zig`, and `checkpoint.zig` is the
CUDA-side checkpoint reader. Static inspection is therefore not a substitute for the unavailable velocity comparison.

## 5. CUDA selftests and the GB10 kernel contract

TOP job `top-7c6183220113` ran `plant-203.sh` (side branch `gpu/203-plant-h3`, `30c9f2302`) on the GB10 and exited 0.
Both selftests pass: `decoder-selftest.txt` and `prompt-selftest.txt` end in `selftest: pass` and `PASS every gate`.
`tf-cuda-test kernel-contract h3` (`contract-h3-top-7c6183220113.txt`) runs the H3 kernels' gates, controls and live limits:

```
(filtered) SUMMARY 34 kernels: 34 pass gates 1-4, 0 fail, 0 known findings; 19 controls: 19 red as planted, 0 missed; 0 limit problems; gates 8447 ms, limits 1 ms
```

The five-SM audit, `kernel-contract-limits` over `-Dsm=80,86,89,120,121` fatbins, reports `105 problems`, none of them H3.
These problems already exist on Ash's main: against `bdcb6e608` this head adds only `h3_image_decoder.cu`, `h3_image_prompt.cu`,
`h3_image_cuda_records.zig` and their line in `registry_joins.zig` under `zig/kernels/cuda` and `zig/src/core/contract`.
Job `top-a4a49aacbcd1` (`plant-203.sh` at `93f592f6b`) ran the full contract without `tf_md_conv3x3`'s reference,
the Lead's ask 5. Its log is `contract.log` in the job's out folder, 2176 lines. Both selftests pass again, and every H3
line is as above. The run still hit the script's 45-minute limit (exit 124) in `window_rows`'s paged cases, before the
GLM, MLA, q35 contiguous, MTP rows, weight-only and family-set suites and the run's SUMMARY. Its only red lines outside
the controls are four `qwen35` nonfinite gates (`head 18 element 0: finite where float64 gives nan`). `qwen35`'s kernels
and `contract_q35_attn.zig` are the same as on Ash's main. The same job's five-SM audit again reports `105 problems`:
stale records on 44 launches, none H3, all in `limits-stale-top-a4a49aacbcd1.txt`.

Job `top-0376995bc84e` (`plant-203.sh` at `14ba0ed56`, `kernel-contract @window_rows -tf_md_conv3x3`) was meant to run
those remaining suites. It started on 23 Nov at 21:30:26 and ended with exit 137 about 56 minutes later. Its log holds
only the build's compiler warnings: the script prints `contract.log` after the contract ends, and the job was killed
before that, so it gives no gate line, no SUMMARY and no per-suite time. The full contract therefore remains partial.
Never run on this head: the rest of `window_rows`'s paged cases, GLM, MLA, q35 contiguous, MTP rows, weight-only and
family-set, and with them the run's SUMMARY. None of these is an H3 suite.

The rest is now split into nine TOP jobs, each running its measured part under `timeout 600`. Each is a wrapper
`docs/recipes/minimax-h3-cuda/plant-203-PART.sh` on side branch `gpu/203-plant-h3` (`6d6faf83f`). Parts `g0` to `g5` run
`kernel-contract @window_rows %I/6 -tf_md_conv3x3`, the gates from `window_rows` on with every sixth kernel from the
Ith. `back`, `after` and `limits` run `kernel-contract-suites PART`, the full run's tail one part at a time; the tail
alone took about 750 s in earlier receipts. Job `g0` ran on the GB10 and its gates finished within the cap (exit 1 in 595 s):

```
(filtered) SUMMARY 87 kernels: 87 pass gates 1-4, 0 fail, 0 known findings; 61 controls: 61 red as planted, 0 missed; 41 limit problems; gates 583904 ms, limits 10597 ms
```

Its exit 1 comes from the 41 live-limit problems: a shard checks every registered launch's live limits. The log was
read through its last 80 lines, so it does not name the 41 records; `top-957270e2461a`'s `limits` run below names
them and none is H3.

Job `limits` ran twice. `top-08c281336b68` (at `6d6faf83f`) measured in 12 s and ended `FAIL kernel-contract-suites
limits: 239 limit problems`; the door's last 200 lines were all `PASS`, so that log named none of them. The rerun,
`top-957270e2461a` (at `89d5f14c6`, which prints every non-PASS line), names all 239 in
`limits-top-957270e2461a.txt`: 198 audit problems across sm_80/86/89/120/121 (153 `no compiled usage recorded` rows on
drama3, qwen35, qwen35_kv8, qwen4_gdn, qwen4_hc, qwen4_ple and qwen4_projection launches, 7 `optional shared-memory
limit` N/As on dsv41_x3gm, plus the stale-record rows the audit counted before) and 41 `FAIL limits live` rows —
the same live-limit drift on qmm_prefill, flashnext_hcn, nemo_audio, muse_*, dfl_*, fn_row_topk, exl3_*,
tf_ctl_glm_old_* and drama3, plus the two planted live controls going `RED`. No line is H3: the audit's rows name
only those non-H3 families and the live FAILs match `g0`'s unverified 41 count for count, confirming them as the
same stale non-H3 records. All of it is pre-existing on Ash's main: this head adds only the h3_image_* sources and
their registry line under `zig/kernels/cuda` and `zig/src/core/contract`.

The rest of the gates ran shard-wise; shards 3 and 5 each needed quarters because the 600 s measure cap stopped
them mid-shard (the capped runs' logs are kept too). Every shard exits 1 only from the same 41 non-H3 live-limit
problems and, in `g2`/`g5c`, one pre-existing non-H3 gate failure each; every planted control in every shard is
red at its planted gate. Logs are `<part>-top-<job>.txt`.

| part | job | gates pass/fail | controls red | notes |
| --- | --- | --- | --- | --- |
| g0 `%0/6` | (earlier) | 87/0 | 61/61 | receipted before |
| g1 `%1/6` | `top-14a6f81fb640` | 83/0 | 63/63 | at `89d5f14c6` |
| g2 `%2/6` | `top-7d5bd5a64223` | 83/1 | 63/63 | 1 fail `idol tf_idol_dec_attention`: idol files identical to base, pre-existing |
| g3 capped `%3/6` | `top-7341f0af31e5`, `top-921df348ef19` | — | — | 600 s cap mid-shard, twice → quartered |
| g3a `%3/24` | `top-963079c78404` | 18/0 | — | |
| g3b `%9/24` | `top-b5a1cc82fd81` | 22/0 | — | |
| g3c `%15/24` | `top-4d9312822d5c` | 23/0 | — | |
| g3d `%21/24` | `top-2f4029305a44` | 20/0 | 16/16 | |
| g4 `%4/6` | `top-6ff1a5489f7d` | 87/0 | 60/60 | at `b4d9984112` |
| g5 capped `%5/6` | `top-8ed47625be2f` | — | — | 600 s cap mid-shard → quartered |
| g5a `%5/24` | `top-cd940a7bd324` | 19/0 | 18/18 | |
| g5b `%11/24` | `top-f51e5ee7dd48` | 20/0 | 17/17 | |
| g5c `%17/24` | `top-192e8adfac07` | 20/1 | 16/16 | 1 fail `moss image encoder tf_me_conv` sums: kernel and records identical to base, pre-existing |
| g5d `%23/24` | `top-c3903f9458c7` | 21/0 | 15/15 | |
| back | `top-a8e24fd8c558` | pass | — | exit 0, `SUMMARY kernel-contract-suites back: pass in 35701 ms` |
| after | `top-6a411e0690ce`, `top-9cc1c34890e4` | — | — | exceeds the 600 s cap both runs (no SUMMARY); the part is indivisible — `suites()` runs its 7 sub-suites serially with no shard/filter argument |

`after` holds the served-to-SwiGLU pair suites (`contract_qwen_served/pages/shared_prefix/pending/learning`,
`contract_fn_lanes`, `contract_prompt_swiglu`); none of those files mentions H3 or MiniMax, so its non-completion
leaves no H3 record unverified. The two `g2`/`g5c` gate failures are both on non-H3 kernels whose sources and
contract records are byte-identical to the base `2f5e1ca37` (`git diff` on every `tf_me_conv`/`tf_idol` file
returns empty), so they are pre-existing there, the same class as `g2`'s idol fail. No H3 record is stale: the
`h3` filter run above and every shard's `SUMMARY` report zero H3 gate failures, zero missed controls and zero
H3 live-limit problems.

On this Mac, `zig build test test-golden` at this head ends `991/998 steps succeeded (5 failed); 1909/1943 tests
passed, 4 failed`. The base `2f5e1ca37` on the same Mac ends `989/998 steps succeeded (7 failed); 17444/17596 tests
passed, 6 failed`. The head's four failing tests are all red on the base too: `vision.test.the release's
vision_config parses to the defaults` and its Metal variant (expected `UnsupportedVision`, found `ZeroHeads`),
`net.test` (`expected AddressUnresolved, found AddressInUse`, a port taken on this machine) and
`persistence_identity_test` (a canonical-prefix hash). The `run sh` step's `_memcpy` linkage check is also red on the
base. None of these is in a file this branch changes (`zig/tests/cuda` contract sources and this receipt).

## CI on #3733

At `2f5e1ca37`, the PR's head (checked 24 Nov), every job passes: `zig`, `zig-b`, `zig-windows`, `zig-macos-dev-tools`, `fmt`, `pytest`,
`release-paths (ubuntu-24.04)`, `comfyui-moss-installer` and `benchmark-clients` on Ubuntu and Windows.
`host-nightly-linux` and `host-nightly-macos` are skipped. At `64fdb3215`, `zig` was red at lean check on a two-line
comment in `contract_limits.zig` from this branch. Ash's main is not red there, and `e27a4ecf4` fixes it.

The PR's head moved to `c433b7902` since this card's code went in (someone pushed more work). Checked 24 Nov
~21:50: every job is pending again on the new head — `zig`, `zig-b`, `zig-windows`, `zig-macos-dev-tools`, `fmt`,
`pytest`, `release-paths (ubuntu-24.04)`, `comfyui-moss-installer`, `benchmark-clients` on Ubuntu and Windows;
`host-nightly-linux` and `host-nightly-macos` skip. No red job at this head.
