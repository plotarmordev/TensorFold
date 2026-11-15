# MiniMax H3 / FastH3 CUDA blocks, one device

This library contains @drowzeys' #958 BF16 transformer blocks and kernel contracts, with the maintainers' boundary,
shared-context, complete-block and allocation repairs. The later shared copy-source owner protects temporary uploads.
Weights and activations retain BF16 storage and products use the runtime's cuBLASLt Linear with FP32 sums.

The host supplies text encoding, sampling, projections, video/audio encoding and decoding. This is a block library,
not a complete native clip-generation endpoint. No weights are included. Code is Apache-2.0; MiniMax H3 / FastH3
weights have the MiniMax H3 Community License and its territorial limits.

Build the real library on a CUDA build machine:

```sh
zig build tf-h3 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast
```

This stage has no rank inputs, transport source, exchange kernel or libibverbs dependency. The C interface is
zig/src/families/h3/cuda_lib.h. Check tf_h3_abi_version() == 1 before passing its 96-byte tf_h3_inputs.
All tiles and rows belong to one device. The library retains the current primary CUDA context through Context.shared;
a host framework's device addresses must belong to that context. If no context is current, the runtime selects zero.

The same entry points support device addresses (tf_h3_forward), host addresses (tf_h3_forward_host), mapped host
allocation, profiling and BF16 projection replacement. The host owns pointer extents and serialized calls on a model.
The fixed 16,384-row cuBLASLt algorithm-selection shape is retained to preserve the original arithmetic policy.

Host tests and compile-only ABI libraries do not load a GPU image:

```sh
zig build test-h3-boundaries test-cuda-host test-cuda-copy-source compile-h3-library
```

Compile-only artifacts are installed under zig-out/host-checks and contain an empty image. They are verification
artifacts, not runnable inference libraries. The Linux library is cross-compiled and is not executed on this Mac.
The numerical contracts, actual kernels, model fidelity, real context interoperability and performance need CUDA hardware.

The original #958 thread reports one/two/three-GB10 results. Those measurements predate this split and do not qualify
this exact source. Sequence-parallel execution is a separate follow-up with an extended ABI and its own transport gates.

The Linux CUDA runner and all H3 contract adapters also have a compile-only gate:

```sh
zig build compile-cuda-tests -Dtarget=x86_64-linux-gnu
```

This step has no NVCC or Metal compiler dependency and does not execute the resulting binary.

## MiniMax-H3-Image on these blocks

MiniMax-H3-Image's `transformer/` uses the clip model's base block names, and the Metal tool reads no
`attn.to_gate_compress` from it (`h3.zig` `baseBlock`): its attention is dense, so no routing gate is needed.
`tf_h3_open` loads such a checkpoint with q, k and v alone in each block's first product (`tf_h3_shape` is
unchanged). The release fuses each head's q, k and v rows (`block.zig`); the load stacks them as `h3_heads` reads
them, all heads' q, then k, then v. A forward with `keep` set is refused with `BadH3Ungated` and
`tf_h3_set_weight` of the gate with `NoGate`, so neither reads past the smaller buffer.

The block arithmetic is the clip model's. Three tools build a picture on a Spark with no Metal and no PyTorch, each
writing the file the matching Metal tool writes (docs/recipes/minimax-h3-image-metal.md), so any stage swaps:

```sh
zig build tf-h3-image-prompt-host tf-h3-image-cuda tf-h3-image-decoder-host -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt-host TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42
zig-out/bin/tf-h3-image-cuda MODEL_DIR out/case.safetensors out/case
zig-out/bin/tf-h3-image-decoder-host MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
```

| Picture stage | Metal | Spark |
|---|---|---|
| Prompt rows (Qwen3-VL to layer 50, `condition_proj`, the two-block refiner) | `prompt.zig` | CPU: `prompt_reference.zig`, a layer's weights in f32 at a time, the case from `prompt_case.zig` |
| Weights | `model.safetensors.index.json` and its shards | `Model.loadSharded` reads the shards in place |
| Rows in: `video_patch_proj`, F32 [5,376, 96] | `h3_rows_in` | host, `rows_reference.rowsIn` |
| AdaLN tables: the time embedder, each block's projection per timestep | `h3_modulation` | CUDA `cuda_modulation.zig`, resident (line, part, c) tables; final-norm table downloaded |
| Rotary: 48 angles over 96 dims | `picture.rotary`, 96 wide, duplicated | `cos`/`sin` 48 wide, `rot` 48: the first half of each row |
| Transformer blocks, dense, tiles of 64 | `block.zig` | these blocks, `keep` 0 |
| Final norm and the step's head, `pdd_fused_heads` | `h3_final_norm`, `h3_rows_out` | host, `rows_reference.finalNorm`, `rowsOut` |
| Euler step, unpatchify | `picture.euler`, `picture.unpatchify` | the same |
| Image decoder | `image_decoder_metal.zig` | CUDA `image_decoder_cuda.zig`, batched tiles; host blending and PNG output |

Gaps between the video and image transformer: the image transformer has no routing gate (dense attention), fuses each
head's q, k and v, has six per-step output heads instead of one, no audio rows, and its rotary covers 96 of 128 dims.
CUDA and Metal are not bit-identical (SwiGLU rounding, RMS sum order): the check is the picture's PSNR.

`tf-h3-image-prompt-cuda` runs the text encoder and refiner on CUDA, streaming one layer's weights at a time.
`tf-h3-image-decoder-cuda` runs the VAE on CUDA. The CPU tools remain independent reference implementations.
[`run-script.sh`](minimax-h3-cuda/run-script.sh) selects these GPU tools and runs their synthetic selftests before
both picture sizes. The measurements below predate this cutover; they are not measurements of these new paths.

### First CUDA pictures (BOTTOM, GB10, 14 Nov 2026)

Prompt `h3-image-pilot/prompt.txt` (the red fox, sha256 cc969bca…), seed 42, the checkpoint's six distilled steps,
BF16 throughout, against the M5 Ultra's Metal pictures of card 34 (`h3-image-runs/34/timing`, run 2, same tools):

| Stage | Spark 2048² | Mac 2048² | Spark 4096² | Mac 4096² |
|---|---:|---:|---:|---:|
| Prompt (CPU on the Spark) | 88 s | 17.5 s | 98 s | 14.8 s |
| Transformer load, AdaLN tables on the host | 137 s | 23.3 s | 136 s | 12.5 s |
| Transformer, 6 steps | 34 s (3.84 s a step settled) | 11.1 s (1.85 s) | 123 s (18.5 s) | 59.4 s (9.8 s) |
| Decoder (CPU on the Spark) | 218 s | 2.8 s | 771 s | 34.2 s |
| PSNR against the Mac picture | 29.2 dB | — | 28.5 dB | — |

The Mac's two runs are bit-identical; the CUDA picture differs by the blocks' bf16 rounding (the latents' cosine
0.996). The Metal decoder on the CUDA latent gives the same PSNR at both sizes, so the CPU decoder matches Metal's and
the gap is the transformer's. The first transformer step pays cuBLASLt's algorithm search (4–7 s).

mmh3 ([`9c0652f`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/docs/image.md#speed))
has no BF16 linear layers on CUDA (`--linear-precision int8|nvfp4`, on its own INT8 ConvRot DiT and text encoder).
The matched run therefore keeps everything else: [`mmh3-job.sh`](minimax-h3-cuda/mmh3-job.sh) runs `mmh3 image` on this
prompt, seed 42, 2048² and 4096² and mmh3's defaults, which are the six distilled steps, BF16 attention and no draft steps
(its logs show six). INT8 linear layers, in the text encoder as in the DiT, and its fp16 VAE are the differences.
Its seed draws other noise, so the comparison is of times, not pixels. The job has no network:
[`mmh3-stage.sh`](minimax-h3-cuda/mmh3-stage.sh) first stages mmh3's source, its vendored crates and Rust in the pilot
folder on BOTTOM's CPUs, and the job builds mmh3 from them for `sm_120f`.

The Desk ran it on BOTTOM (GB10) in the 14 Nov 08:05 window, job `bottom-5733d361f24d`; both pictures are clean foxes:

| Stage | TensorFold 2048² | mmh3 2048² | TensorFold 4096² | mmh3 4096² |
|---|---:|---:|---:|---:|
| Prompt | 88 s (CPU) | 3.6 s | 98 s (CPU) | 3.5 s |
| Transformer load | 137 s | 2.9 s | 136 s | 2.8 s |
| Transformer, 6 steps | 34 s (3.84 s a step settled) | 9.5 s (1.4 s) | 123 s (18.5 s) | 53.1 s (8.5 s) |
| Decoder | 218 s (CPU) | 1.0 s | 771 s (CPU) | 2.7 s |
| Wall | 477 s | 17.6 s | 1,128 s | 63.1 s |

On the GPU stage both share, INT8 linear makes mmh3's step 2.7× faster at 2048² and 2.2× at 4096². The rest of the
gap is TensorFold's CPU prompt, host load and CPU decoder, which mmh3 runs on the GPU. mmh3's quoted ~24 s at 4096²
is three PDMD-LoRA steps as half-size drafts plus one full step with INT8/FP8 attention, a different composition.

### GPU cutover qualification (card 67)

Desk ran commit `96b17d83aef126e088c2b1539668215e3da7e336` on BOTTOM's GB10 (driver 580.173.02)
on 15 November 2026, 10:36–10:45 HKT, with DeepSeek paused. DeepSeek resumed after both jobs ended.
The three CUDA tools were prebuilt successfully, but the candidate **failed qualification**:
`run-script.sh` (job `bottom-f69795061a4e`) exited 1 at the decoder selftest. Its fail-fast gate stopped
before the prompt selftest and both checkpoint renders; the candidate output directory is empty.
This is a completed, failed device run, not a pending window or a passing picture comparison.

| Candidate synthetic check | Result |
|---|---|
| GPU timestep/AdaLN projections | Passed: BF16/F32 weights, ragged K/N, two times, three modalities, five repeats against ordered host sums |
| Decoder, 6×10 cells, block 0 | Passed: 92.997% identical, worst 2 ULP, relative error 8.83e-4 |
| Decoder, 6×10 cells, block 1 | **Failed:** 87.347% identical, worst 3 ULP, relative error 1.15e-3 |
| Decoder, 20×28 cells, block 0 | **Failed:** 89.974% identical, worst 6 ULP, relative error 1.03e-3 |
| Decoder, 20×28 cells, block 1 | Passed: 90.303% identical, worst 3 ULP, relative error 1.01e-3 |
| Decoder batch partitioning | Bit-identical to staged assembly for passes of 1, 3 and 4 tiles |
| Synthetic picture bytes after two blocks | 88.830% / 90.626% identical; maximum difference 1 at both geometries |
| Single-tile 24-block chain | Relative error 1.47e-2; passed its 3e-2 bound |

Both failures are below the existing 90% identical-values block gate; their ULP and relative-error gates
pass. The thresholds remain unchanged. The picture-byte checks use synthetic weights and two blocks,
not the released checkpoint, and do not establish the requested 2048²/4096² PNG tolerance.
No failing operator is isolated by this receipt: each block includes norms, projections, rotary,
attention, SwiGLU and residuals. Changing a reduction or relaxing the bound without isolating it would
not establish a correction.

The beyond-INT8 experiment did execute: BF16 batched decoding of 25 synthetic 16×16-cell tiles through
24 blocks took 0.54 s wall and 0.54 s by GPU events (21.6 ms/tile). This is one synthetic timing,
not a checkpoint decoder measurement, a batch-size sweep, or an end-to-end speedup. GPU AdaLN's
synthetic equality check passed, but checkpoint load time was not reached.

The paired `mmh3-job.sh` (job `bottom-b16fcdc3628f`) exited 0 and produced both PNGs, using the same
prompt file, seed 42, sizes and six-step settings described above. Its 4096² run reused the cuBLASLt
and INT8 tiling caches saved by the 2048² run. One run per size; step totals below sum rounded log values.

| Stage | TensorFold GPU 2048² | mmh3 2048² | TensorFold GPU 4096² | mmh3 4096² |
|---|---:|---:|---:|---:|
| Prompt | Not reached | 3.7 s | Not reached | 3.9 s |
| Transformer load | Not reached | 3.0 s | Not reached | 3.0 s |
| Transformer, 6 steps | Not reached | 9.5 s | Not reached | 53.5 s |
| Decoder | Not reached | 1.0 s | Not reached | 2.8 s |
| Wall | No picture | 18.0 s | No picture | 64.1 s |

Receipt SHA-256 identifiers (the fetched card-67 logs, not newly generated runs):

- `tensorfold.log`: `9fb50ff285ae3889c5896a483eed76147ae3c185dcf0ae46204f12a0fc23f6b3`
- `mmh3/out/mmh3/2048.log`: `8d392be2042b8dce54646272d508e750719f87da0f91e0fa01f23fb77b45eabc`
- `mmh3/out/mmh3/4096.log`: `d76f2232f53af2fcf80203432961f19da1a102ca21186b2e3ed3c004b9959442`

Desk ran continuation `3005ab6b3a2a6398915bc2172f134f5f9d0a966c` on the same GB10 and driver
on 15 November, 11:41–11:49 HKT, then restored DeepSeek. `run-script.sh` (job
`bottom-c90e34da9b1a`) exited 1. The fetched log still has **both** block identity failures above,
including the small-case block 1 at 87.347%; it does not show only the full-size failure.
The independent FFN projection checks passed at 99.948% identical, at most 1 ULP, on every
checked block's GPU norm. SwiGLU was bit-identical on every GPU projection. These checks rule out
a large isolated FFN projection/activation error, not an error elsewhere in the block.
The window script now attempts all three synthetic selftests and refuses checkpoint renders if any
fails, so a decoder failure no longer hides the prompt selftest's result.

The prompt selftest was reached and failed with `H3PromptRows` before reporting numerical results.
Its synthetic fixture created the embedding tensor but omitted it from the lookup map; a
ReleaseFast unchecked optional unwrap is undefined behavior, consistent with that validation error;
the bad slice itself was not captured. The continuation adds the tensor to the map and checks missing
fixture tensors explicitly. This is a fixture repair,
not evidence that the GPU prompt encoder passes. The earlier condition-projection bias repair also
remains unqualified on hardware.

At `3005ab6b`, `nice -n 10 zig build test test-golden --summary all` passed locally: 892/892 build steps,
452/467 executed tests passed and 15 skipped; cached test steps are additional. Server goldens had
322/335 equal with 13 known differences, plus 55/55 authorization goldens equal, zero unexpected
differences. The decoder diagnostic tool compiled without NVCC (empty image, not a device test).
The shell gate was exercised with each selftest failing in turn: all three were attempted, exit 1
prevented renders; the all-pass case proceeded. `sh -n` and changed Zig files' format checks passed.

The next decoder selftest also compares retained Q/K/V against the reference from the GPU block
input, and recomputes attention independently in FP64 on the GPU's own Q/K/V. These readbacks leave
production launches unchanged and retain the existing block bounds; isolated attention uses the
existing 1-ULP/99%-identical product bound. Passing these diagnostics would not prove the remaining
residual path or checkpoint PNG tolerance.

The fixture/attention continuation passed `nice -n 10 zig build test test-golden --summary all`:
894/894 build steps, 454/469 executed tests passed, 15 skipped, with additional cached tests.
Server goldens remained 322/335 equal plus 13 known differences; authorization goldens were 55/55
equal, zero unexpected differences. The new host regression writes and reopens the tiny checkpoint
and checks the embedding exposed to the encoder against its stored bytes. Both CUDA tools compiled
locally without NVCC (empty images, not GPU binaries). Running the prompt CLI's `selftest` constructed
and cleaned up its tiny checkpoint, then stopped with `DriverUnavailable` on this Mac; numerical
execution is still untested. Changed Zig files passed `zig fmt --check`; the plain `zig build`
also passed, 283/283 steps.

The remaining gates are decoder failure isolation and correction, the prompt selftest, kernel-contract
registration and device/compiler receipts, then both checkpoint renders with same-latent CPU decoder
comparisons and matched stage timings. The next device run must use the repaired prompt fixture and
isolated decoder attention check, not repeat `3005ab6b` unchanged. No evidence establishes an
end-to-end win over mmh3's historical 17.6 s.

The paired mmh3 run in the 11:41 window completed both sizes. One run per size, with the 4096²
run reusing the 2048² algorithm and tiling caches; transformer totals sum rounded step logs:

| Stage | TensorFold GPU 2048² | mmh3 2048² | TensorFold GPU 4096² | mmh3 4096² |
|---|---:|---:|---:|---:|
| Prompt | Not reached | 3.7 s | Not reached | 3.9 s |
| Transformer load | Not reached | 3.0 s | Not reached | 3.1 s |
| Transformer, 6 steps | Not reached | 9.5 s | Not reached | 53.7 s |
| Decoder | Not reached | 1.0 s | Not reached | 2.8 s |
| Wall | No picture | 18.1 s | No picture | 64.5 s |

The BF16 batched decoder experiment again took 0.54 s for 25 synthetic tiles through 24 blocks
(21.7 ms/tile); this still does not measure checkpoint decoding. Latest receipt SHA-256 identifiers:

- `3005ab6b/tensorfold.log`: `c37d73c973c84bafed1ee8341028a1c64b235bd8995a2e1a2f4bbbd5646395f4`
- `3005ab6b/mmh3/out/mmh3/2048.log`: `9b70b113ba831ed5cbc459d2b8d1e2ebb3e4009ec462c18ba76640ef93c14025`
- `3005ab6b/mmh3/out/mmh3/4096.log`: `be22c25ffde5cf49322bc21d1aba9d582a0622da533b2b2f4bab8fa0c1576e72`

### Third GPU window: isolated attention and repaired prompt fixture

Desk completed the queued run of `879b27375f2663d09b43128e328b6be70f028d82` on GB10,
driver 580.173.02. The fetched `879b2737/tensorfold.log` records **failed qualification**,
not a resolved numerical gate. The checkpoint render directory is empty. All three synthetic
selftests ran; the modulation check passed, while decoder and prompt checks failed.

- Decoder block identity failures remain 87.347% (6×10, block 1) and 89.974% (20×28, block 0).
  Q/K norm and rotary from each GPU block input pass at 99.724–99.953% identical; V passes at
  99.733–99.958%. These are cumulative projection/head checks, not isolated norm tests.
- Independent FP64 attention on the GPU's own heads passes at 99.986–99.991% identical,
  at most 1 ULP. FFN projection and SwiGLU checks still pass. These results do not isolate
  the output projections, LayerScale residuals or the second RMS norm.
- The repaired prompt fixture reaches numerical execution. Text-only encoding passes
  (cosine 0.9999959, relative error 2.85e-3), and refinement on its GPU rows passes
  (0.9999902, 4.44e-3). Repeated text encoding and refinement are byte-identical.
- Vision/DeepStack encoding fails: cosine 0.9999916, relative error 4.09e-3 versus the
  float32-sum host run's 0.9999980 and 2.02e-3. The existing two-times-host bound remains
  unchanged. This fixture uses equal positions on all three axes; it does not discriminate
  M-RoPE axis selection, despite its log label.

The beyond-INT8 BF16 batched decoder experiment again measured 0.54 s for 25 synthetic tiles
through 24 blocks (21.7 ms/tile). It remains a synthetic result, not checkpoint decoding time.
The paired mmh3 logs contain both completed pictures. Same prompt, seed 42, sizes and six-step
settings as above; 4096² reuses the 2048² algorithm/tiling caches. One run per size:

| Stage | TensorFold GPU 2048² | mmh3 2048² | TensorFold GPU 4096² | mmh3 4096² |
|---|---:|---:|---:|---:|
| Prompt | Not reached | 4.0 s | Not reached | 3.9 s |
| Transformer load | Not reached | 3.2 s | Not reached | 3.1 s |
| Transformer, 6 steps | Not reached | 9.5 s | Not reached | 53.4 s |
| Decoder | Not reached | 1.1 s | Not reached | 2.8 s |
| Wall | No picture | 18.5 s | No picture | 64.1 s |

Transformer totals sum rounded step logs. The 2048² decoder log reports 1.1 s for its stage,
including 1.0 s decoding; its weight read overlaps sampling. No TensorFold matched timing or
same-latent PNG comparison was reached, so the 17.6 s historical target is not beaten.

Receipt SHA-256 identifiers:

- `879b2737/tensorfold.log`: `6bf2edf375e5a53a2ed97ed257ecdc4083c10e6d727c3f1674f5526418443983`
- `879b2737/mmh3/out/mmh3/2048.log`: `962350f7fc5f2bbc188376e09b1a7cb4277cb6baa1a235d8263460a55b9b5111`
- `879b2737/mmh3/out/mmh3/4096.log`: `89a502c5100beadf6b5ce007a6c0db2df1de7a555cc49732292eeda23d53acea`

The next decoder selftest captures the actual attention residual before the FFN, then checks
the attention output projection/LayerScale, norm2, and FFN down projection/LayerScale against
the host reference on each operation's GPU inputs. Capture is compiled out of production passes;
the existing staged-versus-production byte checks remain. These diagnostics keep the existing
whole-block gates and do not constitute an arithmetic fix or a successful device result.

The prompt selftest retains the full failing fixture and also compares an identity vision path
(embedding rows spliced back, zero DeepStack) byte-for-byte against a no-vision run. Separate
one-layer, splice-only and DeepStack-only runs use the same FP64/F32 whole-run gates and the
same original random draws. These are selftest-only probes; the production encoder is unchanged.
They distinguish failing cases, not individual kernel correctness or unequal-axis M-RoPE support.

Local continuation verification passed: `nice -n 10 zig build test test-golden
tf-h3-image-prompt-cuda tf-h3-image-decoder-cuda --summary all` completed 898/898 build steps,
460/475 executed tests passed and 15 skipped (additional tests cached). Server goldens were
322/335 equal with 13 known differences; authorization goldens 55/55 equal, zero unexpected
differences. The default `zig build --summary all` passed 283/283 steps. Changed Zig files
passed `zig fmt --check`. A temporary host smoke exercised the shared reference's LayerScale
rounding, RMS norm and zero-row norm successfully and was removed. Both compiled CUDA CLIs
were launched with `selftest` locally and stopped with `DriverUnavailable`; these builds embed
empty images without NVCC and do not exercise the new GPU diagnostics. The prompt fixture
cleaned up its generated checkpoint. A new Desk run is required for the new diagnostic evidence;
no new window has been scheduled by this continuation.
