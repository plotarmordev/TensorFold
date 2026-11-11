# Imagination Studio on the M5 Ultra: install and MiniMax-H3-Image report

Taken on 11 November 2026 on a Mac17,15 (Apple M5 Ultra, 256 GiB, macOS 27.0) in an ordinary user account, from
[drowzeys/Imagination-Studio-by-keys-Wasserman](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman)
(Apache-2.0) 4.4.1 at `ef88a0747686212749768b72d6a796932e8e583d`. No weights were fetched and no model was loaded.

## Install

`oneshot-setup.sh` was read and not run: even `--image-only` downloads Qwen-Image-2.1-Turbo (33 GB, line 363) and
renders a test image, and without `--no-memguard` it installs a launchd agent (`scripts/memguard.py install`, line 589).
Only its environment step (lines 322-333) was repeated by hand, with its own pins, followed by its small Python extras
(lines 337-339) and the app bundle:

```bash
cd ~/projects/tensorfold/studio
git clone https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman.git && cd Imagination-Studio-by-keys-Wasserman
P=~/.local/opt/tensorfold-studio
uv venv -p 3.11 $P/venv
uv pip install -p $P/venv/bin/python -r requirements.lock \
  "tensorfold @ git+https://github.com/drowzeys/TensorFold.git@a2b592fef5366bf16d4729651b6549094981d3fe" \
  "mflux @ git+https://github.com/mflux-community/mflux.git@add5164e62c07cbcc9aec7a95c2e19c75605880c"
cp h3_generate.py qwen_image_generate.py zig_fasth3.py fasth3_q8.py VERSION $P/
uv pip install -p $P/venv/bin/python faster-whisper cryptography pywebpush
bash scripts/make-app.sh      # ~/Applications/Wasserman Slate AI Studio.app; no Desktop link
```

`uv` supplied Python 3.11.18 for this account; Homebrew was not changed. Left out: every model, the memory guard, motion
graphics, Finish, the native FastH3 engine, ComfyUI, sudo and system services.

## (a) It opens

`bash scripts/app.sh` (what the app bundle runs) serves **http://127.0.0.1:7870**, bound to loopback only:
`GET /` returned 200 (240 KB) and `/api/state` reported version 4.4.1 with one picture model listed (`qwen`, without
weights). The page rendered in Chromium with no script errors: Home, the engine list and "Ready. Nothing rolling." The
app bundle's own launcher could not start the server from a non-GUI shell (`nohup: can't detach from console`), so
only the direct `scripts/app.sh` start was verified; a double-click in this account's Aqua session, which runs the
same script, was not tried. Automatic source updates were turned off
through `POST /api/updates/settings {"automatic": false}` so the clone stays at the reported commit. The server was
stopped after the check.

## (b) TensorFold it builds from

Not ashhart's main: both pins are commits on drowzeys' fork, diverged from `ashhart/TensorFold` main.

| Part | Pin | Fork branch whose head it is | Against ashhart main |
| --- | --- | --- | --- |
| Python engine (`tensorfold` 0.6.5 + `h3`, `qwen_image` families), venv | `a2b592fef5366bf16d4729651b6549094981d3fe` (`oneshot-setup.sh:129-130`) | `ref-video-audio` | 17 ahead, 7,330 behind, base `609ca419a` (3 Oct) |
| Native FastH3 engine `tf-h3-dit` (Zig + Metal, 1.0.4) | `b19753ab01beedb04b3b691c9d50023ccfe09f62` (`scripts/build-zig-engine.sh:13`) | `h3-native-lora`; the docs name `h3-1.0.4`, which has moved on to `316b0b6a9` | 12 ahead, 6,787 behind, base `d4fc1964d` (10 Oct) |

The Python engine installs into `$PREFIX/venv`; the native engine is a release tarball or a build into
`$PREFIX/zig-engine` and was not installed here. Studio keeps no TensorFold checkout of its own.

## (c) How a picture model is wired on a Mac

Natively, not through ComfyUI, with one exception. The page posts a `scout`/`studio`/`edit` job with `image_model`;
`app/server.py` checks it against `IMAGE_MODELS` (1101-1106), `image_models()` (1112-1135) and `image_model_env()`
(1178-1188) and queues `scripts/scout.sh` → `scripts/image.sh`, which dispatches by model:

- Qwen-Image-2.1: `$PREFIX/venv/bin/python qwen_image_generate.py`, TensorFold's MLX `qwen_image` family with mflux's
  prompt encoder.
- FLUX3-K: `$PREFIX/flux3-venv/bin/python flux3_generate.py` (flux3-mlx, `scripts/setup-flux3.sh`).
- MOSS-Image: the exception, a ComfyUI graph from `spark/spark_generate.py` run in the Mac's own ComfyUI on PyTorch
  Metal (`mac/comfy.sh`, `127.0.0.1:8191`), memory-guarded by `MAC_COMFY_PEAK_GB` (`app/server.py:1639`).

A new picture model touches, at minimum (FLUX3-K and MOSS-Image are the templates):

| Area | Files |
| --- | --- |
| Registry and queue | `app/server.py` (`IMAGE_MODELS`, `image_models()`, `image_model_env()`, `MAC_COMFY_PEAK_GB`), `app/model_metadata.py` (model to backend, engine and layout), `app/picture_edit.py`, `app/nodes.py`, `app/revise.py`, `app/director.py`, `app/agents.py`, `mcp/studio-mcp.py` |
| Catalog and install | `app/model_catalog.json`, `app/model_catalog.py`, `app/model_measurements.json`, `app/discover.py`, `app/installs.py`, `oneshot-setup.sh`, `oneshot-setup-cuda.sh`, `scripts/install-checklist.sh`, `scripts/setup-comfy.sh` or a new `scripts/setup-<model>.sh` |
| Render | `scripts/image.sh`, `scripts/edit.sh`, a renderer (`<model>_generate.py`) or `spark/spark_generate.py` / `spark/spark_edit.py` workflows, `spark/progress_file.py`, `mac/comfy_files.py` |
| Page | `app/static/index.html`, `nodes.js`, `edit.js`, `revise.js`, `engines.js`, `home.js`, `settings.js` |
| Windows qualification | `app/windows_image_setup.py`, `app/windows_runtime.py`, `app/windows_image_checks.py` |
| Tests | `tests/test_image_models.py`, `test_model_catalog.py`, `test_discover.py`, `test_install_checklist.py`, `test_picture_edit.py`, `test_studio_mcp_tools.py`, `test_credits.py` |
| Credits and docs | `CREDITS.md` with `app/static/settings.js` (checked by `tests/test_credits.py`), `NOTICE`, `THIRD_PARTY_NOTICES.md`, `docs/changelog.md`, `docs/installation.md`, `docs/cli.md`, `docs/licensing.md` |

## (d) Issue 243, MiniMax-H3-Image, on a Mac

[#243](https://github.com/drowzeys/Imagination-Studio-by-keys-Wasserman/issues/243) (open, no comments, no linked
PR) asks for MiniMax-H3-Image in "Images only" and "Edit a picture", through ComfyUI on DGX Spark workers: the four
Comfy-Org int8 repack files (about 57 GB), ComfyUI with
[Comfy-Org/ComfyUI#17655](https://github.com/Comfy-Org/ComfyUI/pull/17655) (merged as `ede4e5fe8`, first released
in [v0.45.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.45.0) on 10 November), a 4 MP default with a 16 MP
option. The Studio has no H3-Image code yet.

On a Mac there are two routes:

1. **ComfyUI, as MOSS-Image does.** Smallest Studio change: move the ComfyUI pin (`COMFY_COMMIT=b9b290e4`, v0.42.0,
   in `oneshot-setup-cuda.sh`, which `scripts/setup-comfy.sh` reads for the Mac) to a commit with the H3-Image nodes,
   add the repack files to the catalog and `mac/comfy_files.py`,
   a workflow in `spark/spark_generate.py`, a `MAC_COMFY_PEAK_GB` entry and the registry/page/test files above.
   Unverified: whether the `int8_convrot` and `w4a8` files run on PyTorch Metal at all.
2. **Native TensorFold.** The engine has no H3-Image family. The `h3` family is the video DiT (packed
   `text|condition|audio|video` rows, audio projections, FastH3 tile routing); H3-Image is the same 50-layer,
   5376-wide DiT with `audio_enabled: false`, 24 latent channels, a 2x2 patch and a 6-step distilled schedule
   (`pdd_config.json`), plus a new VAE (CNN encoder, 2D ViT decoder). Its text encoder is H3's Qwen3-VL 32B, which a
   video install already holds.

The BF16 weights at `$MODEL_ROOT/MiniMax-H3-Image` (pipeline `MiniMaxH3ImagePipeline`) total about 135 GB on disk by
the download manifest: DiT 66.3 GB, text encoder 66.7 GB, VAE 2.3 GB. They were 17 GB in, mid-download, at the time of
this report; nothing here read or loaded them.

Memory was not measured. A tentative estimate from file sizes alone: BF16 would hold about 135 GB of weights before
activations and workspace, which points to a 192 GB+ Mac; the int8 repack (about 57 GB of files) might fit the
Studio's 96 GB Mac floor as the sole loaded model. Mac fit is unverified for both.

## (e) Contribution rules

- No `CONTRIBUTING` and no pull request template; the rules are in `AGENTS.md`: no personal information
  (`tests/test_no_personal_info.py`), GitHub noreply commit addresses, credits added in the same change to
  `CREDITS.md` and `app/static/settings.js` (`tests/test_credits.py`), engine pins never loosened, every render through
  the Studio queue.
- CI (`.github/workflows/tests.yml`, on `pull_request`): rejects non-noreply author or committer emails, runs
  `pytest -q tests` on Ubuntu with Python 3.11 and Node 20, and the model-free Windows tests on `windows-11-arm`.
- Outside PRs are taken, with drowzeys as the gatekeeper: MiaAI-Lab (#205, #208, #218-#220), newjordan (#209) and
  wassermanproductions (#236) were merged directly; ivberg's #210 and nickmontag's #221 were "carried", re-opened as
  drowzeys' own #211 and #222. Ten PRs are open, none for #243.

## (f) Disk

| Path | Size |
| --- | --- |
| `~/.local/opt/tensorfold-studio` (venv: MLX, PyTorch, TensorFold 0.6.5, mflux 0.21.0) | 1.4 GB |
| `~/projects/tensorfold/studio/Imagination-Studio-by-keys-Wasserman` | 163 MB |
| `~/.local/share/uv/python/cpython-3.11.18-macos-aarch64-none` | 78 MB |
| `~/Applications/Wasserman Slate AI Studio.app` | 280 KB |
| `~/ImaginationStudio` (empty projects, access key) | 8 KB |

About 1.6 GB in all, plus wheels in the shared uv cache (`~/.cache/uv`, 6.7 GB in total, shared with other projects).
