# Checkouts stay small: Zig cache and staged files outside the tree

- Card checkouts reached 15-17 GB (78.4 GB for 13 on 23 November); measured, everything but ~0.2-0.6 GB per checkout was its `.zig-cache/o`: 1,300+ stale `test` binaries and 69 MB `test_zcu.o` objects, one set per revision built, which Zig never prunes. No checkout held model weights; card 178 held a 381 MB ad-hoc `.h3-178-job` folder.
- Zig 0.17 picks the local cache before `build.zig` runs. `tools/zig/setup_cache.sh` links each card checkout's `.zig-cache` to the shared external folder before the first build, so even plain `zig build` stores objects and test scratch outside it. The setup refuses existing cache directories rather than losing them.
- `tools/zig/{bf16_topk,draft_ops,tree_round}_build.sh` and `tools/release/check_package.py` (`test-dist-package`) follow `ZIG_LOCAL_CACHE_DIR` and fall back to `.zig-cache`. The three standalone scripts put explicit executables in per-checkout paths to avoid collisions across simultaneous builds.
- CONTRIBUTING.md documents setup, `--prefix`, and weights/GGUF/fixtures/job output outside the tree; the native recipes skill points to it.
