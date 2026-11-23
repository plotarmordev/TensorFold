# Checkouts stay small: Zig cache and staged files outside the tree

- Card checkouts reached 15-17 GB (78.4 GB for 13 on 23 November); measured, everything but ~0.2-0.6 GB per checkout was its `.zig-cache/o`: 1,300+ stale `test` binaries and 69 MB `test_zcu.o` objects, one set per revision built, which Zig never prunes. No checkout held model weights; card 178 held a 381 MB ad-hoc `.h3-178-job` folder.
- Zig 0.17 picks the local cache before `build.zig` runs, so the fix is `ZIG_LOCAL_CACHE_DIR` pointed at one shared folder outside every checkout. With `ZIG_LOCAL_CACHE_DIR=$HOME/.cache/tensorfold/zig`, `zig build test test-golden` from a clean checkout left it at 235 MB (3.8 GB of cache outside it); a warm rerun took 92 s and exited 0.
- `tools/zig/{bf16_topk,draft_ops,tree_round}_build.sh` and `tools/release/check_package.py` (`test-dist-package`) follow `ZIG_LOCAL_CACHE_DIR` and fall back to the checkout's `.zig-cache` as before.
- CONTRIBUTING.md gains "Keep checkouts small" (shared cache, `--prefix`, weights/GGUF/fixtures/job output outside the tree); the native recipes skill points to it.
