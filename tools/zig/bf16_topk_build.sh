#!/bin/sh
set -eu
: "${ZIG:?set ZIG to the installed pinned 0.17.0 compiler}"
[ "$($ZIG version)" = "0.17.0" ]
bf16_topk_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$bf16_topk_root"
bf16_topk_sdk=$(xcrun --sdk macosx --show-sdk-path)
export ZIG_LOCAL_CACHE_DIR="${ZIG_LOCAL_CACHE_DIR:-$HOME/.cache/tensorfold/zig}"
export ZIG_GLOBAL_CACHE_DIR="$ZIG_LOCAL_CACHE_DIR/global"
bf16_topk_checkout=$(printf '%s' "$bf16_topk_root" | shasum -a 256 | cut -c1-16)
mkdir -p "$ZIG_LOCAL_CACHE_DIR/checkouts/$bf16_topk_checkout"
"$ZIG" test -OReleaseSafe zig/src/core/bf16_topk.zig
"$ZIG" build-exe -OReleaseSafe -target aarch64-native -mcpu=apple_m1 --dep metal --dep bf16_topk --dep bf16_topk_gpu -Mroot=zig/tests/bf16_topk.zig --dep metal --dep bf16_topk --dep bf16_topk_sources -Mbf16_topk_gpu=zig/src/core/bf16_topk_gpu.zig -Mbf16_topk=zig/src/core/bf16_topk.zig --dep profile -Mmetal=zig/src/metal/metal.zig -Mprofile=zig/src/core/profile.zig -Mbf16_topk_sources=zig/bf16_topk_sources.zig -F "$bf16_topk_sdk/System/Library/Frameworks" -L "$bf16_topk_sdk/usr/lib" -framework Metal -framework Foundation -lc -lobjc -femit-bin="$ZIG_LOCAL_CACHE_DIR/checkouts/$bf16_topk_checkout/tf-bf16-topk"
