#!/bin/sh
# Stages the two llama.cpp source trees job.sh builds, whose GPU job has no network: mainline llama.cpp b12651 and
# Unsloth's b12651-mix-c7059d6 (the same b12651 plus Unsloth's 43 pins, scripts/unsloth/pr-set.json; Unsloth's
# `master` holds only that pin list), as tarballs in TOP's pilot folder inkling-llamacpp-251/. The GGUF and CMake are
# card 232's, already in inkling-gguf/ and inkling-llamacpp/. The Desk runs it from any machine with network and the
# top-menu key, never on a Spark:
#   sh docs/recipes/inkling-2-mini-llamacpp/stage.sh
set -eu
# shasum -a 256 -c works on macOS and Linux (perl Digest::SHA); Mac sha256sum has no -c/--quiet. Mac bsdtar also
# writes AppleDouble files and xattrs unless asked not to, so give it the flags GNU tar ignores:
case "$(tar --version 2>/dev/null)" in *bsdtar*) TARFLAGS="--no-mac-metadata --no-xattrs" ;; *) TARFLAGS= ;; esac
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
dir=$work/inkling-llamacpp-251
mkdir "$dir"
curl -sSfL -o "$dir/mainline-b12651.tar.gz" https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12651.tar.gz
curl -sSfL -o "$dir/unsloth-b12651-mix-c7059d6.tar.gz" \
    https://github.com/unslothai/llama.cpp/releases/download/b12651-mix-c7059d6/llama.cpp-source-b12651-mix-c7059d6.tar.gz
(cd "$dir" && shasum -a 256 -c) <<EOF
f98dbe546ef7633d38400648244b981e9ef36edf4ac140c3c9721fb37f483702  mainline-b12651.tar.gz
dedb1c001a6d4af567d9ea152f7fdd0a467cc04486021aa0f126d7dbe2375bb0  unsloth-b12651-mix-c7059d6.tar.gz
EOF
COPYFILE_DISABLE=1 tar ${TARFLAGS:-} -C "$work" -c inkling-llamacpp-251 | ssh top-menu "put top inkling-llamacpp-251"
