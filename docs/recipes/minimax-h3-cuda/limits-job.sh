#!/bin/sh
# The limits audit as its own TOP job: the five SMs' images and kernel-contract-limits alone, without the picture
# tools, CLI and tf-cuda-test that put a timed bench past the Desk's 20 minutes. Its limits.log holds the RECORD
# lines the registry's entries are pasted from.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/limits-job.sh"
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
H3_BUILD=limits exec sh "$here/cuda-build.sh"
