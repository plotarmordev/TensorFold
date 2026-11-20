#!/bin/sh
# Card 144: the --tc decoder's warm 2048² decode kernel by kernel beside mmh3's VAE, in one TOP
# window — bench-trace-job.sh's our-side trace plus mmh3-server-job.sh's, so both engines' decoder
# kernel budgets sit in one job's output.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/decoder-trace-job.sh"
# Our side: bench-script.sh under the CUPTI preload (selftests first, then the resident servers; the
# decoder server's cupti-trace.csv.<pid> holds every kernel of the warm decodes, start and end ns,
# so the trace shows gaps as well as kernel times). mmh3's side: mmh3-server-job.sh unchanged, whose
# cupti-warm-segment.txt is the warm-picture kernel table the VAE section is read from.
# Outputs: /work/out/h3-image-bench-trace (our logs, SUMMARY.md, per-process traces) and
# /work/out/mmh3-server (mmh3's requests.log, server logs, trace and warm-segment table).
set -eu
H3_BENCH_SKIP_COMPARE=1 H3_DECODER=tc \
    sh docs/recipes/minimax-h3-cuda/bench-trace-job.sh
sh docs/recipes/minimax-h3-cuda/mmh3-server-job.sh
echo "decoder-trace-job done"
