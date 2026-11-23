#!/bin/sh
# SQ-10 at 64 streams, one GPU job: see strata-job.sh
STRATA_PACK=sq10 STRATA_STREAMS=64 exec sh "$(dirname "$0")/strata-job.sh"
