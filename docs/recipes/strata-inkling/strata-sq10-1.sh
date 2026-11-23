#!/bin/sh
# SQ-10 at 1 stream, one GPU job: see strata-job.sh
STRATA_PACK=sq10 STRATA_STREAMS=1 exec sh "$(dirname "$0")/strata-job.sh"
