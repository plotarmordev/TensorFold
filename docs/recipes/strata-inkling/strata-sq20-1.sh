#!/bin/sh
# SQ-20 at 1 stream, one GPU job: see strata-job.sh
STRATA_PACK=sq20 STRATA_STREAMS=1 exec sh "$(dirname "$0")/strata-job.sh"
