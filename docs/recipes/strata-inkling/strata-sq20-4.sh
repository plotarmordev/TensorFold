#!/bin/sh
# SQ-20 at 4 streams, one GPU job: see strata-job.sh
STRATA_PACK=sq20 STRATA_STREAMS=4 exec sh "$(dirname "$0")/strata-job.sh"
