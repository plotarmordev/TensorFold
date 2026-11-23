#!/bin/sh
# SQ-20's HTTP bench at 1, 4 and 64 streams, one GPU job without the own bench: see strata-job.sh
STRATA_PACK=sq20 STRATA_OWN=0 exec sh "$(dirname "$0")/strata-job.sh"
