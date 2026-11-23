#!/bin/sh
# SQ-10's HTTP bench at 1, 4 and 64 streams, one GPU job without the own bench: see strata-job.sh
STRATA_PACK=sq10 STRATA_OWN=0 exec sh "$(dirname "$0")/strata-job.sh"
