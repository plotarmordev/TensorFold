#!/bin/sh
# Both packs under Strata's own --bench at 1, 4 and 64 streams, one GPU job without the HTTP bench: see strata-job.sh
STRATA_HTTP=0 exec sh "$(dirname "$0")/strata-job.sh"
