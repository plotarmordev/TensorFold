#!/bin/sh
# kv-pages.sh at 8K context, one GPU job: 1K and 8K do not fit one 600 s job
KV_PAGES_CONTEXT=8192 exec sh "$(dirname "$0")/kv-pages.sh"
