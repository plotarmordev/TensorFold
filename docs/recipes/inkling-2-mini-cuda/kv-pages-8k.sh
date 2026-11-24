#!/bin/sh
# kv-pages.sh at 8K context, one GPU job (TOP builds every job cold, so both contexts cannot share a run)
KV_PAGES_CONTEXT=8192 exec sh "$(dirname "$0")/kv-pages.sh"
