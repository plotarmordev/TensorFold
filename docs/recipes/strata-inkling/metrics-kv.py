"""Print the KV dtype a Strata server reports in /metrics's engine block; fails when the field is absent."""
import json
import sys
import urllib.request

dtype = json.load(urllib.request.urlopen(sys.argv[1]))["engine"]["kv_dtype"]
if not isinstance(dtype, str):
    raise SystemExit("/metrics reports kv_dtype=%r, not a dtype" % dtype)
print(dtype)
