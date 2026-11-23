# MiniMax-H3-Image: the resident server keeps main's idle ticker and fail-closed startup

- The #190 merge had taken the branch's `tf-h3-image-server` whole, dropping upstream's idle ticker and its refusal to serve unwarmed. Both are back beside `--draft-steps`, `--routed-steps` and `profile`: a keepalive commits every 400 ms between requests so the weights' residency set stays wired, and the server exits when the weights cannot be held resident or the ticker cannot start.
- Upstream's gates come with it: resident/fresh latent parity after an idle pause the ticker spans, and both startup refusals.
- The Metal recipe's resident section now says so; every other conflict in that merge was re-read and kept nothing else of upstream's that the branch lacks.
