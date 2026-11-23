# Inkling 2 Mini CUDA: undo rejected draft rows
- Keep the exact convolution state after the last accepted row when a forced draft chain rejects a suffix; the loop's other state is per-forward scratch.
- Check forced accepted and rejected rows against no-draft greedy tokens on the released checkpoint in the Mini CUDA reference gate.
