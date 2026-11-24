# Inkling 2 Mini on Metal: fewer barriers in the verify forward
- Attention prep's query heads run alongside the key and value convolutions, and its key and value heads run
  alongside the relative table. Small-window tensor-unit gate-up tiles build their expert route lists themselves,
  leaving the separate group pass for wider forwards. The router's score and top-8 launches are a single
  dispatch (`tf_inkling2_route`), and on gated visits each controller tuple's normalize-zeta-blend chain plus
  its RMS norm is one dispatch (`tf_inkling2_loop_gate`). At up to 32 rows the forward has 779 barrier-separated
  GPU stages instead of 1,067; larger forwards have 827. Outputs are bit-identical: every fused phase keeps the
  old kernels' reduction order and rounding points.
- `docs/receipts/inkling-2-mini-metal-262/` counts the stages per visit from source, starting from card 248's
  profile, and `run.sh` specifies the device gates and speed window.
