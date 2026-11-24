# Inkling 2 Mini on Metal: one barrier fewer in every layer visit of the forward
- Attention prep's query heads run alongside the key and value convolutions, and its key and value heads run
  alongside the relative table, so a forward has 1,019 barrier-separated GPU stages instead of 1,067. The outputs
  are the same bits.
- `docs/receipts/inkling-2-mini-metal-262/` counts the stages per visit from source, starting from card 248's
  profile, and `run.sh` runs the device gates and the speed window.
