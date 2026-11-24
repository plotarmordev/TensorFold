# Inkling 2 Mini on Metal: fewer barriers in the verify forward
- Attention prep's query heads run alongside the key and value convolutions, and its key and value heads run
  alongside the relative table. Small-window tensor-unit gate-up tiles build their expert route lists themselves,
  leaving the separate group pass for wider forwards. At up to 32 rows, the forward has 971 barrier-separated
  GPU stages instead of 1,067; larger forwards have 1,019.
- `docs/receipts/inkling-2-mini-metal-262/` counts the stages per visit from source, starting from card 248's
  profile, and `run.sh` specifies the device gates and speed window.
