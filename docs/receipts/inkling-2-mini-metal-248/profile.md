# One-stream DSpark round on the M5 Ultra: where the 27 ms goes

No new GPU run: the split below comes from the server logs of this card's window (`/tmp/ink2-bench-248/*.server.log`,
one `done` line a request with `rounds`, `accepted=kept/drafted` and `decode` seconds), receipts 210, 234 and 235,
and the forward's source. M = measured, D = derived (formula given), A = assumed.

## Round split, code prompt, 256 tokens, first control (`90fac7967`, load 26-31)

| part | ms | of the round | source |
|---|---|---|---|
| round | 27.3 | 100% | M: median of 4 one-stream requests, `decode / rounds`; 5.0 drafted rows, 3.2 tokens a round |
| verify (target forward, ~6 rows) | 16.3 | 60% | M: the same session's plain request, 255 rounds in 4.16 s; A: 6 rows cost one plain step (210: "bound by the launch chain") |
| draft hold, keep, absorb | 11.0 | 40% | D: round − verify; 210 measured the same 1.70 plain steps a round |
| host waits | 0.03 | 0.1% | M: two commit-and-wait at 14.2 µs median (`submission.swift`), the two this card removed |

## Against the floors

- Target step: 5.55 GB a token (210, safetensors headers) at 1,021 GB/s (234, `bandwidth.py`) is 5.4 ms (D). The
  plain step's 16.3 ms is 3.0× that: 10.9 ms a step is not reading weights.
- The forward is one chain of command buffers with no host wait inside (`forward.zig` `runShared`: parts are committed
  as encoded, one wait at the end). The rest is the GPU's own dependent chain: a visit encodes about 16 barrier-separated
  stages (8 in `visit`, 5 in `attendSegments`, 3 in `routedExperts` on the tensor-unit route) over 48 visits, about
  770 stages a token. 10.9 ms / 770 ≈ 14 µs a stage (D).
- Draft depth is near break-even. Rows kept over drafted by depth, one stream (235, control): 172/232, 132/226,
  103/224, 57/171, 39/156, 25/153, 7/78. A drafted row costs about 1 ms, 3.7% of a round (D: 11.0 ms over 5.0 rows
  plus a fixed absorb); depth 6 keeps 0.16 of a token (5% of 3.2) and depth 7 0.09 (2.8%). Depth 6 pays about +1.3%,
  depth 7 loses about 0.9%, and it is drafted in only a third of rounds: capping depth moves a round under 0.5%.
- Host submission count: 0.1% of a round (this card's measurement: 113.4 against 118.5 / 120.9, inside load noise).

## #4618

#4618's 123.4 code / 91.5 chat at one stream and 281.1 / 223.0 at four are TensorFold's own published Inkling figures
on this Mac (235's receipt), not another engine's method. Controls on this Mac read 118.5-126.8 at one stream across
windows (234's quiet repeat: 126.6 / 126.8; this card at load 24-33: 118.5 / 120.9). The gap to 123 is machine load,
not a missing technique.

## Biggest gap and predicted gain

The target forward's barrier chain: 10.9 ms of a 16.3 ms verify is not weight reads. Each ms off the verify forward is
27.3 → 26.3 ms a round, +3.8% tok/s (D). +5% (about 6 tok/s at 120) needs 1.4 ms off: at 14 µs a stage, about 100
fewer stages a token, two fewer barriers in each of the 48 visits (8 in `visit` today), each removed barrier a visit
predicted at +2.5% (0.67 ms). This is the only lever here predicted at +5% or more, and the 14 µs is an average no GPU
trace has split by stage: a Metal capture of one decode step, without a benchmark window, names which barriers cost
most before a fusion earns a window.
