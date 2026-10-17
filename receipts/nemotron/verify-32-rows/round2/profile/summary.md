# window-profile, one GB10, alternated base, PR, PR, base (graph ms, median of 5; conv_scan: eager class ms, shared expert forked)

count.ids is 50 ids. A build's windows take the last windowRows + 1 ids after its prompt (16 on the base, 32 on the PR), so
unaligned/ gives the two builds different prompts and windows (base at position 34, PR at 18); aligned/ and tip-aligned/ give
the base count34.ids (the first 34 ids), so both builds run the same prompt and the same window tokens at position 18.

## unaligned

| run | 1 | 3 | 8 | 12 | 16 | conv_scan 8 | conv_scan 16 |
|---|---|---|---|---|---|---|---|
| base.0 | 47.707 | 59.801 | 85.873 | 108.930 | 121.690 | 1.689 | 2.150 |
| base.1 | 47.942 | 59.749 | 86.198 | 109.264 | 121.411 | 1.695 | 2.174 |
| pr.0 | 47.925 | 63.758 | 89.933 | 104.253 | 118.712 | 1.893 | 2.441 |
| pr.1 | 48.050 | 64.035 | 90.390 | 104.079 | 118.762 | 1.894 | 2.481 |

## aligned

| run | 1 | 3 | 8 | 12 | 16 | conv_scan 8 | conv_scan 16 |
|---|---|---|---|---|---|---|---|
| base.0 | 48.947 | 64.117 | 90.348 | 103.715 | 118.256 | 1.703 | 2.169 |
| base.1 | 48.170 | 63.790 | 90.086 | 104.281 | 118.809 | 1.706 | 2.179 |
| pr.0 | 47.580 | 63.790 | 89.914 | 103.872 | 118.320 | 1.902 | 2.442 |
| pr.1 | 47.726 | 64.024 | 89.822 | 104.073 | 118.932 | 1.868 | 2.415 |

## tip-aligned

| run | 1 | 3 | 8 | 12 | 16 | conv_scan 8 | conv_scan 16 |
|---|---|---|---|---|---|---|---|
| base.0 | 47.805 | 64.183 | 89.385 | 103.474 | 117.785 | 1.692 | 2.153 |
| base.1 | 48.080 | 65.015 | 90.034 | 103.715 | 118.740 | 1.710 | 2.161 |
| tip.0 | 48.057 | 64.164 | 89.447 | 103.575 | 117.931 | 1.698 | 2.212 |
| tip.1 | 47.988 | 63.917 | 89.578 | 103.764 | 118.222 | 1.730 | 2.250 |

