# Extra checks: 20261122_ubfc_baseline_s0

Computed from items.jsonl by analysis/baseline_checks.py.

Videos: 42 (subject1 to subject9).

## Heart rate (M3): failed windows

Failed: 16 of 248 windows.

| Quantity | Value |
|---|---|
| Failures with estimate ≈ second harmonic (x2) of the reference | 0 |
| Failures with estimate ≈ half (x0.5) of the reference | 4 |
| Reference below 60 bpm: failed / windows | 0 / 6 |
| Reference 60 bpm and above: failed / windows | 16 / 242 |
| PPG SNR median (dB): failed / succeeded | 8.9 / 12.6 |
| Failures by video | subject41 (5), subject14 (3), subject48 (3), subject22 (1), subject27 (1), subject32 (1), subject33 (1), subject4 (1) |

| Video | Window | Reference (bpm) | Estimate (bpm) | Estimate / reference | PPG SNR (dB) | Oximeter (bpm) |
|---|---|---|---|---|---|---|
| subject14 | 2 | 73.2 | 84.2 | 1.15 | 10.9 | 73 |
| subject14 | 4 | 64.8 | 85.4 | 1.32 | 1.4 | 75 |
| subject14 | 5 | 73.5 | 61.6 | 0.84 | 2.6 | 77 |
| subject22 | 4 | 104.8 | 50.8 | 0.48 | 9.8 | 110 |
| subject27 | 0 | 72.8 | 66.6 | 0.92 | 3.1 | 83 |
| subject32 | 0 | 92.7 | 48.0 | 0.52 | 8.5 | 105 |
| subject33 | 1 | 109.5 | 122.4 | 1.12 | 0.2 | 117 |
| subject4 | 3 | 92.8 | 110.0 | 1.19 | -2.4 | 116 |
| subject41 | 0 | 102.4 | 48.9 | 0.48 | 5.5 | 97 |
| subject41 | 2 | 102.2 | 69.4 | 0.68 | 9.8 | 100 |
| subject41 | 3 | 105.6 | 47.6 | 0.45 | 19.9 | 107 |
| subject41 | 4 | 99.6 | 66.7 | 0.67 | 9.7 | 101 |
| subject41 | 5 | 96.5 | 42.4 | 0.44 | 12.3 | 101 |
| subject48 | 1 | 86.3 | 32.6 | 0.38 | 7.3 | 87 |
| subject48 | 2 | 86.3 | 48.5 | 0.56 | 9.3 | 88 |
| subject48 | 3 | 86.2 | 56.4 | 0.65 | 11.0 | 88 |

## Breathing: camera estimates against the PPG-derived rates

Windows with a reference under the wide_reference rule and both camera estimates: 33. Agreement means within the breathing tolerance.

| Pair | Median abs difference (bpm) | Agree |
|---|---|---|
| M1 vs M2 | 14.2 | 4 of 33 |
| M1 vs BW | 13.6 | 7 of 33 |
| M1 vs AM | 13.8 | 10 of 33 |
| M1 vs FM | 12.5 | 6 of 33 |
| M2 vs BW | 2.7 | 13 of 33 |
| M2 vs AM | 3.7 | 14 of 33 |
| M2 vs FM | 3.8 | 11 of 33 |

| Reference range | Windows | M1 agrees | M2 agrees |
|---|---|---|---|
| 0-12 bpm | 25 | 4 | 12 |
| 12-15 bpm | 3 | 1 | 0 |
| 15-60 bpm | 5 | 3 | 0 |

Quantiles (min / 25% / median / 75% / max, bpm):

- M1: 6.9 / 18.7 / 24.0 / 26.6 / 30.2
- M2: 6.4 / 7.7 / 8.9 / 10.6 / 18.5
- reference: 6.3 / 8.5 / 9.9 / 11.9 / 24.9
