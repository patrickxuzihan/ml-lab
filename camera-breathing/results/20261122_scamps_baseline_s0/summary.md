# Uncompressed baseline: 20261122_scamps_baseline_s0

Dataset: SCAMPS. Videos measured: 100 (subjects: 100); errors: 0.
Videos streamed: 100; skipped for head motion: 0.

Evaluation protocol of configs/protocol.yaml on the uncompressed source: windows of M1 30 s, M2 30 s, M3 10 s; success means an error within 2 bpm (breathing) or 5 bpm (heart rate). Point value [95% CI]; CIs from a cluster bootstrap over subjects.

## Main result

| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r | Success rate (%) | SNR (dB) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 99 | 99 | 1.00 | 0.00 [0.00, 0.01] | 0.01 [0.00, 0.02] | 0.0 [0.0, 0.1] | 1.00 [1.00, 1.00] | 100.0 [100.0, 100.0] | 30.7 [29.7, 31.7] |
| M2 | 99 | 99 | 1.00 | 3.36 [2.26, 4.58] | 6.83 [5.05, 8.42] | 21.5 [14.3, 29.4] | 0.43 [0.25, 0.63] | 70.7 [61.6, 79.8] | 9.1 [7.7, 10.5] |
| M3 | 99 | 198 | 1.00 | 6.67 [4.10, 9.43] | 17.48 [13.13, 21.20] | 11.5 [6.6, 17.1] | 0.84 [0.78, 0.91] | 84.3 [77.8, 90.4] | 3.4 [2.7, 4.1] |

## Windows excluded

- M1: no_region (1)
- M2: no_region (1)
- M3: no_region (2)

## Regions

- Shoulder band from: pose (67), face_anchored (32), none (1)
- Face box from: pose (99), none (1)

## Head-still subset

A video is measured when each of d_pitch, d_roll, d_yaw varies by at most 1e-06 over the video.

| Quantity | Value |
|---|---|
| Videos streamed | 100 |
| Measured | 100 |
| Skipped for head motion | 0 |
| Largest head-pose range, measured (min / 25% / median / 75% / max) | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| Largest head-pose range, skipped | – |
| Measured videos with facial action units | 100 |

## Sensitivity analyses

The same windows without some videos (GUIDE.md section 4.4).

| Analysis | Videos removed | Measure | Windows | MAE (bpm) | Success rate (%) |
|---|---|---|---|---|---|
| Without face-anchored shoulder bands | 32 | M1 | 67 | 0.00 [0.00, 0.01] | 100.0 [100.0, 100.0] |
| Without face-anchored shoulder bands | 32 | M2 | 67 | 2.89 [1.64, 4.44] | 74.6 [64.2, 85.1] |
| Without face-anchored shoulder bands | 32 | M3 | 134 | 7.07 [4.17, 10.54] | 82.1 [73.1, 89.6] |

## For comparison with published results: one window per whole video

Not the protocol: every measurement uses a single window spanning the whole video (the reference too), as in evaluations that report one rate per video.

| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r | Success rate (%) | SNR (dB) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 99 | 99 | 1.00 | 0.00 [0.00, 0.01] | 0.01 [0.00, 0.02] | 0.0 [0.0, 0.1] | 1.00 [1.00, 1.00] | 100.0 [100.0, 100.0] | 30.7 [29.7, 31.7] |
| M2 | 99 | 99 | 1.00 | 3.36 [2.26, 4.58] | 6.83 [5.05, 8.42] | 21.5 [14.3, 29.4] | 0.43 [0.25, 0.63] | 70.7 [61.6, 79.8] | 9.1 [7.7, 10.5] |
| M3 | 99 | 99 | 1.00 | 4.79 [2.25, 7.95] | 15.29 [9.78, 20.04] | 8.0 [3.4, 13.9] | 0.88 [0.80, 0.95] | 89.9 [83.8, 94.9] | 1.8 [1.1, 2.5] |
