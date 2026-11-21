# Uncompressed baseline: 20261122_scamps_baseline-example_s0

Dataset: SCAMPS. Videos measured: 10 (subjects: 10); errors: 0.
Videos streamed: 10; skipped for head motion: 0.

Evaluation protocol of configs/protocol.yaml on the uncompressed source: windows of M1 30 s, M2 30 s, M3 10 s; success means an error within 2 bpm (breathing) or 5 bpm (heart rate). Point value [95% CI]; CIs from a cluster bootstrap over subjects.

## Main result

| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r | Success rate (%) | SNR (dB) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 10 | 10 | 1.00 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.0 [0.0, 0.0] | 1.00 [1.00, 1.00] | 100.0 [100.0, 100.0] | 30.0 [28.3, 31.9] |
| M2 | 10 | 10 | 1.00 | 2.80 [0.09, 7.19] | 6.28 [0.16, 10.45] | 16.8 [0.7, 41.9] | 0.58 [0.13, 1.00] | 80.0 [50.0, 100.0] | 10.7 [6.8, 14.9] |
| M3 | 10 | 20 | 1.00 | 6.66 [0.33, 13.22] | 16.96 [0.40, 25.81] | 11.5 [0.3, 25.3] | 0.87 [0.76, 1.00] | 80.0 [60.0, 100.0] | 2.6 [0.3, 5.1] |

## Windows excluded

None.

## Regions

- Shoulder band from: pose (8), face_anchored (2)
- Face box from: pose (10)

## Head-still subset

A video is measured when each of d_pitch, d_roll, d_yaw varies by at most 1e-06 over the video.

| Quantity | Value |
|---|---|
| Videos streamed | 10 |
| Measured | 10 |
| Skipped for head motion | 0 |
| Largest head-pose range, measured (min / 25% / median / 75% / max) | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| Largest head-pose range, skipped | – |
| Measured videos with facial action units | 10 |

## Sensitivity analyses

The same windows without some videos (GUIDE.md section 4.4).

| Analysis | Videos removed | Measure | Windows | MAE (bpm) | Success rate (%) |
|---|---|---|---|---|---|
| Without face-anchored shoulder bands | 2 | M1 | 8 | 0.00 [0.00, 0.01] | 100.0 [100.0, 100.0] |
| Without face-anchored shoulder bands | 2 | M2 | 8 | 1.27 [0.04, 3.65] | 87.5 [62.5, 100.0] |
| Without face-anchored shoulder bands | 2 | M3 | 16 | 5.14 [0.25, 11.78] | 81.2 [56.2, 100.0] |

## For comparison with published results: one window per whole video

Not the protocol: every measurement uses a single window spanning the whole video (the reference too), as in evaluations that report one rate per video.

| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r | Success rate (%) | SNR (dB) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 10 | 10 | 1.00 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.0 [0.0, 0.0] | 1.00 [1.00, 1.00] | 100.0 [100.0, 100.0] | 30.0 [28.3, 31.9] |
| M2 | 10 | 10 | 1.00 | 2.80 [0.09, 7.19] | 6.28 [0.16, 10.45] | 16.8 [0.7, 41.9] | 0.58 [0.13, 1.00] | 80.0 [50.0, 100.0] | 10.7 [6.8, 14.9] |
| M3 | 10 | 10 | 1.00 | 0.85 [0.11, 2.24] | 2.24 [0.15, 3.86] | 0.9 [0.1, 2.4] | 1.00 [0.99, 1.00] | 90.0 [70.0, 100.0] | 2.1 [0.4, 3.9] |
