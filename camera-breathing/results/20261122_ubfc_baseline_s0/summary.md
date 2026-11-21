# Uncompressed baseline: 20261122_ubfc_baseline_s0

Dataset: UBFC-rPPG. Videos measured: 42 (subjects: 42); errors: 0.

Evaluation protocol of configs/protocol.yaml on the uncompressed source: windows of M1 30 s, M2 30 s, M3 10 s; success means an error within 2 bpm (breathing) or 5 bpm (heart rate). Point value [95% CI]; CIs from a cluster bootstrap over subjects.

## Main result

| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r | Success rate (%) | SNR (dB) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 10 | 12 | 1.00 | 10.02 [6.55, 13.98] | 12.35 [9.45, 15.19] | 121.5 [71.4, 185.5] | 0.13 [-0.56, 0.38] | 33.3 [10.0, 54.5] | 2.5 [-2.2, 6.9] |
| M2 | 10 | 12 | 1.00 | 4.73 [1.70, 7.95] | 7.21 [2.73, 10.09] | 38.3 [17.1, 64.0] | -0.31 [-0.76, 0.34] | 50.0 [18.2, 83.3] | 2.9 [-2.5, 8.1] |
| M3 | 42 | 248 | 1.00 | 2.64 [0.96, 5.05] | 9.58 [3.55, 14.49] | 2.8 [1.0, 5.3] | 0.88 [0.75, 0.98] | 93.5 [87.6, 98.0] | 7.6 [6.4, 8.6] |

## Windows excluded

- M1: no_reference (69)
- M2: no_reference (69)

## Regions

- Shoulder band from: pose (42)
- Face box from: pose (42)

## Breathing reference derived from PPG

Windows of M1: 81 from 42 subjects. Quantiles are min / 25% / median / 75% / max; CIs from a cluster bootstrap over subjects.

| Quantity | Value |
|---|---|
| Usable (BW, AM, FM all present, range ≤ 2.0 bpm; protocol) | 12 of 81 (14.8%) [6.2, 23.5] |
| Usable under the wide rule (sample SD ≤ 4.0 bpm) | 33 of 81 (40.7%) [27.8, 53.7] |
| BW gave a rate | 81 of 81 (100.0%) |
| AM gave a rate | 81 of 81 (100.0%) |
| FM gave a rate | 81 of 81 (100.0%) |
| Range of BW, AM, FM (bpm) | 0.25 / 5.20 / 8.77 / 14.79 / 27.50 |
| BW vs AM: abs difference (bpm) | 0.03 / 0.70 / 2.68 / 9.08 / 24.02 |
| BW vs AM: within 2.0 bpm | 35 of 81 (43.2%) |
| BW vs FM: abs difference (bpm) | 0.03 / 0.98 / 4.80 / 10.98 / 24.20 |
| BW vs FM: within 2.0 bpm | 28 of 81 (34.6%) |
| AM vs FM: abs difference (bpm) | 0.01 / 1.13 / 5.77 / 10.19 / 27.50 |
| AM vs FM: within 2.0 bpm | 26 of 81 (32.1%) |
| Beats per window | 28 / 43 / 49 / 55 / 65 |

## Heart-rate reference

Windows of M3: 248. The reference is the contact PPG waveform; the oximeter's heart-rate channel is only a cross-check.

| Quantity | Value |
|---|---|
| PPG gave a rate | 248 of 248 (100.0%) |
| PPG SNR around its own peak (dB) | -2.4 / 9.2 / 12.3 / 15.5 / 22.6 |
| Oximeter valid fraction per window | 0.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| Windows with both rates | 245 |
| Oximeter vs PPG: abs difference (bpm) | 0.00 / 0.99 / 2.07 / 4.59 / 30.41 |
| Oximeter vs PPG: within 5.0 bpm | 193 of 245 (78.8%) |

## Sensitivity analyses

The same windows without some videos (GUIDE.md section 4.4).

| Analysis | Videos removed | Measure | Windows | MAE (bpm) | Success rate (%) |
|---|---|---|---|---|---|
| Without face-anchored shoulder bands | 0 | M1 | 12 | 10.02 [6.55, 13.98] | 33.3 [10.0, 54.5] |
| Without face-anchored shoulder bands | 0 | M2 | 12 | 4.73 [1.70, 7.95] | 50.0 [18.2, 83.3] |
| Without face-anchored shoulder bands | 0 | M3 | 248 | 2.64 [0.96, 5.05] | 93.5 [87.6, 98.0] |
| Without subjects with flagged timestamps | 14 | M1 | 9 | 10.20 [6.33, 15.39] | 33.3 [12.5, 55.6] |
| Without subjects with flagged timestamps | 14 | M2 | 9 | 5.02 [1.48, 8.77] | 44.4 [10.0, 81.8] |
| Without subjects with flagged timestamps | 14 | M3 | 168 | 3.48 [0.91, 6.78] | 91.7 [83.3, 98.2] |
| Without low-frame-rate subjects (below 25 fps) | 3 | M1 | 11 | 10.01 [6.20, 14.39] | 36.4 [11.1, 58.4] |
| Without low-frame-rate subjects (below 25 fps) | 3 | M2 | 11 | 4.18 [1.15, 7.58] | 54.5 [20.0, 90.0] |
| Without low-frame-rate subjects (below 25 fps) | 3 | M3 | 230 | 2.77 [0.92, 5.25] | 93.5 [87.7, 98.3] |

## Sensitivity: wide breathing-reference rule

Breathing windows use the mean of BW, AM and FM when their sample SD is at most 4.0 bpm (after Karlen et al. 2013); heart-rate windows are unchanged.

| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r | Success rate (%) | SNR (dB) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 23 | 33 | 1.00 | 10.93 [8.16, 13.67] | 13.14 [10.83, 15.15] | 122.3 [88.9, 156.3] | -0.05 [-0.29, 0.13] | 24.2 [11.4, 37.1] | -0.1 [-3.0, 3.0] |
| M2 | 23 | 33 | 1.00 | 3.95 [2.54, 5.54] | 5.76 [3.68, 7.56] | 32.5 [22.6, 43.6] | -0.22 [-0.42, 0.08] | 36.4 [18.2, 54.5] | 5.0 [1.9, 7.9] |
| M3 | 42 | 248 | 1.00 | 2.64 [0.96, 5.05] | 9.58 [3.55, 14.49] | 2.8 [1.0, 5.3] | 0.88 [0.75, 0.98] | 93.5 [87.6, 98.0] | 7.6 [6.4, 8.6] |

## For comparison with published results: one window per whole video

Not the protocol: every measurement uses a single window spanning the whole video (the reference too), as in evaluations that report one rate per video.

| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r | Success rate (%) | SNR (dB) |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 10 | 10 | 1.00 | 6.41 [2.16, 11.15] | 9.60 [5.13, 13.07] | 69.6 [23.2, 121.0] | 0.17 [-0.65, 0.66] | 50.0 [20.0, 80.0] | 0.3 [-4.5, 5.3] |
| M2 | 10 | 10 | 1.00 | 6.02 [2.73, 9.88] | 8.23 [4.25, 11.70] | 34.0 [20.3, 47.6] | -0.01 [-0.62, 0.57] | 30.0 [0.0, 60.0] | -5.4 [-10.1, -1.0] |
| M3 | 42 | 42 | 1.00 | 3.00 [0.97, 5.74] | 8.74 [2.29, 13.48] | 3.2 [1.0, 6.1] | 0.91 [0.80, 0.99] | 88.1 [78.6, 97.6] | -3.2 [-4.6, -1.9] |

## Per subject

| Subject | fps | Frames | Timestamp gap | Duplicate timestamps | Shoulder band | PPG SNR median (dB) | Oximeter valid (mean) | Usable breathing windows |
|---|---|---|---|---|---|---|---|---|
| DATASET_2/subject1 | 29.26 | 1547 | False | True | pose | 9.7 | 1.00 | 0 of 1 |
| DATASET_2/subject3 | 29.55 | 1801 | False | False | pose | 7.2 | 1.00 | 0 of 2 |
| DATASET_2/subject4 | 29.61 | 1368 | False | True | pose | 6.4 | 1.00 | 0 of 1 |
| DATASET_2/subject5 | 29.66 | 1550 | False | True | pose | 11.7 | 1.00 | 1 of 1 |
| DATASET_2/subject8 | 29.89 | 2022 | False | False | pose | 13.7 | 1.00 | 0 of 2 |
| DATASET_2/subject9 | 29.54 | 2016 | True | True | pose | 11.9 | 1.00 | 0 of 2 |
| DATASET_2/subject10 | 29.88 | 2024 | False | True | pose | 12.7 | 1.00 | 0 of 2 |
| DATASET_2/subject11 | 29.70 | 2031 | False | False | pose | 10.7 | 0.77 | 0 of 2 |
| DATASET_2/subject12 | 29.03 | 1989 | False | False | pose | 11.4 | 1.00 | 0 of 2 |
| DATASET_2/subject13 | 29.79 | 2018 | False | False | pose | 12.6 | 1.00 | 0 of 2 |
| DATASET_2/subject14 | 29.37 | 1993 | False | False | pose | 5.8 | 1.00 | 0 of 2 |
| DATASET_2/subject15 | 29.59 | 2020 | True | False | pose | 17.0 | 1.00 | 0 of 2 |
| DATASET_2/subject16 | 29.93 | 2050 | False | False | pose | 13.9 | 1.00 | 2 of 2 |
| DATASET_2/subject17 | 29.93 | 2046 | False | False | pose | 14.8 | 1.00 | 0 of 2 |
| DATASET_2/subject18 | 29.75 | 2034 | False | True | pose | 17.1 | 0.86 | 0 of 2 |
| DATASET_2/subject20 | 29.51 | 2016 | False | False | pose | 10.7 | 0.42 | 1 of 2 |
| DATASET_2/subject22 | 29.75 | 2032 | False | False | pose | 11.2 | 1.00 | 0 of 2 |
| DATASET_2/subject23 | 28.94 | 1958 | True | False | pose | 13.0 | 1.00 | 1 of 2 |
| DATASET_2/subject24 | 29.37 | 2009 | True | False | pose | 13.2 | 0.81 | 0 of 2 |
| DATASET_2/subject25 | 23.21 | 1448 | True | False | pose | 17.1 | 1.00 | 0 of 2 |
| DATASET_2/subject26 | 23.34 | 1485 | False | False | pose | 15.8 | 1.00 | 0 of 2 |
| DATASET_2/subject27 | 23.36 | 1431 | False | False | pose | 10.0 | 1.00 | 1 of 2 |
| DATASET_2/subject30 | 29.78 | 2028 | False | False | pose | 12.9 | 1.00 | 0 of 2 |
| DATASET_2/subject31 | 29.62 | 2027 | False | False | pose | 12.6 | 1.00 | 0 of 2 |
| DATASET_2/subject32 | 29.85 | 2033 | False | False | pose | 11.0 | 1.00 | 2 of 2 |
| DATASET_2/subject33 | 29.21 | 1999 | True | False | pose | 8.8 | 1.00 | 0 of 2 |
| DATASET_2/subject34 | 29.98 | 2035 | False | False | pose | 12.3 | 1.00 | 1 of 2 |
| DATASET_2/subject35 | 29.91 | 2051 | False | False | pose | 14.8 | 1.00 | 0 of 2 |
| DATASET_2/subject36 | 29.91 | 2052 | False | False | pose | 16.7 | 1.00 | 1 of 2 |
| DATASET_2/subject37 | 29.79 | 2032 | False | True | pose | 11.9 | 1.00 | 0 of 2 |
| DATASET_2/subject38 | 29.92 | 2050 | False | False | pose | 15.6 | 1.00 | 0 of 2 |
| DATASET_2/subject39 | 29.69 | 2037 | False | False | pose | 9.6 | 1.00 | 0 of 2 |
| DATASET_2/subject40 | 29.82 | 2028 | False | False | pose | 10.3 | 1.00 | 0 of 2 |
| DATASET_2/subject41 | 29.76 | 2037 | False | False | pose | 11.1 | 1.00 | 1 of 2 |
| DATASET_2/subject42 | 29.95 | 2042 | False | False | pose | 16.1 | 1.00 | 0 of 2 |
| DATASET_2/subject43 | 29.34 | 2009 | False | False | pose | 8.7 | 1.00 | 0 of 2 |
| DATASET_2/subject44 | 29.59 | 2017 | False | False | pose | 9.4 | 1.00 | 0 of 2 |
| DATASET_2/subject45 | 29.95 | 2032 | False | False | pose | 12.4 | 1.00 | 0 of 2 |
| DATASET_2/subject46 | 29.25 | 1991 | False | True | pose | 15.3 | 1.00 | 0 of 2 |
| DATASET_2/subject47 | 29.79 | 2023 | False | False | pose | 13.2 | 1.00 | 0 of 2 |
| DATASET_2/subject48 | 28.77 | 1956 | False | False | pose | 8.8 | 1.00 | 0 of 2 |
| DATASET_2/subject49 | 29.78 | 2034 | False | True | pose | 16.6 | 1.00 | 1 of 2 |
