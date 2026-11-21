# Data check: scamps

Cells show min / median / max over videos, or value (count).

| Quantity | Value |
|---|---|
| Items processed | 1 |
| Videos checked | 1 |
| Errors | 0 |
| Total size of checked videos (GB) | 0.1001 |
| File size (bytes) | 1.001e+08 / 1.001e+08 / 1.001e+08 |
| Frame array as stored | [3, 320, 240, 600] (1) |
| Frame dtype as stored | float64 (1) |
| Interpretation | matlab_order (1) |
| Value range min | 0 / 0 / 0 |
| Value range max | 0.6078 / 0.6078 / 0.6078 |
| Height | 240 (1) |
| Width | 320 (1) |
| Frames | 600 / 600 / 600 |
| Duration (s, at assumed fps) | 20 / 20 / 20 |
| Breathing waveform matches frames | 1 of 1 |
| PPG waveform matches frames | 1 of 1 |
| Breathing rate from waveform (bpm) | 16.03 / 16.03 / 16.03 |
| Heart rate from PPG (bpm) | 126.2 / 126.2 / 126.2 |
| Probe regions from | pose (1) |
| Pose detected (fraction of sampled frames) | 1 / 1 / 1 |
| Nose visible | 1 / 1 / 1 |
| Both shoulders visible | 1 / 1 / 1 |
| A hip visible | 0 / 0 / 0 |
| Shoulder line (fraction of height) | 0.9263 / 0.9263 / 0.9263 |
| Shoulder width (px) | 204.8 / 204.8 / 204.8 |
| Rows below shoulders (px, median) | 17.69 / 17.69 / 17.69 |
| Rows below shoulders (px, min) | 12.58 / 12.58 / 12.58 |
| Rows below shoulders / shoulder width | 0.08665 / 0.08665 / 0.08665 |
| Face width (px) | 84.57 / 84.57 / 84.57 |
| Processing time per video (s) | 14.65 / 14.65 / 14.65 |

## Does breathing show up as body motion?

Motion of each region (global optical flow), band-passed and compared with `d_br`.
A peak match only means something when the motion std is clearly above zero.

| Region | Direction | Peak matches reference | r at zero lag | best abs(r) within max lag | Motion std (px) |
|---|---|---|---|---|---|
| shoulder | vertical | 1 of 1 | -0.9995 / -0.9995 / -0.9995 | 0.9995 / 0.9995 / 0.9995 | 0.04022 / 0.04022 / 0.04022 |
| shoulder | horizontal | 0 of 1 | -0.6251 / -0.6251 / -0.6251 | 0.6325 / 0.6325 / 0.6325 | 0.00714 / 0.00714 / 0.00714 |
| chest | vertical | – | – | – | – |
| chest | horizontal | – | – | – | – |
| head | vertical | 1 of 1 | -0.9428 / -0.9428 / -0.9428 | 0.9491 / 0.9491 / 0.9491 | 0.2016 / 0.2016 / 0.2016 |
| head | horizontal | 1 of 1 | -0.7808 / -0.7808 / -0.7808 | 0.7808 / 0.7808 / 0.7808 | 0.005824 / 0.005824 / 0.005824 |
