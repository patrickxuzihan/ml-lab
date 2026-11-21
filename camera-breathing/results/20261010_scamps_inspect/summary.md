# Data check: scamps

Cells show min / median / max over videos, or value (count).

| Quantity | Value |
|---|---|
| Items processed | 10 |
| Videos checked | 10 |
| Errors | 0 |
| Total size of checked videos (GB) | 1.345 |
| File size (bytes) | 1.001e+08 / 1.346e+08 / 1.691e+08 |
| Frame array as stored | [3, 320, 240, 600] (10) |
| Frame dtype as stored | float64 (10) |
| Interpretation | matlab_order (10) |
| Value range min | 0 / 0 / 0 |
| Value range max | 0.6078 / 1 / 1 |
| Height | 240 (10) |
| Width | 320 (10) |
| Frames | 600 / 600 / 600 |
| Duration (s, at assumed fps) | 20 / 20 / 20 |
| Breathing waveform matches frames | 10 of 10 |
| PPG waveform matches frames | 10 of 10 |
| Breathing rate from waveform (bpm) | 8.471 / 15.93 / 21.66 |
| Heart rate from PPG (bpm) | 48.99 / 124.3 / 141.8 |
| Probe regions from | pose (8), fallback (2) |
| Pose detected (fraction of sampled frames) | 1 / 1 / 1 |
| Nose visible | 1 / 1 / 1 |
| Both shoulders visible | 0 / 1 / 1 |
| A hip visible | 0 / 0 / 0 |
| Shoulder line (fraction of height) | 0.9043 / 0.9531 / 0.9797 |
| Shoulder width (px) | 171.5 / 198.4 / 223.4 |
| Rows below shoulders (px, median) | 4.864 / 11.26 / 22.97 |
| Rows below shoulders (px, min) | 0.9354 / 9.019 / 19.01 |
| Rows below shoulders / shoulder width | 0.02169 / 0.05674 / 0.1279 |
| Face width (px) | 80.54 / 88.67 / 99.02 |
| Processing time per video (s) | 10.83 / 12.64 / 15.06 |

## Does breathing show up as body motion?

Motion of each region (global optical flow), band-passed and compared with `d_br`.
A peak match only means something when the motion std is clearly above zero.

| Region | Direction | Peak matches reference | r at zero lag | best abs(r) within max lag | Motion std (px) |
|---|---|---|---|---|---|
| shoulder | vertical | 10 of 10 | -1 / -0.9999 / -0.9955 | 0.9955 / 0.9999 / 1 | 0.02555 / 0.149 / 0.3451 |
| shoulder | horizontal | 9 of 10 | -0.9989 / 0.03382 / 0.9999 | 0.6325 / 0.9261 / 0.9999 | 0.002996 / 0.009617 / 0.08162 |
| chest | vertical | 2 of 2 | -0.9999 / -0.9999 / -0.9999 | 0.9999 / 0.9999 / 0.9999 | 0.07211 / 0.1042 / 0.1363 |
| chest | horizontal | 2 of 2 | -0.9989 / -0.001776 / 0.9953 | 0.9953 / 0.9971 / 0.9989 | 0.01799 / 0.01966 / 0.02132 |
| head | vertical | 10 of 10 | -0.9989 / -0.9916 / -0.6589 | 0.676 / 0.9926 / 0.9989 | 0.2016 / 0.2736 / 0.5234 |
| head | horizontal | 9 of 10 | -0.8996 / -0.5895 / 0.9967 | 0.6193 / 0.8213 / 0.9967 | 0.005259 / 0.009414 / 0.0205 |
