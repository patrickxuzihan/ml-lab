# Does camera-based breathing measurement survive video compression?

Video from phones, video calls and cloud storage is almost always compressed before an algorithm sees it. This project measures breathing rate from chest motion and heart rate from subtle skin-color changes on the same clips, compresses the clips the way video calls do, and studies which measurement breaks first, why, and how to make breathing measurement more robust.

**Status:** planning. Results will appear here as experiments finish, and every number on this page will link to a file under `results/`.

## Questions

1. How does compression (H.264 and H.265 at different bitrates) change the error of camera-based breathing measurement, and where does it break down?
2. On the same videos, is breathing more robust to compression than heart rate? Is that because of the signal type (motion vs. color) or the frequency (slow vs. fast)?
3. At a fixed bitrate, how should frame rate, resolution and per-frame quality be traded off for each measurement?
4. Can one targeted fix, such as training with compressed videos, recover accuracy at low bitrates?

## Data

Public datasets only: SCAMPS and UBFC-rPPG, plus PURE and COHFACE if access is granted. No images or video of real people are stored in this repository.

## Plan

See [GUIDE.md](GUIDE.md) (in Chinese).
