"""Compression settings (GUIDE.md section 4.3) and the ffmpeg arguments for each.

Pure functions only; `video_io.write_video` runs ffmpeg. Settings come from the
`compression` section of configs/protocol.yaml:

- crf:        storage / recording. libx264 or libx265 at a constant rate factor, preset and
              B-frames left at their defaults.
- bitrate:    video call. Target bitrate with a VBV buffer, `-tune zerolatency`, no B-frames.
- allocation: Q3. The video-call settings at a fixed bitrate, after integer frame decimation
              and downscaling.
- raw:        uncompressed (rawvideo) after the same decimation and downscaling, to separate
              the effect of frame rate and resolution from that of compression.

Every encoded version has a single keyframe (the first frame): no periodic or scene-cut
keyframes. All lossy versions are 4:2:0 (`yuv420p`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

import numpy as np

SHORT_CODEC = {"libx264": "x264", "libx265": "x265", "rawvideo": "raw"}


@dataclass(frozen=True)
class Setting:
    name: str
    ladder: str  # source, crf, bitrate, allocation or raw
    codec: str | None = None  # None: the untouched source
    crf: float | None = None
    kbps: float | None = None
    decimate: int = 1
    scale: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


SOURCE = Setting(name="source", ladder="source")


def _num(value: float) -> str:
    return f"{value:g}"


def make_setting(
    ladder: str,
    codec: str,
    crf: float | None = None,
    kbps: float | None = None,
    decimate: int = 1,
    scale: int = 1,
) -> Setting:
    parts = [SHORT_CODEC[codec]]
    if crf is not None:
        parts.append(f"crf{_num(crf)}")
    if kbps is not None:
        parts.append(f"br{_num(kbps)}")
    if decimate != 1 or scale != 1 or ladder == "allocation":
        parts += [f"d{decimate}", f"s{scale}"]
    return Setting("_".join(parts), ladder, codec, crf, kbps, int(decimate), int(scale))


def expand_ladders(cfg: dict) -> list[Setting]:
    """The source plus every setting of the configured ladders, best quality first."""
    ladders = cfg["ladders"]
    settings = [SOURCE]
    crf = ladders.get("crf")
    if crf:
        for codec in crf["codecs"]:
            settings += [make_setting("crf", codec, crf=v) for v in sorted(crf["crf"])]
    bitrate = ladders.get("bitrate")
    if bitrate:
        for codec in bitrate["codecs"]:
            kbps = sorted(bitrate["kbps"], reverse=True)
            settings += [make_setting("bitrate", codec, kbps=v) for v in kbps]
    alloc = ladders.get("allocation")
    if alloc:
        for codec in alloc["codecs"]:
            for kbps in sorted(alloc["kbps"], reverse=True):
                for d in alloc["decimate"]:
                    for s in alloc["scale"]:
                        settings.append(make_setting("allocation", codec, None, kbps, d, s))
        for d in alloc["decimate"]:
            for s in alloc["scale"]:
                if (d, s) != (1, 1):
                    settings.append(make_setting("raw", "rawvideo", decimate=d, scale=s))
    return settings


def select(settings: list[Setting], names: Iterable[str]) -> list[Setting]:
    """Settings by name, in the given order; unknown names raise KeyError."""
    by_name = {s.name: s for s in settings}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"unknown compression settings: {missing}")
    return [by_name[n] for n in names]


def ladder_groups(settings: list[Setting]) -> dict[str, list[str]]:
    """Ordered rungs (best quality first) of each CRF and bitrate ladder, per codec."""
    groups: dict[str, list[str]] = {}
    for s in settings:
        if s.ladder in ("crf", "bitrate"):
            groups.setdefault(f"{SHORT_CODEC[s.codec]}_{s.ladder}", []).append(s.name)
    return groups


def output_size(width: int, height: int, scale: int) -> tuple[int, int]:
    """(width, height) after downscaling, rounded down to even numbers for 4:2:0."""
    return max(2, (width // scale) // 2 * 2), max(2, (height // scale) // 2 * 2)


def decimate(frames: Iterable[np.ndarray], factor: int) -> Iterator[np.ndarray]:
    """Keep frames 0, k, 2k, ... (original frames only; nothing is interpolated)."""
    for i, frame in enumerate(frames):
        if i % factor == 0:
            yield frame


def n_decimated(n_frames: int, factor: int) -> int:
    return -(-n_frames // factor)


def ffmpeg_args(setting: Setting, width: int, height: int, cfg: dict) -> tuple[list[str], str]:
    """Output arguments for `video_io.write_video` and the file extension.

    width, height: size of the frames that are fed to ffmpeg (before downscaling).
    """
    if setting.codec is None:
        raise ValueError("the source is not encoded")
    out_w, out_h = output_size(width, height, setting.scale)
    args = []
    if (out_w, out_h) != (width, height):
        args += ["-vf", f"scale={out_w}:{out_h}:flags={cfg['scale_flags']}"]
    if setting.codec == "rawvideo":
        return args + ["-c:v", "rawvideo", "-pix_fmt", "bgr24"], ".avi"

    gop = int(cfg["gop_frames"])
    args += ["-c:v", setting.codec, "-preset", cfg["preset"], "-pix_fmt", cfg["pix_fmt"]]
    x265_params = [f"keyint={gop}", "scenecut=0", "log-level=error"]
    if setting.codec == "libx264":
        args += ["-g", str(gop), "-sc_threshold", "0"]
    elif setting.codec != "libx265":
        raise ValueError(f"unsupported codec {setting.codec}")

    if setting.crf is not None:
        args += ["-crf", _num(setting.crf)]
    elif setting.kbps is not None:
        rate = _num(setting.kbps)
        buffer = _num(setting.kbps * cfg["vbv_buffer_s"])
        args += ["-b:v", f"{rate}k", "-tune", "zerolatency"]
        if setting.codec == "libx264":
            args += ["-maxrate", f"{rate}k", "-bufsize", f"{buffer}k", "-bf", "0"]
        else:
            x265_params += [f"vbv-maxrate={rate}", f"vbv-bufsize={buffer}", "bframes=0"]
    else:
        raise ValueError(f"{setting.name}: needs a CRF or a bitrate")
    if setting.codec == "libx265":
        args += ["-x265-params", ":".join(x265_params)]
    return args, ".mp4"
