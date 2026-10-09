"""Video reading and writing through ffmpeg/ffprobe.

`parse_ffprobe` and `parse_rate` are pure; the rest are thin subprocess adapters.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    codec_name: str | None
    codec_tag: str | None
    pix_fmt: str | None
    width: int
    height: int
    fps: float | None  # average frame rate
    r_frame_rate: str | None
    nb_frames: int | None  # from the container header; may be missing
    duration_s: float | None
    bit_rate: int | None
    format_name: str | None
    size_bytes: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_rate(text: str | None) -> float | None:
    """Parse an ffprobe rate such as '30000/1001'. Returns None for '0/0' or missing."""
    if not text:
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return float(value) if value > 0 else None


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ffprobe(data: dict) -> VideoInfo:
    """Turn `ffprobe -show_streams -show_format -of json` output into a VideoInfo."""
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError("no video stream")
    stream = streams[0]
    fmt = data.get("format", {})
    duration = _to_float(stream.get("duration")) or _to_float(fmt.get("duration"))
    bit_rate = _to_int(stream.get("bit_rate")) or _to_int(fmt.get("bit_rate"))
    return VideoInfo(
        codec_name=stream.get("codec_name"),
        codec_tag=stream.get("codec_tag_string"),
        pix_fmt=stream.get("pix_fmt"),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate")),
        r_frame_rate=stream.get("r_frame_rate"),
        nb_frames=_to_int(stream.get("nb_frames")),
        duration_s=duration,
        bit_rate=bit_rate,
        format_name=fmt.get("format_name"),
        size_bytes=_to_int(fmt.get("size")),
    )


def ffprobe(path: str | Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def probe(path: str | Path) -> VideoInfo:
    return parse_ffprobe(ffprobe(path))


def iter_frames(path: str | Path, width: int, height: int) -> Iterator[np.ndarray]:
    """Decode every frame as (H, W, 3) uint8 RGB, without dropping or duplicating frames."""
    cmd = ["ffmpeg", *"-v error -nostdin -i".split(), str(path)]
    cmd += "-vsync passthrough -f rawvideo -pix_fmt rgb24 -".split()
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr.strip()}")


def write_video(
    frames: Iterable[np.ndarray],
    path: str | Path,
    fps: float,
    codec_args: tuple[str, ...] = ("-c:v", "rawvideo", "-pix_fmt", "bgr24"),
) -> None:
    """Encode RGB uint8 frames. The default is uncompressed AVI, like UBFC-rPPG."""
    frames = iter(frames)
    first = next(frames)
    height, width = first.shape[:2]
    cmd = ["ffmpeg", *"-v error -nostdin -y -f rawvideo -pix_fmt rgb24".split()]
    cmd += ["-s", f"{width}x{height}", "-r", str(fps), "-i", "-", *codec_args, str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(np.ascontiguousarray(first, dtype=np.uint8).tobytes())
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    except BrokenPipeError:
        pass  # ffmpeg exited early; its error message is raised below
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed to write {path}: {stderr.strip()}")


def write_png(image: np.ndarray, path: str | Path) -> None:
    """Write one (H, W, 3) uint8 RGB image as PNG."""
    write_video([image], path, fps=1, codec_args=("-frames:v", "1", "-c:v", "png"))


def tool_version(tool: str = "ffmpeg") -> str | None:
    """First line of `<tool> -version`, or None if the tool is missing."""
    try:
        out = subprocess.run([tool, "-version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.splitlines()[0] if out.stdout else None
