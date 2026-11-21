"""Compression settings, ffmpeg arguments, and encoded versions measured end to end."""

from dataclasses import replace

import numpy as np
import pytest
from test_measurements import M_SPEC, regions

from breathing import compression, pipeline
from breathing.synth import breathing_displacement, heart_modulation, make_video

PROTOCOL = pipeline.load_protocol()
CFG = PROTOCOL["compression"]
SETTINGS = compression.expand_ladders(CFG)


def by_name(name: str) -> compression.Setting:
    return compression.select(SETTINGS, [name])[0]


def test_ladders_follow_the_protocol():
    names = [s.name for s in SETTINGS]
    assert len(names) == len(set(names))
    assert names[0] == "source"
    counts = {}
    for s in SETTINGS:
        counts[s.ladder] = counts.get(s.ladder, 0) + 1
    assert counts == {"source": 1, "crf": 14, "bitrate": 10, "allocation": 72, "raw": 11}
    groups = compression.ladder_groups(SETTINGS)
    assert groups["x264_crf"] == [f"x264_crf{v}" for v in (0, 18, 23, 28, 33, 38, 43)]
    assert groups["x265_bitrate"] == [f"x265_br{v}" for v in (2000, 1000, 500, 250, 125)]
    assert "x264_br500_d6_s4" in names and "raw_d3_s1" in names
    with pytest.raises(KeyError):
        compression.select(SETTINGS, ["x264_crf99"])


def test_crf_arguments():
    args, ext = compression.ffmpeg_args(by_name("x264_crf23"), 160, 120, CFG)
    assert ext == ".mp4"
    text = " ".join(args)
    assert "-c:v libx264" in text and "-crf 23" in text and "-pix_fmt yuv420p" in text
    assert "-g 100000 -sc_threshold 0" in text
    assert "zerolatency" not in text and "-vf" not in text

    args, _ = compression.ffmpeg_args(by_name("x265_crf28"), 160, 120, CFG)
    params = args[args.index("-x265-params") + 1]
    assert "keyint=100000" in params and "scenecut=0" in params


def test_bitrate_arguments():
    text = " ".join(compression.ffmpeg_args(by_name("x264_br500"), 160, 120, CFG)[0])
    assert "-b:v 500k" in text and "-maxrate 500k" in text and "-bufsize 500k" in text
    assert "-tune zerolatency" in text and "-bf 0" in text

    args, _ = compression.ffmpeg_args(by_name("x265_br125"), 160, 120, CFG)
    params = args[args.index("-x265-params") + 1]
    assert "vbv-maxrate=125" in params and "vbv-bufsize=125" in params
    assert "bframes=0" in params and "-tune" in args


def test_allocation_and_raw_arguments():
    args, _ = compression.ffmpeg_args(by_name("x264_br250_d2_s4"), 640, 480, CFG)
    assert args[:2] == ["-vf", "scale=160:120:flags=area"]
    args, ext = compression.ffmpeg_args(by_name("raw_d2_s2"), 640, 480, CFG)
    assert ext == ".avi" and "rawvideo" in args and "scale=320:240:flags=area" in args


def test_output_size_and_decimation():
    assert compression.output_size(640, 480, 4) == (160, 120)
    assert compression.output_size(161, 121, 2) == (80, 60)
    assert compression.output_size(320, 240, 1) == (320, 240)
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(7)]
    kept = [int(f[0, 0, 0]) for f in compression.decimate(frames, 3)]
    assert kept == [0, 3, 6] and compression.n_decimated(7, 3) == 3


@pytest.fixture(scope="module")
def short_video():
    spec = replace(M_SPEC, n_frames=300)  # 10 s
    frames = make_video(spec)
    boxes, _ = regions(spec)
    windows = pipeline.analysis_windows(10.0, PROTOCOL)
    waves = {"breathing": breathing_displacement(spec), "heart": heart_modulation(spec)}
    refs = pipeline.waveform_references(waves, spec.fps, windows, PROTOCOL)
    return spec, frames, boxes, windows, refs


@pytest.mark.parametrize("name", ["x264_crf18", "x265_br500", "x264_br250_d2_s2", "raw_d3_s2"])
def test_encoded_versions_are_measured_and_deleted(short_video, tmp_path, name):
    spec, frames, boxes, windows, refs = short_video
    setting = by_name(name)
    record = pipeline.process_version(
        setting,
        lambda: iter(frames),
        spec.fps,
        (spec.height, spec.width),
        spec.n_frames,
        boxes,
        windows,
        refs,
        PROTOCOL,
        tmp_path,
    )
    width, height = compression.output_size(spec.width, spec.height, setting.scale)
    assert (record["width"], record["height"]) == (width, height)
    assert record["fps"] == pytest.approx(spec.fps / setting.decimate)
    assert record["n_frames"] == compression.n_decimated(spec.n_frames, setting.decimate)
    assert record["bytes"] > 0 and record["kbps"] > 0
    if setting.codec != "rawvideo":
        assert record["keyframes"] == 1  # no periodic or scene-cut keyframes
    assert list(tmp_path.iterdir()) == []  # the compressed file is deleted
    m1 = next(w for w in record["windows"] if w["measure"] == "M1")
    assert abs(m1["error_bpm"]) < 0.5  # large, high-contrast motion survives these settings


def test_source_version_is_not_encoded(short_video, tmp_path):
    spec, frames, boxes, windows, refs = short_video
    record = pipeline.process_version(
        compression.SOURCE,
        lambda: iter(frames),
        spec.fps,
        (spec.height, spec.width),
        spec.n_frames,
        boxes,
        windows,
        refs,
        PROTOCOL,
        tmp_path,
    )
    assert "bytes" not in record and record["n_frames"] == spec.n_frames
    assert all(abs(w["error_bpm"]) < 0.5 for w in record["windows"])
