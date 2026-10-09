import numpy as np
import pytest

from breathing.video_io import (
    iter_frames,
    parse_ffprobe,
    parse_rate,
    probe,
    tool_version,
    write_png,
    write_video,
)


@pytest.mark.parametrize(
    "text, expected",
    [("30/1", 30.0), ("30000/1001", 30000 / 1001), ("0/0", None), ("", None), (None, None)],
)
def test_parse_rate(text, expected):
    assert parse_rate(text) == expected


def test_parse_ffprobe_uses_video_stream_and_format_fallbacks():
    data = {
        "streams": [
            {"codec_type": "audio", "codec_name": "aac"},
            {
                "codec_type": "video",
                "codec_name": "rawvideo",
                "codec_tag_string": "[0][0][0][0]",
                "pix_fmt": "bgr24",
                "width": 640,
                "height": 480,
                "avg_frame_rate": "0/0",
                "r_frame_rate": "30/1",
                "nb_frames": "1800",
            },
        ],
        "format": {"format_name": "avi", "duration": "60.0", "bit_rate": "221184000", "size": "9"},
    }
    info = parse_ffprobe(data)
    assert (info.width, info.height, info.fps, info.nb_frames) == (640, 480, 30.0, 1800)
    assert (info.duration_s, info.bit_rate, info.size_bytes) == (60.0, 221184000, 9)
    assert info.codec_name == "rawvideo" and info.pix_fmt == "bgr24"


def test_parse_ffprobe_without_video_stream_fails():
    with pytest.raises(ValueError):
        parse_ffprobe({"streams": [{"codec_type": "audio"}]})


def test_uncompressed_avi_round_trip(tmp_path, spec, frames):
    path = tmp_path / "vid.avi"
    clip = frames[:60]
    write_video(clip, path, spec.fps)
    info = probe(path)
    assert info.codec_name == "rawvideo"
    assert (info.width, info.height) == (spec.width, spec.height)
    assert info.fps == pytest.approx(spec.fps)
    assert info.nb_frames == len(clip)
    decoded = np.array(list(iter_frames(path, info.width, info.height)))
    np.testing.assert_array_equal(decoded, clip)


def test_write_png(tmp_path, frames):
    path = tmp_path / "frame.png"
    write_png(frames[0], path)
    info = probe(path)
    assert info.codec_name == "png"
    np.testing.assert_array_equal(next(iter_frames(path, info.width, info.height)), frames[0])


def test_iter_frames_reports_ffmpeg_errors(tmp_path):
    with pytest.raises(RuntimeError):
        list(iter_frames(tmp_path / "missing.avi", 8, 8))


def test_tool_version():
    assert tool_version("ffmpeg").startswith("ffmpeg version")
    assert tool_version("no-such-tool-xyz") is None
