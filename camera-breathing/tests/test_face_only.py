"""Phase 5 on synthetic stand-ins: the face-only crop, M4 (head motion), model inputs, and
the face-only classic experiment and the training-data script end to end."""

import json
import sys
from dataclasses import replace

import numpy as np
import pytest
import yaml
from conftest import PROJECT_DIR, load_script, write_scamps_mat, write_tar_gz
from test_baseline import SPEC, write_ubfc_style_subject
from test_compression_experiment import counting_pose, experiment, small_protocol

from breathing import compression, model_input, pipeline, roi, sources
from breathing.synth import breathing_displacement, make_video, synth_landmarks

PROTOCOL = pipeline.load_protocol()
FACE_CFG = PROTOCOL["regions"]["face_only"]
# A bigger face than SPEC's, clear of the shoulder band, which moves 0.5 px up and down with
# breathing.
FACE_SPEC = replace(
    SPEC,
    height=96,
    width=96,
    face_center_frac=(0.3, 0.5),
    face_radius_frac=(0.13, 0.13),
    head_breathing_amplitude_px=0.5,
)
experiment_script = load_script("compression_experiment")
data_script = load_script("make_training_data")


def face_only(spec=FACE_SPEC, landmarks=None):
    lm = synth_landmarks(spec) if landmarks is None else landmarks
    lm = np.repeat(lm[None], 10, axis=0)
    return roi.face_only_regions(lm, spec.width, spec.height, 0.5, 0.5, FACE_CFG)


# ---------------------------------------------------------------------------------------
# Crop and M4


def test_face_only_crop_is_a_square_above_the_shoulders():
    crop, boxes, info = face_only()
    r0, r1, c0, c1 = crop
    side = r1 - r0
    assert side == c1 - c0 and side % 2 == 0 and side >= 16
    assert 0 <= r0 and r1 <= FACE_SPEC.height and 0 <= c0 and c1 <= FACE_SPEC.width
    fw = info["face_width_px"]  # ears at the ellipse's sides
    assert fw == pytest.approx(2 * FACE_SPEC.face_radius[1])
    # No shoulder in view: the crop ends above the M1 shoulder band, by the margin.
    lm = np.repeat(synth_landmarks(FACE_SPEC)[None], 10, axis=0)
    band = roi.measurement_regions(lm, FACE_SPEC.width, FACE_SPEC.height)[0]["shoulder"]
    assert r1 <= band[0] - FACE_CFG["shoulder_margin"] * fw
    face_top = FACE_SPEC.face_center[0] - FACE_SPEC.face_radius[0]
    face_bottom = FACE_SPEC.face_center[0] + FACE_SPEC.face_radius[0]
    assert r0 <= face_top and face_bottom <= r1  # the whole face is in view
    assert info["crop_source"] == "pose"
    for name in ("head", "face"):
        b0, b1, d0, d1 = boxes[name]
        assert 0 <= b0 < b1 <= side and 0 <= d0 < d1 <= side, name
    # The face box sits inside the head box, both in crop coordinates.
    (h0, h1, k0, k1), (f0, f1, g0, g1) = boxes["head"], boxes["face"]
    assert h0 <= f0 and f1 <= h1 and k0 <= g0 and g1 <= k1


def test_face_only_crop_keeps_its_bottom_edge_when_the_head_is_near_the_top():
    high = replace(FACE_SPEC, face_center_frac=(0.15, 0.5))  # crop would start above row 0
    crop, _, info = face_only(high)
    lm = np.repeat(synth_landmarks(high)[None], 10, axis=0)
    band = roi.measurement_regions(lm, high.width, high.height)[0]["shoulder"]
    mouth_y = high.face_center[0] + 0.5 * high.face_radius[0]  # synth_landmarks
    bottom = min(mouth_y + FACE_CFG["below_mouth"] * info["face_width_px"],
                 band[0] - FACE_CFG["shoulder_margin"] * info["face_width_px"])  # fmt: skip
    eye_y = high.face_center[0] - 0.25 * high.face_radius[0]
    assert eye_y - FACE_CFG["above_eyes"] * info["face_width_px"] < 0  # the top is cut ...
    assert crop[1] == int(np.floor(bottom)) and crop[0] <= 1  # ... the bottom stays (even side)


def test_face_only_crop_without_shoulders_or_face():
    lm = synth_landmarks(FACE_SPEC)
    lm[[roi.LEFT_SHOULDER, roi.RIGHT_SHOULDER], 2] = 0.0  # not visible: no cap
    crop, _, info = face_only(landmarks=lm)
    capped, _, _ = face_only()
    assert crop[1] - crop[0] >= capped[1] - capped[0]
    assert info["shoulders_visible_frac"] == 0.0
    lm[roi.NOSE, 2] = 0.0
    crop, boxes, info = face_only(landmarks=lm)
    assert crop is None and boxes == {"head": None, "face": None}
    assert info["crop_source"] == "none"


def test_m4_measures_breathing_from_head_motion_on_the_crop():
    frames = make_video(FACE_SPEC)
    crop, boxes, _ = face_only()
    video = sources.Video("v", "v", FACE_SPEC.fps, (FACE_SPEC.height, FACE_SPEC.width),
                          lambda: iter(frames), None)  # fmt: skip
    face = sources.cropped(video, crop)
    assert face.size_hw == (crop[1] - crop[0], crop[3] - crop[2])
    windows = pipeline.analysis_windows(FACE_SPEC.n_frames / FACE_SPEC.fps, PROTOCOL, ("M4",))
    refs = {"M4": [60 * FACE_SPEC.breathing_hz]}
    record = pipeline.process_version(
        compression.SOURCE, face.frames, face.fps, face.size_hw, FACE_SPEC.n_frames, boxes,
        windows, refs, PROTOCOL, "unused", measures=("M4",),
    )  # fmt: skip
    (w,) = record["windows"]
    assert w["measure"] == "M4" and abs(w["error_bpm"]) < 0.5, w
    # Without head motion the crop holds no breathing motion: the shoulders are cut off.
    still = make_video(replace(FACE_SPEC, head_breathing_amplitude_px=0.0))
    still_face = sources.cropped(replace(video, frames=lambda: iter(still)), crop)
    record = pipeline.process_version(
        compression.SOURCE, still_face.frames, face.fps, face.size_hw, FACE_SPEC.n_frames,
        boxes, windows, refs, PROTOCOL, "unused", measures=("M4",),
    )  # fmt: skip
    assert record["windows"][0]["snr_db"] < w["snr_db"]


def test_encoded_version_is_deleted_after_the_block(tmp_path):
    frames = make_video(replace(SPEC, n_frames=30))
    setting = compression.make_setting("bitrate", "libx264", kbps=250)
    args = (lambda: iter(frames), SPEC.fps, (SPEC.height, SPEC.width), 30, PROTOCOL, tmp_path)
    with pipeline.encoded_version(setting, *args) as version:
        assert version.encoded and version.info["keyframes"] == 1
        assert len(list(tmp_path.iterdir())) == 1
        assert sum(1 for _ in version.frames()) == 30
    assert list(tmp_path.iterdir()) == []
    with pipeline.encoded_version(compression.SOURCE, *args) as version:
        assert not version.encoded and version.info == {}
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------
# Model inputs


def test_area_resampling():
    line = np.array([3.0, 6.0, 9.0, 12.0])
    np.testing.assert_allclose(model_input.area_axis(line, 2, 0), [4.5, 10.5])  # pairs
    np.testing.assert_allclose(model_input.area_axis(line[:3], 2, 0), [4.0, 8.0])  # 2/3 + 1/3
    np.testing.assert_allclose(model_input.area_axis(line[:2], 4, 0), [3, 3, 6, 6])  # upscale
    rng = np.random.default_rng(0)
    for n_in, n_out in ((37, 8), (8, 37), (64, 64)):
        x = rng.uniform(0, 255, n_in)
        np.testing.assert_allclose(model_input.area_axis(x, n_out, 0).mean(), x.mean())
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    small = model_input.resize(frame[:, :4], 2)
    np.testing.assert_allclose(small, frame[:, :4].reshape(2, 2, 2, 2, 3).mean(axis=(1, 3)))
    assert model_input.resize(frame, 5).shape == (5, 5, 3)


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        (30.0, [1, 4, 7, 10]),  # the mean of each three frames
        (15.0, [0.5, 2, 3.5, 5]),  # frames 0-1, 2, 3-4, 5
        (5.0, [0, 0, 1, 1]),  # each frame held for two intervals
    ],
)
def test_resampling_to_the_model_frame_rate(fps, expected):
    n = int(round(2 * fps))  # two seconds
    frames = [np.full((5, 7, 3), i, dtype=np.uint8) for i in range(n)]
    n_out = model_input.n_output(n, fps, 10.0)
    assert n_out == 20
    out = model_input.model_frames(frames, fps, n_out, 10.0, 4)
    assert out.shape == (20, 4, 4, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out[:4, 0, 0, 0], np.rint(expected))
    signal = model_input.model_signal(np.arange(n), fps, n_out, 10.0)
    np.testing.assert_allclose(signal[:4], expected)


def test_augmentation_draws_are_lossy_distinct_and_repeatable():
    settings = compression.expand_ladders(PROTOCOL["compression"])
    groups = [{"ladders": ["crf", "bitrate"], "n": 5}, {"ladders": ["allocation"], "n": 3}]
    pools = model_input.augmentation_pools(settings, groups)
    assert all(s.crf != 0 and s.codec in ("libx264", "libx265") for s in pools[0] + pools[1])
    assert len(pools[0]) == 2 * 6 + 2 * 5  # CRF 18-43 and five bitrates, two codecs
    assert all((s.decimate, s.scale) != (1, 1) for s in pools[1])
    a = model_input.draw_settings(pools, [5, 3], 0, "scamps_videos/P000201.mat")
    assert len({s.name for s in a}) == 8
    assert a == model_input.draw_settings(pools, [5, 3], 0, "scamps_videos/P000201.mat")
    draws = {
        tuple(s.name for s in model_input.draw_settings(pools, [5, 3], 0, f"P{i:06d}.mat"))
        for i in range(201, 211)
    }
    assert len(draws) > 1
    with pytest.raises(ValueError):
        model_input.augmentation_pools(settings, [{"ladders": ["crf"], "n": 13}])


# ---------------------------------------------------------------------------------------
# Scripts end to end


def test_face_only_classic_experiment_on_ubfc_style_video(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(experiment_script, "make_pose", counting_pose(FACE_SPEC, calls))
    root = tmp_path / "UBFC"
    write_ubfc_style_subject(root / "DATASET_2" / "subject1", FACE_SPEC)
    config = experiment(tmp_path, "baseline_ubfc.yaml", framing="face",
                        measures=["M2", "M3", "M4"], ladders=["crf"], relocate=[],
                        n_videos=None)  # fmt: skip
    run_id = "20261126_ubfc_face-ladders_s0"
    argv = ["--config", str(config), "--run-id", run_id, "--source", str(root)]
    argv += ["--protocol", str(small_protocol(tmp_path)), "--jobs", "2"]
    argv += ["--results-root", str(tmp_path / "results"), "--work-dir", str(tmp_path / "work")]
    assert experiment_script.main(argv + ["--limit", "1"]) == 0
    run = tmp_path / "results" / run_id
    lines = (run / "items.jsonl").read_text().splitlines()
    records = {r["id"]: r for r in (experiment_script.unpack(json.loads(x)) for x in lines)}
    assert len(records) == 3 and all(r["status"] == "ok" for r in records.values())
    source = records["DATASET_2/subject1/source"]
    crop, _, _ = face_only()
    assert tuple(source["regions"]["crop"]) == crop
    assert source["height"] == source["width"] == crop[1] - crop[0]
    assert set(source["reference"]) == {"M2", "M3", "M4"}
    assert {w["measure"] for r in records.values() for w in r["windows"]} == {"M2", "M3", "M4"}
    m4 = [w for w in source["windows"] if w["measure"] == "M4"]
    assert m4 and all(w["excluded"] is None and abs(w["error_bpm"]) < 1.0 for w in m4), m4

    # A resumed run measures the missing setting on the recorded crop, without pose.
    kept = [x for x in lines if json.loads(x)["id"] != "DATASET_2/subject1/x264_crf38"]
    (run / "items.jsonl").write_text("\n".join(kept) + "\n")
    calls.clear()
    assert experiment_script.main(argv) == 0
    assert calls == []
    last = json.loads((run / "items.jsonl").read_text().splitlines()[-1])
    assert last["id"] == "DATASET_2/subject1/x264_crf38" and last["height"] == crop[1] - crop[0]
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics["per_setting"]["source"]) == {"M2", "M3", "M4"}
    assert "without_face_anchored" not in metrics["sensitivity_breakdown"]
    assert "Framing: face-only crop. Measures: M2, M3, M4." in (run / "summary.md").read_text()

    bad = experiment(tmp_path, "baseline_ubfc.yaml", framing="face", ladders=["crf"],
                     relocate=["x264_crf38"])  # fmt: skip
    with pytest.raises(SystemExit):
        experiment_script.load_config(bad)


def test_real_phase5_configs_load():
    for name in ("face_ladders_scamps.yaml", "face_ladders_ubfc.yaml"):
        config = experiment_script.load_config(PROJECT_DIR / "configs" / name)
        assert experiment_script.measures_of(config, PROTOCOL) == ("M2", "M3", "M4")
        settings = experiment_script.experiment_settings(config, PROTOCOL)
        assert len(settings) == 25
    config = data_script.load_config(PROJECT_DIR / "configs" / "deep_data_scamps.yaml")
    assert data_script.split_of("scamps_videos/P000100.mat", config["splits"]) == "test"
    assert data_script.split_of("scamps_videos/P000101.mat", config["splits"]) == "validation"
    assert data_script.split_of("scamps_videos/P002800.mat", config["splits"]) == "train"
    assert data_script.split_of("scamps_videos/P002801.mat", config["splits"]) is None
    all_settings = compression.expand_ladders(PROTOCOL["compression"])
    fixed = data_script.video_settings("scamps_videos/P000001.mat", "test", config, all_settings)
    assert len(fixed) == 25 and fixed[0] == compression.SOURCE
    drawn = data_script.video_settings("scamps_videos/P000201.mat", "train", config, all_settings)
    assert len(drawn) == 9 and drawn[0] == compression.SOURCE


def test_overlapping_splits_are_refused(tmp_path):
    config = {"dataset_config": str(PROJECT_DIR / "configs" / "baseline_scamps.yaml"),
              "splits": {"a": {"first": 1, "last": 10, "versions": "fixed"},
                         "b": {"first": 10, "last": 20, "versions": "random"}}}  # fmt: skip
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit, match="overlap"):
        data_script.load_config(path)


def face_archive(tmp_path, n_members):
    frames = make_video(FACE_SPEC)
    files = {}
    for i in range(1, n_members + 1):
        path = tmp_path / f"P{i:06d}.mat"
        n = FACE_SPEC.n_frames
        extra = {"d_pitch": np.zeros(n), "d_roll": np.zeros(n), "d_yaw": np.zeros(n)}
        if i == 4:
            extra["d_yaw"] = np.linspace(-20.0, 20.0, n)  # head motion: kept, flagged
        write_scamps_mat(path, frames, FACE_SPEC, extra=extra)
        files[f"scamps_videos/{path.name}"] = path
    archive = tmp_path / "scamps_videos.tar.gz"
    write_tar_gz(archive, files)
    return archive, frames


# numpy and scipy start idle OpenBLAS thread pools when they are imported, so Python warns
# that the process forking the workers is multi-threaded. The workers are forked before the
# main process does any numerical work, and the test checks their output.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_training_data_splits_versions_and_resume(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data_script, "make_pose", counting_pose(FACE_SPEC, calls))
    # Worker processes look up the worker function by module name (from the command line
    # the script is __main__).
    monkeypatch.setitem(sys.modules, "make_training_data", data_script)
    archive, frames = face_archive(tmp_path, 5)
    config = {
        "dataset_config": str(PROJECT_DIR / "configs" / "baseline_scamps.yaml"),
        "model_input": {"size": 16, "fps": 10.0},
        "splits": {  # P000003 belongs to no split
            "test": {"first": 1, "last": 1, "versions": "fixed"},
            "validation": {"first": 2, "last": 2, "versions": "fixed"},
            "train": {"first": 4, "last": 5, "versions": "random"},
        },
        "fixed_ladders": ["crf", "bitrate"],
        "augmentation": {"seed": 0, "groups": [{"ladders": ["crf", "bitrate"], "n": 2},
                                               {"ladders": ["allocation"], "n": 1}]},
        "pose_model_url": (tmp_path / "pose.task").as_uri(),  # recorded only; pose is faked
    }  # fmt: skip
    (tmp_path / "pose.task").write_bytes(b"not a model")
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_id = "20261126_scamps_deep-data_s0"
    data_root = tmp_path / "deep"
    argv = ["--config", str(config_path), "--run-id", run_id, "--source", str(archive)]
    argv += ["--protocol", str(small_protocol(tmp_path)), "--data-root", str(data_root)]
    argv += ["--results-root", str(tmp_path / "results"), "--work-dir", str(tmp_path / "work")]
    assert data_script.main(argv + ["--jobs", "2"]) == 0  # two worker processes

    run = tmp_path / "results" / run_id
    records = {r["id"]: r for r in map(json.loads, (run / "items.jsonl").read_text().splitlines())}
    assert sorted(records) == [f"scamps_videos/P00000{i}.mat" for i in (1, 2, 4, 5)]
    assert all(r["status"] == "ok" for r in records.values())
    files = sorted(p.relative_to(data_root).as_posix() for p in data_root.rglob("*"))
    assert files == ["test", "test/P000001.npz", "train", "train/P000004.npz",
                     "train/P000005.npz", "validation", "validation/P000002.npz"]  # fmt: skip

    crop, _, _ = face_only()
    r0, r1, c0, c1 = crop
    with np.load(data_root / "test" / "P000001.npz") as npz:
        assert list(npz["settings"]) == ["source", "x264_crf18", "x264_crf38", "x264_br250"]
        assert npz["frames"].shape == (4, 200, 16, 16, 3) and npz["frames"].dtype == np.uint8
        assert np.isnan(npz["kbps"][0]) and np.all(npz["kbps"][1:] > 0)
        expected = model_input.model_frames((f[r0:r1, c0:c1] for f in frames), 30.0, 200, 10.0, 16)
        np.testing.assert_array_equal(npz["frames"][0], expected)
        assert np.abs(npz["frames"][1].astype(int) - expected).mean() < 3  # CRF 18: close
        label = model_input.model_signal(breathing_displacement(FACE_SPEC), 30.0, 200, 10.0)
        np.testing.assert_allclose(npz["label_breathing"], label, rtol=1e-6, atol=1e-6)
        assert npz["label_ppg"].shape == (200,)
        meta = json.loads(str(npz["meta"]))
        assert tuple(meta["crop"]) == crop and meta["head_still"] is True
    train = records["scamps_videos/P000004.mat"]
    assert train["head_still"] is False and len(train["settings"]) == 4
    small = pipeline.load_protocol(small_protocol(tmp_path))
    all_settings = compression.expand_ladders(small["compression"])
    loaded = data_script.load_config(config_path)
    drawn = data_script.video_settings(train["id"], "train", loaded, all_settings)
    assert train["settings"] == [s.name for s in drawn]
    with np.load(data_root / "train" / "P000004.npz") as npz:
        versions = json.loads(str(npz["meta"]))["versions"]
    assert [v["setting"] for v in versions] == train["settings"]
    for v in versions:
        setting = next(s for s in all_settings if s.name == v["setting"])
        assert v["fps"] == 30.0 / setting.decimate
        assert v["height"] == compression.output_size(r1 - r0, c1 - c0, setting.scale)[1]

    metrics = json.loads((run / "metrics.json").read_text())
    assert (
        metrics["splits"]["train"]["videos"] == 2 and metrics["splits"]["train"]["head_still"] == 1
    )
    assert sum(metrics["random_versions_drawn"].values()) == 6
    assert list((tmp_path / "work" / "stage" / "held").iterdir()) == []
    assert list((tmp_path / "work" / "versions").iterdir()) == []
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["sessions"][0]["pose_model"]["file"] == "pose.task"

    # Resume in the main process (--jobs 1): a lost video is made again, the others are
    # skipped without pose, and the data are the same as from the worker processes.
    lines = [x for x in (run / "items.jsonl").read_text().splitlines()
             if json.loads(x)["id"] != "scamps_videos/P000005.mat"]  # fmt: skip
    (run / "items.jsonl").write_text("\n".join(lines) + "\n")
    lost = data_root / "train" / "P000005.npz"
    with np.load(lost) as npz:
        before = {k: npz[k] for k in npz.files}
    lost.unlink()
    calls.clear()
    assert data_script.main(argv + ["--jobs", "1"]) == 0
    assert len(calls) == 40  # one video: 600 frames at 30 fps, pose on every 15th
    with np.load(lost) as npz:
        assert sorted(npz.files) == sorted(before)
        for k in npz.files:
            np.testing.assert_array_equal(npz[k], before[k], err_msg=k)
    # Complete: the archive is not opened again.
    archive.unlink()
    assert data_script.main(argv + ["--jobs", "2"]) == 0


def test_pack_split_and_version_choice(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        meta = {"video": f"scamps_videos/P{i:06d}.mat", "head_still": i != 1}
        np.savez_compressed(
            tmp_path / f"P{i:06d}.npz",
            frames=rng.integers(0, 255, (2, 5, 4, 4, 3), dtype=np.uint8),
            settings=np.array(["source", "x264_crf23"]),
            kbps=np.array([np.nan, 10.0]),
            label_breathing=np.full(5, float(i), dtype=np.float32),
            meta=np.array(json.dumps(meta)),
        )
    files = sorted(tmp_path.glob("*.npz"))
    index = model_input.pack_split(files, tmp_path / "packed", "breathing", jobs=2)
    frames, labels, loaded = model_input.load_packed(tmp_path / "packed")
    assert frames.shape == (3, 2, 5, 4, 4, 3) and loaded == index
    with np.load(files[2]) as z:
        np.testing.assert_array_equal(frames[2], z["frames"])
    np.testing.assert_array_equal(labels[:, 0], [0, 1, 2])
    assert [v["head_still"] for v in index["videos"]] == [True, False, True]
    before = (tmp_path / "packed" / "frames.npy").stat().st_mtime_ns
    model_input.pack_split(files, tmp_path / "packed", "breathing")  # same files: reused
    assert (tmp_path / "packed" / "frames.npy").stat().st_mtime_ns == before

    assert {model_input.version_choice("source", 9, 0, e, 3) for e in range(20)} == {0}
    drawn = [model_input.version_choice("all", 9, 0, e, 3) for e in range(50)]
    assert set(drawn) == set(range(9))
    assert drawn == [model_input.version_choice("all", 9, 0, e, 3) for e in range(50)]


def test_a_waveform_goes_through_the_protocol_estimator():
    fs, n = 10.0, 200
    wave = np.sin(2 * np.pi * 0.3 * np.arange(n) / fs)
    (w,) = pipeline.measure_waveform("M4", wave, fs, [(0.0, 20.0)], [18.0], PROTOCOL)
    assert w["measure"] == "M4" and w["excluded"] is None
    assert w["est_bpm"] == pytest.approx(18.0, abs=0.2)
    (w,) = pipeline.measure_waveform("M4", wave, fs, [(0.0, 20.0)], [None], PROTOCOL)
    assert w["excluded"] == "no_reference" and w["est_bpm"] == pytest.approx(18.0, abs=0.2)
