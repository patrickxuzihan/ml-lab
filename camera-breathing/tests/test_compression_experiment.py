"""Phase-4 compression experiments: estimates without a reference, regions found on the
compressed video, and the script end to end on synthetic stand-ins (a reduced protocol keeps
the ladders short; pose is replaced by known landmarks)."""

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml
from conftest import PROJECT_DIR, load_script
from test_baseline import SPEC, scamps_archive, write_ubfc_style_subject

from breathing import compression, pipeline
from breathing.signals import extract_traces
from breathing.synth import make_video, synth_landmarks

PROTOCOL = pipeline.load_protocol()
script = load_script("compression_experiment")


def regions(spec=SPEC):
    from breathing import roi

    landmarks = np.repeat(synth_landmarks(spec)[None], 10, axis=0)
    return roi.measurement_regions(landmarks, spec.width, spec.height)[0]


def test_estimates_are_kept_for_windows_without_a_usable_reference():
    frames = make_video(SPEC)
    boxes = regions()
    traces = extract_traces(iter(frames), SPEC.fps, {"shoulder": boxes["shoulder"]}, boxes)
    windows = pipeline.analysis_windows(SPEC.n_frames / SPEC.fps, PROTOCOL)
    refs = {"M1": [None], "M2": [60 * SPEC.breathing_hz], "M3": [200.0, 60 * SPEC.heart_hz]}
    rows = {
        (r["measure"], r["window"]): r for r in pipeline.measure(traces, windows, refs, PROTOCOL)
    }
    m1, m3_out = rows[("M1", 0)], rows[("M3", 0)]
    assert m1["excluded"] == "no_reference" and m1["error_bpm"] is None and m1["snr_db"] is None
    assert m1["est_bpm"] == pytest.approx(60 * SPEC.breathing_hz, abs=0.5)
    assert m3_out["excluded"] == "reference_out_of_band" and m3_out["est_bpm"] is not None
    # Metrics still use only the windows that are not excluded.
    record = {"video": "v", "subject": "s", "setting": "source", "windows": list(rows.values())}
    agg = pipeline.aggregate([record], [compression.SOURCE], PROTOCOL)
    assert "M1" not in agg["per_setting"]["source"]
    assert agg["per_setting"]["source"]["M3"]["n_windows"] == 1


def test_relocated_regions_on_a_compressed_version(tmp_path):
    frames = make_video(SPEC)
    boxes = regions()
    windows = pipeline.analysis_windows(SPEC.n_frames / SPEC.fps, PROTOCOL)
    refs = {"M1": [60 * SPEC.breathing_hz], "M2": [60 * SPEC.breathing_hz]}
    refs["M3"] = [60 * SPEC.heart_hz] * 2
    setting = compression.make_setting("bitrate", "libx264", kbps=250)
    calls = []

    def same(frames_iter, fps, size_hw):
        calls.append((sum(1 for _ in frames_iter), fps, size_hw))
        return boxes, {"source": "same boxes"}

    args = (setting, lambda: iter(frames), SPEC.fps, (SPEC.height, SPEC.width), SPEC.n_frames)
    record = pipeline.process_version(
        *args, boxes, windows, refs, PROTOCOL, tmp_path, relocate=same
    )
    assert calls == [(SPEC.n_frames, SPEC.fps, (SPEC.height, SPEC.width))]
    assert record["relocated"]["windows"] == record["windows"]  # same regions, same numbers
    assert record["relocated"]["regions"]["source"] == "same boxes"

    def lost(frames_iter, fps, size_hw):
        return {"shoulder": None, "face": None}, {}

    record = pipeline.process_version(
        *args, boxes, windows, refs, PROTOCOL, tmp_path, relocate=lost
    )
    assert {w["excluded"] for w in record["relocated"]["windows"]} == {"no_region"}
    assert list(tmp_path.iterdir()) == []  # the compressed version was deleted


def small_protocol(tmp_path):
    protocol = pipeline.load_protocol()
    protocol["statistics"]["bootstrap_samples"] = 100  # keeps the summaries quick
    protocol["compression"]["ladders"] = {
        "crf": {"codecs": ["libx264"], "crf": [18, 38]},
        "bitrate": {"codecs": ["libx264"], "kbps": [250]},
        "allocation": {"codecs": ["libx264"], "kbps": [250], "decimate": [1, 2], "scale": [1, 2]},
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    return path


def experiment(tmp_path, dataset_config, **keys):
    config = {"dataset_config": str(PROJECT_DIR / "configs" / dataset_config), **keys}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def counting_pose(spec, calls):
    def make_pose(_config, _work_dir):
        def pose(_frame):
            calls.append(1)
            return synth_landmarks(spec)

        return pose, {"pose_model": "synthetic landmarks"}

    return make_pose


def read(run):
    lines = (run / "items.jsonl").read_text().splitlines()
    return [script.unpack(json.loads(line)) for line in lines]


def test_ubfc_compression_runs_in_parallel_and_resumes_mid_video(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(script, "make_pose", counting_pose(SPEC, calls))
    root = tmp_path / "drive" / "UBFC"
    write_ubfc_style_subject(root / "DATASET_2" / "subject1", SPEC)
    write_ubfc_style_subject(root / "DATASET_2" / "subject2", replace(SPEC, seed=5))
    config = experiment(tmp_path, "baseline_ubfc.yaml", ladders=["crf", "bitrate"],
                        relocate=["x264_br250"], n_videos=None)  # fmt: skip
    run_id = "20261123_ubfc_ladders_s0"
    argv = [
        "--config",
        str(config),
        "--run-id",
        run_id,
        "--protocol",
        str(small_protocol(tmp_path)),
    ]
    argv += ["--results-root", str(tmp_path / "results"), "--work-dir", str(tmp_path / "work")]
    argv += ["--source", str(root), "--jobs", "2"]
    names = ["source", "x264_crf18", "x264_crf38", "x264_br250"]

    assert script.main(argv + ["--limit", "1"]) == 0
    run = tmp_path / "results" / run_id
    records = read(run)
    assert sorted(r["id"] for r in records) == sorted(f"DATASET_2/subject1/{n}" for n in names)
    assert all(r["status"] == "ok" for r in records)

    # Drop one finished setting: the next run measures only that one, with the source's
    # regions (no pose), then moves on to the second subject.
    kept = [line for line in (run / "items.jsonl").read_text().splitlines()
            if json.loads(line)["id"] != "DATASET_2/subject1/x264_crf38"]  # fmt: skip
    (run / "items.jsonl").write_text("\n".join(kept) + "\n")
    calls.clear()
    assert script.main(argv + ["--limit", "1"]) == 0
    assert calls == []
    assert read(run)[-1]["id"] == "DATASET_2/subject1/x264_crf38"
    assert script.main(argv) == 0
    assert script.main(argv) == 0  # nothing left to do

    records = {r["id"]: r for r in read(run)}
    assert len(records) == 8 and all(r["status"] == "ok" for r in records.values())
    source = records["DATASET_2/subject1/source"]
    assert source["regions"]["shoulder_source"] == "pose" and "flags" in source
    assert set(source["reference"]) == {"M1", "M2", "M3"}
    for r in records.values():
        assert "codec" not in r and "boxes" not in r  # implied by the setting / the source
        if r["setting"] == "source":  # compressed versions of this tiny video lose the pulse
            for w in r["windows"]:
                assert w["excluded"] is None and abs(w["error_bpm"]) < 1.0, (r["id"], w)
    assert "relocated" in records["DATASET_2/subject2/x264_br250"]
    assert "relocated" not in records["DATASET_2/subject2/x264_crf18"]
    assert records["DATASET_2/subject1/x264_crf18"]["keyframes"] == 1

    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["counts"] == {"videos": 2, "versions_measured": 8, "errors": 0, "settings": 4}
    assert set(metrics["paired_vs_source"]) == {"x264_crf18", "x264_crf38", "x264_br250"}
    assert {"x264_crf", "x264_bitrate"} <= set(metrics["breakdown"])
    assert set(metrics["relocated_minus_fixed"]) == {"x264_br250"}
    assert metrics["sensitivity_breakdown"]["without_timestamp_flags"]["n_removed"] == 0
    assert metrics["encoded_versions_with_more_than_one_keyframe"] == 0
    summary = (run / "summary.md").read_text()
    for heading in ("## Paired difference", "## Breakdown points", "## Regions found on the",
                    "## Breakdown points without some videos", "## Time"):  # fmt: skip
        assert heading in summary, heading
    assert list((tmp_path / "work" / "stage").iterdir()) == []
    assert list((tmp_path / "work" / "versions").iterdir()) == []

    before = (run / "metrics.json").read_text()
    assert script.main(argv + ["--summarize-only"]) == 0
    assert (run / "metrics.json").read_text() == before


def test_scamps_compression_uses_the_head_still_videos(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(script, "make_pose", counting_pose(SPEC, calls))
    archive = scamps_archive(tmp_path, [False, True, False, False], [False, True, False, False])
    config = experiment(tmp_path, "baseline_scamps.yaml", ladders=["crf"], relocate=[],
                        n_videos=2)  # fmt: skip
    argv = ["--config", str(config), "--run-id", "20261123_scamps_ladders_s0"]
    argv += ["--source", str(archive), "--protocol", str(small_protocol(tmp_path))]
    argv += ["--results-root", str(tmp_path / "results"), "--work-dir", str(tmp_path / "work")]
    assert script.main(argv) == 0
    assert script.main(argv) == 0  # complete: the archive is not read again
    run = tmp_path / "results" / "20261123_scamps_ladders_s0"
    records = read(run)
    status = {r["id"].split("/", 1)[-1]: r["status"] for r in records}
    assert status == {
        "P000001.mat/source": "ok",
        "P000001.mat/x264_crf18": "ok",
        "P000001.mat/x264_crf38": "ok",
        "P000002.mat": "skipped",
        "P000003.mat/source": "ok",
        "P000003.mat/x264_crf18": "ok",
        "P000003.mat/x264_crf38": "ok",
    }
    source = next(r for r in records if r["id"].endswith("P000001.mat/source"))
    assert source["layout"]["interpretation"] == "matlab_order"
    for w in source["windows"]:
        assert w["excluded"] is None and abs(w["error_bpm"]) < 0.5
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["counts"]["videos"] == 2
    assert set(metrics["breakdown"]) == {"x264_crf"}
