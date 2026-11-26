"""Phase-3 baseline: references, extra measurements, reference reliability, and the script end
to end on synthetic stand-ins for UBFC-rPPG and SCAMPS (pose replaced by known landmarks)."""

import csv
import json
import re
from dataclasses import replace

import numpy as np
import pytest
import yaml
from conftest import PROJECT_DIR, load_script, write_scamps_mat, write_tar_gz

from breathing import baseline, pipeline, reference
from breathing.synth import SynthSpec, make_video, synth_landmarks, synthetic_ppg
from breathing.video_io import write_video

PROTOCOL = pipeline.load_protocol()
script = load_script("baseline")

# 20 s clips with sub-pixel breathing motion, color changes of about one 8-bit level and noise.
SPEC = SynthSpec(
    n_frames=600,
    fps=30.0,
    height=64,
    width=96,
    breathing_hz=0.27,
    heart_hz=1.35,
    breathing_amplitude_px=0.4,
    heart_amplitude=1.0,
    breathing_color_amplitude=0.8,
    noise_std=1.0,
    seed=3,
    face_center_frac=(0.25, 0.5),
    face_radius_frac=(0.15, 0.12),
    shoulder_half_width_frac=0.15,
)
PPG = {"baseline": 0.1, "amplitude_depth": 0.1, "interval_depth": 0.05, "noise_std": 0.05}


def fake_make_pose(spec: SynthSpec):
    def make_pose(_config, _work_dir):
        return (lambda _frame: synth_landmarks(spec)), {"pose_model": "synthetic landmarks"}

    return make_pose


def test_ubfc_references_from_a_synthetic_contact_ppg():
    fs, n = 30.0, 1800  # one minute, like UBFC-rPPG
    ppg = synthetic_ppg(n, fs, 1.2, 0.25, **PPG)
    hr = np.concatenate([np.full(n // 2, 72.0), np.full(n // 2, 1.0)])  # invalid second half
    windows = pipeline.analysis_windows(n / fs, PROTOCOL)
    refs, details = baseline.ubfc_references(ppg, hr, fs, windows, PROTOCOL)
    assert len(refs["M1"]) == 2 and len(refs["M3"]) == 6
    assert refs["M1"] == refs["M2"]
    assert refs["M1"] == pytest.approx([15.0, 15.0], abs=0.5)
    assert refs["M3"] == pytest.approx([72.0] * 6, abs=0.5)
    heart = details["M3"]
    assert heart[0]["oximeter_bpm"] == 72.0 and heart[-1]["oximeter_bpm"] is None
    assert all(w["ppg_snr_db"] > 3 for w in heart)
    assert {"bw_bpm", "am_bpm", "fm_bpm", "spread_bpm", "n_beats"} <= set(details["M1"][0])

    wide = baseline.wide_references(refs, details, PROTOCOL, 4.0)
    assert wide["M3"] == refs["M3"]
    assert wide["M1"] == pytest.approx([15.0, 15.0], abs=0.5)


def test_wide_rule_uses_the_sample_standard_deviation():
    assert reference.fuse_by_std({"bw_bpm": 12.0, "am_bpm": 15.0, "fm_bpm": 18.0}, 4.0) == 15.0
    assert reference.fuse_by_std({"bw_bpm": 10.0, "am_bpm": 15.0, "fm_bpm": 20.0}, 4.0) is None
    assert reference.fuse_by_std({"bw_bpm": 12.0, "am_bpm": None, "fm_bpm": 12.0}, 4.0) is None


def test_head_pose_selection_and_whole_windows():
    still = baseline.value_ranges({"d_pitch": np.zeros(600), "d_yaw": np.full(600, 3.0)})
    assert still == {"d_pitch": 0.0, "d_yaw": 0.0} and baseline.head_is_still(still, 1e-6)
    moving = baseline.value_ranges({"d_yaw": np.linspace(-10, 10, 600)})
    assert not baseline.head_is_still(moving, 1e-6)
    whole = baseline.whole_windows(68.2, PROTOCOL)
    assert whole == {m: [(0.0, 68.2)] for m in ("M1", "M2", "M3")}
    assert baseline.whole_windows(0.0, PROTOCOL)["M1"] == []


def test_reference_reliability_counts_usable_windows_and_agreement():
    def breathing(bw, am, fm, fused):
        values = [v for v in (bw, am, fm) if v is not None]
        spread = max(values) - min(values) if len(values) == 3 else None
        return {"bw_bpm": bw, "am_bpm": am, "fm_bpm": fm, "spread_bpm": spread, "bpm": fused,
                "n_beats": 30}  # fmt: skip

    def heart(ppg_bpm, oximeter_bpm):
        return {"bpm": ppg_bpm, "ppg_snr_db": 10.0, "oximeter_bpm": oximeter_bpm,
                "oximeter_valid_frac": 0.0 if oximeter_bpm is None else 1.0}  # fmt: skip

    agree = breathing(15.0, 15.5, 16.0, 15.5)  # range 1 bpm: usable
    loose = breathing(12.0, 15.0, 18.0, None)  # range 6 bpm, SD 3 bpm: wide rule only
    partial = breathing(12.0, None, 12.5, None)  # AM missing: never usable
    records = [
        {
            "subject": "s1",
            "reference": {"M1": [agree, loose], "M3": [heart(70, 72), heart(70, 80)]},
        },
        {"subject": "s2", "reference": {"M1": [partial], "M3": [heart(90, None)]}},
    ]
    rel = baseline.reference_reliability(records, PROTOCOL, 4.0)
    b, h = rel["breathing"], rel["heart"]
    assert (b["n_windows"], b["n_subjects"]) == (3, 2)
    assert b["usable"]["count"] == 1 and b["usable_wide"]["count"] == 2
    assert len(b["usable"]["ci"]) == 2
    assert b["estimate_available"]["am"]["count"] == 2
    assert b["pairwise"]["bw-am"]["within_agreement"]["count"] == 1  # 0.5 bpm yes, 3 bpm no
    assert b["pairwise"]["bw-fm"]["abs_diff_bpm"][0] == pytest.approx(0.5)
    assert b["spread_bpm"] == [1.0, 2.25, 3.5, 4.75, 6.0]
    assert h["oximeter_vs_ppg"]["n_windows"] == 2
    assert h["oximeter_vs_ppg"]["within_tolerance"]["count"] == 1  # 2 bpm yes, 10 bpm no


def write_ubfc_style_subject(folder, spec: SynthSpec, duplicate_timestamp: bool = False):
    """vid.avi (uncompressed) and ground_truth.txt with a contact PPG that carries breathing."""
    folder.mkdir(parents=True, exist_ok=True)
    write_video(make_video(spec), folder / "vid.avi", spec.fps)
    ppg = synthetic_ppg(spec.n_frames, spec.fps, spec.heart_hz, spec.breathing_hz, **PPG)
    hr = np.full(spec.n_frames, 60.0 * spec.heart_hz)
    t = np.arange(spec.n_frames) / spec.fps
    if duplicate_timestamp:
        t[5] = t[4]
    lines = [" ".join(f"{v:.8e}" for v in row) for row in (ppg, hr, t)]
    (folder / "ground_truth.txt").write_text("\n".join(lines) + "\n")


def test_ubfc_baseline_runs_resumes_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "make_pose", fake_make_pose(SPEC))
    root = tmp_path / "drive" / "UBFC"
    write_ubfc_style_subject(root / "DATASET_2" / "subject1", SPEC)
    low_fps = replace(SPEC, fps=24.0, n_frames=480, breathing_hz=0.3, heart_hz=1.1, seed=4)
    write_ubfc_style_subject(root / "DATASET_2" / "subject2", low_fps, duplicate_timestamp=True)
    run_id = "20261122_ubfc_baseline_s0"
    argv = ["--config", str(PROJECT_DIR / "configs" / "baseline_ubfc.yaml"), "--run-id", run_id]
    argv += ["--results-root", str(tmp_path / "results"), "--work-dir", str(tmp_path / "work")]
    argv += ["--source", str(root)]

    assert script.main(argv + ["--limit", "1"]) == 0
    run = tmp_path / "results" / run_id
    assert len((run / "items.jsonl").read_text().splitlines()) == 1
    assert script.main(argv) == 0
    assert script.main(argv) == 0  # nothing left to do
    records = [json.loads(line) for line in (run / "items.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == ["DATASET_2/subject1", "DATASET_2/subject2"]
    assert all(r["status"] == "ok" for r in records)
    assert len(json.loads((run / "manifest.json").read_text())["sessions"]) == 3
    assert list((tmp_path / "work" / "stage").iterdir()) == []  # staged copies were deleted

    for r in records:
        assert r["regions"]["shoulder_source"] == "pose" and r["regions"]["face_source"] == "pose"
        assert set(r["variants"]) == {"whole_video", "wide_reference"}
        for windows in (r["windows"], r["variants"]["whole_video"]):
            for w in windows:
                assert w["excluded"] is None, w
                assert abs(w["error_bpm"]) < (0.5 if w["measure"] == "M3" else 1.0), w
    assert records[0]["flags"]["duplicates"] is False and records[1]["flags"]["duplicates"]
    assert records[1]["fps"] == pytest.approx(24.0)

    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["counts"] == {"measured": 2, "subjects": 2, "skipped": 0, "errors": 0}
    assert metrics["main"]["per_measure"]["M3"]["n_windows"] == 4  # 2 + 2 windows of 10 s
    assert metrics["main"]["per_measure"]["M1"]["success_rate"]["value"] == 1.0
    assert metrics["sensitivity"]["without_low_fps"]["n_removed"] == 1
    assert metrics["sensitivity"]["without_timestamp_flags"]["n_removed"] == 1
    assert metrics["sensitivity"]["without_face_anchored"]["n_removed"] == 0
    assert metrics["reference_reliability"]["breathing"]["usable"]["count"] == 2
    assert [s["subject"] for s in metrics["subjects"]] == [
        "DATASET_2/subject1",
        "DATASET_2/subject2",
    ]

    summary = (run / "summary.md").read_text()
    for heading in ("## Main result", "## Breathing reference derived from PPG",
                    "## Heart-rate reference", "## Sensitivity analyses",
                    "## Sensitivity: wide breathing-reference rule",
                    "## For comparison with published results", "## Per subject"):  # fmt: skip
        assert heading in summary, heading
    with open(run / "windows.csv") as f:
        variants = {row["variant"] for row in csv.DictReader(f)}
    assert variants == {"main", "whole_video", "wide_reference"}

    # The summary is a function of items.jsonl alone.
    before = (run / "metrics.json").read_text()
    assert script.main(argv + ["--summarize-only"]) == 0
    assert (run / "metrics.json").read_text() == before


def scamps_archive(tmp_path, head_motion: list[bool], action_units: list[bool]):
    frames = make_video(SPEC)
    files = {}
    for i, (moving, acting) in enumerate(zip(head_motion, action_units, strict=True), start=1):
        n = SPEC.n_frames
        extra = {
            "d_pitch": np.zeros(n),
            "d_roll": np.full(n, 2.0),
            "d_yaw": np.linspace(-20.0, 20.0, n) if moving else np.zeros(n),
            "au12": np.linspace(0.0, 1.0, n) if acting else np.zeros(n),
        }
        path = tmp_path / f"P{i:06d}.mat"
        write_scamps_mat(path, frames, SPEC, extra=extra)
        files[f"scamps_videos/{path.name}"] = path
    archive = tmp_path / "scamps_videos.tar.gz"
    write_tar_gz(archive, files)
    return archive


def scamps_argv(tmp_path, archive, **selection):
    config = yaml.safe_load((PROJECT_DIR / "configs" / "baseline_scamps.yaml").read_text())
    config["source"] = str(archive)
    config["selection"].update(selection)
    path = tmp_path / "scamps.yaml"
    path.write_text(yaml.safe_dump(config))
    argv = ["--config", str(path), "--run-id", "20261122_scamps_baseline_s0"]
    return argv + [
        "--results-root",
        str(tmp_path / "results"),
        "--work-dir",
        str(tmp_path / "work"),
    ]


def test_scamps_baseline_streams_the_head_still_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "make_pose", fake_make_pose(SPEC))
    archive = scamps_archive(tmp_path, [False, True, False, False], [False, False, True, False])
    argv = scamps_argv(tmp_path, archive, n_videos=2)

    assert script.main(argv + ["--limit", "1"]) == 0
    assert script.main(argv) == 0
    assert script.main(argv) == 0  # target reached: the archive is not opened again
    run = tmp_path / "results" / "20261122_scamps_baseline_s0"
    records = [json.loads(line) for line in (run / "items.jsonl").read_text().splitlines()]
    assert [(r["id"].split("/")[-1], r["status"]) for r in records] == [
        ("P000001.mat", "ok"),
        ("P000002.mat", "skipped"),
        ("P000003.mat", "ok"),
    ]
    assert records[1]["head_pose_range"]["d_yaw"] == pytest.approx(40.0)
    assert records[2]["action_units_max_range"] == pytest.approx(1.0)
    for r in (records[0], records[2]):
        assert r["subject"] == r["id"].split("/")[-1].removesuffix(".mat")
        for w in r["windows"]:
            assert w["excluded"] is None and abs(w["error_bpm"]) < 0.5, w

    metrics = json.loads((run / "metrics.json").read_text())
    selection = metrics["selection"]
    assert (selection["streamed"], selection["measured"], selection["skipped_head_motion"]) == (
        3,
        2,
        1,
    )
    assert selection["action_units_active"] == 1
    assert "## Head-still subset" in (run / "summary.md").read_text()
    assert list((tmp_path / "work" / "stage").iterdir()) == []


def test_scamps_baseline_gives_up_when_no_head_is_still(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "make_pose", fake_make_pose(SPEC))
    archive = scamps_archive(tmp_path, [True, True, True], [False] * 3)
    argv = scamps_argv(tmp_path, archive, give_up_after=2)
    with pytest.raises(SystemExit, match="still head"):
        script.main(argv)
    run = tmp_path / "results" / "20261122_scamps_baseline_s0"
    assert len((run / "items.jsonl").read_text().splitlines()) == 2
    assert "| Measured | 0 |" in (run / "summary.md").read_text()


def test_colab_notebook_is_a_thin_launcher():
    notebook = json.loads((PROJECT_DIR / "notebooks" / "colab_runner.ipynb").read_text())
    assert notebook["nbformat"] == 4
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code
    for cell in code:
        assert cell.get("outputs") == [] and cell.get("execution_count") is None
    text = "\n".join("".join(c["source"]) for c in code)
    paths = re.findall(r"(?:scripts|configs)/[\w.-]+\.(?:py|yaml)", text)
    assert "scripts/compression_experiment.py" in paths  # the current phase
    for path in paths:
        assert (PROJECT_DIR / path).exists(), path
    assert 'userdata.get("GH_TOKEN")' in text  # the token comes from Colab secrets
    assert "breathing" not in text.replace("camera-breathing", "")  # no project logic inline
