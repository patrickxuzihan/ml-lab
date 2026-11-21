"""The phase-2 scripts on tiny synthetic inputs, and the no-torch rule for core modules."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml
from conftest import PROJECT_DIR, load_script, write_tar_gz

TINY = {
    "dataset": "synth",
    "seed": 0,
    "n_subjects": 1,
    "videos_per_subject": 1,
    "video": {
        "duration_s": 20.0,  # breathing windows are never shorter (SCAMPS clips)
        "fps": 30.0,
        "height": 64,
        "width": 96,
        "noise_std": 1.0,
        "face_center_frac": [0.25, 0.5],
        "face_radius_frac": [0.15, 0.12],
        "shoulder_half_width_frac": 0.15,
        "breathing_bpm": [12.0, 20.0],
        "heart_bpm": [60.0, 100.0],
        "breathing_amplitude_px": [0.4, 0.5],
        "heart_amplitude": [1.0, 1.5],
        "breathing_color_amplitude": [0.8, 1.0],
    },
    "ppg": {"baseline": 0.1, "amplitude_depth": 0.1, "interval_depth": 0.05, "noise_std": 0.05},
    "settings": ["source", "x264_crf18"],
}


def test_synth_experiment_runs_and_resumes(tmp_path):
    script = load_script("synth_experiment")
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY))
    argv = ["--config", str(config), "--run-id", "20261121_synth_tiny_s0"]
    argv += ["--results-root", str(tmp_path / "results"), "--work-dir", str(tmp_path / "work")]
    assert script.main(argv) == 0
    run = tmp_path / "results" / "20261121_synth_tiny_s0"
    for name in ("config.yaml", "manifest.json", "items.jsonl", "windows.csv", "metrics.json"):
        assert (run / name).exists(), name
    records = [json.loads(line) for line in (run / "items.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == ["s0_v0/source", "s0_v0/x264_crf18"]
    assert all(r["status"] == "ok" for r in records)
    for w in records[0]["windows"]:
        assert abs(w["error_bpm"]) < 0.5
    assert records[0]["ppg_reference"][0]["bpm"] is not None
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics["per_setting"]) == {"source", "x264_crf18"}
    assert "x264_crf" in metrics["breakdown"]
    assert "Paired difference" in (run / "summary.md").read_text()

    # Resuming measures nothing new and keeps one record per version.
    assert script.main(argv) == 0
    assert len((run / "items.jsonl").read_text().splitlines()) == 2
    assert len(json.loads((run / "manifest.json").read_text())["sessions"]) == 2


def test_scamps_timing_from_labels_csv(tmp_path):
    files = {}
    for i, fps in enumerate((30.0, 30.0, 25.0)):
        path = tmp_path / f"{i:06d}.csv"
        t = np.arange(600) / fps
        rows = ["t_ppg,d_ppg,d_br"] + [f"{v:.6f},0.1,0.2" for v in t]
        path.write_text("\n".join(rows) + "\n")
        files[f"scamps_waveforms_csv/{path.name}"] = path
    bad = tmp_path / "bad.csv"
    bad.write_text("d_ppg,d_br\n0.1,0.2\n")
    files["scamps_waveforms_csv/bad.csv"] = bad
    archive = tmp_path / "labels.tar.gz"
    write_tar_gz(archive, files)

    script = load_script("check_scamps_timing")
    argv = ["--run-id", "20261121_scamps_timing", "--source", str(archive)]
    argv += ["--results-root", str(tmp_path / "results")]
    assert script.main(argv) == 0
    summary = json.loads((tmp_path / "results/20261121_scamps_timing/summary.json").read_text())
    assert summary["files"] == 4 and summary["errors"] == 1
    assert summary["rows"] == [600, 600, 600]
    assert summary["fs_mean_hz"] == pytest.approx([25.0, 30.0, 30.0])
    assert summary["files_off_assumed_fps"] == ["scamps_waveforms_csv/000002.csv"]


def test_core_modules_do_not_import_torch():
    modules = [
        "breathing.synth",
        "breathing.roi",
        "breathing.signals",
        "breathing.rate",
        "breathing.reference",
        "breathing.metrics",
        "breathing.compression",
        "breathing.video_io",
        "breathing.pipeline",
        "breathing.baseline",
        "breathing.sources",
    ]
    code = "import sys\n" + "".join(f"import {m}\n" for m in modules)
    code += "print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONPATH": str(PROJECT_DIR / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"
