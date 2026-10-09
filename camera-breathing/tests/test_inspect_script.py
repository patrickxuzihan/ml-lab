"""End-to-end runs of scripts/inspect_dataset.py on synthetic stand-ins (pose disabled)."""

import json

import pytest
import yaml
from conftest import PROJECT_DIR, load_script, write_scamps_mat, write_tar_gz, write_ubfc_subject

from breathing.runinfo import ConfigMismatchError

inspect_dataset = load_script("inspect_dataset")


def run(*args):
    return inspect_dataset.main([str(a) for a in args])


def test_ubfc_run_resumes_one_video_at_a_time(tmp_path, spec, frames):
    root = tmp_path / "drive" / "UBFC"
    for name in ("subject1", "subject2"):
        write_ubfc_subject(root / "DATASET_2" / name, frames[:90], spec)
    (root / "DATASET_1" / "x").mkdir(parents=True)
    (root / "DATASET_1" / "x" / "vid.avi").write_bytes(b"")
    common = [
        "--config", PROJECT_DIR / "configs" / "inspect_ubfc.yaml",
        "--run-id", "20261012_ubfc_inspect",
        "--results-root", tmp_path / "results",
        "--work-dir", tmp_path / "work",
        "--source", root,
        "--no-pose",
    ]  # fmt: skip
    run(*common, "--limit", "1")
    out = tmp_path / "results" / "20261012_ubfc_inspect"
    lines = (out / "items.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["DATASET_2/subject1"]

    run(*common)
    run(*common)  # nothing left to do
    records = [json.loads(line) for line in (out / "items.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == ["DATASET_2/subject1", "DATASET_2/subject2"]
    assert all(r["status"] == "ok" for r in records)
    assert records[0]["video"]["n_decoded_frames"] == 90

    assert json.loads((out / "folders.json").read_text())["skipped_no_ground_truth"] == [
        "DATASET_1/x"
    ]
    assert len(json.loads((out / "manifest.json").read_text())["sessions"]) == 3
    assert yaml.safe_load((out / "config.yaml").read_text())["source"] == str(root)
    assert "| Videos checked | 2 |" in (out / "summary.md").read_text()
    assert (out / "items.csv").exists()
    assert list((tmp_path / "work" / "stage").iterdir()) == []  # staged copies were deleted

    with pytest.raises(ConfigMismatchError):
        run(*common[:-3], "--source", tmp_path, "--no-pose")


def test_scamps_run_from_local_archive(tmp_path, spec, frames):
    mats = tmp_path / "mats"
    mats.mkdir()
    write_scamps_mat(mats / "P000001.mat", frames, spec)
    write_scamps_mat(mats / "P000002.mat", frames, spec)
    archive = tmp_path / "scamps_videos_example.tar.gz"
    write_tar_gz(
        archive,
        {"scamps/P000001.mat": mats / "P000001.mat", "scamps/P000002.mat": mats / "P000002.mat"},
    )
    (tmp_path / "P000001.csv").write_text("d_ppg,d_br\n1,2\n")
    labels = tmp_path / "labels.tar.gz"
    write_tar_gz(labels, {"csv/P000001.csv": tmp_path / "P000001.csv"})

    config = yaml.safe_load((PROJECT_DIR / "configs" / "inspect_scamps.yaml").read_text())
    config.update(labels_csv=str(labels), head_urls=[archive.as_uri()])
    config_path = tmp_path / "scamps.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run(
        "--config", config_path,
        "--run-id", "20261012_scamps_inspect",
        "--results-root", tmp_path / "results",
        "--work-dir", tmp_path / "work",
        "--source", archive,
        "--no-pose",
    )  # fmt: skip
    out = tmp_path / "results" / "20261012_scamps_inspect"
    records = [json.loads(line) for line in (out / "items.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["ok", "ok"]
    assert records[0]["motion_probe"]["region_source"] == "fallback"
    assert (out / "example_frame_regions.png").exists()
    assert json.loads((out / "labels_csv.json").read_text())[0]["n_rows"] == 1
    downloads = json.loads((out / "downloads.json").read_text())
    assert downloads[0]["size_bytes"] == archive.stat().st_size
    summary = (out / "summary.md").read_text()
    assert "Does breathing show up as body motion?" in summary
    assert "| Videos checked | 2 |" in summary


def test_real_person_frames_are_never_saved(tmp_path):
    config = yaml.safe_load((PROJECT_DIR / "configs" / "inspect_ubfc.yaml").read_text())
    config["save_example_figure"] = True
    path = tmp_path / "ubfc.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit, match="refusing"):
        run("--config", path, "--run-id", "20261012_ubfc_inspect", "--results-root", tmp_path)
