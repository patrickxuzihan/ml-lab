import json
import math

import numpy as np
import pytest

from breathing.runinfo import (
    ConfigMismatchError,
    RunDir,
    check_run_id,
    environment_snapshot,
    to_jsonable,
)


@pytest.mark.parametrize(
    "run_id", ["20261012_scamps_inspect", "20261101_scamps_x264-ladder", "20261101_ubfc_train_s0"]
)
def test_valid_run_ids(run_id):
    assert check_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id", ["scamps_inspect", "2026-10-12_scamps_inspect", "20261012_UBFC_x"]
)
def test_invalid_run_ids(run_id):
    with pytest.raises(ValueError):
        check_run_id(run_id)


def test_run_dir_resume_and_config_check(tmp_path):
    config = {"dataset": "ubfc", "pose": {"enabled": False}}
    run = RunDir(tmp_path, "20261012_ubfc_inspect")
    run.init(config)
    run.init(dict(config))  # same config: fine
    with pytest.raises(ConfigMismatchError):
        run.init({**config, "dataset": "scamps"})

    run.append({"id": "a", "status": "error", "error": "boom"})
    run.append({"id": "b", "status": "ok", "value": np.float64(1.5)})
    run.append({"id": "a", "status": "ok", "value": float("nan")})
    assert run.done_ids() == {"a", "b"}
    assert [r["id"] for r in run.records()] == ["a", "b"]
    assert run.records()[0]["value"] is None


def test_manifest_keeps_one_entry_per_session(tmp_path):
    run = RunDir(tmp_path, "20261012_ubfc_inspect")
    run.init({})
    run.add_session({"n": 1})
    run.add_session({"n": 2})
    manifest = json.loads(run.manifest_path.read_text())
    assert [s["n"] for s in manifest["sessions"]] == [1, 2]


def test_to_jsonable():
    data = {"a": np.int64(3), "b": [np.float32(0.5), math.inf], "c": np.arange(2), 1: None}
    assert to_jsonable(data) == {"a": 3, "b": [0.5, None], "c": [0, 1], "1": None}


def test_environment_snapshot_records_versions(tmp_path):
    snapshot = environment_snapshot(tmp_path, {"extra": 1})
    assert snapshot["extra"] == 1
    assert snapshot["git"]["commit"] is None  # tmp_path is not a git repository
    assert snapshot["packages"]["numpy"] == np.__version__
    assert snapshot["ffmpeg"].startswith("ffmpeg version")
    assert snapshot["hardware"]["cpu_count"] >= 1
