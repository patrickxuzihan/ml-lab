"""Phase-5 deep model: CPU smoke tests on tiny synthetic inputs (skipped without torch).

A tiny model, a few synthetic videos and a few steps: the model's shapes and loss, that it
can fit one clip, resumable training, and the scripts end to end (model inputs -> classic run
-> tune -> final -> evaluation on SCAMPS-style and UBFC-style data).
"""

import json

import numpy as np
import pytest
import yaml
from conftest import PROJECT_DIR, load_script

torch = pytest.importorskip("torch")

from test_baseline import write_ubfc_style_subject  # noqa: E402
from test_compression_experiment import counting_pose, experiment, small_protocol  # noqa: E402
from test_face_only import FACE_SPEC, face_archive  # noqa: E402

from breathing import deep, model_input, pipeline  # noqa: E402
from breathing.synth import breathing_displacement, make_video  # noqa: E402

PROTOCOL = pipeline.load_protocol()
FORK_WARNING = "ignore:This process .* is multi-threaded:DeprecationWarning"


def test_shapes_and_loss():
    model = deep.BreathingNet(width=2, dilations=(1, 2))
    frames = torch.randint(0, 255, (2, 40, 16, 16, 3), dtype=torch.uint8)
    assert model(frames).shape == (2, 40)
    assert deep.model_input_tensor(frames).shape == (2, 40, 6, 16, 16)
    x = torch.randn(3, 50)
    assert float(deep.pearson_loss(x, x)) == pytest.approx(0.0, abs=1e-5)
    assert float(deep.pearson_loss(x, -x)) == pytest.approx(2.0, abs=1e-5)
    assert float(deep.pearson_loss(x, 3 * x + 1)) == pytest.approx(0.0, abs=1e-5)


def test_the_model_can_fit_a_moving_head():
    """Gradients flow from the loss to the frames: a tiny model fits one synthetic clip."""
    torch.manual_seed(0)
    spec = FACE_SPEC.__class__(**{**FACE_SPEC.__dict__, "n_frames": 300, "noise_std": 0.0})
    video = make_video(spec)
    frames = model_input.model_frames(iter(video), 30.0, 100, 10.0, 16)
    label = model_input.model_signal(breathing_displacement(spec), 30.0, 100, 10.0)
    x, y = torch.from_numpy(frames)[None], torch.from_numpy(label)[None]
    model = deep.BreathingNet(width=4, dilations=(1, 2, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = deep.pearson_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 < losses[0]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_split(folder, videos, versions, n_out=20, size=8):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(videos):
        meta = {"video": f"scamps_videos/P{i:06d}.mat", "head_still": True}
        np.savez_compressed(
            folder / f"P{i:06d}.npz",
            frames=rng.integers(0, 255, (versions, n_out, size, size, 3), dtype=np.uint8),
            settings=np.array(["source"] + [f"v{k}" for k in range(1, versions)]),
            kbps=np.full(versions, np.nan),
            label_breathing=np.sin(np.arange(n_out) + i).astype(np.float32),
            meta=np.array(json.dumps(meta)),
        )


def test_training_resumes_at_the_next_epoch(tmp_path):
    write_split(tmp_path / "train", 3, 2)
    write_split(tmp_path / "val", 2, 2)
    train_dir, val_dir = tmp_path / "cache" / "train", tmp_path / "cache" / "val"
    model_input.pack_split(sorted((tmp_path / "train").glob("*.npz")), train_dir, "breathing")
    model_input.pack_split(sorted((tmp_path / "val").glob("*.npz")), val_dir, "breathing")
    job = {"variant": "B", "versions": "all", "width": 2, "lr": 1e-3, "seed": 0}
    model_cfg = {"dilations": [1, 2], "kernel": 3}
    train_cfg = {"epochs": 1, "batch_size": 2, "weight_decay": 0.0, "fps": 10.0}
    args = (model_cfg, train_cfg, train_dir, val_dir, tmp_path / "ckpt", PROTOCOL, "M4")
    first = deep.train(job, *args, torch.device("cpu"))
    assert len(first["history"]) == 1 and first["best"]["epoch"] == 0
    log = []
    second = deep.train(job, model_cfg, {**train_cfg, "epochs": 2}, *args[2:],
                        torch.device("cpu"), log=log.append)  # fmt: skip
    assert log[0] == "resuming at epoch 1"
    assert [h["epoch"] for h in second["history"]] == [0, 1]
    assert (tmp_path / "ckpt" / "best.pt").exists() and (tmp_path / "ckpt" / "last.pt").exists()
    model = deep.load_model(tmp_path / "ckpt" / "best.pt", model_cfg, torch.device("cpu"))
    frames, _, _ = model_input.load_packed(val_dir)
    assert deep.predict(model, frames[:, 0], torch.device("cpu")).shape == (2, 20)


@pytest.mark.filterwarnings(FORK_WARNING)
def test_scripts_end_to_end(tmp_path, monkeypatch):
    data_script = load_script("make_training_data")
    classic_script = load_script("compression_experiment")
    train_script = load_script("train_model")
    eval_script = load_script("evaluate_model")
    calls = []
    for script in (data_script, classic_script):
        monkeypatch.setattr(script, "make_pose", counting_pose(FACE_SPEC, calls))
    protocol_path = small_protocol(tmp_path)
    results, work = tmp_path / "results", tmp_path / "work"
    common = ["--protocol", str(protocol_path), "--results-root", str(results)]

    # Model inputs: one test, one validation and three training videos.
    archive, _ = face_archive(tmp_path, 5)
    (tmp_path / "pose.task").write_bytes(b"not a model")
    data_cfg = {
        "dataset_config": str(PROJECT_DIR / "configs" / "baseline_scamps.yaml"),
        "model_input": {"size": 16, "fps": 10.0},
        "splits": {"test": {"first": 1, "last": 1, "versions": "fixed"},
                   "validation": {"first": 2, "last": 2, "versions": "fixed"},
                   "train": {"first": 3, "last": 5, "versions": "random"}},
        "fixed_ladders": ["crf", "bitrate"],
        "augmentation": {"seed": 0, "groups": [{"ladders": ["crf", "bitrate"], "n": 1}]},
        "pose_model_url": (tmp_path / "pose.task").as_uri(),
    }  # fmt: skip
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(data_cfg))
    data_root = tmp_path / "deep"
    argv = ["--config", str(tmp_path / "data.yaml"), "--run-id", "20261126_scamps_deep-data_s0"]
    argv += ["--source", str(archive), "--data-root", str(data_root), "--work-dir", str(work)]
    assert data_script.main(argv + common) == 0

    # The classic face-only run on the test video gives the references and M2, M4.
    classic_cfg = experiment(tmp_path, "baseline_scamps.yaml", framing="face",
                             measures=["M2", "M3", "M4"], ladders=["crf", "bitrate"],
                             relocate=[], n_videos=1)  # fmt: skip
    argv = ["--config", str(classic_cfg), "--run-id", "20261126_scamps_face-ladders_s0"]
    argv += ["--source", str(archive), "--work-dir", str(work)]
    assert classic_script.main(argv + common) == 0

    # Tune (two learning rates) and final (two seeds) with a tiny model.
    train_cfg = yaml.safe_load((PROJECT_DIR / "configs" / "deep_train.yaml").read_text())
    train_cfg["input"]["size"] = 16
    train_cfg["model"]["dilations"] = [1, 2]
    train_cfg["train"].update(epochs=2, batch_size=2)
    train_cfg["tuning"] = {"width": [2], "lr": [1e-3, 3e-3]}
    train_cfg["seeds"] = [0, 1]
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(train_cfg))
    ckpt = tmp_path / "checkpoints"
    argv = ["--config", str(tmp_path / "train.yaml"), "--data-root", str(data_root)]
    argv += ["--cache-dir", str(work / "cache"), "--checkpoint-root", str(ckpt)]
    argv += ["--workers", "0", *common]
    assert (
        train_script.main(argv + ["--stage", "tune", "--run-id", "20261126_scamps_deep-tune_s0"])
        == 0
    )
    tune = read_jsonl(results / "20261126_scamps_deep-tune_s0" / "items.jsonl")
    assert sorted(r["id"] for r in tune) == [
        "A_w2_lr0.001_s0", "A_w2_lr0.003_s0", "B_w2_lr0.001_s0", "B_w2_lr0.003_s0"
    ]  # fmt: skip
    assert all(len(r["history"]) == 2 for r in tune)
    final_args = argv + ["--stage", "final", "--tune-run", "20261126_scamps_deep-tune_s0"]
    final_args += ["--run-id", "20261126_scamps_deep-final_s0"]
    assert train_script.main(final_args) == 0
    final_run = results / "20261126_scamps_deep-final_s0"
    final = read_jsonl(final_run / "items.jsonl")
    assert sorted((r["variant"], r["seed"]) for r in final) == [
        ("A", 0),
        ("A", 1),
        ("B", 0),
        ("B", 1),
    ]
    assert "## Over seeds" in (final_run / "summary.md").read_text()
    assert train_script.main(final_args) == 0  # complete: nothing is trained again

    # Evaluation on the SCAMPS-style test video.
    argv = ["--final-run", "20261126_scamps_deep-final_s0", "--checkpoint-root", str(ckpt)]
    argv += ["--data-root", str(data_root), "--work-dir", str(work), *common[2:]]
    scamps_args = argv + ["--dataset", "scamps", "--run-id", "20261126_scamps_deep-eval_s0",
                          "--classic-run", "20261126_scamps_face-ladders_s0"]  # fmt: skip
    assert eval_script.main(scamps_args) == 0
    run = results / "20261126_scamps_deep-eval_s0"
    records = [eval_script.unpack(r) for r in read_jsonl(run / "items.jsonl")]
    assert sorted(r["setting"] for r in records) == [
        "source",
        "x264_br250",
        "x264_crf18",
        "x264_crf38",
    ]
    measures = {w["measure"] for w in records[0]["windows"]}
    assert measures == {"A0", "A1", "B0", "B1"}
    for r in records:  # the classic run's reference
        assert all(w["ref_bpm"] == pytest.approx(60 * FACE_SPEC.breathing_hz, abs=0.5)
                   for w in r["windows"])  # fmt: skip
    metrics = json.loads((run / "metrics.json").read_text())
    source = metrics["settings"]["source"]
    assert set(source["metrics"]) == {"M2", "M4", "A", "B"}
    assert {"B-A", "B-M4", "B-M2"} <= set(source["paired_success_rate"])
    text = (run / "summary.md").read_text()
    assert "## Success rate (%)" in text and "## Paired difference" in text
    assert eval_script.main(scamps_args) == 0  # complete: nothing to add
    assert len((run / "items.jsonl").read_text().splitlines()) == 4

    # Evaluation on a UBFC-style subject: inputs made on the fly, agreement only.
    root = tmp_path / "UBFC"
    write_ubfc_style_subject(root / "DATASET_2" / "subject1", FACE_SPEC)
    ubfc_cfg = experiment(tmp_path, "baseline_ubfc.yaml", framing="face",
                          measures=["M2", "M3", "M4"], ladders=["crf"], relocate=[],
                          n_videos=None)  # fmt: skip
    argv_c = ["--config", str(ubfc_cfg), "--run-id", "20261126_ubfc_face-ladders_s0"]
    argv_c += ["--source", str(root), "--work-dir", str(work)]
    assert classic_script.main(argv_c + common) == 0
    ubfc_args = argv + ["--dataset", "ubfc", "--run-id", "20261126_ubfc_deep-eval_s0"]
    ubfc_args += ["--classic-run", "20261126_ubfc_face-ladders_s0", "--jobs", "2"]
    assert eval_script.main(ubfc_args) == 0
    run = results / "20261126_ubfc_deep-eval_s0"
    records = read_jsonl(run / "items.jsonl")
    assert sorted(r["setting"] for r in records) == ["source", "x264_crf18", "x264_crf38"]
    agreement = json.loads((run / "metrics.json").read_text())["settings"]["x264_crf18"][
        "agreement"
    ]
    assert set(agreement) == {"M2", "M4", "A", "B"}
    assert "## Agreement with the uncompressed estimate" in (run / "summary.md").read_text()
    assert list((work / "stage").glob("*.avi")) == []
    assert list((work / "versions").glob("*")) == []
