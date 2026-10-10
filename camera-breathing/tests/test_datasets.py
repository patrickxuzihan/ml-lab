from dataclasses import replace

import h5py
import numpy as np
import pytest
from conftest import write_scamps_mat, write_tar_gz, write_ubfc_subject

from breathing.datasets import scamps, ubfc


def test_ubfc_ground_truth_and_subject_discovery(tmp_path, spec, frames):
    root = tmp_path / "UBFC"
    write_ubfc_subject(root / "DATASET_2" / "subject3", frames[:30], spec)
    write_ubfc_subject(root / "DATASET_2" / "subject1", frames[:30], spec)
    (root / "DATASET_1" / "old").mkdir(parents=True)
    (root / "DATASET_1" / "old" / "vid.avi").write_bytes(b"")

    subjects, skipped = ubfc.find_subjects(root)
    assert [s.id for s in subjects] == ["DATASET_2/subject1", "DATASET_2/subject3"]
    assert skipped == ["DATASET_1/old"]

    gt = ubfc.read_ground_truth(subjects[0].ground_truth)
    assert len(gt["t"]) == len(gt["ppg"]) == len(gt["hr"]) == 30
    assert gt["t"][1] == pytest.approx(1 / spec.fps)
    assert gt["hr"][0] == pytest.approx(60 * spec.heart_hz)


def test_ubfc_ground_truth_rejects_bad_files(tmp_path):
    path = tmp_path / "ground_truth.txt"
    path.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="3 lines"):
        ubfc.read_ground_truth(path)
    path.write_text("1 2 3\n4 5\n7 8 9\n")
    with pytest.raises(ValueError, match="lengths differ"):
        ubfc.read_ground_truth(path)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_scamps_reader_undoes_matlab_axis_order(tmp_path, spec, frames, dtype):
    clip = frames[:20]
    short = replace(spec, n_frames=20)
    path = tmp_path / "P000001.mat"
    write_scamps_mat(path, clip, short, dtype=dtype)

    video = scamps.read_video(path, "RawFrames", {"ppg": "d_ppg", "breathing": "d_br"})
    assert video.layout["interpretation"] == "matlab_order"
    assert video.layout["stored_shape"] == [3, spec.width, spec.height, 20]
    np.testing.assert_array_equal(video.frames, clip)
    assert video.waveforms["breathing"].shape == (20,)

    with h5py.File(path, "r") as h5:
        names = {v["name"] for v in scamps.list_variables(h5)}
        assert {"RawFrames", "d_br", "d_ppg", "hr_label"} <= names
        assert scamps.small_numeric_values(h5) == {"hr_label": [60 * spec.heart_hz]}


def test_frames_already_in_time_first_order_are_detected_by_size(frames):
    clip = frames[:7]
    out, layout = scamps.frames_to_thwc(clip, n_frames=7)
    assert layout["interpretation"] == "by_size"
    np.testing.assert_array_equal(out, clip)


def test_frames_without_time_axis_fail(frames):
    with pytest.raises(ValueError, match="cannot find time"):
        scamps.frames_to_thwc(frames[:7], n_frames=99)


def test_to_uint8():
    np.testing.assert_array_equal(scamps.to_uint8(np.array([0.0, 0.5, 1.0])), [0, 128, 255])
    np.testing.assert_array_equal(scamps.to_uint8(np.array([-3.0, 12.4, 300.0])), [0, 12, 255])


def test_tar_members_are_streamed_one_at_a_time(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("P1.mat", "P2.mat", "README.txt"):
        (src / name).write_bytes(name.encode())
    archive = tmp_path / "videos.tar.gz"
    write_tar_gz(
        archive,
        {
            "scamps/P1.mat": src / "P1.mat",
            "scamps/README.txt": src / "README.txt",
            "../P2.mat": src / "P2.mat",  # must not escape the work dir
        },
    )
    work = tmp_path / "work"
    seen = []
    for name, local in scamps.iter_tar_members(str(archive), work):
        assert local.parent == work and local.exists()
        assert list(work.iterdir()) == [local]  # only the current member is on disk
        seen.append((name, local.read_bytes()))
    assert seen == [("scamps/P1.mat", b"P1.mat"), ("../P2.mat", b"P2.mat")]
    assert list(work.iterdir()) == []

    skipped = [n for n, _ in scamps.iter_tar_members(archive.as_uri(), work, skip={"../P2.mat"})]
    assert skipped == ["scamps/P1.mat"]


def test_summarize_csv_tar(tmp_path):
    csv_file = tmp_path / "a.csv"
    csv_file.write_text("d_ppg,d_br\n1,2\n3,4\n5,6\n")
    (tmp_path / "b.txt").write_text("not a csv")
    archive = tmp_path / "labels.tar.gz"
    write_tar_gz(archive, {"csv/P1.csv": csv_file, "csv/notes.txt": tmp_path / "b.txt"})
    summary = scamps.summarize_csv_tar(str(archive))
    size = csv_file.stat().st_size
    assert summary == [
        {"name": "csv/P1.csv", "size_bytes": size, "columns": ["d_ppg", "d_br"], "n_rows": 3}
    ]


def test_http_head(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"12345")
    assert scamps.http_head(path.as_uri())["size_bytes"] == 5
    assert "error" in scamps.http_head((tmp_path / "missing.tar.gz").as_uri())
