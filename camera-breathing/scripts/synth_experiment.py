"""Run the evaluation protocol on synthetic videos with known rates (phase-2 smoke run).

For each synthetic video: regions from its known landmarks, reference rates from its known
waveforms, then every configured compression setting is encoded, decoded and measured.
Results go to <results-root>/<run-id>/; re-running the same command resumes.

  python scripts/synth_experiment.py --config configs/synth_smoke.yaml \
      --run-id 20261121_synth_pipeline-smoke_s0
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from breathing import compression, pipeline, reference, roi
from breathing.runinfo import RunDir, environment_snapshot
from breathing.synth import (
    SynthSpec,
    breathing_displacement,
    heart_modulation,
    make_video,
    synth_landmarks,
    synthetic_ppg,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, default=pipeline.PROTOCOL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "work")
    return parser.parse_args(argv)


def make_specs(cfg: dict) -> list[dict]:
    """Synthetic videos: per-subject amplitudes, per-video rates (seeded)."""
    rng = np.random.default_rng(cfg["seed"])
    v = cfg["video"]
    base = SynthSpec(
        n_frames=int(round(v["duration_s"] * v["fps"])),
        fps=v["fps"],
        height=v["height"],
        width=v["width"],
        noise_std=v["noise_std"],
        face_center_frac=tuple(v["face_center_frac"]),
        face_radius_frac=tuple(v["face_radius_frac"]),
        shoulder_half_width_frac=v["shoulder_half_width_frac"],
    )
    videos = []
    for s in range(cfg["n_subjects"]):
        subject = replace(
            base,
            breathing_amplitude_px=float(rng.uniform(*v["breathing_amplitude_px"])),
            heart_amplitude=float(rng.uniform(*v["heart_amplitude"])),
            breathing_color_amplitude=float(rng.uniform(*v["breathing_color_amplitude"])),
        )
        for k in range(cfg["videos_per_subject"]):
            spec = replace(
                subject,
                breathing_hz=float(rng.uniform(*v["breathing_bpm"])) / 60,
                heart_hz=float(rng.uniform(*v["heart_bpm"])) / 60,
                seed=int(rng.integers(0, 2**31)),
            )
            videos.append({"video": f"s{s}_v{k}", "subject": f"s{s}", "spec": spec})
    return videos


def ppg_reference_check(spec: SynthSpec, windows: list, cfg: dict, protocol: dict) -> list[dict]:
    """Breathing reference derived from a synthetic contact PPG, next to the true rate."""
    p = cfg["ppg"]
    ppg = synthetic_ppg(
        spec.n_frames,
        spec.fps,
        spec.heart_hz,
        spec.breathing_hz,
        baseline=p["baseline"],
        amplitude_depth=p["amplitude_depth"],
        interval_depth=p["interval_depth"],
        noise_std=p["noise_std"],
        seed=spec.seed,
    )
    m1 = protocol["measurements"]["M1"]
    return reference.respiration_from_ppg(
        ppg,
        spec.fps,
        windows,
        tuple(m1["band_hz"]),
        protocol["rate"],
        protocol["reference"]["ppg_respiration"],
    )


def run_video(item: dict, settings, protocol: dict, cfg: dict, run: RunDir, work_dir: Path):
    spec: SynthSpec = item["spec"]
    done = run.done_ids()
    todo = [s for s in settings if f"{item['video']}/{s.name}" not in done]
    if not todo:
        return
    frames = make_video(spec)
    reg = protocol["regions"]
    n_sampled = max(1, int(spec.n_frames / spec.fps * reg["pose_sample_fps"]))
    landmarks = np.repeat(synth_landmarks(spec)[None], n_sampled, axis=0)
    boxes, region_info = roi.measurement_regions(
        landmarks, spec.width, spec.height, reg["visibility_threshold"], reg["min_shoulder_frac"]
    )
    windows = pipeline.analysis_windows(spec.n_frames / spec.fps, protocol)
    waves = {"breathing": breathing_displacement(spec), "heart": heart_modulation(spec)}
    refs = pipeline.waveform_references(waves, spec.fps, windows, protocol)
    truth = {"breathing_bpm": 60 * spec.breathing_hz, "heart_bpm": 60 * spec.heart_hz}
    ppg_check = ppg_reference_check(spec, windows["M1"], cfg, protocol)
    for setting in todo:
        item_id = f"{item['video']}/{setting.name}"
        print(f"measuring {item_id} ...", flush=True)
        base = {"id": item_id, "video": item["video"], "subject": item["subject"]}
        try:
            record = pipeline.process_version(
                setting,
                lambda: iter(frames),
                spec.fps,
                (spec.height, spec.width),
                spec.n_frames,
                boxes,
                windows,
                refs,
                protocol,
                work_dir,
            )
            record = {**base, "status": "ok", **record, "truth": truth}
            record["regions"] = {"boxes": boxes, **region_info}
            record["synth"] = {
                "breathing_amplitude_px": spec.breathing_amplitude_px,
                "heart_amplitude": spec.heart_amplitude,
                "breathing_color_amplitude": spec.breathing_color_amplitude,
            }
            if setting.name == "source":
                record["ppg_reference"] = ppg_check
        except Exception as err:  # recorded and retried on the next run
            traceback.print_exc()
            record = {**base, "status": "error", "error": f"{type(err).__name__}: {err}"}
        run.append(record)


def write_windows_csv(records: list[dict], path: Path) -> None:
    rows = pipeline.window_rows(records)
    columns = ["video", "subject", "setting", "measure", "window", "t0", "t1", "ref_bpm"]
    columns += ["est_bpm", "error_bpm", "snr_db", "excluded"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ppg_check_markdown(records: list[dict]) -> str:
    lines = [
        "## PPG-derived breathing reference (synthetic contact PPG)",
        "",
        "| Video | True (bpm) | BW | AM | FM | Spread | Reference |",
        "|---|---|---|---|---|---|---|",
    ]

    def f(v):
        return "–" if v is None else f"{v:.2f}"

    for r in records:
        for w in r.get("ppg_reference", []):
            lines.append(
                f"| {r['video']} | {f(r['truth']['breathing_bpm'])} | {f(w['bw_bpm'])} "
                f"| {f(w['am_bpm'])} | {f(w['fm_bpm'])} | {f(w['spread_bpm'])} | {f(w['bpm'])} |"
            )
    return "\n".join(lines) + "\n"


def summarize(run: RunDir, settings, protocol: dict) -> None:
    records = [r for r in run.records() if r.get("status") == "ok"]
    errors = [r for r in run.records() if r.get("status") != "ok"]
    write_windows_csv(records, run.path / "windows.csv")
    agg = pipeline.aggregate(records, settings, protocol)
    run.write_json("metrics.json", agg)
    header = [
        f"# Synthetic pipeline run: {run.run_id}",
        "",
        f"Videos: {len({r['video'] for r in records})}; "
        f"subjects: {len({r['subject'] for r in records})}; "
        f"versions measured: {len(records)}; errors: {len(errors)}.",
        "",
    ]
    text = "\n".join(header) + pipeline.summary_markdown(agg, records, settings)
    text += "\n" + ppg_check_markdown([r for r in records if r["setting"] == "source"])
    (run.path / "summary.md").write_text(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    protocol = pipeline.load_protocol(args.protocol)
    settings = compression.select(
        compression.expand_ladders(protocol["compression"]), cfg["settings"]
    )
    run = RunDir(args.results_root, args.run_id)
    run.init({"experiment": cfg, "protocol": protocol})
    run.add_session(environment_snapshot(PROJECT_DIR, {"argv": sys.argv[1:]}))
    try:
        for item in make_specs(cfg):
            run_video(item, settings, protocol, cfg, run, args.work_dir / "versions")
    finally:
        summarize(run, settings, protocol)
    print(f"results in {run.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
