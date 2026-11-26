"""Phase-5 deep model evaluation: the final models through the evaluation protocol, next to
the classic M2 and M4 of the same face-only videos and compression settings.

- SCAMPS (--dataset scamps): the test split of scripts/make_training_data.py (the source
  and 24 compressed versions of each video). References and the classic estimates come from
  the face-only classic run (scripts/compression_experiment.py, configs/face_ladders_scamps.yaml).
- UBFC-rPPG (--dataset ubfc): no usable breathing reference, so the summary is the agreement
  of each version's estimate with the uncompressed one. The inputs are made on the fly from
  the videos (same crop as the classic run, same settings, same downscaling) and deleted;
  only numbers are kept.

  python scripts/evaluate_model.py --dataset scamps --run-id 20261129_scamps_deep-eval_s0 \
      --final-run 20261128_scamps_deep-final_s0 --classic-run 20261127_scamps_face-ladders_s0 \
      --data-root /content/drive/MyDrive/camera-breathing/deep_data \
      --checkpoint-root /content/drive/MyDrive/camera-breathing/checkpoints \
      --results-root /content/drive/MyDrive/camera-breathing/results

  python scripts/evaluate_model.py --dataset ubfc --run-id 20261129_ubfc_deep-eval_s0 \
      --final-run 20261128_scamps_deep-final_s0 --classic-run 20261127_ubfc_face-ladders_s0 \
      --checkpoint-root ... --results-root ... --work-dir /content/work --jobs 8

Each model is a measurement named by its variant and seed (A0 ... B4), with the band,
windows and tolerance of the training config's `measure` (M4). One record per video and
setting, windows stored compactly as in compression_experiment.py; a run resumes at the
next unfinished video.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from breathing import compression, deep, metrics, model_input, pipeline, sources
from breathing.datasets import ubfc
from breathing.runinfo import RunDir, environment_snapshot

PROJECT_DIR = Path(__file__).resolve().parents[1]
WINDOW_FIELDS = ("measure", "window", "t0", "t1", "ref_bpm", "est_bpm", "error_bpm", "snr_db")
WINDOW_FIELDS += ("excluded",)  # the order of compression_experiment.py
CLASSIC = ("M2", "M4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=("scamps", "ubfc"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--final-run", required=True, help="run id of the final training stage")
    parser.add_argument("--classic-run", required=True, help="run id of the face-only classic run")
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data" / "deep")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=PROJECT_DIR / "work" / "checkpoints"
    )
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "work")
    parser.add_argument("--source", help="UBFC-rPPG location (default: the classic run's)")
    parser.add_argument("--jobs", type=int, default=1, help="versions encoded at once (UBFC)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument(
        "--summarize-only", action="store_true", help="only rebuild the summary from items.jsonl"
    )
    return parser.parse_args(argv)


def rounded(value, digits: int = 3):
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def pack(record: dict) -> dict:
    out = dict(record)
    out["windows"] = [[w[f] for f in WINDOW_FIELDS] for w in record["windows"]]
    return rounded(out)


def unpack(record: dict) -> dict:
    out = dict(record)
    if "windows" in out:
        out["windows"] = [dict(zip(WINDOW_FIELDS, w, strict=True)) for w in out["windows"]]
    return out


def model_name(record: dict) -> str:
    return f"{record['variant']}{record['seed']}"


def eval_protocol(protocol: dict, measure: str, names: list[str]) -> dict:
    """The protocol with one measurement per model, configured like `measure`."""
    out = copy.deepcopy(protocol)
    for name in names:
        out["measurements"][name] = {**protocol["measurements"][measure], "signal": "model"}
    return out


def reference_windows(source_record: dict, measure: str) -> tuple[list, list]:
    """Windows (seconds) and references of `measure` on the classic run's source record."""
    rows = [w for w in source_record["windows"] if w["measure"] == measure]
    return [(w["t0"], w["t1"]) for w in rows], [w["ref_bpm"] for w in rows]


def version_records(video, subject, settings, kbps, waves, fs, windows, refs, protocol):
    """One record per version: the windows of every model."""
    out = []
    for v, setting in enumerate(settings):
        rows = []
        for name, wave in waves.items():
            rows += pipeline.measure_waveform(name, wave[v], fs, windows, refs, protocol)
        k = None if kbps[v] is None or not np.isfinite(kbps[v]) else float(kbps[v])
        base = {"id": f"{video}/{setting}", "status": "ok", "video": video, "subject": subject}
        out.append({**base, "setting": setting, "kbps": k, "windows": rows})
    return out


def load_models(final: RunDir, checkpoint_root: Path, device: torch.device) -> dict:
    model_cfg = final.saved_config()["train"]["model"]
    records = sorted((r for r in final.records() if r["status"] == "ok"), key=model_name)
    return {
        model_name(r): deep.load_model(checkpoint_root / r["checkpoint"], model_cfg, device)
        for r in records
    }


def run_scamps(run, args, classic, models, train_cfg, protocol, device) -> int:
    fs, measure = float(train_cfg["input"]["fps"]), train_cfg["measure"]
    done = {r["id"] for r in run.records()}
    videos = 0
    for path in sorted((args.data_root / train_cfg["splits"]["test"]).glob("*.npz")):
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            settings = [str(s) for s in z["settings"]]
            if all(f"{meta['video']}/{s}" in done for s in settings):
                continue
            frames, kbps = z["frames"], list(z["kbps"])
        video = meta["video"]
        source = classic.get(f"{video}/source")
        if source is None:
            print(f"{video}: not in the classic run; skipped", flush=True)
            continue
        windows, refs = reference_windows(source, measure)
        waves = {name: deep.predict(m, frames, device) for name, m in models.items()}
        subject = PurePosixPath(video).stem
        for record in version_records(video, subject, settings, kbps, waves, fs, windows, refs,
                                      protocol):  # fmt: skip
            if record["id"] not in done:
                run.append(pack(record))
        videos += 1
        print(f"{video}: {len(settings)} versions", flush=True)
    return videos


def run_ubfc(run, args, classic, classic_cfg, models, train_cfg, protocol, device) -> int:
    fs, measure = float(train_cfg["input"]["fps"]), train_cfg["measure"]
    size = int(train_cfg["input"]["size"])
    settings = [
        s
        for s in compression.expand_ladders(protocol["compression"])
        if s.ladder == "source" or s.ladder in classic_cfg["ladders"]
    ]
    subjects, _ = ubfc.find_subjects(args.source or classic_cfg["source"])
    done = {r["id"] for r in run.records()}
    videos = 0
    for subject in subjects:
        source = classic.get(f"{subject.id}/source")
        todo = [s for s in settings if f"{subject.id}/{s.name}" not in done]
        if source is None or not todo:
            continue
        crop, n_frames = tuple(source["regions"]["crop"]), int(source["n_frames"])
        with sources.ubfc_video(subject, protocol, args.work_dir / "stage") as video:
            face = sources.cropped(video, crop)
            n_out = model_input.n_output(n_frames, video.fps, fs)
            versions = args.work_dir / "versions"

            def one(setting, face=face, n_out=n_out, n_frames=n_frames, versions=versions):
                with pipeline.encoded_version(
                    setting, face.frames, face.fps, face.size_hw, n_frames, protocol, versions
                ) as v:
                    x = model_input.model_frames(v.frames(), v.fps, n_out, fs, size)
                    return x, v.info.get("kbps")

            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                made = list(pool.map(one, todo))
        frames = np.stack([x for x, _ in made])
        windows, refs = reference_windows(source, measure)
        waves = {name: deep.predict(m, frames, device) for name, m in models.items()}
        kbps = [k for _, k in made]
        names = [s.name for s in todo]
        for record in version_records(subject.id, subject.id, names, kbps, waves, fs, windows,
                                      refs, protocol):  # fmt: skip
            run.append(pack(record))
        videos += 1
        print(f"{subject.id}: {len(todo)} versions", flush=True)
    return videos


# ---------------------------------------------------------------------------------------
# Summary


def window_table(records: list[dict]) -> dict:
    """{(setting, measure): {(video, window): window}}"""
    table: dict = {}
    for r in records:
        for w in r["windows"]:
            table.setdefault((r["setting"], w["measure"]), {})[(r["video"], w["window"])] = w
    return table


def paired(table, setting, a, b, subjects, protocol) -> dict | None:
    """b minus a on the windows both measured (neither excluded)."""
    cells_a, cells_b = table.get((setting, a), {}), table.get((setting, b), {})
    keys = sorted(k for k in cells_a if k in cells_b and cells_a[k]["excluded"] is None
                  and cells_b[k]["excluded"] is None)  # fmt: skip
    if not keys:
        return None
    tolerance = protocol["tolerance_bpm"][protocol["measurements"][a]["kind"]]
    return metrics.paired_difference(
        pipeline._arrays([cells_a[k] for k in keys]),
        pipeline._arrays([cells_b[k] for k in keys]),
        [subjects[k[0]] for k in keys],
        tolerance,
        protocol["statistics"],
    )


def agreement(table, setting, measure, tolerance) -> float | None:
    """Fraction of windows whose estimate is within the tolerance of the uncompressed one
    (a missing estimate disagrees); windows without an uncompressed estimate are left out."""
    source, cells = table.get(("source", measure), {}), table.get((setting, measure), {})
    hits = []
    for key, w in source.items():
        if w["est_bpm"] is None or w["excluded"] == "no_region" or key not in cells:
            continue
        est = cells[key]["est_bpm"]
        hits.append(est is not None and abs(est - w["est_bpm"]) <= tolerance)
    return float(np.mean(hits)) if hits else None


def spread(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}


def summarize(run, dataset, classic, classic_cfg, train_cfg, protocol, names) -> str:
    records = [unpack(r) for r in run.records() if r["status"] == "ok"]
    videos = {r["video"] for r in records}
    classic_records = [r for r in classic.values() if r["video"] in videos]
    settings = [
        s
        for s in compression.expand_ladders(protocol["compression"])
        if s.ladder == "source" or s.ladder in classic_cfg["ladders"]
    ]
    variants = sorted({n[0] for n in names})
    by_variant = {v: [n for n in names if n[0] == v] for v in variants}
    table = window_table(records + classic_records)
    tolerance = protocol["tolerance_bpm"][protocol["measurements"][train_cfg["measure"]]["kind"]]
    summary: dict = {"videos": len(videos), "models": names, "settings": {}}
    if dataset == "scamps":
        agg = {n: pipeline.aggregate(records, settings, protocol, (n,)) for n in names}
        classic_agg = pipeline.aggregate(classic_records, settings, protocol, CLASSIC)
        subjects = {r["video"]: r["subject"] for r in records}
        for s in settings:
            cell: dict = {}
            for m in CLASSIC:
                c = classic_agg["per_setting"].get(s.name, {}).get(m)
                if c:
                    cell[m] = {k: c[k]["value"] for k in ("success_rate", "mae")}
            for v, members in by_variant.items():
                cells = [agg[n]["per_setting"].get(s.name, {}).get(n) for n in members]
                cell[v] = {
                    k: spread([c[k]["value"] for c in cells if c]) for k in ("success_rate", "mae")
                }
            comparisons = {}
            for a, b in (("A", "B"), ("M4", "B"), ("M2", "B")):
                if b not in by_variant:
                    continue
                deltas = []
                for i, nb in enumerate(by_variant[b]):
                    na = by_variant[a][i] if a in by_variant else a
                    d = paired(table, s.name, na, nb, subjects, protocol)
                    if d:
                        deltas.append(d["success_rate"])
                if deltas:
                    comparisons[f"{b}-{a}"] = {
                        "delta": spread([d["delta"] for d in deltas]),
                        "significant_up": sum(d["significant"] and d["delta"] > 0 for d in deltas),
                        "significant_down": sum(
                            d["significant"] and d["delta"] < 0 for d in deltas
                        ),
                        "models": len(deltas),
                    }
            summary["settings"][s.name] = {"metrics": cell, "paired_success_rate": comparisons}
    else:
        for s in settings:
            cell = {m: agreement(table, s.name, m, tolerance) for m in CLASSIC}
            for v, members in by_variant.items():
                cell[v] = spread([agreement(table, s.name, n, tolerance) for n in members])
            summary["settings"][s.name] = {"agreement": cell}
    run.write_json("metrics.json", summary)
    text = markdown(summary, run.run_id, dataset, variants, records)
    (run.path / "summary.md").write_text(text)
    return text


def _pct(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, dict):
        return f"{100 * value['mean']:.1f} [{100 * value['min']:.1f}, {100 * value['max']:.1f}]"
    return f"{100 * value:.1f}"


def _num(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, dict):
        return f"{value['mean']:.2f} [{value['min']:.2f}, {value['max']:.2f}]"
    return f"{value:.2f}"


def markdown(summary, run_id, dataset, variants, records) -> str:
    kbps: dict = {}
    for r in records:
        if r.get("kbps") is not None:
            kbps.setdefault(r["setting"], []).append(r["kbps"])
    lines = [
        f"# Deep model evaluation ({dataset}): {run_id}",
        "",
        f"Videos: {summary['videos']}. Models: {', '.join(summary['models'])}. For A and B: mean "
        "over seeds [min, max]. M2 and M4: the face-only classic run on the same videos.",
        "",
    ]
    heads = " | ".join(["M2", "M4", *variants])
    rule = "|".join(["---"] * (2 + 2 + len(variants)))
    if dataset == "scamps":
        for key, title, fmt in (("success_rate", "Success rate (%)", _pct),
                                ("mae", "MAE (bpm)", _num)):  # fmt: skip
            lines += [f"## {title}", "", f"| Setting | Median kbps | {heads} |", f"|{rule}|"]
            for name, s in summary["settings"].items():
                m = s["metrics"]
                rate_text = f"{np.median(kbps[name]):.0f}" if name in kbps else "–"
                cols = [fmt(m.get(c, {}).get(key)) for c in ("M2", "M4", *variants)]
                lines.append(f"| {name} | {rate_text} | {' | '.join(cols)} |")
            lines.append("")
        lines += [
            "## Paired difference in success rate (pp)",
            "",
            "Per model (seed): b minus a on the windows both measured; mean over models [min, "
            "max], and how many models have a 95% CI above / below zero.",
            "",
            "| Setting | Comparison | Δ success rate (pp) | CI above / below zero |",
            "|---|---|---|---|",
        ]
        for name, s in summary["settings"].items():
            for comp, d in s["paired_success_rate"].items():
                text = _pct(d["delta"])
                lines.append(
                    f"| {name} | {comp} | {text} | {d['significant_up']} / {d['significant_down']} "
                    f"of {d['models']} |"
                )
    else:
        lines += [
            "## Agreement with the uncompressed estimate (%)",
            "",
            "Windows whose estimate is within the tolerance of the same measurement's estimate "
            "on the uncompressed video (no reference needed).",
            "",
            f"| Setting | Median kbps | {heads} |",
            f"|{rule}|",
        ]
        for name, s in summary["settings"].items():
            a = s["agreement"]
            rate_text = f"{np.median(kbps[name]):.0f}" if name in kbps else "–"
            cols = [_pct(a.get(c)) for c in ("M2", "M4", *variants)]
            lines.append(f"| {name} | {rate_text} | {' | '.join(cols)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    final = RunDir(args.results_root, args.final_run)
    classic_run = RunDir(args.results_root, args.classic_run)
    if final.saved_config() is None or classic_run.saved_config() is None:
        raise SystemExit("the final training run and the classic run must exist")
    train_cfg = final.saved_config()["train"]
    classic_saved = classic_run.saved_config()
    classic_cfg, protocol = classic_saved["experiment"], classic_saved["protocol"]
    if classic_cfg["dataset"] != args.dataset or classic_cfg.get("framing") != "face":
        raise SystemExit(f"{args.classic_run} is not a face-only {args.dataset} run")
    classic = {
        r["id"]: unpack(r)
        for r in classic_run.records()
        if r.get("status") == "ok" and "setting" in r
    }
    names = sorted(model_name(r) for r in final.records() if r["status"] == "ok")
    if not names:
        raise SystemExit(f"{args.final_run} has no trained models")
    protocol = eval_protocol(protocol, train_cfg["measure"], names)
    run = RunDir(args.results_root, args.run_id)
    config = {"dataset": args.dataset, "final_run": args.final_run,
              "classic_run": args.classic_run, "models": names, "train": train_cfg,
              "protocol": protocol}  # fmt: skip
    if args.summarize_only:
        print(summarize(run, args.dataset, classic, classic_cfg, train_cfg, protocol, names))
        return 0
    run.init(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    extra = {
        "argv": sys.argv[1:],
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    run.add_session(environment_snapshot(PROJECT_DIR, extra))
    models = load_models(final, args.checkpoint_root, device)
    try:
        if args.dataset == "scamps":
            videos = run_scamps(run, args, classic, models, train_cfg, protocol, device)
        else:
            videos = run_ubfc(run, args, classic, classic_cfg, models, train_cfg, protocol, device)
    finally:
        summarize(run, args.dataset, classic, classic_cfg, train_cfg, protocol, names)
    print(f"worked on {videos} video(s); results in {run.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
