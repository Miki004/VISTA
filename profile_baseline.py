"""Profile the QwenYoloPipeline baseline before submitting.

Measures, without needing ground-truth annotations:
  - End-to-end FPS (deep FPS) per video and overall.
  - Shallow FPS estimate (YOLO + decode + post, no Qwen).
  - Per-stage latency (median / p95) for YOLO, Qwen, frame decode, total forward.
  - Tracking stats: number of tracks, length distribution, per-category counts.
  - Caption stability (IDF1 proxy): flip rate per track, unique captions/track,
    tracks that never received a caption.
  - Detection stats: mean detections per frame, frames with zero detections.

Outputs to <output>/:
  profile.json             — all numeric stats (machine-readable)
  profile_report.md        — human-readable summary tables
  caption_histories.json   — per-track (frame_idx, caption) sequences

Usage:
  python profile_baseline.py --config config/qwenyolo/cfg10.yaml
  python profile_baseline.py --config <cfg> --videos-dir <dir> --max-videos 3
  python profile_baseline.py --config <cfg> --max-frames 300        # quick run
  python profile_baseline.py --config <cfg> --no-qwen               # YOLO only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import cv2
import yaml
from PIL import Image
from ultralytics import YOLO


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ── timing utilities ──────────────────────────────────────────────────────────

class Timings:
    def __init__(self):
        self.stages: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def time(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[stage].append(time.perf_counter() - t0)

    def summary(self, stage: str) -> dict | None:
        ts = self.stages.get(stage, [])
        if not ts:
            return None
        ts_ms = sorted(t * 1000 for t in ts)
        n = len(ts_ms)
        idx95 = min(n - 1, int(n * 0.95))
        return {
            "n": n,
            "total_s": round(sum(ts), 3),
            "mean_ms": round(statistics.fmean(ts_ms), 2),
            "median_ms": round(statistics.median(ts_ms), 2),
            "p95_ms": round(ts_ms[idx95], 2),
            "min_ms": round(min(ts_ms), 2),
            "max_ms": round(max(ts_ms), 2),
        }


def _wrap_method(obj, method_name: str, timings: Timings, stage: str) -> None:
    """Replace obj.method with a timed version. Idempotent if called once."""
    original = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        with timings.time(stage):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)


# ── pipeline construction ────────────────────────────────────────────────────

def build_pipeline(cfg: dict, use_qwen: bool):
    """Construct a QwenYoloPipeline (Qwen optional)."""
    from vista.pipeline.qwen_yolo import QwenYoloPipeline

    print(f"[INFO] Loading YOLO: {cfg['yolo']['model']}")
    yolo_model = YOLO(cfg["yolo"]["model"])

    qwen = None
    if use_qwen:
        from vista.qwen import get_model
        print(f"[INFO] Loading Qwen: {cfg['qwen']['model_id']}")
        qwen = get_model(cfg)

    pipeline = QwenYoloPipeline(
        yolo_model=yolo_model,
        qwen_model=qwen,
        caption_stride=cfg["qwen"].get("every_n_frames", 30),
        iou_threshold=cfg["yolo"].get("iou_match_threshold", 0.3),
        yolo_conf=cfg["yolo"].get("conf", None),
    )
    return pipeline


# ── caption stability ────────────────────────────────────────────────────────

def caption_stats(caption_history: dict[int, list[tuple[int, str]]]) -> dict:
    """Per-track caption stability — proxy for IDF1.

    A 'flip' is any frame where the caption changed vs. the previous frame
    that had a caption.  Lower flip rate ≈ more consistent identity captions.
    """
    flip_rates: list[float] = []
    unique_counts: list[int] = []
    stable = 0  # tracks with exactly one unique caption across all updates
    most_common: dict[int, str] = {}

    for tid, hist in caption_history.items():
        captions = [c for _, c in hist]
        if not captions:
            continue
        if len(set(captions)) == 1:
            stable += 1
        most_common[tid] = Counter(captions).most_common(1)[0][0]
        if len(captions) >= 2:
            flips = sum(1 for i in range(1, len(captions)) if captions[i] != captions[i - 1])
            flip_rates.append(flips / (len(captions) - 1))
            unique_counts.append(len(set(captions)))

    return {
        "tracks_with_caption": len(caption_history),
        "stable_tracks": stable,
        "caption_flip_rate_mean": round(statistics.fmean(flip_rates), 3) if flip_rates else 0.0,
        "unique_captions_per_track_mean": round(statistics.fmean(unique_counts), 2) if unique_counts else 0.0,
        "most_common_per_track": {str(k): v for k, v in most_common.items()},
    }


# ── per-video profiling ──────────────────────────────────────────────────────

def profile_video(pipeline, video_path: Path, max_frames: int | None) -> tuple[dict, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pipeline.reset()
    timings = Timings()
    _wrap_method(pipeline.yolo, "track", timings, "yolo_track")
    if pipeline.qwen is not None:
        _wrap_method(pipeline.qwen, "generate", timings, "qwen_generate")

    caption_history: dict[int, list[tuple[int, str]]] = defaultdict(list)
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    category_counts: Counter = Counter()
    detection_counts: list[int] = []

    t_start = time.perf_counter()
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, bgr = cap.read()
        if not ret:
            break

        with timings.time("forward_total"):
            with timings.time("frame_decode"):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(rgb)
            result = pipeline(frame, frame_idx)

        detection_counts.append(len(result.detections))
        for det in result.detections:
            if det.track_id is None:
                continue
            first_seen.setdefault(det.track_id, frame_idx)
            last_seen[det.track_id] = frame_idx
            category_counts[det.category] += 1
            if det.caption:
                caption_history[det.track_id].append((frame_idx, det.caption))

        frame_idx += 1

    wall = time.perf_counter() - t_start
    cap.release()

    track_lengths = {tid: last_seen[tid] - first_seen[tid] + 1 for tid in first_seen}

    yolo_sum = timings.summary("yolo_track")
    qwen_sum = timings.summary("qwen_generate")
    decode_sum = timings.summary("frame_decode")
    fwd_sum = timings.summary("forward_total")

    # Shallow FPS estimate: time of a forward call where Qwen did NOT fire.
    # forward_total = yolo + decode + (matching + db update).  Subtract Qwen.
    shallow_ms = None
    if yolo_sum and decode_sum:
        shallow_ms = yolo_sum["median_ms"] + decode_sum["median_ms"] + 5.0  # +5ms post-fudge

    return {
        "video": str(video_path),
        "video_fps": round(video_fps, 2),
        "video_frames_meta": n_frames_meta,
        "frames_processed": frame_idx,
        "wall_time_s": round(wall, 2),
        "end_to_end_fps": round(frame_idx / wall, 2) if wall > 0 else None,
        "shallow_fps_est": round(1000 / shallow_ms, 2) if shallow_ms else None,
        "stages": {
            "yolo_track": yolo_sum,
            "qwen_generate": qwen_sum,
            "frame_decode": decode_sum,
            "forward_total": fwd_sum,
        },
        "tracking": {
            "n_tracks": len(track_lengths),
            "track_length_mean": round(statistics.fmean(track_lengths.values()), 1) if track_lengths else 0,
            "track_length_median": int(statistics.median(track_lengths.values())) if track_lengths else 0,
            "track_length_max": max(track_lengths.values()) if track_lengths else 0,
            "category_counts": dict(category_counts),
        },
        "captions": caption_stats(caption_history),
        "detections": {
            "mean_per_frame": round(statistics.fmean(detection_counts), 2) if detection_counts else 0,
            "frames_with_zero_detections": sum(1 for c in detection_counts if c == 0),
        },
    }, {str(k): v for k, v in caption_history.items()}


# ── reporting ────────────────────────────────────────────────────────────────

def _stage_md(stage: dict | None, key: str) -> str:
    return str(stage[key]) if stage else "—"


def write_report(profile: dict, out_path: Path) -> None:
    lines = ["# Baseline Profile Report", ""]
    lines.append(f"- Config: `{profile['config']}`")
    lines.append(f"- Device: `{profile['device']}`")
    lines.append(f"- YOLO model: `{profile['yolo_model']}`")
    qwen_id = profile['qwen_model'] or "(disabled)"
    lines.append(f"- Qwen model: `{qwen_id}`  (stride={profile['caption_stride']})")
    lines.append("")

    lines.append("## Throughput")
    lines.append("")
    lines.append("| Video | Frames | Wall (s) | End-to-end FPS | Shallow FPS est | ≥ 2 FPS? |")
    lines.append("|-------|--------|----------|----------------|-----------------|----------|")
    for v in profile["videos"]:
        ok = "✓" if (v.get("end_to_end_fps") or 0) >= 2.0 else "✗"
        lines.append(
            f"| {Path(v['video']).name} | {v['frames_processed']} | {v['wall_time_s']} "
            f"| {v.get('end_to_end_fps')} | {v.get('shallow_fps_est')} | {ok} |"
        )
    lines.append("")

    lines.append("## Per-stage latency (median ms)")
    lines.append("")
    lines.append("| Video | YOLO | Qwen | Decode | Forward total |")
    lines.append("|-------|------|------|--------|---------------|")
    for v in profile["videos"]:
        s = v["stages"]
        lines.append(
            f"| {Path(v['video']).name} "
            f"| {_stage_md(s.get('yolo_track'), 'median_ms')} "
            f"| {_stage_md(s.get('qwen_generate'), 'median_ms')} "
            f"| {_stage_md(s.get('frame_decode'), 'median_ms')} "
            f"| {_stage_md(s.get('forward_total'), 'median_ms')} |"
        )
    lines.append("")

    lines.append("## Per-stage latency (p95 ms)")
    lines.append("")
    lines.append("| Video | YOLO | Qwen | Forward total |")
    lines.append("|-------|------|------|---------------|")
    for v in profile["videos"]:
        s = v["stages"]
        lines.append(
            f"| {Path(v['video']).name} "
            f"| {_stage_md(s.get('yolo_track'), 'p95_ms')} "
            f"| {_stage_md(s.get('qwen_generate'), 'p95_ms')} "
            f"| {_stage_md(s.get('forward_total'), 'p95_ms')} |"
        )
    lines.append("")

    lines.append("## Tracking")
    lines.append("")
    lines.append("| Video | Tracks | Mean len | Median len | Max len | Categories |")
    lines.append("|-------|--------|----------|------------|---------|------------|")
    for v in profile["videos"]:
        t = v["tracking"]
        cats = ", ".join(f"{k}={vv}" for k, vv in t.get("category_counts", {}).items())
        lines.append(
            f"| {Path(v['video']).name} | {t['n_tracks']} | "
            f"{t['track_length_mean']} | {t['track_length_median']} | "
            f"{t['track_length_max']} | {cats} |"
        )
    lines.append("")

    lines.append("## Caption stability (IDF1 proxy)")
    lines.append("")
    lines.append("Lower flip rate and lower unique-captions-per-track ≈ more consistent identity captions.")
    lines.append("")
    lines.append("| Video | Tracks w/ caption | Stable (no flip) | Flip rate | Unique/track |")
    lines.append("|-------|-------------------|------------------|-----------|--------------|")
    for v in profile["videos"]:
        c = v["captions"]
        lines.append(
            f"| {Path(v['video']).name} | {c['tracks_with_caption']} | "
            f"{c['stable_tracks']} | {c['caption_flip_rate_mean']} | "
            f"{c['unique_captions_per_track_mean']} |"
        )
    lines.append("")

    lines.append("## Detections")
    lines.append("")
    lines.append("| Video | Mean/frame | Frames with 0 detections |")
    lines.append("|-------|------------|--------------------------|")
    for v in profile["videos"]:
        d = v["detections"]
        lines.append(
            f"| {Path(v['video']).name} | {d['mean_per_frame']} | "
            f"{d['frames_with_zero_detections']} |"
        )
    lines.append("")

    agg = profile["aggregate"]
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Total videos: {agg['total_videos']}")
    lines.append(f"- Total frames: {agg['total_frames']}")
    lines.append(f"- Total wall time: {agg['total_wall_time_s']} s")
    lines.append(f"- Overall end-to-end FPS: **{agg['overall_fps']}**")
    if agg.get("overall_fps") is not None:
        meets = "✓ meets" if agg["overall_fps"] >= 2.0 else "✗ BELOW"
        lines.append(f"- Challenge two-stage minimum (≥ 2 FPS): {meets}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────────

def _collect_videos(args, cfg: dict) -> list[Path]:
    if args.video:
        return [Path(args.video)]
    if args.videos_dir:
        d = Path(args.videos_dir)
        return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    cfg_video = cfg.get("input", {}).get("video")
    if cfg_video:
        return [Path(cfg_video)]
    return []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config (same shape as config/qwenyolo/*.yaml)")
    p.add_argument("--videos-dir", help="Directory of videos to profile")
    p.add_argument("--video", help="Single video file (overrides --videos-dir and config)")
    p.add_argument("--output", default="out/profile", help="Output directory")
    p.add_argument("--max-videos", type=int, default=None, help="Cap number of videos")
    p.add_argument("--max-frames", type=int, default=None, help="Per-video frame cap (quick smoke run)")
    p.add_argument("--no-qwen", action="store_true", help="Skip Qwen, profile YOLO + decode only")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    videos = _collect_videos(args, cfg)
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        print("[ERROR] No videos found. Set --video / --videos-dir or `input.video` in the config.",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(cfg, use_qwen=not args.no_qwen)
    print(f"[INFO] Profiling {len(videos)} video(s) → {out_dir}")

    video_results: list[dict] = []
    all_histories: dict[str, dict] = {}
    for i, vp in enumerate(videos):
        print(f"\n[{i + 1}/{len(videos)}] {vp.name}")
        try:
            result, history = profile_video(pipeline, vp, max_frames=args.max_frames)
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            continue
        video_results.append(result)
        all_histories[vp.name] = history
        print(
            f"  → {result['frames_processed']} frames in {result['wall_time_s']}s "
            f"= {result['end_to_end_fps']} FPS"
        )
        s = result["stages"]
        if s.get("yolo_track"):
            print(f"     YOLO median {s['yolo_track']['median_ms']}ms, p95 {s['yolo_track']['p95_ms']}ms")
        if s.get("qwen_generate"):
            print(f"     Qwen median {s['qwen_generate']['median_ms']}ms, p95 {s['qwen_generate']['p95_ms']}ms")
        t = result["tracking"]
        c = result["captions"]
        print(
            f"     tracks={t['n_tracks']}  with-caption={c['tracks_with_caption']}  "
            f"flip-rate={c['caption_flip_rate_mean']}"
        )

    total_frames = sum(v["frames_processed"] for v in video_results)
    total_wall = sum(v["wall_time_s"] for v in video_results)
    profile = {
        "config": str(args.config),
        "device": cfg.get("device", "unknown"),
        "yolo_model": cfg["yolo"]["model"],
        "qwen_model": (cfg["qwen"]["model_id"] if not args.no_qwen else None),
        "caption_stride": cfg["qwen"].get("every_n_frames", 30),
        "no_qwen": args.no_qwen,
        "videos": video_results,
        "aggregate": {
            "total_videos": len(video_results),
            "total_frames": total_frames,
            "total_wall_time_s": round(total_wall, 2),
            "overall_fps": round(total_frames / total_wall, 2) if total_wall > 0 else None,
        },
    }

    (out_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    write_report(profile, out_dir / "profile_report.md")
    (out_dir / "caption_histories.json").write_text(
        json.dumps(all_histories, indent=2), encoding="utf-8"
    )

    print("\n=== Summary ===")
    print(f"  videos:      {len(video_results)}")
    print(f"  frames:      {total_frames}")
    print(f"  wall time:   {total_wall:.1f}s")
    if total_wall > 0:
        fps = total_frames / total_wall
        print(f"  overall FPS: {fps:.2f}   ({'OK ≥2' if fps >= 2.0 else 'BELOW 2 FPS minimum'})")
    print(f"\n  report:      {out_dir / 'profile_report.md'}")
    print(f"  json:        {out_dir / 'profile.json'}")
    print(f"  histories:   {out_dir / 'caption_histories.json'}")


if __name__ == "__main__":
    main()
