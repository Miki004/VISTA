"""Generate a submission.csv from the qwen_yolo baseline.

Runs a VistaPipeline over every video in a folder, aggregates per-track
results, and writes the CSV consumed by ``vista.eval_era``:

    video_id, track_id, frame_start, frame_end, caption

``video_id`` is the file stem (e.g. TrafficCollision_001), which matches the
CapERA ids after eval_era normalizes them. For each track the first/last frame
it appears on are recorded, and the last non-empty caption is kept.

CLI:
    python -m vista.run_submission --videos data/ERA/TrafficCollision \
        --out workspace/vista_output/submission.csv \
        --config config/qwenyolo/cfg.yaml --yolo-weights yolov8s.pt
    # quick smoke test, tracking only, first 150 frames per video:
    python -m vista.run_submission --videos VIDS --out sub.csv --no-qwen --max-frames 150

Then:
    python -m vista.eval_era --references CapERA_DATASET_train.json --submission sub.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vista.pipeline.base import VistaPipeline
from vista.annotate_video import _build_qwenyolo_from_config

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def generate_submission(
    pipeline: VistaPipeline,
    videos_dir: str | Path,
    out_csv: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int | None = None,
    progress_every: int = 30,
) -> Path:
    """Run ``pipeline`` over all videos in ``videos_dir`` and write submission.csv.

    Args:
        pipeline:      Any VistaPipeline (e.g. QwenYoloPipeline).
        videos_dir:    Folder containing the video files.
        out_csv:       Destination CSV path.
        start_frame:   First frame to process per video (inclusive).
        end_frame:     Last frame to process per video (exclusive); None = end.
        max_frames:    Cap frames processed per video (from start_frame); the
                       earlier of end_frame / max_frames wins.
        progress_every: Log progress every N frames (0 = silent).

    Returns:
        The output CSV path.
    """
    videos_dir, out_csv = Path(videos_dir), Path(out_csv)
    videos = sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise RuntimeError(f"No videos found in {videos_dir}")

    stop_at = end_frame
    if max_frames is not None:
        limit = start_frame + max_frames
        stop_at = limit if stop_at is None else min(stop_at, limit)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "track_id", "frame_start", "frame_end", "caption"])

        for video_path in videos:
            video_id = video_path.stem
            print(f"=== {video_id} ===", flush=True)

            tracks: dict[int, dict] = {}
            for result in pipeline.process_video(
                str(video_path), start_frame=start_frame, end_frame=stop_at
            ):
                if progress_every and result.frame_idx % progress_every == 0:
                    print(f"  frame {result.frame_idx}: {len(result.detections)} detections",
                          flush=True)
                for det in result.detections:
                    if det.track_id is None:
                        continue
                    rec = tracks.setdefault(
                        det.track_id,
                        {"frame_start": result.frame_idx, "frame_end": result.frame_idx,
                         "caption": None},
                    )
                    rec["frame_end"] = result.frame_idx
                    if det.caption:
                        rec["caption"] = det.caption

            for tid, rec in tracks.items():
                writer.writerow([video_id, tid, rec["frame_start"], rec["frame_end"],
                                 rec["caption"] or ""])

    print(f"Wrote {out_csv}", flush=True)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="Folder of input videos")
    parser.add_argument("--out", required=True, help="Output submission.csv path")
    parser.add_argument("--config", default=None, help="QwenYolo config yaml (for the captioner)")
    parser.add_argument("--yolo-weights", default="yolov8s.pt", help="YOLO weights")
    parser.add_argument("--caption-stride", type=int, default=30, help="Run the VLM every N frames")
    parser.add_argument("--no-qwen", action="store_true", help="Tracking only, skip the captioner")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames processed per video (from start-frame)")
    args = parser.parse_args()

    pipeline = _build_qwenyolo_from_config(
        config_path=args.config,
        yolo_weights=args.yolo_weights,
        caption_stride=args.caption_stride,
        use_qwen=not args.no_qwen,
    )
    generate_submission(
        pipeline, args.videos, args.out,
        start_frame=args.start_frame, end_frame=args.end_frame, max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
