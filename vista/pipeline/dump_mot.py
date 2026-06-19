#!/usr/bin/env python3
"""Run MyPipeline over VISTA video sequences and dump predictions_mot.csv.

Output: one row per detection per frame, columns in this exact order:
    video_id, frame_id, track_id, x1, y1, x2, y2, conf, category

These are the boxes + identities needed to compute mAP@50, MOTA and IDF1.
Captions are NOT part of this CSV, so the VLM captioner is disabled by default
(much faster). Pass --use-qwen if you want the caption-derived category
remapping (e.g. car -> emergency_vehicle via LABEL_TO_CATEGORY) to take effect.

Usage examples
--------------
# directory of .mp4 sequences, fine-tuned YOLOe weights
python dump_predictions_mot.py \
    --videos-dir data/VISTA/test \
    --yolo-weights runs/yoloe11m_lp/weights/best.pt \
    --yolo-model YOLOE \
    --category-map yoloe \
    --out predictions_mot.csv

# directory whose subfolders each contain ordered frame images
python dump_predictions_mot.py \
    --videos-dir data/VISTA/test_frames \
    --yolo-weights runs/yoloe11m_lp/weights/best.pt \
    --yolo-model YOLOE --category-map yoloe
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from PIL import Image

from mypipeline import (
    build_mypipeline_from_config,
    COCO_CATEGORY_MAP,
    YOLOE_CATEGORY_MAP,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

CATEGORY_MAPS = {
    "yoloe": YOLOE_CATEGORY_MAP,   # crashed car / car / person
    "coco":  COCO_CATEGORY_MAP,    # car,truck,bus,motorcycle / person
}


# ── frame iterators ──────────────────────────────────────────────────────────

def iter_frames_from_video(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yield idx, Image.fromarray(rgb)
        idx += 1
    cap.release()


def iter_frames_from_dir(path: Path):
    files = sorted(
        (p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS),
        key=lambda p: p.name,
    )
    for idx, f in enumerate(files):
        with Image.open(f) as im:
            yield idx, im.convert("RGB")


def discover_sequences(root: Path):
    """Return a list of (video_id, kind, path) tuples.

    Supports three layouts:
      1) root/ contains video files                -> one sequence per video
      2) root/<seq>/ contains ordered frame images -> one sequence per subdir
      3) root/ itself contains ordered frame images -> a single sequence
    """
    vids = sorted(p for p in root.iterdir() if p.suffix.lower() in VID_EXTS)
    if vids:
        return [(v.stem, "video", v) for v in vids]

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    seqs = []
    for d in subdirs:
        if any(p.suffix.lower() in IMG_EXTS for p in d.iterdir()):
            seqs.append((d.name, "frames", d))
    if seqs:
        return seqs

    if any(p.suffix.lower() in IMG_EXTS for p in root.iterdir()):
        return [(root.name, "frames", root)]

    raise SystemExit(f"No videos or frame sequences found under {root}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True, type=Path,
                    help="Root with video files OR per-sequence frame subdirs.")
    ap.add_argument("--yolo-weights", required=True, type=str,
                    help="Ultralytics weights (.pt) for the detector/tracker.")
    ap.add_argument("--yolo-model", default="YOLOE", choices=["YOLO", "YOLOE"])
    ap.add_argument("--category-map", default="yoloe", choices=list(CATEGORY_MAPS))
    ap.add_argument("--config", default="", type=str,
                    help="YAML config (only needed with --use-qwen).")
    ap.add_argument("--use-qwen", action="store_true",
                    help="Enable the VLM captioner (slow; not needed for MOT metrics).")
    ap.add_argument("--yolo-conf", default=None, type=float,
                    help="Detector confidence threshold (None = Ultralytics default).")
    ap.add_argument("--min-hits", default=3, type=int)
    ap.add_argument("--caption-stride", default=30, type=int)
    ap.add_argument("--out", default="predictions_mot.csv", type=Path)
    ap.add_argument("--coord-decimals", default=2, type=int)
    ap.add_argument("--conf-decimals", default=4, type=int)
    args = ap.parse_args()

    if args.use_qwen and not args.config:
        ap.error("--use-qwen requires --config (path to the qwen yaml).")

    pipeline = build_mypipeline_from_config(
        config_path=args.config or None,
        yolo_weights=args.yolo_weights,
        yolo_model=args.yolo_model,
        use_qwen=args.use_qwen,
        yolo_conf=args.yolo_conf,
        min_hits=args.min_hits,
        caption_stride=args.caption_stride,
        category_map=CATEGORY_MAPS[args.category_map],
    )
    print(f"[pipeline] {pipeline!r}")

    sequences = discover_sequences(args.videos_dir)
    print(f"[discover] {len(sequences)} sequence(s): "
          f"{', '.join(v for v, _, _ in sequences)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cd, fd = args.coord_decimals, args.conf_decimals
    n_rows = 0
    t0 = time.time()

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["video_id", "frame_id", "track_id", "x1", "y1", "x2", "y2",
             "conf", "category"]
        )

        for video_id, kind, path in sequences:
            pipeline.reset()  # harness contract: reset before each sequence
            it = (iter_frames_from_video(path) if kind == "video"
                  else iter_frames_from_dir(path))

            seq_rows, seq_frames = 0, 0
            for frame_idx, frame in it:
                result = pipeline.forward(frame, frame_idx)
                for det in result.detections:
                    x1, y1, x2, y2 = det.bbox
                    writer.writerow([
                        video_id,
                        frame_idx,
                        det.track_id,
                        round(float(x1), cd),
                        round(float(y1), cd),
                        round(float(x2), cd),
                        round(float(y2), cd),
                        round(float(det.confidence), fd),
                        det.category,
                    ])
                    seq_rows += 1
                seq_frames += 1
            fh.flush()
            n_rows += seq_rows
            print(f"  [{video_id}] {seq_frames} frames -> {seq_rows} detections")

    dt = time.time() - t0
    print(f"[done] {n_rows} rows -> {args.out}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()