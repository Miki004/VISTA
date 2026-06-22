#!/usr/bin/env python3
"""Measure DEEP FPS of MyPipeline: end-to-end throughput WITH the VLM captioner.

Deep FPS = (timed frames) / (wall-clock time of the full pipeline run, captioner
ON). This naturally "accounts for stride": during the run the VLM fires only on
stride-aligned frames plus caption-on-confirmation births, so the per-frame cost
is already amortised over the stride.

It also reports the VLM's share of the time and the captioning load (crops per
VLM call), since crop-captioning cost grows with the number of tracks per
frame -- the crowding dependence claimed in the paper.
Usage
-----
python deep.py \
    --video data/VISTA/test/seq01.mp4 \
    --config config/qwenyolo/cfg.yaml \
    --yolo-weights runs/yoloe11m_lp/weights/best.pt \
    --yolo-model YOLOE --category-map yoloe --warmup 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from mypipeline import (
    build_mypipeline_from_config,
    COCO_CATEGORY_MAP,
    YOLOE_CATEGORY_MAP,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CATEGORY_MAPS = {"yoloe": YOLOE_CATEGORY_MAP, "coco": COCO_CATEGORY_MAP}


def iter_frames(path: Path):
    if path.suffix.lower() in VID_EXTS:
        import cv2
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield i, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
    else:
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
        for i, f in enumerate(files):
            with Image.open(f) as im:
                yield i, im.convert("RGB")


class _TimedCaptioner:
    """Wrap the real captioner to count VLM forward passes / crops and the time
    spent inside the VLM, with CUDA synchronisation for accurate GPU timing."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0      # batched VLM forward passes (one per crop chunk)
        self.crops = 0      # total crops captioned
        self.time = 0.0     # seconds spent inside the VLM

    def reset(self):
        self.calls = self.crops = 0
        self.time = 0.0

    def __call__(self, crops):
        import torch
        self.calls += 1
        self.crops += len(crops)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.inner(crops)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time += time.perf_counter() - t0
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--config", required=True, type=str,
                    help="qwen yaml config; the captioner must be ON.")
    ap.add_argument("--yolo-weights", required=True, type=str)
    ap.add_argument("--yolo-model", default="YOLOE", choices=["YOLO", "YOLOE"])
    ap.add_argument("--category-map", default="yoloe", choices=list(CATEGORY_MAPS))
    ap.add_argument("--yolo-conf", default=None, type=float)
    ap.add_argument("--min-hits", default=3, type=int)
    ap.add_argument("--warmup", default=30, type=int,
                    help="Frames to run untimed first (model/CUDA warmup).")
    args = ap.parse_args()

    pipe = build_mypipeline_from_config(
        config_path=args.config,
        yolo_weights=args.yolo_weights,
        yolo_model=args.yolo_model,
        use_qwen=True,
        yolo_conf=args.yolo_conf,
        min_hits=args.min_hits,
        category_map=CATEGORY_MAPS[args.category_map],
    )
    if pipe.captioner is None:
        raise SystemExit("captioner is None: deep FPS needs the VLM on "
                         "(check --config and that use_qwen built it).")

    timed = _TimedCaptioner(pipe.captioner)
    pipe.captioner = timed
    pipe.reset()

    import torch
    t_start = None
    frames_timed = 0
    n_total = 0

    for idx, frame in iter_frames(args.video):
        if idx == args.warmup:                      # begin the timed region
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timed.reset()                           # count only the timed region
            t_start = time.perf_counter()
        pipe.forward(frame, idx)
        n_total += 1
        if t_start is not None:
            frames_timed += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if t_start is None:
        raise SystemExit(f"video shorter than warmup ({args.warmup}); lower --warmup.")
    elapsed = time.perf_counter() - t_start

    deep_fps = frames_timed / elapsed if elapsed else 0.0
    vlm_t = timed.time
    other_t = max(elapsed - vlm_t, 1e-9)
    shallow_equiv = frames_timed / other_t

    print(f"frames (total / timed) : {n_total} / {frames_timed}  (warmup {args.warmup})")
    print(f"wall time (timed)      : {elapsed:.2f} s")
    print(f"DEEP FPS               : {deep_fps:.2f}   ({1000*elapsed/frames_timed:.1f} ms/frame)")
    print("-" * 48)
    print(f"VLM forward passes     : {timed.calls}")
    print(f"crops captioned        : {timed.crops}"
          + (f"  ({timed.crops/timed.calls:.1f} crops/call)" if timed.calls else ""))
    print(f"VLM time               : {vlm_t:.2f} s  ({100*vlm_t/elapsed:.1f}% of wall time)")
    print(f"detect+track time      : {other_t:.2f} s  -> shallow-equivalent {shallow_equiv:.1f} FPS")


if __name__ == "__main__":
    main()